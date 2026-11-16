import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import AIMessage

from app.bq_tools import (
    validate_query_dry_run,
    execute_safe_query,
    MAX_BYTES_BILLED
)
from app.agent import (
    route_after_validation,
    dry_run_validate,
    build_graph
)
from app.state import SQLAgentState


class TestSQLGuardrails:
    """Tests for validating SQL queries against mutations and byte scan limits."""

    @pytest.mark.parametrize("mutating_sql,keyword", [
        ("DROP TABLE `bigquery-public-data.thelook_ecommerce.orders`", "DROP"),
        ("DELETE FROM `bigquery-public-data.thelook_ecommerce.orders` WHERE id = 10", "DELETE"),
        ("UPDATE `bigquery-public-data.thelook_ecommerce.orders` SET status = 'Complete' WHERE id = 1", "UPDATE"),
        ("INSERT INTO `bigquery-public-data.thelook_ecommerce.users` (id) VALUES (999)", "INSERT"),
        ("TRUNCATE TABLE `bigquery-public-data.thelook_ecommerce.products`", "TRUNCATE"),
        ("ALTER TABLE `bigquery-public-data.thelook_ecommerce.orders` DROP COLUMN num_of_item", "ALTER"),
        ("MERGE dataset.target T USING dataset.source S ON T.id = S.id WHEN MATCHED THEN UPDATE SET T.val = S.val", "MERGE"),
    ])
    def test_mutating_sql_blocked_immediately(self, mutating_sql, keyword):
        """Verify that any DDL/DML mutation is rejected without contacting BigQuery."""
        result = validate_query_dry_run(mutating_sql)
        assert result["valid"] is False
        assert f"'{keyword}'" in result["error"] or keyword in result["error"]
        assert result["estimated_bytes"] == 0

    def test_bytes_exceeded_fails_validation(self):
        """Verify that a query exceeding MAX_BYTES_BILLED (1GB) fails validation."""
        mock_client = MagicMock()
        mock_job = MagicMock()
        # 2 GB > 1 GB limit
        mock_job.total_bytes_processed = 2 * 1024 * 1024 * 1024
        mock_client.query.return_value = mock_job

        safe_select = "SELECT * FROM `bigquery-public-data.thelook_ecommerce.orders`"
        result = validate_query_dry_run(safe_select, client=mock_client)

        assert result["valid"] is False
        assert "exceeds maximum allowed" in result["error"]
        assert result["estimated_bytes"] == mock_job.total_bytes_processed

    def test_valid_dry_run_passes(self):
        """Verify that a valid SELECT query within byte limit passes validation."""
        mock_client = MagicMock()
        mock_job = MagicMock()
        # 50 MB
        mock_job.total_bytes_processed = 50 * 1024 * 1024
        mock_client.query.return_value = mock_job

        safe_select = "SELECT id, status FROM `bigquery-public-data.thelook_ecommerce.orders` LIMIT 10"
        result = validate_query_dry_run(safe_select, client=mock_client)

        assert result["valid"] is True
        assert result["error"] is None
        assert result["estimated_bytes"] == 50 * 1024 * 1024


class TestLangGraphRoutingAndLoop:
    """Tests LangGraph conditional edge routing and self-correction retry loop."""

    def test_route_to_execute_sql_when_valid(self):
        """When validation_error is None, edge routes to execute_sql."""
        state: SQLAgentState = {
            "question": "Show top orders",
            "dataset_id": "thelook",
            "schema_context": None,
            "generated_sql": "SELECT * FROM orders",
            "validation_error": None,
            "estimated_bytes": 1000,
            "retry_count": 0,
            "query_results": None,
            "final_answer": None
        }
        next_node = route_after_validation(state)
        assert next_node == "execute_sql"

    def test_route_to_generate_sql_when_retry_less_than_3(self):
        """When validation_error is present and retry_count < 3, edge loops back to generate_sql."""
        for retries in (1, 2):
            state: SQLAgentState = {
                "question": "Show top orders",
                "dataset_id": "thelook",
                "schema_context": None,
                "generated_sql": "SELECT * FROM orders",
                "validation_error": "Syntax error in query",
                "estimated_bytes": 0,
                "retry_count": retries,
                "query_results": None,
                "final_answer": None
            }
            next_node = route_after_validation(state)
            assert next_node == "generate_sql"

    def test_route_to_summarize_answer_when_retry_3_or_more(self):
        """When validation_error is present and retry_count >= 3, edge terminates at summarize_answer."""
        for retries in (3, 4):
            state: SQLAgentState = {
                "question": "Show top orders",
                "dataset_id": "thelook",
                "schema_context": None,
                "generated_sql": "SELECT * FROM orders",
                "validation_error": "Syntax error",
                "estimated_bytes": 0,
                "retry_count": retries,
                "query_results": None,
                "final_answer": None
            }
            next_node = route_after_validation(state)
            assert next_node == "summarize_answer"

    @patch("app.agent.get_llm")
    @patch("app.agent.execute_safe_query")
    @patch("app.agent.validate_query_dry_run")
    @patch("app.agent.fetch_dataset_schema")
    def test_end_to_end_graph_successful_path(
        self, mock_fetch_schema, mock_validate, mock_execute, mock_get_llm
    ):
        """Simulates full LangGraph pipeline on valid query execution."""
        mock_fetch_schema.return_value = "Table: `orders` Columns: id INT64, status STRING"
        mock_validate.return_value = {"valid": True, "error": None, "estimated_bytes": 1024}
        mock_execute.return_value = [{"category": "Intimates", "revenue": 150000.0}]

        mock_llm_instance = MagicMock()
        # First call is generate_sql, second call is summarize_answer
        mock_llm_instance.invoke.side_effect = [
            AIMessage(content="SELECT category, SUM(sale_price) as revenue FROM order_items GROUP BY category"),
            AIMessage(content="The top product category by revenue is Intimates with $150,000.")
        ]
        mock_get_llm.return_value = mock_llm_instance

        graph = build_graph()
        initial_state: SQLAgentState = {
            "question": "Top category by revenue?",
            "dataset_id": "bigquery-public-data.thelook_ecommerce",
            "schema_context": None,
            "generated_sql": None,
            "validation_error": None,
            "estimated_bytes": None,
            "retry_count": 0,
            "query_results": None,
            "final_answer": None
        }

        result = graph.invoke(initial_state)

        assert result["generated_sql"] is not None
        assert result["validation_error"] is None
        assert result["query_results"] == [{"category": "Intimates", "revenue": 150000.0}]
        assert "Intimates" in result["final_answer"]
        assert mock_execute.called

    @patch("app.agent.get_llm")
    @patch("app.agent.execute_safe_query")
    @patch("app.agent.validate_query_dry_run")
    @patch("app.agent.fetch_dataset_schema")
    def test_self_correction_retry_loop_recovers(
        self, mock_fetch_schema, mock_validate, mock_execute, mock_get_llm
    ):
        """Simulates self-correction: first dry run fails, retry fixes SQL and succeeds."""
        mock_fetch_schema.return_value = "Table: `orders`"

        # Validation sequence: first attempt fails, second attempt succeeds
        mock_validate.side_effect = [
            {"valid": False, "error": "Unrecognized name: rev at [1:8]", "estimated_bytes": 0},
            {"valid": True, "error": None, "estimated_bytes": 2048}
        ]
        mock_execute.return_value = [{"revenue": 5000}]

        mock_llm_instance = MagicMock()
        # LLM calls:
        # 1. generate_sql (bad sql)
        # 2. generate_sql retry (fixed sql)
        # 3. summarize_answer
        mock_llm_instance.invoke.side_effect = [
            AIMessage(content="SELECT rev FROM orders"),
            AIMessage(content="SELECT revenue FROM orders"),
            AIMessage(content="Total revenue is $5,000.")
        ]
        mock_get_llm.return_value = mock_llm_instance

        graph = build_graph()
        initial_state: SQLAgentState = {
            "question": "Total revenue?",
            "dataset_id": "bigquery-public-data.thelook_ecommerce",
            "schema_context": None,
            "generated_sql": None,
            "validation_error": None,
            "estimated_bytes": None,
            "retry_count": 0,
            "query_results": None,
            "final_answer": None
        }

        result = graph.invoke(initial_state)

        # Confirm it corrected and succeeded
        assert result["generated_sql"] == "SELECT revenue FROM orders"
        assert result["validation_error"] is None
        assert result["retry_count"] == 1
        assert result["query_results"] == [{"revenue": 5000}]
        assert "5,000" in result["final_answer"]
        assert mock_llm_instance.invoke.call_count == 3
