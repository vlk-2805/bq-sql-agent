import os
import re
from typing import Any, Dict, List, Optional
import dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END

from app.state import SQLAgentState
from app.bq_tools import fetch_dataset_schema, validate_query_dry_run, execute_safe_query

# Load environment variables
dotenv.load_dotenv()

DEFAULT_DATASET = "bigquery-public-data.thelook_ecommerce"
DEFAULT_TABLES = ["orders", "order_items", "products", "users"]


def extract_text_content(content: Any) -> str:
    """Extracts raw text from LLM response content (str or list of content blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(item.get("text", ""))
            elif hasattr(item, "text"):
                parts.append(getattr(item, "text", ""))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content)


def clean_sql(text: str) -> str:
    """Removes Markdown code block ticks and leading/trailing whitespace."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def get_llm():
    """Initializes ChatGoogleGenerativeAI without Vertex AI."""
    api_key = os.getenv("GOOGLE_API_KEY")
    return ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        temperature=0.0,
        google_api_key=api_key
    )


# Node 1: Inspect Schema
def inspect_schema(state: SQLAgentState) -> Dict[str, Any]:
    dataset_id = state.get("dataset_id") or DEFAULT_DATASET
    schema_context = state.get("schema_context")

    if not schema_context:
        schema_context = fetch_dataset_schema(dataset_id=dataset_id, tables=DEFAULT_TABLES)

    return {
        "dataset_id": dataset_id,
        "schema_context": schema_context,
        "retry_count": state.get("retry_count", 0),
        "validation_error": state.get("validation_error"),
        "estimated_bytes": state.get("estimated_bytes"),
        "generated_sql": state.get("generated_sql"),
        "query_results": state.get("query_results"),
        "final_answer": state.get("final_answer")
    }


# Node 2: Generate SQL
def generate_sql(state: SQLAgentState) -> Dict[str, Any]:
    llm = get_llm()
    question = state["question"]
    dataset_id = state["dataset_id"]
    schema = state.get("schema_context", "")
    retry_count = state.get("retry_count", 0)
    validation_error = state.get("validation_error")
    prev_sql = state.get("generated_sql")

    system_prompt = (
        "You are an expert Google BigQuery data analyst. Your job is to convert natural language "
        "questions into accurate, optimal GoogleSQL queries. Only output the raw SQL query without "
        "any markdown fences, comments, or explanations. Only SELECT statements are permitted."
    )

    if validation_error and retry_count > 0:
        user_prompt = f"""
We previously generated this SQL query for dataset `{dataset_id}`:
```sql
{prev_sql}
```

However, it failed validation with this error:
{validation_error}

Schema Context:
{schema}

User Question:
{question}

Please correct the SQL query to resolve the validation error while faithfully answering the user question.
Return ONLY the raw SQL query.
"""
    else:
        user_prompt = f"""
Schema Context for `{dataset_id}`:
{schema}

User Question:
{question}

Write a standard GoogleSQL query to answer this question.
Return ONLY the raw SQL query.
"""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])

    raw_sql = extract_text_content(response.content)
    clean_query = clean_sql(raw_sql)

    return {
        "generated_sql": clean_query
    }


# Node 3: Dry Run Validate
def dry_run_validate(state: SQLAgentState) -> Dict[str, Any]:
    sql = state.get("generated_sql", "")
    validation = validate_query_dry_run(sql)

    if validation["valid"]:
        return {
            "validation_error": None,
            "estimated_bytes": validation["estimated_bytes"]
        }
    else:
        current_retries = state.get("retry_count", 0)
        return {
            "validation_error": validation["error"],
            "estimated_bytes": validation["estimated_bytes"],
            "retry_count": current_retries + 1
        }


# Conditional Edge Router
def route_after_validation(state: SQLAgentState) -> str:
    """
    Routes flow:
    - If valid -> execute_sql
    - If invalid and retry_count < 3 -> generate_sql
    - If invalid and retry_count >= 3 -> summarize_answer
    """
    if not state.get("validation_error"):
        return "execute_sql"
    if state.get("retry_count", 0) < 3:
        return "generate_sql"
    return "summarize_answer"


# Node 4: Execute SQL
def execute_sql(state: SQLAgentState) -> Dict[str, Any]:
    sql = state.get("generated_sql", "")
    results = execute_safe_query(sql)
    return {
        "query_results": results
    }


# Node 5: Summarize Answer
def summarize_answer(state: SQLAgentState) -> Dict[str, Any]:
    question = state["question"]
    validation_error = state.get("validation_error")
    retry_count = state.get("retry_count", 0)
    sql = state.get("generated_sql", "")
    results = state.get("query_results")

    # If all retries failed
    if validation_error and retry_count >= 3:
        final_msg = (
            f"Unable to execute a valid query after {retry_count} attempts. "
            f"Last validation error encountered: {validation_error}"
        )
        return {"final_answer": final_msg}

    llm = get_llm()
    system_prompt = (
        "You are an insightful data analyst. Synthesize the provided query results "
        "into a clear, professional, natural language answer that directly addresses the user's question."
    )
    user_prompt = f"""
User Question:
{question}

Executed SQL Query:
{sql}

Query Results (first {len(results or [])} rows):
{results}

Provide a concise, direct, and helpful answer for business stakeholders.
"""
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])
    final_text = extract_text_content(response.content)
    return {"final_answer": final_text.strip()}


# Build LangGraph StateGraph
def build_graph():
    graph_builder = StateGraph(SQLAgentState)

    # Add Nodes
    graph_builder.add_node("inspect_schema", inspect_schema)
    graph_builder.add_node("generate_sql", generate_sql)
    graph_builder.add_node("dry_run_validate", dry_run_validate)
    graph_builder.add_node("execute_sql", execute_sql)
    graph_builder.add_node("summarize_answer", summarize_answer)

    # Add Edges
    graph_builder.set_entry_point("inspect_schema")
    graph_builder.add_edge("inspect_schema", "generate_sql")
    graph_builder.add_edge("generate_sql", "dry_run_validate")

    # Conditional routing after dry run validation
    graph_builder.add_conditional_edges(
        "dry_run_validate",
        route_after_validation,
        {
            "execute_sql": "execute_sql",
            "generate_sql": "generate_sql",
            "summarize_answer": "summarize_answer"
        }
    )

    graph_builder.add_edge("execute_sql", "summarize_answer")
    graph_builder.add_edge("summarize_answer", END)

    return graph_builder.compile()


sql_agent_graph = build_graph()
