import os
import re
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional
from google.cloud import bigquery
from google.api_core.exceptions import GoogleAPICallError

# 1 GB default maximum billed bytes
MAX_BYTES_BILLED = 1073741824

# Disallowed mutating SQL keywords
MUTATING_KEYWORDS = ["DROP", "DELETE", "UPDATE", "INSERT", "MERGE", "ALTER", "TRUNCATE"]
MUTATING_PATTERN = re.compile(
    r"\b(" + "|".join(MUTATING_KEYWORDS) + r")\b",
    re.IGNORECASE
)


def get_bq_client(project: Optional[str] = None) -> bigquery.Client:
    """Instantiate a BigQuery client using project from env if available."""
    project_id = project or os.getenv("GOOGLE_CLOUD_PROJECT") or None
    return bigquery.Client(project=project_id)


def _serialize_value(val: Any) -> Any:
    """Serialize BigQuery cell values to JSON-safe types."""
    if isinstance(val, (datetime, date)):
        return val.isoformat()
    if isinstance(val, Decimal):
        return float(val)
    if isinstance(val, bytes):
        return val.decode("utf-8", errors="replace")
    if isinstance(val, list):
        return [_serialize_value(v) for v in val]
    if isinstance(val, dict):
        return {k: _serialize_value(v) for k, v in val.items()}
    return val


def fetch_dataset_schema(
    dataset_id: str,
    tables: Optional[List[str]] = None,
    client: Optional[bigquery.Client] = None
) -> str:
    """
    Fetches table and column names and types from BigQuery.
    Returns a formatted string representing the schema context.
    """
    bq_client = client or get_bq_client()
    lines = [f"Dataset: {dataset_id}"]

    target_tables = tables or []
    if not target_tables:
        dataset_ref = bigquery.DatasetReference.from_string(dataset_id)
        listed_tables = bq_client.list_tables(dataset_ref)
        target_tables = [t.table_id for t in listed_tables]

    for table_name in target_tables:
        full_table_ref = f"{dataset_id}.{table_name}"
        try:
            table_obj = bq_client.get_table(full_table_ref)
            lines.append(f"\nTable: `{full_table_ref}`")
            lines.append("Columns:")
            for schema_field in table_obj.schema:
                mode_str = f", {schema_field.mode}" if schema_field.mode else ""
                lines.append(f"  - `{schema_field.name}` ({schema_field.field_type}{mode_str})")
        except Exception as e:
            lines.append(f"\nTable `{full_table_ref}`: Error retrieving schema: {str(e)}")

    return "\n".join(lines)


def validate_query_dry_run(
    sql: str,
    client: Optional[bigquery.Client] = None
) -> Dict[str, Any]:
    """
    Validates a SQL query:
    1. Blocks mutating keywords (DROP, DELETE, UPDATE, INSERT, MERGE, ALTER, TRUNCATE).
    2. Runs a BigQuery dry run with QueryJobConfig(dry_run=True, use_query_cache=False).
    3. Fails if syntax/schema is invalid OR if total_bytes_processed > MAX_BYTES_BILLED.
    """
    # Guardrail 1: Check for mutating keywords
    match = MUTATING_PATTERN.search(sql)
    if match:
        forbidden_word = match.group(0).upper()
        return {
            "valid": False,
            "error": f"Mutating SQL statements are forbidden: detected '{forbidden_word}'. Only read-only SELECT queries are allowed.",
            "estimated_bytes": 0
        }

    bq_client = client or get_bq_client()
    job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)

    try:
        dry_run_job = bq_client.query(sql, job_config=job_config)
        total_bytes = dry_run_job.total_bytes_processed or 0

        # Guardrail 2: Byte scan limit check
        if total_bytes > MAX_BYTES_BILLED:
            mb_scanned = round(total_bytes / (1024 * 1024), 2)
            limit_mb = round(MAX_BYTES_BILLED / (1024 * 1024), 2)
            return {
                "valid": False,
                "error": f"Query estimated bytes {total_bytes} (~{mb_scanned} MB) exceeds maximum allowed {MAX_BYTES_BILLED} bytes (~{limit_mb} MB). Please optimize filters or columns.",
                "estimated_bytes": total_bytes
            }

        return {
            "valid": True,
            "error": None,
            "estimated_bytes": total_bytes
        }

    except (GoogleAPICallError, Exception) as e:
        return {
            "valid": False,
            "error": f"BigQuery dry-run validation error: {str(e)}",
            "estimated_bytes": 0
        }


def execute_safe_query(
    sql: str,
    client: Optional[bigquery.Client] = None,
    max_results: int = 50
) -> List[Dict[str, Any]]:
    """
    Runs validated SQL with maximum_bytes_billed=MAX_BYTES_BILLED
    and returns up to max_results rows as a list of dicts.
    """
    bq_client = client or get_bq_client()
    job_config = bigquery.QueryJobConfig(maximum_bytes_billed=MAX_BYTES_BILLED)

    query_job = bq_client.query(sql, job_config=job_config)
    results = query_job.result(max_results=max_results)

    rows = []
    for row in results:
        row_dict = {}
        for key, value in row.items():
            row_dict[key] = _serialize_value(value)
        rows.append(row_dict)

    return rows
