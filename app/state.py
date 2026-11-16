from typing import TypedDict, Optional, List, Dict, Any


class SQLAgentState(TypedDict):
    question: str
    dataset_id: str
    schema_context: Optional[str]
    generated_sql: Optional[str]
    validation_error: Optional[str]
    estimated_bytes: Optional[int]
    retry_count: int
    query_results: Optional[List[Dict[str, Any]]]
    final_answer: Optional[str]
