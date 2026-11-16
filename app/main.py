import os
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.agent import sql_agent_graph, DEFAULT_DATASET

app = FastAPI(
    title="Self-Correcting Text-to-SQL BigQuery Agent",
    description="LangGraph-powered Text-to-SQL Agent with Gemini 2.5 Flash and BigQuery Guardrails",
    version="1.0.0"
)

# Enable CORS for local testing and web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str = Field(..., description="Natural language question to query BigQuery")
    dataset_id: Optional[str] = Field(
        default=DEFAULT_DATASET,
        description="BigQuery dataset identifier (e.g. bigquery-public-data.thelook_ecommerce)"
    )


class AskResponse(BaseModel):
    answer: str
    sql: str
    attempts: int
    estimated_mb_scanned: float
    rows: List[Dict[str, Any]]


@app.get("/health")
def health_check():
    """Health check endpoint for Cloud Run and monitoring."""
    return {"status": "healthy"}


@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest):
    """
    Accepts a question and optional dataset_id, invokes the LangGraph agent,
    and returns the synthesized answer, SQL, attempts, and estimated MB scanned.
    """
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="The 'question' field cannot be empty.")

    try:
        initial_state = {
            "question": request.question.strip(),
            "dataset_id": request.dataset_id or DEFAULT_DATASET,
            "schema_context": None,
            "generated_sql": None,
            "validation_error": None,
            "estimated_bytes": None,
            "retry_count": 0,
            "query_results": None,
            "final_answer": None
        }

        output = sql_agent_graph.invoke(initial_state)

        retry_count = output.get("retry_count", 0)
        estimated_bytes = output.get("estimated_bytes") or 0
        estimated_mb = round(estimated_bytes / (1024 * 1024), 4)

        return AskResponse(
            answer=output.get("final_answer") or "No answer generated.",
            sql=output.get("generated_sql") or "",
            attempts=retry_count + 1,
            estimated_mb_scanned=estimated_mb,
            rows=output.get("query_results") or []
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Agent execution encountered an error: {str(e)}"
        )
