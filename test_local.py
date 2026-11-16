import os
import json
import dotenv
from app.agent import sql_agent_graph, DEFAULT_DATASET

# Load local environment variables
dotenv.load_dotenv()


def main():
    print("=" * 70)
    print("Running Self-Correcting BigQuery SQL Agent Local Test")
    print("=" * 70)

    google_api_key = os.getenv("GOOGLE_API_KEY")
    project = os.getenv("GOOGLE_CLOUD_PROJECT")

    if not google_api_key:
        print("[WARNING] GOOGLE_API_KEY is not set in environment or .env file.")
    else:
        masked_key = google_api_key[:6] + "..." + google_api_key[-4:]
        print(f"[INFO] Using GOOGLE_API_KEY: {masked_key}")

    print(f"[INFO] Target Dataset: {DEFAULT_DATASET}")
    print(f"[INFO] Target Project: {project or '(Default ADC project)'}")

    question = "What are the top 5 product categories by total revenue for completed orders?"
    print(f"\n[QUERY QUESTION]:\n{question}\n")

    initial_state = {
        "question": question,
        "dataset_id": DEFAULT_DATASET,
        "schema_context": None,
        "generated_sql": None,
        "validation_error": None,
        "estimated_bytes": None,
        "retry_count": 0,
        "query_results": None,
        "final_answer": None
    }

    print("[INFO] Invoking LangGraph SQL Agent...")
    try:
        final_state = sql_agent_graph.invoke(initial_state)

        generated_sql = final_state.get("generated_sql")
        retry_count = final_state.get("retry_count", 0)
        attempts = retry_count + 1
        estimated_bytes = final_state.get("estimated_bytes") or 0
        mb_scanned = round(estimated_bytes / (1024 * 1024), 4)
        final_answer = final_state.get("final_answer")
        rows = final_state.get("query_results") or []

        print("\n" + "=" * 70)
        print("AGENT EXECUTION RESULTS")
        print("=" * 70)
        print(f"Attempts: {attempts}")
        print(f"Estimated MB Scanned: {mb_scanned} MB ({estimated_bytes} bytes)")
        print("\nGenerated SQL:")
        print("-" * 40)
        print(generated_sql)
        print("-" * 40)
        print(f"\nRetrieved Rows ({len(rows)}):")
        print(json.dumps(rows[:5], indent=2))
        print("\nFinal Answer:")
        print("-" * 40)
        print(final_answer)
        print("-" * 40)

    except Exception as e:
        print(f"\n[ERROR] Test run failed: {e}")
        raise


if __name__ == "__main__":
    main()
