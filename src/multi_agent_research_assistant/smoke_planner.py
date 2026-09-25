import json
import os
from time import perf_counter

from langchain_openai import ChatOpenAI

from .models import TokenUsage
from .planner import create_plan


QUESTION = (
    "Compare Redis and PostgreSQL for storing agent run state "
    "in a small FastAPI service, focusing on durability, "
    "recovery after crashes, and operational complexity."
)


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY before running this command.")

    model_name = os.environ.get("OPENAI_MODEL", "gpt-5.4-mini")

    llm = ChatOpenAI(
        model=model_name,
        timeout=30.0,
        max_retries=0,
        max_completion_tokens=2000,
        reasoning_effort="none",
        use_responses_api=True,
    )

    recorded_usage: list[TokenUsage | None] = []

    result = {
        "model": model_name,
        "status": "failed",
        "plan": None,
    }

    started = perf_counter()

    try:
        plan = create_plan(
            QUESTION,
            llm=llm,
            max_subquestions=3,
            record_usage=recorded_usage.append,
        )

        result["plan"] = plan.model_dump(mode="json")
        result["status"] = "completed"

    finally:
        result["elapsed_seconds"] = round(
            perf_counter() - started,
            3,
        )
        result["usage"] = [
            usage.model_dump(mode="json") if usage is not None else None
            for usage in recorded_usage
        ]

        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()