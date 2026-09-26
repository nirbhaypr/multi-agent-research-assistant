import argparse
import json
import os
from datetime import datetime, timezone
from time import perf_counter

from langchain_openai import ChatOpenAI

from .models import SourceDocument, SubQuestion, TokenUsage
from .researcher import research_subquestion


FIXTURES = {
    "complete": (
        "Completed runs are normally retained for seven days. "
        "Failed runs are retained for thirty days unless an administrator "
        "deletes them earlier."
    ),
    "partial": "Completed runs are normally retained for seven days.",
    "irrelevant": "The dashboard supports light and dark themes.",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("case", choices=tuple(FIXTURES))
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY before running this command.")

    subquestion = SubQuestion(
        id="sq_retention",
        question="How long are completed and failed runs retained?",
        completion_criteria=(
            "Identify retention periods for both completed and failed runs, "
            "including any qualifications or exceptions."
        ),
    )
    source = SourceDocument(
        id="source_1",
        url=f"https://example.com/fixtures/{args.case}",
        text=FIXTURES[args.case],
        # Synthetic fixture metadata; no page was fetched.
        retrieved_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )

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
    output = {
        "model": model_name,
        "case": args.case,
        "synthetic_source": True,
        "status": "failed",
        "research": None,
    }
    started = perf_counter()

    try:
        research = research_subquestion(
            subquestion,
            [source],
            llm=llm,
            record_usage=recorded_usage.append,
        )
        output["research"] = research.model_dump(mode="json")
        output["status"] = "completed"
    finally:
        output["elapsed_seconds"] = round(perf_counter() - started, 3)
        output["usage"] = [
            usage.model_dump(mode="json") if usage is not None else None
            for usage in recorded_usage
        ]
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
