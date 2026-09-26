import argparse
import json
import os
from datetime import datetime, timezone
from time import perf_counter

from langchain_openai import ChatOpenAI

from .models import Finding, SubQuestion, TokenUsage
from .reviewer import review_research


COMPLETED = "Completed runs are normally retained for seven days."
FAILED = (
    "Failed runs are retained for thirty days unless an administrator "
    "deletes them earlier."
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("case", choices=("complete", "partial", "unsupported"))
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY before running this command.")

    subquestion = SubQuestion(
        id="sq_retention",
        question="How long are completed and failed runs retained?",
        completion_criteria=(
            "Identify both retention periods, including qualifications "
            "and exceptions."
        ),
    )

    rows = [(COMPLETED, COMPLETED)]
    if args.case != "partial":
        rows.append((FAILED, FAILED))
    if args.case == "unsupported":
        rows[0] = (
            "Completed runs are always retained for seven days.",
            COMPLETED,
        )

    # Synthetic findings and metadata; no webpages were retrieved.
    findings = [
        Finding(
            id=f"finding_{index}",
            subquestion_id=subquestion.id,
            claim=claim,
            source_url="https://example.com/fixtures/retention",
            snippet=snippet,
            retrieved_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
        for index, (claim, snippet) in enumerate(rows, start=1)
    ]

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
        "subquestion": subquestion.model_dump(mode="json"),
        "synthetic_findings": True,
        "status": "failed",
        "review": None,
    }
    started = perf_counter()

    try:
        review = review_research(
            subquestion,
            findings,
            llm=llm,
            record_usage=recorded_usage.append,
        )
        output["review"] = review.model_dump(mode="json")
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