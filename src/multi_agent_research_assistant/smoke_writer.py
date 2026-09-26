import argparse
import json
import os
from datetime import datetime, timezone
from time import perf_counter

from langchain_openai import ChatOpenAI

from .models import Finding, TokenUsage
from .writer import write_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("case", choices=("complete", "partial", "leading"))
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY before running this command.")

    question = "How long are completed and failed runs retained?"
    if args.case == "leading":
        question = "Why are completed runs always retained for seven days?"

    texts = ["Completed runs are normally retained for seven days."]
    if args.case == "complete":
        texts.append(
            "Failed runs are retained for thirty days unless an administrator "
            "deletes them earlier."
        )

    # Synthetic evidence and metadata; no webpages were retrieved.
    findings = [
        Finding(
            id=f"finding_{index}",
            subquestion_id="sq_retention",
            claim=text,
            source_url="https://example.com/fixtures/retention",
            snippet=text,
            retrieved_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
        for index, text in enumerate(texts, start=1)
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
        "question": question,
        "synthetic_findings": True,
        "status": "failed",
        "report": None,
    }
    started = perf_counter()

    try:
        report = write_report(
            question,
            findings,
            llm=llm,
            record_usage=recorded_usage.append,
        )
        output["report"] = report.model_dump(mode="json")
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