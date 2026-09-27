import json
import time

from multi_agent_research_assistant.config import Settings
from multi_agent_research_assistant.domain.runs import RunLimits
from multi_agent_research_assistant.runtime import connect_store
from multi_agent_research_assistant.worker import work_once


def main():
    settings = Settings().model_copy(update={"research_mode": "live"})
    settings.require_providers()
    store = connect_store(settings)
    try:
        started = time.monotonic()
        run, _ = store.enqueue(
            "How does Redis AOF help recover after a process crash? Use official Redis documentation.",
            RunLimits(
                max_subquestions=1,
                max_searches_per_subquestion=1,
                max_total_tokens=12000,
                max_output_tokens=1000,
                max_seconds=120,
                max_sources_per_search=2,
            ),
            mode="live",
            model_name=settings.openai_model,
        )
        while not store.get(run.run_id).terminal and time.monotonic() - started < 130:
            if not work_once(settings):
                time.sleep(0.2)
        result = store.get(run.run_id)
        print(
            json.dumps(
                {
                    "run_id": run.run_id,
                    "mode": "live",
                    "status": result.status,
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                    "budget": result.budget.model_dump(),
                    "report": result.report_markdown,
                    "trace_steps": len(store.traces(run.run_id)),
                },
                indent=2,
            )
        )
    finally:
        store.redis.close()


if __name__ == "__main__":
    main()
