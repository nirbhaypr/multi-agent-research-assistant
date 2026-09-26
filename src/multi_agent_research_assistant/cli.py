import argparse
import json
import time

from multi_agent_research_assistant.config import Settings
from multi_agent_research_assistant.domain.runs import RunLimits
from multi_agent_research_assistant.runtime import connect_store
from multi_agent_research_assistant.worker import main as worker_main
from multi_agent_research_assistant.worker import work_once


def main():
    parser = argparse.ArgumentParser(
        description="Durable multi-agent research assistant"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve", help="Run the HTTP API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", default=8000, type=int)
    worker = commands.add_parser("worker", help="Run the durable queue worker")
    worker.add_argument("--once", action="store_true")
    commands.add_parser(
        "demo", help="Run synthetic research against local Redis without provider calls"
    )
    args = parser.parse_args()
    if args.command == "serve":
        import uvicorn

        uvicorn.run(
            "multi_agent_research_assistant.api.app:create_app",
            factory=True,
            host=args.host,
            port=args.port,
        )
    elif args.command == "worker":
        worker_main(["--once"] if args.once else [])
    else:
        settings = Settings().model_copy(update={"research_mode": "demo"})
        store = connect_store(settings)
        try:
            started = time.monotonic()
            run, _ = store.enqueue(
                "How long are completed and failed runs retained?",
                RunLimits(),
                mode="demo",
                model_name="synthetic-fixture",
            )
            while (
                not store.get(run.run_id).terminal and time.monotonic() - started < 300
            ):
                if not work_once(settings):
                    time.sleep(0.2)
            result = store.get(run.run_id)
            print(
                json.dumps(
                    {
                        "run_id": run.run_id,
                        "mode": "demo",
                        "synthetic_sources": True,
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
