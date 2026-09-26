"""Recoverable Redis queue worker; each run executes in a killable process."""

import argparse
import json
import multiprocessing
import signal
import threading
import time
from datetime import UTC, datetime

from multi_agent_research_assistant.config import Settings
from multi_agent_research_assistant.domain.models import TokenBudgetState
from multi_agent_research_assistant.domain.runs import StepTrace
from multi_agent_research_assistant.orchestration.budgets import settle_tokens
from multi_agent_research_assistant.orchestration.workflow import finalize_run
from multi_agent_research_assistant.runtime import connect_store, execute_run
from multi_agent_research_assistant.storage.redis_store import LeaseLost, RedisJournal


def stop_run(store, run_id, token, reason):
    run = store.get(run_id)
    if run is None or run.terminal:
        return
    pending = []
    for raw in store.redis.hvals(store.key(run_id, "operations")):
        value = json.loads(raw)
        if value["status"] == "started":
            pending.append(value)
    if pending:
        run.budget = TokenBudgetState.model_validate(pending[-1]["budget"])
        if run.budget.reserved_tokens:
            run.budget = settle_tokens(run.budget, None)
    run.stop_reason = reason
    produced = finalize_run(run)
    trace = StepTrace(
        step_id=f"{run.step:03d}:worker",
        agent="worker",
        status="failed",
        received=pending[-1].get("received", {}) if pending else {"run_id": run_id},
        produced=produced,
        usage=[None] if run.budget.blocked_reason == "usage_unknown" else [],
        started_at=datetime.now(UTC),
        latency_ms=0,
        error=reason,
    )
    journal = RedisJournal(store, run_id, token)
    journal.begin(trace.step_id, trace.received, run.budget.model_dump())
    run.step += 1
    journal.complete(
        trace.step_id, run.model_dump(mode="json"), trace.model_dump(mode="json")
    )


def work_once(settings: Settings) -> bool:
    store = connect_store(settings)
    try:
        for run_id in store.pending():
            run = store.get(run_id)
            if run is None:
                store.redis.zrem(f"{store.prefix}:pending", run_id)
                continue
            token = store.claim(run_id, settings.lease_seconds)
            if token is None:
                continue
            process = None
            deadline_timer = None
            deadline_expired = threading.Event()
            try:
                if time.time() >= run.deadline:
                    stop_run(store, run_id, token, "deadline")
                    return True
                process = multiprocessing.get_context("spawn").Process(
                    target=execute_run, args=(settings, run_id, token), daemon=True
                )
                process.start()

                def enforce_deadline():
                    deadline_expired.set()
                    try:
                        if process.is_alive():
                            process.kill()
                    except ProcessLookupError:
                        pass

                deadline_timer = threading.Timer(
                    max(0, run.deadline - time.time()), enforce_deadline
                )
                deadline_timer.daemon = True
                deadline_timer.start()
                while process.is_alive():
                    remaining = run.deadline - time.time()
                    if remaining <= 0:
                        process.terminate()
                        process.join(0.5)
                        if process.is_alive():
                            process.kill()
                            process.join(0.5)
                        stop_run(store, run_id, token, "deadline")
                        return True
                    process.join(min(1.0, remaining))
                    store.renew(run_id, token, settings.lease_seconds)
                if deadline_expired.is_set() or time.time() >= run.deadline:
                    stop_run(store, run_id, token, "deadline")
                    return True
                current = store.get(run_id)
                if current and not current.terminal:
                    # Retry startup failures only; journal prevents replay of external calls.
                    attempts = store.redis.incr(store.key(run_id, "worker_failures"))
                    store.redis.expire(
                        store.key(run_id, "worker_failures"), store.retention
                    )
                    if attempts >= 3:
                        stop_run(store, run_id, token, "worker_failure")
                return True
            except LeaseLost:
                if process and process.is_alive():
                    process.terminate()
                    process.join(0.5)
                return True
            finally:
                if deadline_timer:
                    deadline_timer.cancel()
                if process and process.is_alive():
                    process.terminate()
                    process.join(0.5)
                store.release(run_id, token)
        return False
    finally:
        store.redis.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--once", action="store_true", help="Process one available run, then exit."
    )
    args = parser.parse_args(argv)
    settings = Settings()
    settings.require_providers()

    def shutdown(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, shutdown)
    try:
        while True:
            worked = work_once(settings)
            if args.once:
                break
            if not worked:
                time.sleep(0.5)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
