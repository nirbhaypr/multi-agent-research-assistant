"""Redis run queue, fenced ownership, atomic step completion, and idempotency."""

import hashlib
import json
import time
from uuid import uuid4

from redis import Redis, WatchError

from multi_agent_research_assistant.domain.runs import RunLimits, RunState


class LeaseLost(RuntimeError):
    pass


class IdempotencyConflict(ValueError):
    pass


class RedisRunStore:
    def __init__(self, redis: Redis, *, prefix="research:v1", retention_seconds=604800):
        self.redis, self.prefix, self.retention = redis, prefix, retention_seconds

    def key(self, run_id, suffix):
        return f"{self.prefix}:run:{run_id}:{suffix}"

    def enqueue(
        self,
        question: str,
        limits: RunLimits,
        idempotency_key: str | None = None,
        *,
        mode: str = "live",
        model_name: str = "gpt-5.4-mini",
    ):
        digest = hashlib.sha256(
            json.dumps(
                [question, limits.model_dump(), mode, model_name], sort_keys=True
            ).encode()
        ).hexdigest()
        idem = (
            f"{self.prefix}:idem:{hashlib.sha256(idempotency_key.encode()).hexdigest()}"
            if idempotency_key
            else None
        )
        while True:
            with self.redis.pipeline() as pipe:
                try:
                    if idem:
                        pipe.watch(idem)
                        previous = pipe.get(idem)
                        if previous:
                            value = json.loads(previous)
                            if value["digest"] != digest:
                                raise IdempotencyConflict(
                                    "Idempotency key was used for another request"
                                )
                            existing = self.get(value["run_id"])
                            if existing:
                                return existing, False
                    run = RunState.new(question, limits)
                    run.mode = mode
                    run.model_name = model_name
                    pipe.multi()
                    pipe.set(
                        self.key(run.run_id, "state"),
                        run.model_dump_json(),
                        ex=self.retention,
                    )
                    pipe.zadd(f"{self.prefix}:pending", {run.run_id: time.time()})
                    if idem:
                        pipe.set(
                            idem,
                            json.dumps({"run_id": run.run_id, "digest": digest}),
                            ex=self.retention,
                        )
                    pipe.execute()
                    return run, True
                except WatchError:
                    continue

    def get(self, run_id):
        value = self.redis.get(self.key(run_id, "state"))
        return RunState.model_validate_json(value) if value else None

    def pending(self, limit=100):
        return [
            v.decode() if isinstance(v, bytes) else v
            for v in self.redis.zrange(f"{self.prefix}:pending", 0, limit - 1)
        ]

    def claim(self, run_id, lease_seconds=15):
        token = uuid4().hex
        return (
            token
            if self.redis.set(
                self.key(run_id, "lease"), token, nx=True, ex=lease_seconds
            )
            else None
        )

    def fenced(self, run_id, token, commands):
        lock = self.key(run_id, "lease")
        while True:
            with self.redis.pipeline() as pipe:
                try:
                    pipe.watch(lock)
                    current = pipe.get(lock)
                    if (
                        current is None
                        or (current.decode() if isinstance(current, bytes) else current)
                        != token
                    ):
                        raise LeaseLost("Run lease no longer belongs to this worker")
                    pipe.multi()
                    commands(pipe)
                    return pipe.execute()
                except WatchError:
                    continue

    def renew(self, run_id, token, lease_seconds=15):
        self.fenced(
            run_id, token, lambda p: p.expire(self.key(run_id, "lease"), lease_seconds)
        )

    def release(self, run_id, token):
        try:
            self.fenced(run_id, token, lambda p: p.delete(self.key(run_id, "lease")))
        except LeaseLost:
            pass

    def traces(self, run_id, start=0):
        return [
            json.loads(v)
            for v in self.redis.lrange(self.key(run_id, "trace"), start, -1)
        ]

    def save_terminal(self, run: RunState, token: str):
        def commands(pipe):
            pipe.set(
                self.key(run.run_id, "state"), run.model_dump_json(), ex=self.retention
            )
            pipe.zrem(f"{self.prefix}:pending", run.run_id)

        self.fenced(run.run_id, token, commands)


class RedisJournal:
    def __init__(self, store: RedisRunStore, run_id: str, token: str):
        self.store, self.run_id, self.token = store, run_id, token
        self.ops = store.key(run_id, "operations")

    def get(self, step_id):
        data = self.store.redis.hget(self.ops, step_id)
        return json.loads(data) if data else None

    def _save(self, step_id, value):
        def commands(pipe):
            pipe.hset(self.ops, step_id, json.dumps(value))
            pipe.expire(self.ops, self.store.retention)

        self.store.fenced(self.run_id, self.token, commands)

    def begin(self, step_id, received, budget):
        self._save(
            step_id, {"status": "started", "received": received, "budget": budget}
        )

    def budget(self, step_id, budget):
        value = self.get(step_id)
        value["budget"] = budget
        self._save(step_id, value)

    def complete(self, step_id, state, trace):
        value = self.get(step_id)
        value.update(status="completed", state=state, trace=trace)

        def commands(pipe):
            pipe.hset(self.ops, step_id, json.dumps(value))
            pipe.expire(self.ops, self.store.retention)
            pipe.set(
                self.store.key(self.run_id, "state"),
                json.dumps(state),
                ex=self.store.retention,
            )
            pipe.rpush(self.store.key(self.run_id, "trace"), json.dumps(trace))
            pipe.expire(self.store.key(self.run_id, "trace"), self.store.retention)
            if state["next_node"] == "done":
                pipe.zrem(f"{self.store.prefix}:pending", self.run_id)

        self.store.fenced(self.run_id, self.token, commands)
