"""A synchronous LangGraph checkpointer using ordinary Redis primitives.

Stores full snapshots (bounded sequential graphs), plus pending writes. No Redis
Search modules are required. A worker lease fences every checkpoint write.
"""

import base64
import hashlib
import json

from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    CheckpointTuple,
    get_checkpoint_metadata,
)


class RedisCheckpointer(BaseCheckpointSaver):
    def __init__(self, store, run_id, token):
        super().__init__()
        self.store, self.run_id, self.token = store, run_id, token

    def _base(self, config):
        if config["configurable"]["thread_id"] != self.run_id:
            raise ValueError("Checkpointer is scoped to one run")
        namespace = config["configurable"].get("checkpoint_ns", "")
        return self.store.key(
            self.run_id,
            "checkpoints:" + hashlib.sha256(namespace.encode()).hexdigest()[:16],
        )

    def _encode(self, value):
        kind, data = self.serde.dumps_typed(value)
        return [kind, base64.b64encode(data).decode()]

    def _decode(self, value):
        return self.serde.loads_typed((value[0], base64.b64decode(value[1])))

    def get_tuple(self, config):
        base = self._base(config)
        checkpoint_id = config["configurable"].get("checkpoint_id")
        if not checkpoint_id:
            checkpoint_id = self.store.redis.get(base + ":latest")
            if isinstance(checkpoint_id, bytes):
                checkpoint_id = checkpoint_id.decode()
        if not checkpoint_id:
            return None
        raw = self.store.redis.hget(base, checkpoint_id)
        if raw is None:
            return None
        data = json.loads(raw)
        current = {
            "configurable": {**config["configurable"], "checkpoint_id": checkpoint_id}
        }
        parent = (
            {
                "configurable": {
                    **config["configurable"],
                    "checkpoint_id": data["parent"],
                }
            }
            if data["parent"]
            else None
        )
        writes = [
            self._decode(json.loads(v))
            for v in self.store.redis.hvals(base + ":writes:" + checkpoint_id)
        ]
        return CheckpointTuple(
            config=current,
            checkpoint=self._decode(data["checkpoint"]),
            metadata=self._decode(data["metadata"]),
            parent_config=parent,
            pending_writes=writes,
        )

    def put(self, config, checkpoint, metadata, new_versions):
        base = self._base(config)
        data = {
            "checkpoint": self._encode(checkpoint),
            "metadata": self._encode(get_checkpoint_metadata(config, metadata)),
            "parent": config["configurable"].get("checkpoint_id"),
        }

        def commands(pipe):
            pipe.hset(base, checkpoint["id"], json.dumps(data))
            pipe.expire(base, self.store.retention)
            pipe.set(base + ":latest", checkpoint["id"], ex=self.store.retention)

        self.store.fenced(self.run_id, self.token, commands)
        return {
            "configurable": {
                **config["configurable"],
                "checkpoint_id": checkpoint["id"],
            }
        }

    def put_writes(self, config, writes, task_id, task_path=""):
        key = self._base(config) + ":writes:" + config["configurable"]["checkpoint_id"]

        def commands(pipe):
            for i, (channel, value) in enumerate(writes):
                index = WRITES_IDX_MAP.get(channel, i)
                field = f"{task_id}:{index}"
                payload = json.dumps(self._encode((task_id, channel, value)))
                if index < 0:
                    pipe.hset(key, field, payload)
                else:
                    pipe.hsetnx(key, field, payload)
            pipe.expire(key, self.store.retention)

        self.store.fenced(self.run_id, self.token, commands)

    def list(self, config, *, filter=None, before=None, limit=None):
        ids = sorted(
            (
                x.decode() if isinstance(x, bytes) else x
                for x in self.store.redis.hkeys(self._base(config))
            ),
            reverse=True,
        )
        count = 0
        for checkpoint_id in ids:
            if before and checkpoint_id >= before["configurable"]["checkpoint_id"]:
                continue
            item = self.get_tuple(
                {
                    "configurable": {
                        **config["configurable"],
                        "checkpoint_id": checkpoint_id,
                    }
                }
            )
            if filter and any(item.metadata.get(k) != v for k, v in filter.items()):
                continue
            if limit is not None and count >= limit:
                break
            yield item
            count += 1
