"""Record the exact specialist prompt and parsed response without provider secrets."""

import time
from collections.abc import Callable

from langchain_core.messages import BaseMessage
from pydantic import BaseModel

from multi_agent_research_assistant.domain.models import TokenUsage
from multi_agent_research_assistant.llm.interface import StructuredModel


class TracedModel:
    def __init__(self, wrapped: StructuredModel, events: list[dict], model_name: str):
        self.wrapped, self.events, self.model_name = wrapped, events, model_name

    def invoke[T: BaseModel](
        self,
        *,
        schema: type[T],
        messages: list[BaseMessage],
        record_usage: Callable[[TokenUsage | None], None] | None = None,
    ) -> T:
        started = time.monotonic()
        event = {
            "tool": "structured_model_call",
            "input": {
                "model": self.model_name,
                "schema": schema.model_json_schema(),
                "messages": [{"role": m.type, "content": m.content} for m in messages],
            },
            "usage": [],
        }

        def observe(usage):
            event["usage"].append(usage.model_dump() if usage is not None else None)
            if record_usage:
                record_usage(usage)

        try:
            result = self.wrapped.invoke(
                schema=schema, messages=messages, record_usage=observe
            )
            event["output"] = {
                "status": "completed",
                "parsed": result.model_dump(mode="json"),
            }
            return result
        except Exception as exc:
            event["output"] = {"status": "failed", "error": type(exc).__name__}
            raise
        finally:
            event["latency_ms"] = round((time.monotonic() - started) * 1000, 3)
            self.events.append(event)
