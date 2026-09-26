"""Specialists depend on structured calls, not on a provider or budget policy."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from multi_agent_research_assistant.domain.models import TokenUsage
from multi_agent_research_assistant.llm.model_calls import invoke_structured


class StructuredModel(Protocol):
    def invoke[T: BaseModel](
        self,
        *,
        schema: type[T],
        messages: list[BaseMessage],
        record_usage: Callable[[TokenUsage | None], None] | None = None,
    ) -> T: ...


@dataclass
class LangChainModel:
    """Unbudgeted adapter for isolated specialist smoke tests only."""

    llm: ChatOpenAI

    def invoke[T: BaseModel](
        self,
        *,
        schema: type[T],
        messages: list[BaseMessage],
        record_usage: Callable[[TokenUsage | None], None] | None = None,
    ) -> T:
        return invoke_structured(
            llm=self.llm,
            schema=schema,
            messages=messages,
            record_usage=record_usage,
        )
