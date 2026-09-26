import time
from collections.abc import Callable

from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from openai import OpenAI
from pydantic import BaseModel

from multi_agent_research_assistant.domain.models import TokenBudgetState, TokenUsage
from multi_agent_research_assistant.llm.model_calls import invoke_structured
from multi_agent_research_assistant.llm.token_counting import count_structured_input
from multi_agent_research_assistant.orchestration.budgets import (
    reserve_tokens,
    settle_tokens,
)


class BudgetedModel:
    """One run's sequential, budgeted structured calls."""

    def __init__(
        self,
        *,
        client: OpenAI,
        model: str,
        budget: TokenBudgetState,
        max_output_tokens: int = 2000,
        deadline: float | None = None,
        on_budget: Callable[[TokenBudgetState], None] | None = None,
    ) -> None:
        if type(max_output_tokens) is not int or max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be a positive integer")

        self._budget = budget
        self._deadline = deadline
        self._on_budget = on_budget
        self._max_output_tokens = max_output_tokens
        self._client = client.with_options(max_retries=0, timeout=30.0)

        self._llm = ChatOpenAI(
            model=model,
            api_key=self._client.api_key,
            base_url=str(self._client.base_url),
            client=self._client.chat.completions,
            root_client=self._client,
            use_responses_api=True,
            reasoning_effort="none",
            max_completion_tokens=max_output_tokens,
            max_retries=0,
            timeout=30.0,
        )

    def _check_deadline(self) -> None:
        remaining = 30.0 if self._deadline is None else self._deadline - time.time()
        if remaining <= 0:
            raise TimeoutError("Run deadline exceeded")
        self._client = self._client.with_options(timeout=min(30.0, remaining))
        self._llm.root_client = self._client
        self._llm.client = self._client.chat.completions

    def _publish_budget(self) -> None:
        if self._on_budget is not None:
            self._on_budget(self._budget)

    @property
    def budget(self) -> TokenBudgetState:
        return self._budget

    def invoke[T: BaseModel](
        self,
        *,
        schema: type[T],
        messages: list[BaseMessage],
        record_usage: Callable[[TokenUsage | None], None] | None = None,
    ) -> T:
        # Dry check: if even the output allowance cannot fit, skip counting.
        # reserve_tokens returns a new state; this does not change our ledger.
        reserve_tokens(self._budget, tokens=self._max_output_tokens)

        self._check_deadline()
        input_tokens = count_structured_input(
            client=self._client,
            model=self._llm.model_name,
            schema=schema,
            messages=messages,
        )

        self._check_deadline()
        self._budget = reserve_tokens(
            self._budget,
            tokens=input_tokens + self._max_output_tokens,
        )

        def settle_usage(usage: TokenUsage | None) -> None:
            self._budget = settle_tokens(self._budget, usage)
            self._publish_budget()

            if record_usage is not None:
                record_usage(usage)

        try:
            self._publish_budget()
            return invoke_structured(
                llm=self._llm,
                schema=schema,
                messages=messages,
                record_usage=settle_usage,
            )
        finally:
            # Covers failures that occur before the usage callback runs.
            if self._budget.reserved_tokens:
                self._budget = settle_tokens(self._budget, None)
                self._publish_budget()
