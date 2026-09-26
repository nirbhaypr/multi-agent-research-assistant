from collections.abc import Callable

from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from openai import OpenAI
from pydantic import BaseModel

from .budgets import reserve_tokens, settle_tokens
from .model_calls import invoke_structured
from .models import TokenBudgetState, TokenUsage
from .token_counting import count_structured_input


class BudgetedModel:
    """One run's sequential, budgeted structured calls."""

    def __init__(
        self,
        *,
        client: OpenAI,
        model: str,
        budget: TokenBudgetState,
        max_output_tokens: int = 2000,
    ) -> None:
        if type(max_output_tokens) is not int or max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be a positive integer")

        self._budget = budget
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

        input_tokens = count_structured_input(
            client=self._client,
            model=self._llm.model_name,
            schema=schema,
            messages=messages,
        )

        self._budget = reserve_tokens(
            self._budget,
            tokens=input_tokens + self._max_output_tokens,
        )

        def settle_usage(usage: TokenUsage | None) -> None:
            self._budget = settle_tokens(self._budget, usage)

            if record_usage is not None:
                record_usage(usage)

        try:
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