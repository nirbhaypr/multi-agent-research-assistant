from .models import TokenBudgetState, TokenUsage


class BudgetBlocked(RuntimeError):
    """The next model call cannot be admitted."""


def reserve_tokens(
    budget: TokenBudgetState,
    *,
    tokens: int,
) -> TokenBudgetState:
    """Reserve a token ceiling for one call in a sequential run."""
    if tokens <= 0:
        raise ValueError("tokens must be positive")
    if budget.blocked_reason is not None:
        raise BudgetBlocked(budget.blocked_reason)
    if budget.reserved_tokens:
        raise BudgetBlocked("call_pending")
    if budget.recorded_tokens + tokens > budget.max_total_tokens:
        raise BudgetBlocked("token_limit")

    return TokenBudgetState.model_validate({
        **budget.model_dump(),
        "reserved_tokens": tokens,
    })


def settle_tokens(
    budget: TokenBudgetState,
    usage: TokenUsage | None,
) -> TokenBudgetState:
    """Record reported usage and close the current reservation."""
    if budget.reserved_tokens == 0:
        raise ValueError("No token reservation to settle")

    recorded_tokens = budget.recorded_tokens
    blocked_reason = budget.blocked_reason

    if usage is None:
        blocked_reason = "usage_unknown"
    else:
        recorded_tokens += usage.total_tokens
        if usage.total_tokens > budget.reserved_tokens:
            blocked_reason = "reservation_exceeded"

    return TokenBudgetState.model_validate({
        **budget.model_dump(),
        "recorded_tokens": recorded_tokens,
        "reserved_tokens": 0,
        "blocked_reason": blocked_reason,
    })