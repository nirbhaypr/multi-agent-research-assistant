import pytest

from multi_agent_research_assistant.domain.models import TokenBudgetState, TokenUsage
from multi_agent_research_assistant.orchestration.budgets import (
    BudgetBlocked,
    reserve_tokens,
    settle_tokens,
)


def reported_usage(total):
    return TokenUsage(
        input_tokens=total,
        output_tokens=0,
        total_tokens=total,
    )


def test_reservation_is_not_reported_usage():
    original = TokenBudgetState(max_total_tokens=1000)

    reserved = reserve_tokens(original, tokens=400)

    assert reserved.reserved_tokens == 400
    assert reserved.recorded_tokens == 0
    assert original.reserved_tokens == 0


def test_settlement_releases_unused_reservation():
    budget = reserve_tokens(
        TokenBudgetState(max_total_tokens=1000),
        tokens=400,
    )

    settled = settle_tokens(budget, reported_usage(250))

    assert settled.recorded_tokens == 250
    assert settled.reserved_tokens == 0
    assert settled.blocked_reason is None

    # Exactly the remaining budget can be reserved.
    next_call = reserve_tokens(settled, tokens=750)
    assert next_call.reserved_tokens == 750


def test_usage_accumulates_across_calls():
    budget = TokenBudgetState(max_total_tokens=1000)

    budget = reserve_tokens(budget, tokens=400)
    budget = settle_tokens(budget, reported_usage(250))
    budget = reserve_tokens(budget, tokens=400)
    budget = settle_tokens(budget, reported_usage(300))

    assert budget.recorded_tokens == 550
    assert budget.reserved_tokens == 0


def test_rejects_call_that_would_exceed_limit():
    budget = TokenBudgetState(
        max_total_tokens=1000,
        recorded_tokens=750,
    )

    with pytest.raises(BudgetBlocked, match="token_limit"):
        reserve_tokens(budget, tokens=251)


def test_only_one_call_can_have_a_reservation():
    budget = reserve_tokens(
        TokenBudgetState(max_total_tokens=1000),
        tokens=400,
    )

    with pytest.raises(BudgetBlocked, match="call_pending"):
        reserve_tokens(budget, tokens=1)


def test_missing_usage_preserves_known_total_and_blocks_calls():
    budget = TokenBudgetState(
        max_total_tokens=1000,
        recorded_tokens=250,
    )
    budget = reserve_tokens(budget, tokens=100)

    settled = settle_tokens(budget, None)

    assert settled.recorded_tokens == 250
    assert settled.reserved_tokens == 0
    assert settled.blocked_reason == "usage_unknown"
    with pytest.raises(BudgetBlocked, match="usage_unknown"):
        reserve_tokens(settled, tokens=1)


def test_reported_zero_usage_does_not_block_calls():
    budget = TokenBudgetState(
        max_total_tokens=1000,
        recorded_tokens=250,
    )
    budget = reserve_tokens(budget, tokens=100)

    settled = settle_tokens(budget, reported_usage(0))

    assert settled.recorded_tokens == 250
    assert settled.blocked_reason is None
    assert reserve_tokens(settled, tokens=750).reserved_tokens == 750


@pytest.mark.parametrize("limit", [1000, 500])
def test_overrun_records_actual_usage_and_blocks_calls(limit):
    budget = reserve_tokens(
        TokenBudgetState(max_total_tokens=limit),
        tokens=400,
    )

    settled = settle_tokens(budget, reported_usage(600))

    # Keep the reported total even when it exceeds the run limit.
    assert settled.recorded_tokens == 600
    assert settled.reserved_tokens == 0
    assert settled.blocked_reason == "reservation_exceeded"
    with pytest.raises(BudgetBlocked, match="reservation_exceeded"):
        reserve_tokens(settled, tokens=1)


def test_cannot_settle_the_current_state_twice():
    budget = reserve_tokens(
        TokenBudgetState(max_total_tokens=1000),
        tokens=100,
    )
    budget = settle_tokens(budget, reported_usage(50))

    with pytest.raises(ValueError, match="No token reservation to settle"):
        settle_tokens(budget, reported_usage(50))


@pytest.mark.parametrize("tokens", [0, -1])
def test_rejects_nonpositive_reservation(tokens):
    budget = TokenBudgetState(max_total_tokens=1000)

    with pytest.raises(ValueError, match="tokens must be positive"):
        reserve_tokens(budget, tokens=tokens)


def test_inconsistent_provider_usage_cannot_understate_spending():
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="total_tokens must equal"):
        TokenUsage(input_tokens=100, output_tokens=40, total_tokens=0)
