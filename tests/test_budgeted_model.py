import json

import httpx
import pytest
from langchain_core.messages import HumanMessage
from openai import InternalServerError, OpenAI

from multi_agent_research_assistant.budgeted_model import BudgetedModel
from multi_agent_research_assistant.budgets import BudgetBlocked
from multi_agent_research_assistant.models import ResearchDraft, TokenBudgetState


@pytest.fixture
def harness():
    requests = []
    settings = {
        "count_status": 200,
        "generation_status": 200,
        "status": "completed",
        "usage": {
            "input_tokens": 100,
            "output_tokens": 40,
            "total_tokens": 140,
        },
    }
    models = []

    def respond(request):
        path = request.url.path
        requests.append((path, json.loads(request.content)))

        if path == "/v1/responses/input_tokens":
            code = settings["count_status"]
            body = {
                "object": "response.input_tokens",
                "input_tokens": 100,
            }
        else:
            assert path == "/v1/responses"

            # The reservation must exist before generation reaches HTTP.
            assert models[-1].budget.reserved_tokens == 300

            code = settings["generation_status"]
            body = {
                "id": "resp_test",
                "object": "response",
                "created_at": 0,
                "status": settings["status"],
                "model": "gpt-5.4-mini",
                "output": [
                    {
                        "id": "msg_test",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {
                                "type": "output_text",
                                "text": '{"findings": []}',
                                "annotations": [],
                            }
                        ],
                    }
                ],
                "usage": settings["usage"],
            }

            if settings["status"] == "incomplete":
                body["output"] = []
                body["incomplete_details"] = {
                    "reason": "max_output_tokens"
                }

        if code != 200:
            body = {
                "error": {
                    "message": "unavailable",
                    "type": "server_error",
                }
            }

        return httpx.Response(code, json=body)

    with httpx.Client(transport=httpx.MockTransport(respond)) as http:
        client = OpenAI(
            api_key="test-only",
            base_url="https://example.test/v1",
            http_client=http,
            max_retries=2,
        )

        def build(budget=None):
            model = BudgetedModel(
                client=client,
                model="gpt-5.4-mini",
                budget=budget or TokenBudgetState(max_total_tokens=1000),
                max_output_tokens=200,
            )
            models.append(model)
            return model

        yield build, requests, settings


def invoke(model, record_usage=None):
    return model.invoke(
        schema=ResearchDraft,
        messages=[HumanMessage(content="Research retention.")],
        record_usage=record_usage,
    )


def test_reserves_before_generation_and_settles_actual_usage(harness):
    build, requests, _ = harness
    model = build(TokenBudgetState(max_total_tokens=300))
    recorded = []

    assert invoke(model, recorded.append) == ResearchDraft(findings=[])

    assert model.budget.recorded_tokens == 140
    assert model.budget.reserved_tokens == 0
    assert model.budget.blocked_reason is None
    assert [usage.total_tokens for usage in recorded] == [140]
    assert len(requests) == 2
    assert requests[1][1]["max_output_tokens"] == 200


def test_rejects_full_reservation_before_generation(harness):
    build, requests, _ = harness
    original = TokenBudgetState(max_total_tokens=299)
    model = build(original)

    with pytest.raises(BudgetBlocked, match="token_limit"):
        invoke(model)

    assert [path for path, _ in requests] == [
        "/v1/responses/input_tokens"
    ]
    assert model.budget == original


@pytest.mark.parametrize(
    ("budget", "reason"),
    [
        (
            TokenBudgetState(max_total_tokens=199),
            "token_limit",
        ),
        (
            TokenBudgetState(
                max_total_tokens=1000,
                reserved_tokens=300,
            ),
            "call_pending",
        ),
        (
            TokenBudgetState(
                max_total_tokens=1000,
                blocked_reason="usage_unknown",
            ),
            "usage_unknown",
        ),
    ],
)
def test_existing_budget_blocks_before_any_http_request(
    harness, budget, reason,
):
    build, requests, _ = harness
    model = build(budget)

    with pytest.raises(BudgetBlocked, match=reason):
        invoke(model)

    assert requests == []
    assert model.budget == budget


@pytest.mark.parametrize(
    "failure_stage",
    ["count_status", "generation_status"],
)
def test_http_failure_is_not_retried_and_accounts_for_its_stage(
    harness, failure_stage,
):
    build, requests, settings = harness
    model = build()
    original = model.budget
    recorded = []
    settings[failure_stage] = 500

    with pytest.raises(InternalServerError):
        invoke(model, recorded.append)

    if failure_stage == "count_status":
        assert len(requests) == 1
        assert model.budget == original
        assert recorded == []
    else:
        assert len(requests) == 2
        assert model.budget.recorded_tokens == 0
        assert model.budget.reserved_tokens == 0
        assert model.budget.blocked_reason == "usage_unknown"
        assert recorded == [None]


def test_incomplete_response_still_settles_reported_usage(harness):
    build, _, settings = harness
    model = build()
    settings["status"] = "incomplete"

    with pytest.raises(RuntimeError, match="not completed"):
        invoke(model)

    assert model.budget.recorded_tokens == 140
    assert model.budget.reserved_tokens == 0
    assert model.budget.blocked_reason is None


@pytest.mark.parametrize(
    ("usage", "reason", "total"),
    [
        (None, "usage_unknown", 0),
        (
            {
                "input_tokens": 350,
                "output_tokens": 50,
                "total_tokens": 400,
            },
            "reservation_exceeded",
            400,
        ),
    ],
)
def test_unknown_usage_or_overrun_blocks_the_next_call(
    harness, usage, reason, total,
):
    build, requests, settings = harness
    model = build()
    settings["usage"] = usage

    invoke(model)

    assert model.budget.recorded_tokens == total
    assert model.budget.reserved_tokens == 0
    assert model.budget.blocked_reason == reason

    with pytest.raises(BudgetBlocked, match=reason):
        invoke(model)

    assert len(requests) == 2


def test_repeated_calls_share_the_same_budget(harness):
    build, requests, _ = harness
    model = build(TokenBudgetState(max_total_tokens=440))

    invoke(model)
    invoke(model)

    assert model.budget.recorded_tokens == 280

    with pytest.raises(BudgetBlocked, match="token_limit"):
        invoke(model)

    assert len(requests) == 4


def test_observer_failure_does_not_erase_or_double_settle_usage(harness):
    build, _, _ = harness
    model = build()

    def broken_observer(usage):
        raise RuntimeError("observer failed")

    with pytest.raises(RuntimeError, match="observer failed"):
        invoke(model, broken_observer)

    assert model.budget.recorded_tokens == 140
    assert model.budget.reserved_tokens == 0
    assert model.budget.blocked_reason is None


def test_failure_before_usage_callback_blocks_further_calls(
    harness, monkeypatch,
):
    build, _, _ = harness
    model = build()

    def broken_invocation(**kwargs):
        raise RuntimeError("invocation failed before callback")

    monkeypatch.setattr(
        "multi_agent_research_assistant.budgeted_model.invoke_structured",
        broken_invocation,
    )

    with pytest.raises(RuntimeError, match="before callback"):
        invoke(model)

    assert model.budget.recorded_tokens == 0
    assert model.budget.reserved_tokens == 0
    assert model.budget.blocked_reason == "usage_unknown"