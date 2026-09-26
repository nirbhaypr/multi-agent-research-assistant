import json
from unittest.mock import Mock

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from openai import OpenAI

from multi_agent_research_assistant.model_calls import invoke_structured
from multi_agent_research_assistant.models import (
    ReportClaim,
    ResearchDraft,
    ResearchPlan,
    ResearchReport,
    ResearchReview,
    SubQuestion,
)
from multi_agent_research_assistant.token_counting import count_structured_input


@pytest.mark.parametrize(
    "expected",
    [
        ResearchPlan(
            question="How long are runs retained?",
            subquestions=[
                SubQuestion(
                    id="sq_1",
                    question="What is the retention period?",
                    completion_criteria="Identify the documented period.",
                )
            ],
        ),
        ResearchDraft(findings=[]),
        ResearchReport(
            title="Retention",
            claims=[
                ReportClaim(
                    text="Runs are retained.",
                    finding_ids=["finding_1"],
                )
            ],
        ),
        ResearchReview(
            status="insufficient",
            accepted_finding_ids=[],
            gaps=["Retention evidence is missing."],
            reason="No findings were supplied.",
        ),
    ],
)
def test_counted_input_matches_actual_langchain_request(expected):
    captured = {}
    schema = type(expected)

    def respond(request):
        captured[request.url.path] = json.loads(request.content)

        if request.url.path == "/v1/responses/input_tokens":
            return httpx.Response(
                200,
                json={
                    "object": "response.input_tokens",
                    "input_tokens": 123,
                },
            )

        assert request.url.path == "/v1/responses"

        return httpx.Response(
            200,
            json={
                "id": "resp_test",
                "object": "response",
                "created_at": 0,
                "status": "completed",
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
                                "text": expected.model_dump_json(),
                                "annotations": [],
                            }
                        ],
                    }
                ],
                "usage": {
                    "input_tokens": 123,
                    "output_tokens": 20,
                    "total_tokens": 143,
                },
            },
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as http:
        client = OpenAI(
            api_key="test-only",
            base_url="https://example.test/v1",
            http_client=http,
            max_retries=0,
        )

        llm = ChatOpenAI(
            api_key="test-only",
            base_url="https://example.test/v1",
            http_client=http,
            model="gpt-5.4-mini",
            use_responses_api=True,
            reasoning_effort="none",
            max_completion_tokens=2000,
            max_retries=0,
        )

        messages = [
            SystemMessage(
                content="Return the requested structured result."
            ),
            HumanMessage(content="Research retention."),
        ]

        count = count_structured_input(
            client=client,
            model=llm.model_name,
            schema=schema,
            messages=messages,
        )

        result = invoke_structured(
            llm=llm,
            schema=schema,
            messages=messages,
        )

    assert count == 123
    assert result == expected

    counted = captured["/v1/responses/input_tokens"]
    generated = captured["/v1/responses"]

    for key in ("model", "input", "text", "reasoning"):
        assert counted[key] == generated[key]

    assert generated["max_output_tokens"] == 2000


@pytest.mark.parametrize(
    "message",
    [
        AIMessage(content="Earlier answer"),
        HumanMessage(
            content=[{"type": "text", "text": "Hello"}]
        ),
        HumanMessage(content="Hello", name="named_user"),
        SystemMessage(
            content="Hello",
            additional_kwargs={"custom": True},
        ),
    ],
)
def test_rejects_unsupported_messages_before_count_request(message):
    client = Mock()

    with pytest.raises(ValueError):
        count_structured_input(
            client=client,
            model="gpt-5.4-mini",
            schema=ResearchDraft,
            messages=[message],
        )

    client.responses.input_tokens.count.assert_not_called()


def test_rejects_invalid_count_instead_of_using_it():
    client = Mock()
    client.responses.input_tokens.count.return_value.input_tokens = -1

    with pytest.raises(ValueError, match="nonnegative integer"):
        count_structured_input(
            client=client,
            model="gpt-5.4-mini",
            schema=ResearchDraft,
            messages=[HumanMessage(content="Research retention.")],
        )


def test_count_failure_is_not_replaced_with_zero():
    client = Mock()
    client.responses.input_tokens.count.side_effect = TimeoutError(
        "count unavailable"
    )

    with pytest.raises(TimeoutError, match="count unavailable"):
        count_structured_input(
            client=client,
            model="gpt-5.4-mini",
            schema=ResearchDraft,
            messages=[HumanMessage(content="Research retention.")],
        )