from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage

from multi_agent_research_assistant.models import ResearchPlan, TokenUsage
from multi_agent_research_assistant.planner import create_plan


QUESTION = "How does Redis persist data?"


@pytest.fixture
def planner_llm():
    plan = ResearchPlan.model_validate(
        {
            "question": QUESTION,
            "subquestions": [
                {
                    "id": "sq_1",
                    "question": "How do RDB and AOF differ?",
                    "completion_criteria": (
                        "Explain recovery behavior and data-loss tradeoffs."
                    ),
                }
            ],
        }
    )

    llm = Mock()
    llm.with_structured_output.return_value.invoke.return_value = {
        "raw": AIMessage(
            content="",
            response_metadata={"status": "completed"},
            usage_metadata={
                "input_tokens": 120,
                "output_tokens": 40,
                "total_tokens": 160,
                "input_token_details": {"cache_read": 20},
            },
        ),
        "parsed": plan,
        "parsing_error": None,
    }

    return llm


def test_records_usage_for_accepted_plan(planner_llm):
    recorded = []

    create_plan(
        QUESTION,
        llm=planner_llm,
        record_usage=recorded.append,
    )

    assert recorded == [
        TokenUsage(
            input_tokens=120,
            output_tokens=40,
            total_tokens=160,
        )
    ]


def test_records_usage_before_output_rejection(planner_llm):
    recorded = []
    response = (
        planner_llm.with_structured_output.return_value.invoke.return_value
    )
    response["parsed"] = None
    response["parsing_error"] = ValueError("invalid output")

    with pytest.raises(RuntimeError, match="could not be parsed"):
        create_plan(
            QUESTION,
            llm=planner_llm,
            record_usage=recorded.append,
        )

    assert recorded == [
        TokenUsage(
            input_tokens=120,
            output_tokens=40,
            total_tokens=160,
        )
    ]


@pytest.mark.parametrize(
    ("usage_metadata", "expected"),
    [
        (None, None),
        (
            {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            },
            TokenUsage(
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
            ),
        ),
    ],
)
def test_distinguishes_unknown_usage_from_zero(
    planner_llm,
    usage_metadata,
    expected,
):
    recorded = []
    response = (
        planner_llm.with_structured_output.return_value.invoke.return_value
    )
    response["raw"] = AIMessage(
        content="",
        response_metadata={"status": "completed"},
        usage_metadata=usage_metadata,
    )

    create_plan(
        QUESTION,
        llm=planner_llm,
        record_usage=recorded.append,
    )

    assert recorded == [expected]


def test_records_unknown_usage_when_invocation_fails(planner_llm):
    recorded = []
    invocation = planner_llm.with_structured_output.return_value.invoke
    invocation.side_effect = TimeoutError("provider timed out")

    with pytest.raises(TimeoutError, match="provider timed out"):
        create_plan(
            QUESTION,
            llm=planner_llm,
            record_usage=recorded.append,
        )

    assert recorded == [None]


def test_does_not_record_usage_for_invalid_input(planner_llm):
    recorder = Mock()

    with pytest.raises(ValueError, match="blank"):
        create_plan(
            "   ",
            llm=planner_llm,
            record_usage=recorder,
        )

    recorder.assert_not_called()
    planner_llm.with_structured_output.assert_not_called()