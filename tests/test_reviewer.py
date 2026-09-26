import json
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage

from multi_agent_research_assistant.models import (
    Finding,
    ResearchReview,
    SubQuestion,
    TokenUsage,
)
from multi_agent_research_assistant.reviewer import review_research


@pytest.fixture
def case():
    subquestion = SubQuestion(
        id="sq_1",
        question="How long are completed runs retained?",
        completion_criteria="Identify the retention period and qualifications.",
    )
    finding = Finding.model_validate({
        "id": "finding_1",
        "subquestion_id": "sq_1",
        "claim": "Completed runs are normally retained for seven days.",
        "source_url": "https://example.com/retention",
        "snippet": "Completed runs are normally retained for seven days.",
        "retrieved_at": "2026-01-01T00:00:00Z",
    })
    review = ResearchReview(
        status="sufficient",
        accepted_finding_ids=[finding.id],
        gaps=[],
        reason="The finding covers the period and its qualification.",
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
            },
        ),
        "parsed": review,
        "parsing_error": None,
    }
    return subquestion, finding, review, llm


def test_returns_review_with_criteria_and_records_usage(case):
    subquestion, finding, expected, llm = case
    recorded = []

    result = review_research(
        subquestion, [finding], llm=llm, record_usage=recorded.append,
    )

    assert result == expected
    assert recorded == [
        TokenUsage(input_tokens=120, output_tokens=40, total_tokens=160),
    ]
    llm.with_structured_output.assert_called_once_with(
        ResearchReview, method="json_schema", strict=True, include_raw=True,
    )
    invocation = llm.with_structured_output.return_value.invoke
    invocation.assert_called_once()
    payload = json.loads(invocation.call_args.args[0][-1].content)
    assert payload["subquestion"] == subquestion.model_dump(mode="json")
    assert payload["findings"][0]["id"] == finding.id
    assert payload["findings"][0]["snippet"] == finding.snippet


def test_no_findings_returns_insufficient_without_model_call(case):
    subquestion, _, _, llm = case
    recorder = Mock()

    result = review_research(
        subquestion, [], llm=llm, record_usage=recorder,
    )

    assert result.status == "insufficient"
    assert result.accepted_finding_ids == []
    assert result.gaps == [subquestion.completion_criteria]
    llm.with_structured_output.assert_not_called()
    recorder.assert_not_called()


def test_rejects_duplicate_input_ids_before_model_call(case):
    subquestion, finding, _, llm = case

    with pytest.raises(ValueError, match="Finding IDs must be unique"):
        review_research(subquestion, [finding, finding], llm=llm)

    llm.with_structured_output.assert_not_called()


def test_rejects_findings_from_another_subquestion(case):
    subquestion, finding, _, llm = case
    wrong_finding = Finding.model_validate({
        **finding.model_dump(),
        "subquestion_id": "sq_other",
    })

    with pytest.raises(ValueError, match="belong to the reviewed subquestion"):
        review_research(subquestion, [wrong_finding], llm=llm)

    llm.with_structured_output.assert_not_called()


def test_insufficient_review_can_preserve_partial_findings(case):
    subquestion, finding, _, llm = case
    broader_question = SubQuestion(
        id=subquestion.id,
        question="How long are completed and failed runs retained?",
        completion_criteria="Identify both retention periods and qualifications.",
    )
    expected = ResearchReview(
        status="insufficient",
        accepted_finding_ids=[finding.id],
        gaps=["Failed-run retention is not established."],
        reason="Only completed-run retention is supported.",
    )
    llm.with_structured_output.return_value.invoke.return_value["parsed"] = expected

    result = review_research(broader_question, [finding], llm=llm)

    assert result == expected


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"accepted_finding_ids": ["missing"]}, "unknown findings"),
        (
            {"accepted_finding_ids": ["finding_1", "finding_1"]},
            "Accepted finding IDs must be unique",
        ),
        (
            {"accepted_finding_ids": []},
            "Sufficient review requires accepted findings and no gaps",
        ),
        (
            {"gaps": ["The retention period is missing."]},
            "Sufficient review requires accepted findings and no gaps",
        ),
        (
            {"status": "insufficient"},
            "Insufficient review must describe gaps",
        ),
    ],
)
def test_rejects_invalid_review_after_recording_usage(case, changes, message):
    subquestion, finding, original, llm = case
    invalid = ResearchReview.model_validate({
        **original.model_dump(),
        **changes,
    })
    llm.with_structured_output.return_value.invoke.return_value["parsed"] = invalid
    recorded = []

    with pytest.raises(ValueError, match=message):
        review_research(
            subquestion, [finding], llm=llm, record_usage=recorded.append,
        )

    assert recorded == [
        TokenUsage(input_tokens=120, output_tokens=40, total_tokens=160),
    ]