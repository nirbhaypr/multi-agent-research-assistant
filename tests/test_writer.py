import json
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage

from multi_agent_research_assistant.models import (
    Finding,
    ReportClaim,
    ResearchReport,
    TokenUsage,
)
from multi_agent_research_assistant.writer import write_report


@pytest.fixture
def case():
    question = "How long are completed runs retained?"
    finding = Finding.model_validate({
        "id": "finding_1",
        "subquestion_id": "sq_1",
        "claim": "Completed runs are normally retained for seven days.",
        "source_url": "https://example.com/retention",
        "snippet": "Completed runs are normally retained for seven days.",
        "retrieved_at": "2026-01-01T00:00:00Z",
    })
    report = ResearchReport(
        title="Completed-run retention",
        claims=[ReportClaim(text=finding.claim, finding_ids=[finding.id])],
    )
    llm = Mock()
    llm.with_structured_output.return_value.invoke.return_value = {
        "raw": AIMessage(
            content="",
            response_metadata={"status": "completed"},
            usage_metadata={
                "input_tokens": 150,
                "output_tokens": 50,
                "total_tokens": 200,
            },
        ),
        "parsed": report,
        "parsing_error": None,
    }
    return question, finding, report, llm


def test_returns_report_with_valid_citations_and_records_usage(case):
    question, finding, expected, llm = case
    recorded = []

    result = write_report(
        f"  {question}  ",
        [finding],
        llm=llm,
        record_usage=recorded.append,
    )

    assert result == expected
    assert recorded == [
        TokenUsage(input_tokens=150, output_tokens=50, total_tokens=200),
    ]
    llm.with_structured_output.assert_called_once_with(
        ResearchReport, method="json_schema", strict=True, include_raw=True,
    )
    invocation = llm.with_structured_output.return_value.invoke
    invocation.assert_called_once()
    payload = json.loads(invocation.call_args.args[0][-1].content)
    assert payload["question"] == question
    assert payload["findings"][0]["id"] == finding.id
    assert payload["findings"][0]["snippet"] == finding.snippet


@pytest.mark.parametrize("question", ["", " \t "])
def test_rejects_blank_question_before_model_call(case, question):
    _, finding, _, llm = case
    recorder = Mock()

    with pytest.raises(ValueError, match="question must not be blank"):
        write_report(question, [finding], llm=llm, record_usage=recorder)

    llm.with_structured_output.assert_not_called()
    recorder.assert_not_called()


def test_rejects_empty_findings_before_model_call(case):
    question, _, _, llm = case
    recorder = Mock()

    with pytest.raises(ValueError, match="At least one finding is required"):
        write_report(question, [], llm=llm, record_usage=recorder)

    llm.with_structured_output.assert_not_called()
    recorder.assert_not_called()


def test_rejects_duplicate_finding_ids_before_model_call(case):
    question, finding, _, llm = case
    recorder = Mock()

    with pytest.raises(ValueError, match="Finding IDs must be unique"):
        write_report(
            question, [finding, finding], llm=llm, record_usage=recorder,
        )

    llm.with_structured_output.assert_not_called()
    recorder.assert_not_called()


def test_rejects_unknown_citation_after_recording_usage(case):
    question, finding, _, llm = case
    response = llm.with_structured_output.return_value.invoke.return_value
    response["parsed"] = ResearchReport(
        title="Retention",
        claims=[ReportClaim(
            text=finding.claim,
            finding_ids=[finding.id, "finding_missing"],
        )],
    )
    recorded = []

    with pytest.raises(ValueError, match="unknown findings: finding_missing"):
        write_report(
            question, [finding], llm=llm, record_usage=recorded.append,
        )

    assert recorded == [
        TokenUsage(input_tokens=150, output_tokens=50, total_tokens=200),
    ]