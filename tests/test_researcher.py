import json
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage

from multi_agent_research_assistant.models import (
    FindingDraft,
    ResearchDraft,
    SourceDocument,
    SubQuestion,
    TokenUsage,
)
from multi_agent_research_assistant.researcher import research_subquestion


@pytest.fixture
def case():
    subquestion = SubQuestion(
        id="sq_1",
        question="How long does Service Alpha retain completed runs?",
        completion_criteria="Find the documented retention period.",
    )
    source = SourceDocument.model_validate({
        "id": "source_1",
        "url": "https://example.com/retention-policy",
        "text": "Completed runs are retained for seven days.",
        "retrieved_at": "2026-09-26T10:00:00Z",
    })
    draft = ResearchDraft(findings=[FindingDraft(
        source_id=source.id,
        claim="Service Alpha retains completed runs for seven days.",
        snippet=source.text,
    )])
    llm = Mock()
    llm.with_structured_output.return_value.invoke.return_value = {
        "raw": AIMessage(
            content="",
            response_metadata={"status": "completed"},
            usage_metadata={"input_tokens": 120, "output_tokens": 40, "total_tokens": 160},
        ),
        "parsed": draft,
        "parsing_error": None,
    }
    return subquestion, source, llm


def test_returns_findings_with_source_metadata(case):
    subquestion, source, llm = case
    recorded = []

    result = research_subquestion(
        subquestion, [source], llm=llm, record_usage=recorded.append,
    )

    assert result.status == "findings_found"
    assert result.subquestion_id == subquestion.id
    assert len(result.findings) == 1
    assert result.findings[0].source_url == source.url
    assert result.findings[0].retrieved_at == source.retrieved_at
    assert recorded == [TokenUsage(input_tokens=120, output_tokens=40, total_tokens=160)]
    llm.with_structured_output.assert_called_once_with(
        ResearchDraft, method="json_schema", strict=True, include_raw=True,
    )
    invocation = llm.with_structured_output.return_value.invoke
    invocation.assert_called_once()
    messages = invocation.call_args.args[0]
    payload = json.loads(messages[-1].content)
    assert payload["subquestion"] == subquestion.model_dump(mode="json")
    assert payload["sources"] == [{"id": source.id, "text": source.text}]


def test_skips_model_when_no_sources_are_available(case):
    subquestion, _, llm = case
    recorder = Mock()

    result = research_subquestion(subquestion, [], llm=llm, record_usage=recorder)

    assert result.status == "no_evidence"
    assert result.findings == []
    llm.with_structured_output.assert_not_called()
    recorder.assert_not_called()


def test_accepts_no_supported_findings(case):
    subquestion, source, llm = case
    llm.with_structured_output.return_value.invoke.return_value["parsed"] = ResearchDraft(
        findings=[],
    )

    result = research_subquestion(subquestion, [source], llm=llm)

    assert result.status == "no_evidence"
    assert result.findings == []
    llm.with_structured_output.return_value.invoke.assert_called_once()


def test_rejects_duplicate_source_ids_before_model_call(case):
    subquestion, source, llm = case

    with pytest.raises(ValueError, match="Source IDs must be unique"):
        research_subquestion(subquestion, [source, source], llm=llm)

    llm.with_structured_output.assert_not_called()


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"source_id": "missing_source"}, "Unknown source ID"),
        ({"snippet": "Completed runs are retained for thirty days."}, "Snippet not found"),
    ],
)
def test_rejects_invalid_evidence_after_recording_usage(case, changes, message):
    subquestion, source, llm = case
    response = llm.with_structured_output.return_value.invoke.return_value
    original = response["parsed"].findings[0]
    invalid = FindingDraft.model_validate({**original.model_dump(), **changes})
    response["parsed"] = ResearchDraft(findings=[invalid])
    recorded = []

    with pytest.raises(ValueError, match=message):
        research_subquestion(subquestion, [source], llm=llm, record_usage=recorded.append)

    assert recorded == [TokenUsage(input_tokens=120, output_tokens=40, total_tokens=160)]
