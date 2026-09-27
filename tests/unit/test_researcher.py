import json
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage

from multi_agent_research_assistant.agents.researcher import research_subquestion
from multi_agent_research_assistant.domain.models import (
    FindingDraft,
    ResearchDraft,
    SourceDocument,
    SubQuestion,
    TokenUsage,
)
from multi_agent_research_assistant.llm.interface import LangChainModel


@pytest.fixture
def case():
    subquestion = SubQuestion(
        id="sq_1",
        question="How long does Service Alpha retain completed runs?",
        completion_criteria="Find the documented retention period.",
    )
    source = SourceDocument.model_validate(
        {
            "id": "source_1",
            "url": "https://example.com/retention-policy",
            "text": "Completed runs are retained for seven days.",
            "retrieved_at": "2026-09-26T10:00:00Z",
        }
    )
    draft = ResearchDraft(
        findings=[
            FindingDraft(
                source_id=source.id,
                claim="Service Alpha retains completed runs for seven days.",
                snippet=source.text,
            )
        ]
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
        "parsed": draft,
        "parsing_error": None,
    }
    return subquestion, source, llm


def test_returns_findings_with_source_metadata(case):
    subquestion, source, llm = case
    recorded = []

    result = research_subquestion(
        subquestion,
        [source],
        model=LangChainModel(llm),
        record_usage=recorded.append,
    )

    assert result.status == "findings_found"
    assert result.subquestion_id == subquestion.id
    assert len(result.findings) == 1
    assert result.findings[0].source_url == source.url
    assert result.findings[0].retrieved_at == source.retrieved_at
    assert recorded == [
        TokenUsage(input_tokens=120, output_tokens=40, total_tokens=160)
    ]
    llm.with_structured_output.assert_called_once_with(
        ResearchDraft,
        method="json_schema",
        strict=True,
        include_raw=True,
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

    result = research_subquestion(
        subquestion, [], model=LangChainModel(llm), record_usage=recorder
    )

    assert result.status == "no_evidence"
    assert result.findings == []
    llm.with_structured_output.assert_not_called()
    recorder.assert_not_called()


def test_accepts_no_supported_findings(case):
    subquestion, source, llm = case
    llm.with_structured_output.return_value.invoke.return_value["parsed"] = (
        ResearchDraft(
            findings=[],
        )
    )

    result = research_subquestion(subquestion, [source], model=LangChainModel(llm))

    assert result.status == "no_evidence"
    assert result.findings == []
    llm.with_structured_output.return_value.invoke.assert_called_once()


def test_rejects_duplicate_source_ids_before_model_call(case):
    subquestion, source, llm = case

    with pytest.raises(ValueError, match="Source IDs must be unique"):
        research_subquestion(subquestion, [source, source], model=LangChainModel(llm))

    llm.with_structured_output.assert_not_called()


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"source_id": "missing_source"}, "unknown_source"),
        (
            {"snippet": "Completed runs are retained for thirty days."},
            "snippet_not_in_source",
        ),
    ],
)
def test_reports_invalid_evidence_after_recording_usage(case, changes, reason):
    subquestion, source, llm = case
    response = llm.with_structured_output.return_value.invoke.return_value
    original = response["parsed"].findings[0]
    invalid = FindingDraft.model_validate({**original.model_dump(), **changes})
    response["parsed"] = ResearchDraft(findings=[invalid])
    recorded = []

    result = research_subquestion(
        subquestion,
        [source],
        model=LangChainModel(llm),
        record_usage=recorded.append,
    )

    assert result.status == "no_evidence"
    assert result.findings == []
    assert [r.model_dump() for r in result.rejected_findings] == [
        {"draft_index": 0, "source_id": invalid.source_id, "reason": reason}
    ]
    assert recorded == [
        TokenUsage(input_tokens=120, output_tokens=40, total_tokens=160)
    ]
    llm.with_structured_output.return_value.invoke.assert_called_once()


@pytest.mark.parametrize(
    "invalid_snippet",
    [
        "Completed runs ... seven days.",
        "Completed runs are never retained for seven days.",
        "Completed runs are retained for seven days.\u0019",
    ],
)
def test_preserves_valid_drafts_on_both_sides_of_invalid_excerpt(case, invalid_snippet):
    subquestion, source, llm = case
    source.text += " Backups are kept separately."
    response = llm.with_structured_output.return_value.invoke.return_value
    first = response["parsed"].findings[0]
    last = FindingDraft(
        source_id=source.id,
        claim="Backups are kept separately.",
        snippet="Backups are kept separately.",
    )
    invalid = first.model_copy(update={"snippet": invalid_snippet})
    response["parsed"] = ResearchDraft(findings=[first, invalid, last])
    recorded = []

    result = research_subquestion(
        subquestion, [source], model=LangChainModel(llm), record_usage=recorded.append
    )

    assert result.status == "findings_found"
    assert [f.claim for f in result.findings] == [first.claim, last.claim]
    assert all(f.source_url == source.url for f in result.findings)
    assert [r.model_dump() for r in result.rejected_findings] == [
        {"draft_index": 1, "source_id": source.id, "reason": "snippet_not_in_source"}
    ]
    assert sum(u.total_tokens for u in recorded) == 160
    llm.with_structured_output.return_value.invoke.assert_called_once()


def test_does_not_reassign_a_quote_to_another_source(case):
    subquestion, source, llm = case
    other_source = source.model_copy(
        update={"id": "source_2", "text": "Backups are kept separately."}
    )
    response = llm.with_structured_output.return_value.invoke.return_value
    original = response["parsed"].findings[0]
    response["parsed"] = ResearchDraft(
        findings=[original.model_copy(update={"source_id": other_source.id}), original]
    )

    result = research_subquestion(
        subquestion, [source, other_source], model=LangChainModel(llm)
    )

    assert len(result.findings) == 1
    assert result.findings[0].snippet == source.text
    assert result.rejected_findings[0].draft_index == 0
    assert result.rejected_findings[0].source_id == other_source.id
    assert result.rejected_findings[0].reason == "snippet_not_in_source"


def test_sends_source_unicode_punctuation_without_json_escape_sequences(case):
    subquestion, source, llm = case
    source.text += " The service’s policy uses ‘normally’—not ‘always’."

    research_subquestion(subquestion, [source], model=LangChainModel(llm))

    messages = llm.with_structured_output.return_value.invoke.call_args.args[0]
    assert source.text in messages[-1].content
    assert "\\u2019" not in messages[-1].content


def test_model_errors_are_not_swallowed_as_evidence_rejections(case):
    subquestion, source, llm = case
    llm.with_structured_output.return_value.invoke.side_effect = ValueError("provider")

    with pytest.raises(ValueError, match="provider"):
        research_subquestion(subquestion, [source], model=LangChainModel(llm))
