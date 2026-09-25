import pytest

from multi_agent_research_assistant.evidence import build_findings
from multi_agent_research_assistant.models import (
    FindingDraft,
    SourceDocument,
    SubQuestion,
)


@pytest.fixture
def subquestion():
    return SubQuestion(
        id="sq_1",
        question="How long does Service Alpha retain completed runs?",
        completion_criteria="Find the documented retention period.",
    )


@pytest.fixture
def source():
    return SourceDocument.model_validate(
        {
            "id": "source_1",
            "url": "https://example.com/retention-policy",
            "text": (
                "Completed runs are retained for seven days.\n"
                "Backups are kept separately."
            ),
            "retrieved_at": "2026-09-26T10:00:00Z",
        }
    )


@pytest.fixture
def draft():
    return FindingDraft(
        source_id="source_1",
        claim="Service Alpha retains completed runs for seven days.",
        snippet="Completed runs are retained for seven days.",
    )


def test_builds_finding_with_source_metadata(subquestion, source, draft):
    findings = build_findings(subquestion, [draft], [source])

    assert len(findings) == 1

    finding = findings[0]
    assert finding.id
    assert finding.subquestion_id == subquestion.id
    assert finding.claim == draft.claim
    assert finding.snippet == draft.snippet
    assert finding.source_url == source.url
    assert finding.retrieved_at == source.retrieved_at


def test_rejects_unknown_source(subquestion, source, draft):
    invalid_draft = FindingDraft.model_validate(
        {
            **draft.model_dump(),
            "source_id": "missing_source",
        }
    )

    with pytest.raises(ValueError, match="missing_source"):
        build_findings(subquestion, [invalid_draft], [source])


def test_rejects_excerpt_absent_from_source(subquestion, source, draft):
    invalid_draft = FindingDraft.model_validate(
        {
            **draft.model_dump(),
            "snippet": "Completed runs are retained for thirty days.",
        }
    )

    with pytest.raises(ValueError, match="Snippet not found"):
        build_findings(subquestion, [invalid_draft], [source])


def test_rejects_excerpt_from_a_different_source(
    subquestion,
    source,
    draft,
):
    other_source = SourceDocument.model_validate(
        {
            **source.model_dump(),
            "id": "source_2",
            "url": "https://example.com/another-policy",
            "text": "Completed runs are retained for thirty days.",
        }
    )

    invalid_draft = FindingDraft.model_validate(
        {
            **draft.model_dump(),
            "snippet": other_source.text,
        }
    )

    with pytest.raises(ValueError, match="Snippet not found"):
        build_findings(
            subquestion,
            [invalid_draft],
            [source, other_source],
        )


def test_rejects_duplicate_source_ids(subquestion, source, draft):
    with pytest.raises(ValueError, match="Source IDs must be unique"):
        build_findings(subquestion, [draft], [source, source])


def test_accepts_whitespace_differences(subquestion, source, draft):
    wrapped_source = SourceDocument.model_validate(
        {
            **source.model_dump(),
            "text": "Completed runs are\n retained  for seven days.",
        }
    )

    findings = build_findings(subquestion, [draft], [wrapped_source])

    assert len(findings) == 1
    assert findings[0].snippet == draft.snippet


def test_accepts_no_findings(subquestion, source):
    findings = build_findings(subquestion, [], [source])

    assert findings == []


def test_assigns_distinct_finding_ids(subquestion, source, draft):
    second_draft = FindingDraft(
        source_id=source.id,
        claim="Service Alpha keeps backups separately.",
        snippet="Backups are kept separately.",
    )

    findings = build_findings(
        subquestion,
        [draft, second_draft],
        [source],
    )

    assert len(findings) == 2
    assert findings[0].id != findings[1].id