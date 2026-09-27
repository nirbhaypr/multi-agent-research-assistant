from uuid import uuid4

from multi_agent_research_assistant.domain.models import (
    EvidenceRejectionReason,
    Finding,
    FindingDraft,
    SourceDocument,
    SubQuestion,
)


class EvidenceValidationError(ValueError):
    """Expected rejection of one generated draft, distinct from internal failures."""

    def __init__(self, reason: EvidenceRejectionReason, message: str):
        self.reason = reason
        super().__init__(message)


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def build_findings(
    subquestion: SubQuestion, drafts: list[FindingDraft], sources: list[SourceDocument]
) -> list[Finding]:

    sources_by_id = {source.id: source for source in sources}

    if len(sources_by_id) != len(sources):
        raise ValueError("Source IDs must be unique")

    findings = []

    for draft in drafts:
        source = sources_by_id.get(draft.source_id)

        if source is None:
            raise EvidenceValidationError(
                "unknown_source", f"Unknown source ID: {draft.source_id}"
            )

        snippet = _normalize_whitespace(draft.snippet)
        source_text = _normalize_whitespace(source.text)

        if snippet not in source_text:
            raise EvidenceValidationError(
                "snippet_not_in_source", f"Snippet not found in source {source.id}"
            )

        findings.append(
            Finding(
                id=f"finding_{uuid4().hex}",
                subquestion_id=subquestion.id,
                claim=draft.claim,
                source_url=source.url,
                snippet=draft.snippet,
                retrieved_at=source.retrieved_at,
            )
        )

    return findings
