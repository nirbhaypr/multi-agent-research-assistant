from typing import Annotated, Literal

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    model_validator,
)


class SubQuestion(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    completion_criteria: str = Field(min_length=1)


class ResearchPlan(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    question: str = Field(min_length=1)
    subquestions: list[SubQuestion] = Field(min_length=1)


class SourceDocument(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    id: str = Field(min_length=1)
    url: HttpUrl
    text: str = Field(min_length=1)
    retrieved_at: AwareDatetime


class FindingDraft(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    source_id: str = Field(min_length=1)
    claim: str = Field(min_length=1)
    snippet: str = Field(min_length=1)


class Finding(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    id: str = Field(min_length=1)
    subquestion_id: str = Field(min_length=1)
    claim: str = Field(min_length=1)
    source_url: HttpUrl
    snippet: str = Field(min_length=1)
    retrieved_at: AwareDatetime


class ReportClaim(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    text: str = Field(min_length=1)
    finding_ids: list[str] = Field(min_length=1)


class ResearchReport(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    title: str = Field(min_length=1)
    claims: list[ReportClaim] = Field(min_length=1)


class ResearchDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    findings: list[FindingDraft]


EvidenceRejectionReason = Literal["unknown_source", "snippet_not_in_source"]


class RejectedFinding(BaseModel):
    """A draft rejected by provenance checks; indices refer to ResearchDraft.findings."""

    model_config = ConfigDict(extra="forbid")

    draft_index: int = Field(ge=0)
    source_id: str = Field(min_length=1)
    reason: EvidenceRejectionReason


class ResearchResult(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    subquestion_id: str = Field(min_length=1)
    status: Literal["findings_found", "no_evidence"]
    findings: list[Finding]
    rejected_findings: list[RejectedFinding] = Field(default_factory=list)


class TokenUsage(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
    )

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)

    @model_validator(mode="after")
    def check_total(self):
        if self.total_tokens != self.input_tokens + self.output_tokens:
            raise ValueError("total_tokens must equal input_tokens plus output_tokens")
        return self


class ResearchReview(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    status: Literal["sufficient", "insufficient"]
    accepted_finding_ids: list[str]
    gaps: list[Annotated[str, Field(min_length=1)]]
    reason: str = Field(min_length=1)


class ReviewDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["accept", "revise", "finish_incomplete"]
    reason: Literal[
        "evidence_sufficient",
        "revision_allowed",
        "revision_limit_reached",
        "retry_budget_unavailable",
    ]


class TokenBudgetState(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    max_total_tokens: int = Field(gt=0)
    recorded_tokens: int = Field(default=0, ge=0)
    reserved_tokens: int = Field(default=0, ge=0)
    blocked_reason: Literal["usage_unknown", "reservation_exceeded"] | None = None
