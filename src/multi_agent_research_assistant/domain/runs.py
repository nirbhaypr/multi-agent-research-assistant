"""Serializable run state: the supervisor's single source of truth."""

from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from .models import (
    Finding,
    ResearchPlan,
    ResearchReport,
    ResearchReview,
    SourceDocument,
    TokenBudgetState,
)


class RunLimits(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    max_subquestions: int = Field(default=3, ge=1, le=6)
    max_searches_per_subquestion: int = Field(default=2, ge=1, le=3)
    max_total_tokens: int = Field(default=24000, ge=500, le=100000)
    max_output_tokens: int = Field(default=2000, ge=100, le=4000)
    max_seconds: int = Field(default=240, ge=1, le=300)
    max_sources_per_search: int = Field(default=3, ge=1, le=5)
    max_sources_per_domain: int = Field(default=2, ge=1, le=3)


class RetrievalIssue(BaseModel):
    status: Literal[
        "no_results",
        "rate_limited",
        "paywalled",
        "timed_out",
        "unavailable",
        "invalid_source",
    ]
    url: str | None = None
    detail: str


class RetrievalResult(BaseModel):
    sources: list[SourceDocument] = Field(default_factory=list)
    issues: list[RetrievalIssue] = Field(default_factory=list)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)


class StepTrace(BaseModel):
    step_id: str
    agent: str
    status: str
    received: dict[str, Any]
    produced: dict[str, Any]
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    usage: list[dict[str, int] | None] = Field(default_factory=list)
    started_at: datetime
    latency_ms: float
    error: str | None = None


class RunState(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_id: str = Field(default_factory=lambda: uuid4().hex)
    question: str = Field(min_length=1, max_length=4000)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    deadline: float
    limits: RunLimits
    budget: TokenBudgetState
    status: Literal[
        "queued", "running", "completed", "partial", "failed", "timed_out"
    ] = "queued"
    next_node: Literal[
        "planner",
        "search",
        "researcher",
        "reviewer",
        "supervisor",
        "writer",
        "finish",
        "done",
    ] = "planner"
    step: int = 0
    plan: ResearchPlan | None = None
    index: int = 0
    revisions_used: int = 0
    searches: dict[str, int] = Field(default_factory=dict)
    sources: list[SourceDocument] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)
    accepted_ids: list[str] = Field(default_factory=list)
    reviews: dict[str, ResearchReview] = Field(default_factory=dict)
    gaps: dict[str, list[str]] = Field(default_factory=dict)
    seen_urls: list[str] = Field(default_factory=list)
    domain_counts: dict[str, int] = Field(default_factory=dict)
    source_cache: dict[str, SourceDocument] = Field(default_factory=dict)
    retrieval_issues: list[RetrievalIssue] = Field(default_factory=list)
    report: ResearchReport | None = None
    report_markdown: str | None = None
    stop_reason: str | None = None
    writer_fallback: bool = False

    @property
    def terminal(self) -> bool:
        return self.next_node == "done"

    @classmethod
    def new(cls, question: str, limits: RunLimits) -> "RunState":
        now = datetime.now(UTC)
        return cls(
            question=question.strip(),
            limits=limits,
            created_at=now,
            deadline=now.timestamp() + limits.max_seconds,
            budget=TokenBudgetState(max_total_tokens=limits.max_total_tokens),
        )
