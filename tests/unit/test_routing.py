import pytest

from multi_agent_research_assistant.domain.models import ResearchReview
from multi_agent_research_assistant.orchestration.routing import decide_after_review


@pytest.fixture
def review():
    return ResearchReview(
        status="insufficient",
        accepted_finding_ids=["finding_1"],
        gaps=["Failed-run retention is not established."],
        reason="Only completed-run retention is supported.",
    )


@pytest.mark.parametrize(
    ("status", "revisions_used", "budget_available", "action", "reason"),
    [
        ("sufficient", 0, True, "accept", "evidence_sufficient"),
        ("sufficient", 1, False, "accept", "evidence_sufficient"),
        ("insufficient", 0, True, "revise", "revision_allowed"),
        ("insufficient", 0, False, "finish_incomplete", "retry_budget_unavailable"),
        ("insufficient", 1, True, "finish_incomplete", "revision_limit_reached"),
        ("insufficient", 1, False, "finish_incomplete", "revision_limit_reached"),
        ("insufficient", 2, True, "finish_incomplete", "revision_limit_reached"),
    ],
)
def test_routing_decisions(
    review,
    status,
    revisions_used,
    budget_available,
    action,
    reason,
):
    assessment = ResearchReview.model_validate(
        {
            **review.model_dump(),
            "status": status,
            "gaps": [] if status == "sufficient" else review.gaps,
            "reason": "Controlled assessment for the routing test.",
        }
    )

    decision = decide_after_review(
        assessment,
        revisions_used=revisions_used,
        retry_budget_available=budget_available,
    )

    assert decision.action == action
    assert decision.reason == reason


@pytest.mark.parametrize(
    ("revisions_used", "expected_action"),
    [(0, "revise"), (1, "finish_incomplete")],
)
def test_empty_accepted_findings_still_follow_revision_limit(
    review,
    revisions_used,
    expected_action,
):
    assessment = ResearchReview.model_validate(
        {
            **review.model_dump(),
            "accepted_finding_ids": [],
            "gaps": ["No supported retention findings are available."],
            "reason": "No findings were accepted.",
        }
    )

    decision = decide_after_review(
        assessment,
        revisions_used=revisions_used,
        retry_budget_available=True,
    )

    assert decision.action == expected_action


def test_routing_preserves_accepted_findings_and_gaps(review):
    before = review.model_dump()

    decision = decide_after_review(
        review,
        revisions_used=1,
        retry_budget_available=True,
    )

    assert decision.action == "finish_incomplete"
    assert review.model_dump() == before


def test_rejects_negative_revision_count(review):
    with pytest.raises(ValueError, match="revisions_used must not be negative"):
        decide_after_review(
            review,
            revisions_used=-1,
            retry_budget_available=True,
        )
