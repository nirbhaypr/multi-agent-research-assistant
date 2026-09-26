from multi_agent_research_assistant.domain.models import ResearchReview, ReviewDecision


def decide_after_review(
    review: ResearchReview,
    *,
    revisions_used: int,
    retry_budget_available: bool,
) -> ReviewDecision:
    """Route a validated review; revisions_used is shared across the whole run."""
    if revisions_used < 0:
        raise ValueError("revisions_used must not be negative")

    if review.status == "sufficient":
        return ReviewDecision(
            action="accept",
            reason="evidence_sufficient",
        )

    if revisions_used >= 1:
        return ReviewDecision(
            action="finish_incomplete",
            reason="revision_limit_reached",
        )

    if not retry_budget_available:
        return ReviewDecision(
            action="finish_incomplete",
            reason="retry_budget_unavailable",
        )

    return ReviewDecision(
        action="revise",
        reason="revision_allowed",
    )
