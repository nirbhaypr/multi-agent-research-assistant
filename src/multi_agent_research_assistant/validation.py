from .models import Finding, ResearchReport, ResearchPlan


def validate_report_citations(
        report: ResearchReport,
        findings: list[Finding]
) -> None:
    available_ids = {finding.id for finding in findings}

    for idx, claim in enumerate(report.claims, start=1):
        unknown_ids = set(claim.finding_ids) - available_ids

        if unknown_ids:
            references = ", ".join(sorted(unknown_ids))
            raise ValueError(
                f"Claim {idx} references unknown findings: {references}"
            )


def validate_research_plan(
        plan: ResearchPlan,
        *,
        question: str,
        max_subquestions: int
) -> None:
    if max_subquestions < 1:
        raise ValueError("max_subquestions must be at least 1")

    if plan.question != question.strip():
        raise ValueError("Plan must preserve the original question")

    if len(plan.subquestions) > max_subquestions:
        raise ValueError(f"Plan exceeds the limit of {max_subquestions} sub_questions")

    ids = [subquestion.id for subquestion in plan.subquestions]

    if len(ids) != len(set(ids)):
        raise ValueError("Subquestion IDs must be unique")