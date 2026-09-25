from .models import Finding, ResearchReport


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