import pytest
from pydantic import ValidationError
from multi_agent_research_assistant.models import ResearchPlan, Finding, ReportClaim, ResearchReport
from multi_agent_research_assistant.validation import validate_report_citations, validate_research_plan

def test_accepts_valid_report():
    data = {
        "id": "finding_1",
        "subquestion_id": "sq_1",
        "claim": "Service Alpha retains completed runs for seven days.",
        "source_url": "https://example.com/retention-policy",
        "snippet": "Completed runs are retained for seven days.",
        "retrieved_at": "2026-09-25T10:00:00Z",
    }

    finding = Finding.model_validate(data)
    claims = ReportClaim.model_validate(
        {
            "text": "Test Claim",
            "finding_ids": ["finding_1"]
        }
    )
    report = ResearchReport.model_validate(
        {
            "title": "Test Title 1",
            "claims": [claims]
        }
    )

    validate_report_citations(report, [finding])
    assert report.claims[0].finding_ids[0] == finding.id


def test_rejects_report_with_invalid_findings():
    data = {
        "id": "finding_1",
        "subquestion_id": "sq_1",
        "claim": "Service Alpha retains completed runs for seven days.",
        "source_url": "https://example.com/retention-policy",
        "snippet": "Completed runs are retained for seven days.",
        "retrieved_at": "2026-09-25T10:00:00Z",
    }

    finding = Finding.model_validate(data)
    claims = ReportClaim.model_validate(
        {
            "text": "Test Claim",
            "finding_ids": ["finding_1", "finding_missing"]
        }
    )
    report = ResearchReport.model_validate(
        {
            "title": "Test Title 1",
            "claims": [claims]
        }
    )

    with pytest.raises(ValueError, match="finding_missing") as error:
        validate_report_citations(report, [finding])


def test_accepts_valid_research_plan():
    plan = ResearchPlan.model_validate(
            {
                "question": "q_1",
                "subquestions": [
                    {
                        "id": "sq_1",
                        "question": "Test sub question 1",
                        "completion_criteria": (
                            "Completion criteria 1"
                        ),
                    },
                    {
                        "id": "sq_2",
                        "question": "Test sub question 2",
                        "completion_criteria": (
                            "Completion criteria 2"
                        ),
                    }
                ],
            }
        )

    validate_research_plan(plan, question="q_1", max_subquestions=2)


def test_rejects_research_plan_with_duplicate_ids():
    plan = ResearchPlan.model_validate(
            {
                "question": "q_1",
                "subquestions": [
                    {
                        "id": "sq_1",
                        "question": "Test sub question 1",
                        "completion_criteria": (
                            "Completion criteria 1"
                        ),
                    },
                    {
                        "id": "sq_1",
                        "question": "Test sub question 2",
                        "completion_criteria": (
                            "Completion criteria 2"
                        ),
                    }
                ],
            }
        )

    with pytest.raises(ValueError, match="unique"):
        validate_research_plan(plan, question="q_1", max_subquestions=2)


def test_rejects_research_plan_with_too_many_subquestions():
    plan = ResearchPlan.model_validate(
                {
                    "question": "q_1",
                    "subquestions": [
                        {
                            "id": "sq_1",
                            "question": "Test sub question 1",
                            "completion_criteria": (
                                "Completion criteria 1"
                            ),
                        },
                        {
                            "id": "sq_2",
                            "question": "Test sub question 2",
                            "completion_criteria": (
                                "Completion criteria 2"
                            ),
                        }
                    ],
                }
            )   

    with pytest.raises(ValueError, match="exceeds"):
            validate_research_plan(plan, question="q_1", max_subquestions=1)


def test_rejects_research_plan_with_changed_question():
    plan = ResearchPlan.model_validate(
                {
                    "question": "q_1",
                    "subquestions": [
                        {
                            "id": "sq_1",
                            "question": "Test sub question 1",
                            "completion_criteria": (
                                "Completion criteria 1"
                            ),
                        }
                    ],
                }
            )   

    with pytest.raises(ValueError, match="preserve"):
            validate_research_plan(plan, question="q_2", max_subquestions=1)


def test_rejects_research_plan_with_invalid_configuration():
    plan = ResearchPlan.model_validate(
                {
                    "question": "q_1",
                    "subquestions": [
                        {
                            "id": "sq_1",
                            "question": "Test sub question 1",
                            "completion_criteria": (
                                "Completion criteria 1"
                            ),
                        }
                    ],
                }
            )   

    with pytest.raises(ValueError, match="at least 1"):
            validate_research_plan(plan, question="q_2", max_subquestions=0)

    

    