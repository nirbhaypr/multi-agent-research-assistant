import pytest
from pydantic import ValidationError

from multi_agent_research_assistant.models import ResearchPlan, SubQuestion, Finding, ResearchReport, ReportClaim
from multi_agent_research_assistant.validation import validate_report_citations

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
                }
            ],
        }
    )

    assert len(plan.subquestions) == 1
    assert isinstance(plan.subquestions[0], SubQuestion)
    assert plan.subquestions[0].id == "sq_1"


def test_rejects_research_plan_without_subquestions():
    with pytest.raises(ValidationError) as error:
        ResearchPlan.model_validate(
            {
                "question": "Test question 2",
                "subquestions": []
            }
        )

    assert error.value.errors()[0]["loc"] == ("subquestions",)

def test_rejects_reserch_plan_subquestion_with_empty_question():
    with pytest.raises(ValidationError) as error:
        ResearchPlan.model_validate(
                {
                    "question": "Test question 3",
                    "subquestions": [
                        {
                            "id": "sq_2",
                            "question": " ",
                            "completion_criteria": "Completion criteria 2"
                        }
                    ]
                }
            )

    assert error.value.errors()[0]["loc"] == ("subquestions", 0, "question")


def test_rejects_research_plan_subquestion_with_empty_completion_criteria():
    with pytest.raises(ValidationError) as error:
        ResearchPlan.model_validate(
                {
                    "question": "Test question 3",
                    "subquestions": [
                        {
                            "id": "sq_3",
                            "question": "Test sub question 2",
                            "completion_criteria": " "
                        }
                    ]
                }
            )

    assert error.value.errors()[0]["loc"] == ("subquestions", 0, "completion_criteria")


def test_rejects_research_plan_with_empty_question():
    with pytest.raises(ValidationError) as error:
            ResearchPlan.model_validate(
                    {
                        "question": " ",
                        "subquestions": [
                            {
                                "id": "sq_4",
                                "question": "Test sub question 3",
                                "completion_criteria": "Completion criteria 4"
                            }
                        ]
                    }
                )
    
    assert error.value.errors()[0]["loc"] == ("question",)


def test_accepts_valid_finding():
    data = {
        "id": "finding_1",
        "subquestion_id": "sq_1",
        "claim": "Service Alpha retains completed runs for seven days.",
        "source_url": "https://example.com/retention-policy",
        "snippet": "Completed runs are retained for seven days.",
        "retrieved_at": "2026-09-25T10:00:00Z",
    }
    finding = Finding.model_validate(data)

    assert finding.id == "finding_1"
    assert finding.subquestion_id == "sq_1"


def test_rejects_finding_with_missing_source():
    data = {
            "id": "finding_1",
            "subquestion_id": "sq_1",
            "claim": "Service Alpha retains completed runs for seven days.",
            "snippet": "Completed runs are retained for seven days.",
            "retrieved_at": "2026-09-25T10:00:00Z",
        }

    with pytest.raises(ValidationError) as error:
        Finding.model_validate(data)

    assert error.value.errors()[0]["loc"] == ("source_url", )


def test_rejects_finding_with_invalid_source():
    data = {
            "id": "finding_1",
            "subquestion_id": "sq_1",
            "claim": "Service Alpha retains completed runs for seven days.",
            "source_url": "not-a-url",
            "snippet": "Completed runs are retained for seven days.",
            "retrieved_at": "2026-09-25T10:00:00Z",
        }

    with pytest.raises(ValidationError) as error:
        Finding.model_validate(data)

    assert error.value.errors()[0]["loc"] == ("source_url", )


def test_rejects_finding_with_blank_snippet():
    data = {
            "id": "finding_1",
            "subquestion_id": "sq_1",
            "claim": "Service Alpha retains completed runs for seven days.",
            "source_url": "https://example.com/retention-policy",
            "snippet": " ",
            "retrieved_at": "2026-09-25T10:00:00Z",
        }

    with pytest.raises(ValidationError) as error:
        Finding.model_validate(data)
    
    assert error.value.errors()[0]["loc"] == ("snippet", )


def test_rejects_finding_with_ambiguous_timestamp():
    data = {
             "id": "finding_1",
             "subquestion_id": "sq_1",
             "claim": "Service Alpha retains completed runs for seven days.",
             "source_url": "https://example.com/retention-policy",
             "snippet": "Completed runs are retained for seven days.",
             "retrieved_at": "2026-09-25T10:00:00",
        }
    with pytest.raises(ValidationError) as error:
        Finding.model_validate(data)
        
    assert error.value.errors()[0]["loc"] == ("retrieved_at", )


def test_rejects_report_claim_with_empty_finding_ids():
    with pytest.raises(ValidationError) as error:
        claims = ReportClaim.model_validate(
            {
                "text": "Test Claim",
                "finding_ids": []
            }
        )

    assert error.value.errors()[0]["loc"] == ("finding_ids", )


def test_rejects_research_report_with_empty_claims():
    with pytest.raises(ValidationError) as error:
        ResearchReport.model_validate(
            {
                "title": "Test Title 1",
                "claims": []
            }
        )

    assert error.value.errors()[0]["loc"] == ("claims",)