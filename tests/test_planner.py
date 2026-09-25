import pytest
from unittest.mock import Mock
from langchain_core.messages import AIMessage

from multi_agent_research_assistant.models import ResearchPlan
from multi_agent_research_assistant.planner import create_plan


def test_returns_valid_generated_plan():
    question = "How does Redis persist data?"

    expected_plan = ResearchPlan.model_validate(
        {
            "question": question,
            "subquestions": [
                {
                    "id": "sq_1",
                    "question": "How do RDB and AOF differ?",
                    "completion_criteria": (
                        "Explain recovery behavior and data-loss tradeoffs."
                    ),
                }
            ],
        }
    )

    llm = Mock()
    structured_llm = llm.with_structured_output.return_value

    structured_llm.invoke.return_value = {
        "raw": AIMessage(
            content="",
            response_metadata={"status": "completed"},
        ),
        "parsed": expected_plan,
        "parsing_error": None,
    }

    result = create_plan(question, llm=llm)

    llm.with_structured_output.assert_called_once_with(
        ResearchPlan,
        method="json_schema",
        strict=True,
        include_raw=True,
    )

    assert result == expected_plan
    structured_llm.invoke.assert_called_once()

    request = structured_llm.invoke.call_args

    assert request[0][0][-1].content == question


def test_rejects_plan_input_with_empty_question():
    llm = Mock()

    with pytest.raises(ValueError, match="blank"):
        create_plan("   ", llm=llm)

    llm.with_structured_output.assert_not_called()


def test_rejects_plan_input_with_zero_max_subquestions():
    llm = Mock()

    with pytest.raises(ValueError, match="at least 1"):
        create_plan(
            "How does Redis persist data?",
            llm=llm,
            max_subquestions=0,
        )

    llm.with_structured_output.assert_not_called()


def test_rejects_generated_plan_with_parsed_none():
    question = "How does Redis persist data?"

    expected_plan = ResearchPlan.model_validate(
        {
            "question": question,
            "subquestions": [
                {
                    "id": "sq_1",
                    "question": "How do RDB and AOF differ?",
                    "completion_criteria": (
                        "Explain recovery behavior and data-loss tradeoffs."
                    ),
                }
            ],
        }
    )

    llm = Mock()
    structured_llm = llm.with_structured_output.return_value

    structured_llm.invoke.return_value = {
        "raw": AIMessage(
            content="",
            response_metadata={"status": "completed"},
        ),
        "parsed": None,
        "parsing_error": None,
    }

    with pytest.raises(RuntimeError, match="no parsed output"):
        result = create_plan(question, llm=llm)


def test_rejects_generated_plan_with_incomplete_status():
    question = "How does Redis persist data?"

    expected_plan = ResearchPlan.model_validate(
        {
            "question": question,
            "subquestions": [
                {
                    "id": "sq_1",
                    "question": "How do RDB and AOF differ?",
                    "completion_criteria": (
                        "Explain recovery behavior and data-loss tradeoffs."
                    ),
                }
            ],
        }
    )

    llm = Mock()
    structured_llm = llm.with_structured_output.return_value

    structured_llm.invoke.return_value = {
        "raw": AIMessage(
            content="",
            response_metadata={"status": "incomplete"},
        ),
        "parsed": expected_plan,
        "parsing_error": None,
    }


    with pytest.raises(RuntimeError, match="not completed"):
            result = create_plan(question, llm=llm)



def test_rejects_generated_plan_with_more_than_max_subquestions():
    question = "How does Redis persist data?"

    expected_plan = ResearchPlan.model_validate(
        {
            "question": question,
            "subquestions": [
                {
                    "id": "sq_1",
                    "question": "How do RDB and AOF differ?",
                    "completion_criteria": (
                        "Explain recovery behavior and data-loss tradeoffs."
                    ),
                },
                {
                    "id": "sq_2",
                    "question": "How do RDB and AOF not differ?",
                    "completion_criteria": (
                        "Explain tradeoffs."
                    ),
                }
            ],
        }
    )

    llm = Mock()
    structured_llm = llm.with_structured_output.return_value

    structured_llm.invoke.return_value = {
        "raw": AIMessage(
            content="",
            response_metadata={"status": "completed"},
        ),
        "parsed": expected_plan,
        "parsing_error": None,
    }


    with pytest.raises(ValueError, match="exceeds the limit"):
            result = create_plan(question, llm=llm, max_subquestions=1)


def test_rejects_generated_plan_with_parsing_error():
    llm = Mock()
    structured_llm = llm.with_structured_output.return_value
    parsing_error = ValueError("invalid output")

    structured_llm.invoke.return_value = {
        "raw": AIMessage(
            content="",
            response_metadata={"status": "completed"},
        ),
        "parsed": None,
        "parsing_error": parsing_error,
    }

    with pytest.raises(
        RuntimeError,
        match="could not be parsed",
    ) as error:
        create_plan(
            "How does Redis persist data?",
            llm=llm,
        )

    assert error.value.__cause__ is parsing_error