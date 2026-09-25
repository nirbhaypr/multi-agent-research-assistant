from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from collections.abc import Callable

from .models import ResearchPlan, TokenUsage
from .validation import validate_research_plan
from .model_calls import invoke_structured


def create_plan(
        question: str,
        *,
        llm: ChatOpenAI,
        max_subquestions: int = 3,
        record_usage: Callable[[TokenUsage | None], None] | None = None
) -> ResearchPlan:
    question = question.strip()

    if not question:
        raise ValueError("question must not be blank")

    if max_subquestions < 1:
        raise ValueError("max_subquestions must be at least 1")

    instructions = SystemMessage(
        content = (
            "Break the research question into focused, independently "
            "researchable subquestions. "
            f"Return between 1 and {max_subquestions} subquestions. "
            "Preserve the original question exactly. "
            "Preserve its scope and time constraints. "
            "Use unique subquestion IDs such as sq_1 and sq_2. "
            "Give each subquestion concrete completion criteria describing "
            "the evidence needed to answer it. "
            "Avoid overlapping subquestions. "
            "Produce the plan without answering the research question."
        )
    )
    query = HumanMessage(content=question)

    plan = invoke_structured(
        llm=llm,
        schema=ResearchPlan,
        messages=[instructions, query],
        record_usage=record_usage,
    )

    validate_research_plan(
        plan,
        question=question,
        max_subquestions=max_subquestions,
    )

    return plan