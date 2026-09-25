from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from .models import ResearchPlan
from .validation import validate_research_plan


def create_plan(
        question: str,
        *,
        llm: ChatOpenAI,
        max_subquestions: int = 3,
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

    structured_llm = llm.with_structured_output(
        ResearchPlan,
        method="json_schema",
        strict=True,
        include_raw=True
    )

    response = structured_llm.invoke(
        [
            instructions,
            query
        ]
    )

    metadata = response["raw"].response_metadata

    if metadata.get("status") not in (None, "completed"):
        raise RuntimeError("Planner response was not completed")

    if metadata.get("finish_reason") in ("length", "content_filter"):
        raise RuntimeError("Planner response was not completed")

    if response["parsing_error"] is not None:
        raise RuntimeError(
            "Planner output could not be parsed"
        ) from response["parsing_error"]

    plan = response["parsed"]

    if plan is None:
        raise RuntimeError("Planner returned no parsed plan")

    validate_research_plan(
        plan,
        question=question,
        max_subquestions=max_subquestions
    )

    return plan
    