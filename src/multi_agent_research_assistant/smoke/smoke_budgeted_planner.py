import json
import time

from openai import OpenAI

from multi_agent_research_assistant.agents.planner import create_plan
from multi_agent_research_assistant.config import Settings
from multi_agent_research_assistant.domain.models import TokenBudgetState
from multi_agent_research_assistant.llm.budgeted_model import BudgetedModel


def main():
    settings = Settings()
    if settings.openai_api_key is None:
        raise SystemExit("Set OPENAI_API_KEY locally")
    usage = []
    started = time.monotonic()
    with OpenAI(
        api_key=settings.openai_api_key.get_secret_value(), max_retries=0
    ) as client:
        model = BudgetedModel(
            client=client,
            model=settings.openai_model,
            budget=TokenBudgetState(max_total_tokens=4000),
            max_output_tokens=600,
            deadline=time.time() + 45,
        )
        try:
            plan = create_plan(
                "How does Redis AOF help recover after a process crash?",
                model=model,
                max_subquestions=1,
                record_usage=usage.append,
            )
            output = {"status": "completed", "plan": plan.model_dump(mode="json")}
        except Exception as exc:
            output = {"status": "failed", "error": type(exc).__name__}
        output.update(
            model=settings.openai_model,
            budget=model.budget.model_dump(),
            elapsed_seconds=round(time.monotonic() - started, 3),
            usage=[u.model_dump() if u is not None else None for u in usage],
        )
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
