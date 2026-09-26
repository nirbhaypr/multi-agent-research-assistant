"""Construct providers and resume a single durable run under a worker lease."""

from contextlib import ExitStack

import httpx
from openai import OpenAI
from redis import Redis

from multi_agent_research_assistant.config import Settings
from multi_agent_research_assistant.demo import DemoModel, DemoRetriever
from multi_agent_research_assistant.llm.budgeted_model import BudgetedModel
from multi_agent_research_assistant.orchestration.workflow import ResearchWorkflow
from multi_agent_research_assistant.retrieval.tavily import TavilyRetriever
from multi_agent_research_assistant.storage.checkpoints import RedisCheckpointer
from multi_agent_research_assistant.storage.redis_store import (
    RedisJournal,
    RedisRunStore,
)


def connect_store(settings: Settings) -> RedisRunStore:
    return RedisRunStore(
        Redis.from_url(settings.redis_url, socket_connect_timeout=5, socket_timeout=5),
        prefix=settings.redis_prefix,
        retention_seconds=settings.retention_seconds,
    )


def execute_run(settings: Settings, run_id: str, token: str):
    store = connect_store(settings)
    try:
        run = store.get(run_id)
        if run is None or run.terminal:
            return
        with ExitStack() as stack:
            if run.mode == "demo":
                retriever = DemoRetriever()

                def factory(budget, deadline, on_budget):
                    return DemoModel(
                        budget,
                        deadline,
                        on_budget,
                        output_cap=run.limits.max_output_tokens,
                    )
            else:
                settings.require_providers()
                client = stack.enter_context(
                    OpenAI(
                        api_key=settings.openai_api_key.get_secret_value(),
                        max_retries=0,
                    )
                )
                http = stack.enter_context(httpx.Client())
                retriever = TavilyRetriever(
                    http, settings.tavily_api_key.get_secret_value()
                )

                def factory(budget, deadline, on_budget):
                    return BudgetedModel(
                        client=client,
                        model=run.model_name,
                        budget=budget,
                        max_output_tokens=run.limits.max_output_tokens,
                        deadline=deadline,
                        on_budget=on_budget,
                    )

            saver = RedisCheckpointer(store, run_id, token)
            graph = ResearchWorkflow(
                model_factory=factory,
                retriever=retriever,
                journal=RedisJournal(store, run_id, token),
            ).build(saver)
            config = {"configurable": {"thread_id": run_id}, "recursion_limit": 128}
            initial = (
                None
                if saver.get_tuple(config)
                else {"run": run.model_dump(mode="json")}
            )
            graph.invoke(initial, config, durability="sync")
    finally:
        store.redis.close()
