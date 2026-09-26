import time

from langgraph.checkpoint.memory import InMemorySaver

from multi_agent_research_assistant.demo import DemoModel, DemoRetriever
from multi_agent_research_assistant.domain.runs import RunLimits, RunState
from multi_agent_research_assistant.orchestration.workflow import (
    MemoryJournal,
    ResearchWorkflow,
)


def setup_run(*, partial=False, tokens=24000):
    calls = []
    journal = MemoryJournal()

    def factory(budget, deadline, on_budget):
        return DemoModel(budget, deadline, on_budget, calls=calls, partial=partial)

    workflow = ResearchWorkflow(
        model_factory=factory, retriever=DemoRetriever(partial=partial), journal=journal
    )
    run = RunState.new(
        "How long are completed and failed runs retained?",
        RunLimits(max_total_tokens=tokens),
    )
    return workflow, run, journal, calls


def test_complete_graph_citations_trace_and_cumulative_budget():
    workflow, run, journal, calls = setup_run()
    result = RunState.model_validate(
        workflow.build().invoke({"run": run.model_dump(mode="json")})["run"]
    )
    assert result.status == "completed"
    assert len(result.report.claims) == 2
    assert result.budget.recorded_tokens == 140 * len(calls)
    assert "https://example.com/fixtures/retention" in result.report_markdown
    assert [r["trace"]["agent"] for r in journal.records.values()] == [
        "planner",
        "search",
        "researcher",
        "reviewer",
        "supervisor",
        "writer",
        "finish",
    ]
    assert all(r["trace"]["latency_ms"] >= 0 for r in journal.records.values())


def test_partial_evidence_allows_exactly_one_revision():
    workflow, run, journal, calls = setup_run(partial=True)
    result = RunState.model_validate(
        workflow.build().invoke({"run": run.model_dump(mode="json")})["run"]
    )
    assert result.status == "partial"
    assert result.revisions_used == 1
    assert result.searches == {"sq_retention": 2}
    assert calls.count("ResearchDraft") == 2
    assert "Failed-run retention is missing." in result.report_markdown


def test_budget_exhaustion_stops_calls_without_fabricating_a_report():
    workflow, run, journal, calls = setup_run(tokens=500)
    result = RunState.model_validate(
        workflow.build().invoke({"run": run.model_dump(mode="json")})["run"]
    )
    assert result.status == "failed"
    assert result.stop_reason == "token_limit"
    assert result.report is None
    assert calls == []


def test_expired_run_performs_no_model_calls():
    workflow, run, journal, calls = setup_run()
    run.deadline = time.time() - 1
    result = RunState.model_validate(
        workflow.build().invoke({"run": run.model_dump(mode="json")})["run"]
    )
    assert result.status == "timed_out"
    assert calls == []


def test_completed_node_is_cached_when_reentered():
    workflow, run, journal, calls = setup_run()
    node = workflow._node("planner")
    first = node({"run": run.model_dump(mode="json")})
    second = node({"run": run.model_dump(mode="json")})
    assert first == second
    assert calls == ["ResearchPlan"]
    assert len(journal.records) == 1


def test_inflight_reservation_is_not_replayed():
    workflow, run, journal, calls = setup_run()
    budget = run.budget.model_copy(update={"reserved_tokens": 2100})
    journal.begin("000:planner", {}, budget.model_dump())
    result = RunState.model_validate(
        workflow.build().invoke({"run": run.model_dump(mode="json")})["run"]
    )
    assert result.status == "failed"
    assert result.budget.blocked_reason == "usage_unknown"
    assert result.stop_reason == "interrupted_operation"
    assert calls == []


def test_langgraph_resumes_checkpoint_without_replaying_planner():
    workflow, run, journal, calls = setup_run()
    saver = InMemorySaver()
    graph = workflow.build(saver)
    config = {"configurable": {"thread_id": run.run_id}}
    graph.invoke(
        {"run": run.model_dump(mode="json")}, config, interrupt_after=["planner"]
    )
    result = RunState.model_validate(graph.invoke(None, config)["run"])
    assert result.status == "completed"
    assert calls.count("ResearchPlan") == 1


def test_trace_contains_prompt_schema_parsed_output_and_usage():
    workflow, run, journal, calls = setup_run()
    workflow.build().invoke({"run": run.model_dump(mode="json")})
    events = journal.records["000:planner"]["trace"]["tool_calls"]
    event = next(e for e in events if e["tool"] == "structured_model_call")
    assert events[0]["output"]["reserved_tokens"] == 2100
    assert event["input"]["messages"][0]["role"] == "system"
    assert "subquestions" in event["input"]["schema"]["properties"]
    assert event["output"]["parsed"]["question"] == run.question
    assert event["usage"][0]["total_tokens"] == 140
    assert event["latency_ms"] >= 0
