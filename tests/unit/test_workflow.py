import time

import pytest
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


@pytest.mark.parametrize(
    ("partial", "all_invalid", "status", "claim_count", "revisions"),
    [
        (False, False, "completed", 2, 0),
        (True, False, "partial", 1, 1),
        (False, True, "failed", 0, 1),
    ],
)
def test_invalid_excerpts_are_traced_and_valid_findings_reach_review_and_writer(
    partial, all_invalid, status, claim_count, revisions
):
    calls = []
    journal = MemoryJournal()

    class MixedEvidenceModel(DemoModel):
        def invoke(self, **kwargs):
            result = super().invoke(**kwargs)
            if kwargs["schema"].__name__ == "ResearchDraft":
                invalid = result.findings[0].model_copy(
                    update={"snippet": "Completed runs ... seven days."}
                )
                result.findings = [invalid] + ([] if all_invalid else result.findings)
            return result

    workflow = ResearchWorkflow(
        model_factory=lambda budget, deadline, on_budget: MixedEvidenceModel(
            budget, deadline, on_budget, calls=calls, partial=partial
        ),
        retriever=DemoRetriever(partial=partial),
        journal=journal,
    )
    run = RunState.new("How long are completed and failed runs retained?", RunLimits())
    result = RunState.model_validate(
        workflow.build().invoke({"run": run.model_dump(mode="json")})["run"]
    )

    assert result.status == status
    assert len(result.report.claims if result.report else []) == claim_count
    assert result.revisions_used == revisions
    assert calls.count("ResearchDraft") == 1 + revisions
    assert result.budget.recorded_tokens == 140 * len(calls)
    traces = [r["trace"] for r in journal.records.values()]
    assert all(t["error"] is None for t in traces)
    assert "reviewer" in [t["agent"] for t in traces]
    for trace in (t for t in traces if t["agent"] == "researcher"):
        rejection = trace["produced"]["rejected_findings"][0]
        assert rejection == {
            "draft_index": 0,
            "source_id": trace["received"]["sources"][0]["id"],
            "reason": "snippet_not_in_source",
        }
    assert all("..." not in f.snippet for f in result.findings)
    assert "researcher failed" not in result.report_markdown
