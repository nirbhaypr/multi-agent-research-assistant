import fakeredis
import pytest

from multi_agent_research_assistant.demo import DemoModel, DemoRetriever
from multi_agent_research_assistant.domain.runs import RunLimits, RunState
from multi_agent_research_assistant.orchestration.workflow import ResearchWorkflow
from multi_agent_research_assistant.storage.checkpoints import RedisCheckpointer
from multi_agent_research_assistant.storage.redis_store import (
    IdempotencyConflict,
    LeaseLost,
    RedisJournal,
    RedisRunStore,
)


def store():
    return RedisRunStore(fakeredis.FakeRedis())


def workflow(db, run, token, calls):
    def factory(budget, deadline, on_budget):
        return DemoModel(budget, deadline, on_budget, calls=calls)

    return ResearchWorkflow(
        model_factory=factory,
        retriever=DemoRetriever(),
        journal=RedisJournal(db, run.run_id, token),
    ).build(RedisCheckpointer(db, run.run_id, token))


def test_idempotency_and_conflict():
    db = store()
    first, created = db.enqueue("retention", RunLimits(), "request-1")
    second, created_again = db.enqueue("retention", RunLimits(), "request-1")
    assert first == second and created and not created_again
    assert db.pending() == [first.run_id]
    with pytest.raises(IdempotencyConflict):
        db.enqueue("different", RunLimits(), "request-1")


def test_only_lease_owner_can_write():
    db = store()
    run, _ = db.enqueue("retention", RunLimits())
    token = db.claim(run.run_id)
    assert db.claim(run.run_id) is None
    db.redis.delete(db.key(run.run_id, "lease"))
    replacement = db.claim(run.run_id)
    with pytest.raises(LeaseLost):
        RedisJournal(db, run.run_id, token).begin("0", {}, run.budget.model_dump())
    db.release(run.run_id, token)
    db.renew(run.run_id, replacement)


def test_redis_checkpoint_resumes_with_new_graph_and_no_duplicate_calls():
    db = store()
    run, _ = db.enqueue("retention", RunLimits())
    calls = []
    token = db.claim(run.run_id)
    config = {"configurable": {"thread_id": run.run_id}}
    graph = workflow(db, run, token, calls)
    graph.invoke(
        {"run": run.model_dump(mode="json")},
        config,
        interrupt_after=["planner"],
        durability="sync",
    )
    assert calls == ["ResearchPlan"]
    assert db.get(run.run_id).next_node == "search"
    db.release(run.run_id, token)
    token = db.claim(run.run_id)
    restored = workflow(db, run, token, calls)
    result = RunState.model_validate(
        restored.invoke(None, config, durability="sync")["run"]
    )
    assert result.status == "completed"
    assert calls.count("ResearchPlan") == 1
    assert db.get(run.run_id) == result
    assert db.pending() == []
    assert len(db.traces(run.run_id)) == 7
    saver = RedisCheckpointer(db, run.run_id, token)
    assert len(list(saver.list(config))) >= 7


def test_inflight_budget_is_visible_to_status_readers():
    db = store()
    run, _ = db.enqueue("retention", RunLimits())
    token = db.claim(run.run_id)
    journal = RedisJournal(db, run.run_id, token)
    journal.begin("000:planner", {}, run.budget.model_dump())
    assert db.get(run.run_id).status == "running"
    reserved = run.budget.model_copy(update={"reserved_tokens": 2100})
    journal.budget("000:planner", reserved.model_dump())
    assert db.get(run.run_id).budget.reserved_tokens == 2100
