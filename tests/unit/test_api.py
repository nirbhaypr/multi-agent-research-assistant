import fakeredis
import pytest
from fastapi.testclient import TestClient

from multi_agent_research_assistant.api.app import create_app
from multi_agent_research_assistant.config import Settings
from multi_agent_research_assistant.demo import DemoModel, DemoRetriever
from multi_agent_research_assistant.orchestration.workflow import ResearchWorkflow
from multi_agent_research_assistant.storage.checkpoints import RedisCheckpointer
from multi_agent_research_assistant.storage.redis_store import (
    RedisJournal,
    RedisRunStore,
)


@pytest.fixture
def client_store():
    store = RedisRunStore(fakeredis.FakeRedis())
    with TestClient(
        create_app(Settings(_env_file=None, research_mode="demo"), store)
    ) as client:
        yield client, store


def finish(store, run_id):
    run = store.get(run_id)
    token = store.claim(run_id)
    graph = ResearchWorkflow(
        model_factory=lambda budget, deadline, on_budget: DemoModel(
            budget, deadline, on_budget
        ),
        retriever=DemoRetriever(),
        journal=RedisJournal(store, run_id, token),
    ).build(RedisCheckpointer(store, run_id, token))
    graph.invoke(
        {"run": run.model_dump(mode="json")},
        {"configurable": {"thread_id": run_id}},
        durability="sync",
    )


def test_async_submission_status_report_trace_and_resumable_events(client_store):
    client, store = client_store
    response = client.post(
        "/research", json={"question": "retention?"}, headers={"Idempotency-Key": "one"}
    )
    assert response.status_code == 202
    run_id = response.json()["run_id"]
    assert response.headers["location"] == f"/research/{run_id}"
    assert client.get(f"/research/{run_id}").json()["status"] == "queued"
    replay = client.post(
        "/research", json={"question": "retention?"}, headers={"Idempotency-Key": "one"}
    )
    assert replay.json()["run_id"] == run_id
    assert replay.json()["created"] is False
    finish(store, run_id)
    result = client.get(f"/research/{run_id}").json()
    assert result["status"] == "completed" and result["mode"] == "demo"
    assert len(result["citations"]) == 2
    assert result["report_markdown"]
    assert len(client.get(f"/research/{run_id}/trace").json()["steps"]) == 7
    stream = client.get(f"/research/{run_id}/events", headers={"Last-Event-ID": "4"})
    assert "id: 5" in stream.text and "id: 4" not in stream.text
    assert "event: done" in stream.text
    assert (
        client.get(
            f"/research/{run_id}/events", headers={"Last-Event-ID": "bad"}
        ).status_code
        == 400
    )


def test_validation_missing_runs_and_idempotency_conflict(client_store):
    client, _ = client_store
    assert client.post("/research", json={"question": " "}).status_code == 422
    assert (
        client.post(
            "/research", json={"question": "ok", "limits": {"max_seconds": 301}}
        ).status_code
        == 422
    )
    assert client.get("/research/missing").status_code == 404
    client.post(
        "/research", json={"question": "one"}, headers={"Idempotency-Key": "same"}
    )
    assert (
        client.post(
            "/research", json={"question": "two"}, headers={"Idempotency-Key": "same"}
        ).status_code
        == 409
    )


def test_optional_authentication_and_queue_admission():
    store = RedisRunStore(fakeredis.FakeRedis())
    settings = Settings(
        _env_file=None, research_mode="demo", api_token="test-token", max_pending_runs=1
    )
    with TestClient(create_app(settings, store)) as client:
        assert client.post("/research", json={"question": "ok"}).status_code == 401
        headers = {"Authorization": "Bearer test-token"}
        assert (
            client.post(
                "/research", json={"question": "one"}, headers=headers
            ).status_code
            == 202
        )
        assert (
            client.post(
                "/research", json={"question": "two"}, headers=headers
            ).status_code
            == 429
        )


def test_live_mode_requires_both_provider_keys():
    store = RedisRunStore(fakeredis.FakeRedis())
    settings = Settings(
        _env_file=None, research_mode="live", openai_api_key=None, tavily_api_key=None
    )
    with TestClient(create_app(settings, store)) as client:
        assert client.post("/research", json={"question": "ok"}).status_code == 503
