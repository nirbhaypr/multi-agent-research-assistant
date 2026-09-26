"""Run with TEST_REDIS_URL=redis://localhost:6397/0."""

import os
import socket
import subprocess
import sys
import time
from uuid import uuid4

import httpx
import pytest
from redis import Redis

from multi_agent_research_assistant.config import Settings
from multi_agent_research_assistant.domain.runs import RunLimits
from multi_agent_research_assistant.runtime import connect_store
from multi_agent_research_assistant.worker import work_once

pytestmark = pytest.mark.integration


@pytest.fixture
def settings():
    url = os.environ.get("TEST_REDIS_URL")
    if not url:
        pytest.skip("Set TEST_REDIS_URL to run real Redis/process tests")
    config = Settings(
        _env_file=None,
        redis_url=url,
        redis_prefix=f"test:{uuid4().hex}",
        research_mode="demo",
    )
    Redis.from_url(url).ping()
    yield config
    client = Redis.from_url(url)
    for key in client.scan_iter(config.redis_prefix + ":*"):
        client.delete(key)
    client.close()


def hang(settings, run_id, token):
    time.sleep(60)


def test_worker_completes_a_cited_run_under_five_minutes(settings):
    store = connect_store(settings)
    run, _ = store.enqueue(
        "How long are completed and failed runs retained?", RunLimits(), mode="demo"
    )
    started = time.monotonic()
    assert work_once(settings)
    result = store.get(run.run_id)
    assert result.status == "completed"
    assert time.monotonic() - started < 300
    assert len(result.report.claims) == 2
    assert result.budget.recorded_tokens == 560
    assert len(store.traces(run.run_id)) == 7


@pytest.mark.parametrize("boundary", ["completed", "reserved"])
def test_killed_process_recovers_without_replaying_completed_or_unknown_calls(
    settings, tmp_path, boundary
):
    store = connect_store(settings)
    run, _ = store.enqueue("retention?", RunLimits(), mode="demo")
    token = store.claim(run.run_id, lease_seconds=2)
    ready = tmp_path / "ready"
    script = """
import sys,time
from pathlib import Path
from multi_agent_research_assistant.config import Settings
from multi_agent_research_assistant.runtime import execute_run
from multi_agent_research_assistant.storage.redis_store import RedisJournal
boundary,ready,run_id,token=sys.argv[1:]
if boundary=='completed':
    original=RedisJournal.complete
    def complete(self,step_id,state,trace):
        original(self,step_id,state,trace)
        if trace['agent']=='planner':
            Path(ready).touch()
            time.sleep(60)
    RedisJournal.complete=complete
else:
    original=RedisJournal.budget
    def budget(self,step_id,value):
        original(self,step_id,value)
        if value['reserved_tokens']:
            Path(ready).touch()
            time.sleep(60)
    RedisJournal.budget=budget
execute_run(Settings(),run_id,token)
"""
    env = {
        **os.environ,
        "REDIS_URL": settings.redis_url,
        "REDIS_PREFIX": settings.redis_prefix,
        "RESEARCH_MODE": "demo",
    }
    process = subprocess.Popen(
        [sys.executable, "-c", script, boundary, str(ready), run.run_id, token], env=env
    )
    try:
        end = time.monotonic() + 10
        while not ready.exists() and time.monotonic() < end:
            assert process.poll() is None
            time.sleep(0.05)
        assert ready.exists()
        process.kill()
        process.wait(timeout=5)
        while store.redis.exists(store.key(run.run_id, "lease")):
            time.sleep(0.05)
        assert work_once(settings)
        result = store.get(run.run_id)
        traces = store.traces(run.run_id)
        if boundary == "completed":
            assert result.status == "completed"
            assert result.budget.recorded_tokens == 560
            assert sum(t["agent"] == "planner" for t in traces) == 1
        else:
            assert result.status == "failed"
            assert result.stop_reason == "interrupted_operation"
            assert result.budget.blocked_reason == "usage_unknown"
            assert result.budget.recorded_tokens == 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


def test_parent_enforces_wall_clock_even_when_child_hangs(settings, monkeypatch):
    store = connect_store(settings)
    run, _ = store.enqueue("retention?", RunLimits(max_seconds=1), mode="demo")
    monkeypatch.setattr("multi_agent_research_assistant.worker.execute_run", hang)
    started = time.monotonic()
    assert work_once(settings)
    assert time.monotonic() - started < 3
    assert store.get(run.run_id).status == "timed_out"
    assert store.traces(run.run_id)[-1]["error"] == "deadline"


def test_real_http_api_and_worker_return_report_and_trace(settings):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = {
        **os.environ,
        "REDIS_URL": settings.redis_url,
        "REDIS_PREFIX": settings.redis_prefix,
        "RESEARCH_MODE": "demo",
        "API_TOKEN": "",
    }
    server = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "multi_agent_research_assistant.api.app:create_app",
            "--factory",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "error",
        ],
        env=env,
    )
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=5) as client:
            end = time.monotonic() + 10
            while True:
                try:
                    if client.get("/health").status_code == 200:
                        break
                except httpx.ConnectError:
                    pass
                assert server.poll() is None and time.monotonic() < end
                time.sleep(0.05)
            response = client.post("/research", json={"question": "retention?"})
            assert response.status_code == 202
            run_id = response.json()["run_id"]
            assert work_once(settings)
            result = client.get(f"/research/{run_id}").json()
            assert result["status"] == "completed"
            assert result["mode"] == "demo"
            assert len(result["citations"]) == 2
            assert len(client.get(f"/research/{run_id}/trace").json()["steps"]) == 7
            assert "event: done" in client.get(f"/research/{run_id}/events").text
    finally:
        server.terminate()
        server.wait(timeout=5)
