"""FastAPI submits durable jobs; a separate worker performs the research."""

import asyncio
import json
import re
import secrets
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from redis.exceptions import RedisError

from multi_agent_research_assistant.config import Settings
from multi_agent_research_assistant.domain.runs import RunLimits
from multi_agent_research_assistant.domain.validation import validate_report_citations
from multi_agent_research_assistant.runtime import connect_store
from multi_agent_research_assistant.storage.redis_store import (
    IdempotencyConflict,
    QueueFull,
)


class ResearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    question: str = Field(min_length=1, max_length=4000)
    limits: RunLimits = Field(default_factory=RunLimits)


def create_app(settings: Settings | None = None, store=None) -> FastAPI:
    settings = settings or Settings()
    owned = store is None
    store = store or connect_store(settings)

    @asynccontextmanager
    async def lifespan(app):
        yield
        if owned:
            store.redis.close()

    app = FastAPI(
        title="Multi-Agent Research Assistant",
        version="1.0.0",
        lifespan=lifespan,
        description="Durable research runs, validated citations, bounded spending, and step traces.",
    )
    app.state.store = store

    def authorize(authorization: Annotated[str | None, Header()] = None):
        if settings.api_token:
            expected = "Bearer " + settings.api_token.get_secret_value()
            if not authorization or not secrets.compare_digest(authorization, expected):
                raise HTTPException(
                    401,
                    "A valid bearer token is required",
                    headers={"WWW-Authenticate": "Bearer"},
                )

    def get_run(run_id):
        if re.fullmatch(r"[a-f0-9]{32}", run_id) is None:
            raise HTTPException(404, "Run not found")
        run = store.get(run_id)
        if run is None:
            raise HTTPException(404, "Run not found")
        return run

    @app.exception_handler(RedisError)
    async def redis_unavailable(request, exc):
        return JSONResponse(
            status_code=503, content={"detail": "Research storage is unavailable"}
        )

    @app.get("/health")
    def health():
        store.redis.ping()
        return {"status": "ok", "mode": settings.research_mode}

    @app.post("/research", status_code=202, dependencies=[Depends(authorize)])
    def submit(
        body: ResearchRequest,
        response: Response,
        idempotency_key: Annotated[
            str | None, Header(min_length=1, max_length=200)
        ] = None,
    ):
        try:
            settings.require_providers()
        except ValueError as exc:
            raise HTTPException(503, str(exc)) from exc
        try:
            run, created = store.enqueue(
                body.question,
                body.limits,
                idempotency_key,
                mode=settings.research_mode,
                model_name="synthetic-fixture"
                if settings.research_mode == "demo"
                else settings.openai_model,
                max_pending=settings.max_pending_runs,
            )
        except IdempotencyConflict as exc:
            raise HTTPException(409, str(exc)) from exc
        except QueueFull as exc:
            raise HTTPException(429, str(exc), headers={"Retry-After": "5"}) from exc
        response.headers["Location"] = f"/research/{run.run_id}"
        return {
            "run_id": run.run_id,
            "status": run.status,
            "created": created,
            "mode": run.mode,
            "status_url": f"/research/{run.run_id}",
            "events_url": f"/research/{run.run_id}/events",
            "trace_url": f"/research/{run.run_id}/trace",
        }

    @app.get("/research/{run_id}", dependencies=[Depends(authorize)])
    def status(run_id: str):
        run = get_run(run_id)
        accepted = [f for f in run.findings if f.id in run.accepted_ids]
        if run.report:
            validate_report_citations(run.report, accepted)
        return {
            "run_id": run.run_id,
            "question": run.question,
            "status": run.status,
            "mode": run.mode,
            "model": run.model_name,
            "created_at": run.created_at,
            "deadline": run.deadline,
            "progress": {
                "next_node": run.next_node,
                "completed_steps": run.step,
                "processed_subquestions": run.index,
                "completed_subquestions": sum(
                    r.status == "sufficient" for r in run.reviews.values()
                ),
                "planned_subquestions": len(run.plan.subquestions) if run.plan else 0,
            },
            "budget": run.budget,
            "usage_complete": run.budget.blocked_reason != "usage_unknown",
            "searches": run.searches,
            "revisions_used": run.revisions_used,
            "report": run.report,
            "report_markdown": run.report_markdown,
            "citations": accepted,
            "gaps": run.gaps,
            "stop_reason": run.stop_reason,
            "writer_fallback": run.writer_fallback,
            "retrieval_issues": run.retrieval_issues,
        }

    @app.get("/research/{run_id}/trace", dependencies=[Depends(authorize)])
    def trace(run_id: str):
        get_run(run_id)
        return {"run_id": run_id, "steps": store.traces(run_id)}

    @app.get("/research/{run_id}/events", dependencies=[Depends(authorize)])
    async def events(
        run_id: str,
        request: Request,
        last_event_id: Annotated[str | None, Header()] = None,
    ):
        await asyncio.to_thread(get_run, run_id)
        try:
            offset = int(last_event_id) + 1 if last_event_id is not None else 0
            if offset < 0:
                raise ValueError
        except ValueError as exc:
            raise HTTPException(
                400, "Last-Event-ID must be a nonnegative integer"
            ) from exc

        async def stream():
            nonlocal offset
            while not await request.is_disconnected():
                items = await asyncio.to_thread(store.traces, run_id, offset)
                for item in items:
                    yield f"id: {offset}\nevent: step\ndata: {json.dumps(item)}\n\n"
                    offset += 1
                run = await asyncio.to_thread(get_run, run_id)
                if run.terminal:
                    yield f"event: done\ndata: {json.dumps({'status': run.status, 'run_id': run_id})}\n\n"
                    return
                yield f"event: progress\ndata: {json.dumps({'status': run.status, 'next_node': run.next_node})}\n\n"
                await asyncio.sleep(0.5)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app
