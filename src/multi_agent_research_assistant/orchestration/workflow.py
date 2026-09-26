"""Sequential LangGraph workflow with explicit supervisor decisions."""

import time
from datetime import UTC, datetime
from typing import Callable, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph

from multi_agent_research_assistant.agents.planner import create_plan
from multi_agent_research_assistant.agents.researcher import research_subquestion
from multi_agent_research_assistant.agents.reviewer import review_research
from multi_agent_research_assistant.agents.writer import write_report
from multi_agent_research_assistant.domain.models import (
    ReportClaim,
    ResearchReport,
    TokenBudgetState,
)
from multi_agent_research_assistant.domain.runs import (
    RetrievalResult,
    RunState,
    StepTrace,
)
from multi_agent_research_assistant.domain.validation import validate_report_citations
from multi_agent_research_assistant.llm.tracing import TracedModel
from multi_agent_research_assistant.orchestration.budgets import (
    BudgetBlocked,
    settle_tokens,
)
from multi_agent_research_assistant.orchestration.routing import decide_after_review


class Retriever(Protocol):
    def retrieve(self, query: str, *, state: RunState) -> RetrievalResult: ...


class Journal(Protocol):
    def get(self, step_id: str) -> dict | None: ...
    def begin(self, step_id: str, received: dict, budget: dict) -> None: ...
    def budget(self, step_id: str, budget: dict) -> None: ...
    def complete(self, step_id: str, state: dict, trace: dict) -> None: ...


class MemoryJournal:
    """Offline fixture; deployed runs use RedisJournal."""

    def __init__(self):
        self.records = {}

    def get(self, step_id):
        return self.records.get(step_id)

    def begin(self, step_id, received, budget):
        self.records[step_id] = {
            "status": "started",
            "received": received,
            "budget": budget,
        }

    def budget(self, step_id, budget):
        self.records[step_id]["budget"] = budget

    def complete(self, step_id, state, trace):
        self.records[step_id].update(status="completed", state=state, trace=trace)


class GraphState(TypedDict):
    run: dict


def render_report(run: RunState) -> str:
    if run.report is None:
        body = "# Research incomplete\n\nNo supported claims were available."
    else:
        findings = {f.id: f for f in run.findings}
        lines = [f"# {run.report.title}", ""]
        for claim in run.report.claims:
            links = " ".join(
                f"[{fid}]({findings[fid].source_url})" for fid in claim.finding_ids
            )
            lines.append(f"- {claim.text} {links}")
        body = "\n".join(lines)
    gaps = [gap for items in run.gaps.values() for gap in items]
    if run.stop_reason:
        gaps.append(f"Run stopped: {run.stop_reason}.")
    if gaps:
        body += "\n\n## Remaining gaps\n\n" + "\n".join(
            f"- {g}" for g in dict.fromkeys(gaps)
        )
    return body


class ResearchWorkflow:
    def __init__(
        self, *, model_factory: Callable, retriever: Retriever, journal: Journal
    ):
        self.model_factory = model_factory
        self.retriever = retriever
        self.journal = journal

    def build(self, checkpointer=None):
        builder = StateGraph(GraphState)
        names = (
            "planner",
            "search",
            "researcher",
            "reviewer",
            "supervisor",
            "writer",
            "finish",
        )
        for name in names:
            builder.add_node(name, self._node(name))
        routes = {name: name for name in names} | {"done": END}

        def choose(state):
            return state["run"]["next_node"]

        builder.add_conditional_edges(START, choose, routes)
        for name in names:
            builder.add_conditional_edges(name, choose, routes)
        return builder.compile(checkpointer=checkpointer)

    def _node(self, name):
        def execute(state):
            run = RunState.model_validate(state["run"])
            step_id = f"{run.step:03d}:{name}"
            existing = self.journal.get(step_id)
            if existing and existing["status"] == "completed":
                return {"run": existing["state"]}
            received = self._received(name, run)
            started = datetime.now(UTC)
            usage, tools = [], []
            error = None
            if existing:
                run.budget = TokenBudgetState.model_validate(existing["budget"])
                if run.budget.reserved_tokens:
                    run.budget = settle_tokens(run.budget, None)
                run.stop_reason = "interrupted_operation"
                run.next_node = "finish"
                produced = {
                    "recovery": "Incomplete external operation was not replayed."
                }
                error = "interrupted_operation"
            else:
                self.journal.begin(step_id, received, run.budget.model_dump())
                run.status = "running"

                def update_budget(budget):
                    run.budget = budget
                    tools.append(
                        {"tool": "token_budget", "output": budget.model_dump()}
                    )
                    self.journal.budget(step_id, budget.model_dump())

                model = TracedModel(
                    self.model_factory(run.budget, run.deadline, update_budget),
                    tools,
                    run.model_name,
                )
                try:
                    if name != "finish" and time.time() >= run.deadline:
                        raise TimeoutError("Run deadline exceeded")
                    produced = self._act(name, run, model, usage, tools)
                except (BudgetBlocked, TimeoutError) as exc:
                    error = type(exc).__name__
                    run.stop_reason = (
                        "deadline" if isinstance(exc, TimeoutError) else str(exc)
                    )
                    run.next_node = "finish"
                    produced = {"stop_reason": run.stop_reason}
                except Exception as exc:
                    # Store a safe category; provider exception strings can contain credentials.
                    error = type(exc).__name__
                    produced = {"failure": error}
                    if name == "planner":
                        run.stop_reason = f"planner_{error}"
                        run.next_node = "finish"
                    elif name == "writer":
                        run.writer_fallback = True
                        run.next_node = "finish"
                    elif name in ("researcher", "reviewer"):
                        sq = run.plan.subquestions[run.index]
                        run.gaps[sq.id] = [f"{name} failed: {error}"]
                        run.index += 1
                        run.next_node = (
                            "search"
                            if run.index < len(run.plan.subquestions)
                            else "writer"
                        )
                    else:
                        run.stop_reason = f"{name}_{error}"
                        run.next_node = "finish"
            run.step += 1
            trace = StepTrace(
                step_id=step_id,
                agent=name,
                status="failed" if error else "completed",
                received=received,
                produced=produced,
                tool_calls=tools,
                usage=usage,
                started_at=started,
                latency_ms=round(
                    (datetime.now(UTC) - started).total_seconds() * 1000, 3
                ),
                error=error,
            )
            result = run.model_dump(mode="json")
            self.journal.complete(step_id, result, trace.model_dump(mode="json"))
            return {"run": result}

        return execute

    def _received(self, name, run):
        payload = {"question": run.question}
        if run.plan and run.index < len(run.plan.subquestions):
            sq = run.plan.subquestions[run.index]
            payload["subquestion"] = sq.model_dump(mode="json")
            payload["gaps"] = run.gaps.get(sq.id, [])
        if name == "researcher":
            payload["sources"] = [s.model_dump(mode="json") for s in run.sources]
        if name in ("reviewer", "writer"):
            payload["findings"] = [
                f.model_dump(mode="json")
                for f in run.findings
                if (
                    f.id in run.accepted_ids
                    if name == "writer"
                    else f.subquestion_id == run.plan.subquestions[run.index].id
                )
            ]
        if name == "supervisor":
            payload["reviews"] = {k: v.model_dump() for k, v in run.reviews.items()}
            payload["revisions_used"] = run.revisions_used
        return payload

    def _act(self, name, run, model, usage, tools):
        def record(value):
            return usage.append(value.model_dump() if value is not None else None)

        if name == "planner":
            run.plan = create_plan(
                run.question,
                model=model,
                max_subquestions=run.limits.max_subquestions,
                record_usage=record,
            )
            run.next_node = "search"
            return {"plan": run.plan.model_dump(mode="json")}
        if name == "search":
            sq = run.plan.subquestions[run.index]
            count = run.searches.get(sq.id, 0)
            if count >= run.limits.max_searches_per_subquestion:
                run.gaps[sq.id] = ["Search allowance exhausted."]
                run.index += 1
                run.next_node = (
                    "search" if run.index < len(run.plan.subquestions) else "writer"
                )
                return {"search_limit": sq.id}
            # Attempts, including failures, consume the search allowance.
            run.searches[sq.id] = count + 1
            query = sq.question
            if run.gaps.get(sq.id):
                query += " Focus on: " + " ".join(run.gaps[sq.id])
            result = self.retriever.retrieve(query[:1000], state=run)
            tools.extend(result.tool_calls)
            # Retain old extracted pages so a revision can correct a claim without new search hits.
            known = {str(s.url): s for s in run.sources}
            known.update({str(s.url): s for s in result.sources})
            run.sources = list(known.values())
            run.retrieval_issues.extend(result.issues)
            run.next_node = "researcher"
            return result.model_dump(mode="json")
        if name == "researcher":
            sq = run.plan.subquestions[run.index]
            result = research_subquestion(
                sq, run.sources, model=model, record_usage=record
            )
            known = {
                (f.subquestion_id, str(f.source_url), f.claim, f.snippet)
                for f in run.findings
            }
            for finding in result.findings:
                key = (
                    finding.subquestion_id,
                    str(finding.source_url),
                    finding.claim,
                    finding.snippet,
                )
                if key not in known:
                    run.findings.append(finding)
                    known.add(key)
            run.next_node = "reviewer"
            return result.model_dump(mode="json")
        if name == "reviewer":
            sq = run.plan.subquestions[run.index]
            findings = [f for f in run.findings if f.subquestion_id == sq.id]
            review = review_research(sq, findings, model=model, record_usage=record)
            run.reviews[sq.id] = review
            run.accepted_ids = [
                fid for fid in run.accepted_ids if fid not in {f.id for f in findings}
            ]
            run.accepted_ids.extend(review.accepted_finding_ids)
            run.gaps[sq.id] = review.gaps
            run.next_node = "supervisor"
            return review.model_dump(mode="json")
        if name == "supervisor":
            sq = run.plan.subquestions[run.index]
            # Admission to the actual model request still performs exact input counting.
            available = (
                run.budget.blocked_reason is None
                and run.budget.max_total_tokens - run.budget.recorded_tokens
                > run.limits.max_output_tokens
                and run.searches.get(sq.id, 0) < run.limits.max_searches_per_subquestion
                and time.time() < run.deadline
            )
            decision = decide_after_review(
                run.reviews[sq.id],
                revisions_used=run.revisions_used,
                retry_budget_available=available,
            )
            if decision.action == "revise":
                run.revisions_used += 1
                run.next_node = "search"
            else:
                run.index += 1
                run.sources = []
                run.next_node = (
                    "search" if run.index < len(run.plan.subquestions) else "writer"
                )
            return decision.model_dump()
        if name == "writer":
            accepted = [f for f in run.findings if f.id in run.accepted_ids]
            if accepted:
                run.report = write_report(
                    run.question, accepted, model=model, record_usage=record
                )
            run.next_node = "finish"
            return {
                "report": run.report.model_dump(mode="json") if run.report else None
            }
        return finalize_run(run)


def finalize_run(run: RunState) -> dict:
    # A deterministic fallback can present already accepted claims without new spending.
    accepted = [f for f in run.findings if f.id in run.accepted_ids]
    if not run.report and accepted:
        run.writer_fallback = True
        run.report = ResearchReport(
            title="Supported research findings",
            claims=[ReportClaim(text=f.claim, finding_ids=[f.id]) for f in accepted],
        )
    if run.report:
        validate_report_citations(run.report, accepted)
    if run.plan:
        for sq in run.plan.subquestions:
            if sq.id not in run.reviews:
                run.gaps.setdefault(sq.id, [sq.completion_criteria])
    run.status = (
        "timed_out"
        if run.stop_reason == "deadline"
        else "failed"
        if not run.report
        else "partial"
        if run.stop_reason or any(run.gaps.values()) or run.writer_fallback
        else "completed"
    )
    run.report_markdown = render_report(run)
    run.next_node = "done"
    return {
        "status": run.status,
        "report": run.report.model_dump(mode="json") if run.report else None,
    }
