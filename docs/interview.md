# Interview walkthrough

## A two-minute explanation

“I built a research service with typed Planner, Researcher, Reviewer, and Writer
specialists. A LangGraph supervisor controls progress, retries, and budgets.
The difficult part was handling failures: a timed-out request may still cost tokens,
and a worker may die after an API call succeeds but before the graph saves state.

I reserve a full request ceiling before generation, settle actual usage afterward,
and stop further calls when usage is unknown. Redis checkpoints preserve routing
state, while a separate operation journal caches completed steps. Worker leases fence
stale writes, and a parent process enforces the wall-clock deadline.

Every report claim refers to a validated finding record. The trace shows prompts,
evidence, decisions, token usage, and latency, so I can investigate a bad answer
instead of guessing which agent caused it.”

## Show the implementation

1. Start Redis and run `uv run multi-agent-research-assistant demo`.
2. Explain that the fixture is synthetic and does not measure model quality.
3. Open the run's status and trace through the API.
4. Point to source excerpts, accepted IDs, the Writer's citations, and budget transitions.
5. Run `TEST_REDIS_URL=redis://localhost:6379/0 uv run pytest -q tests/integration`.
6. Explain the completed-node kill test and the in-flight reservation kill test.
7. If both provider keys are configured, run the bounded live pipeline smoke command.

## Questions to practice

**Why multiple specialists?** Each has a narrow typed responsibility and independent
tests. This makes handoffs and failures observable. It does not imply multiple agents
are always cheaper or more accurate than one agent; that needs comparative evaluation.

**What does Pydantic prove?** Shape, types, required fields, and local invariants.
It does not prove a claim is true, relevant, or fully supported.

**Why fetch pages separately?** Search snippets may be truncated or misleading.
A finding should point to text actually extracted from its source.

**Why is a valid citation insufficient?** An existing finding ID proves a reference
exists. It does not prove the report's wording follows from that finding.

**How do you cap spending?** Count the complete structured input, reserve it plus
the output limit, persist the reservation, and settle actual usage. Enforce search
and revision counts as well. These are resource caps, not an exact dollar forecast.

**What happens after a timeout?** The provider might have processed the request.
Unknown usage blocks more generation, while known usage is charged even if parsing
or validation fails.

**Why only one revision?** It bounds feedback loops. The reviewer records specific gaps,
and the supervisor alone decides whether the remaining allowance permits a revision.
The allowance is global per run in this implementation.

**How does restart recovery work?** LangGraph restores the checkpoint. A node whose
result was already journaled returns its cached result. An ambiguous in-flight
operation stops conservatively; remote exactly-once execution is not claimed.

**Why both a checkpoint and an operation journal?** They cover different moments.
A checkpoint captures graph progress. The journal also records pre-call reservations
and closes the result/checkpoint crash gap.

**What prevents two workers from updating the same run?** An expiring ownership token,
heartbeat renewal, and transactional token checks on writes. The token fences an old
worker after lease replacement.

**Why a separate worker?** A durable queue survives API restarts. A process supervisor
can terminate a blocked child at the wall-clock limit, whereas a synchronous HTTP
timeout alone does not bound all work.

**How would you debug a bad answer?** Follow the report citation to the finding,
compare claim with snippet, inspect the extracted page, review the acceptance decision,
then examine the Writer prompt/output and budget/retry trace.

**What would parallel research change?** The current ledger allows one pending call.
Parallel calls need atomic multi-reservation admission, deterministic state merging,
and per-operation recovery. Reusing the serial ledger would be incorrect.

**What remains uncertain?** Source truth, semantic entailment, and model reliability.
Live observations are examples, not a statistically meaningful evaluation.

## Evidence before claims

Use the test names, code, and trace as evidence. Avoid claiming “hallucination-free,”
“exactly once,” “high availability,” or an established five-minute live success rate.
The watchdog enforces a ceiling; it may return a partial or timed-out result.

