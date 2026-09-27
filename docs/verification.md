# Verification record

Date: 2026-09-26. Results below distinguish synthetic system checks from live model
observations. These are individual observations, not performance or reliability benchmarks.

## Automated checks

**143 tests passed** with real Redis enabled (9.22 seconds in the final observed
run). Ruff lint and formatting checks passed. Without TEST_REDIS_URL, the five
real-process integration cases are explicitly skipped.

The suite covers typed handoffs, evidence matching, citation IDs, reviewer consistency,
routing, reservations, settlement, input-request equivalence, provider failures,
Tavily extraction and source limits, API/idempotency/auth/SSE behavior, and Redis recovery.

Real-process integration checks:
- a complete cited synthetic run finishes within the five-minute acceptance ceiling;
- killing after a node's result is journaled does not repeat the Planner or its spend;
- killing with an open reservation marks usage unknown and stops generation;
- a parent watchdog stops a hung child at its persisted deadline;
- a real HTTP API enqueues work and returns the worker's report, citations, trace, and SSE events.

The Docker image builds successfully with the locked dependencies. Its synthetic
CLI demo completed in **1.644 seconds**, with two cited claims and seven trace steps.
The reported 560 tokens are fixture values, not provider usage. The captured
[demo output](example-demo.json) is included for reproducibility.

## Live observations

The budgeted Planner used `gpt-5.4-mini` for:
“How does Redis AOF help recover after a process crash?”

Observed result: completed in **3.014 seconds**, with **212 input tokens + 115 output
tokens = 327 total**. The reservation was settled and no budget block remained.
This run used a 4,000-token allowance, a 600-token output cap, and a 45-second deadline.

Earlier specialist smoke observations are preserved in the local teaching progress log.
They used synthetic evidence with a real model; they did not fetch webpages.

## Researcher failure regression — 2026-09-27

A user-run live request completed search/extraction and all three Researcher model
calls, but each pass failed during excerpt validation. The recorded drafts included
ellipses joining passages, altered punctuation/control characters, and an excerpt
assigned to a different supplied source. One invalid draft discarded each whole batch.

Replaying the captured Researcher inputs and parsed outputs locally reproduced all
three failures without provider calls. After the fix, the same outputs preserve
5, 2, and 3 provenance-valid findings respectively, and record 4, 8, and 4 rejections.
This verifies recovery of valid drafts; their relevance and claim support still need
Reviewer assessment. The original terminal run remains an unchanged audit record.

The updated offline suite passes **147 tests**, with **5 real-Redis integration tests
skipped**. New regression coverage includes mixed valid/invalid drafts, mismatched
source IDs, all-invalid results, graph routing through Reviewer/Writer, one bounded
revision, unchanged usage accounting, and rejection details in the step trace.
The prompt now requests short contiguous quotations and sends Unicode punctuation
directly rather than JSON ASCII escapes. No fresh live run was made for this fix,
so the effect of the prompt change on generation quality has not been measured.

## Remaining external verification

Successful completion of the full Tavily + OpenAI live pipeline remains unverified:
the key was absent at initial verification, and the later live run exposed the
Researcher failure described above.
The Tavily adapter is tested against the documented HTTP contract and explicit failure
fixtures. With both keys configured, restart the worker with the fix and run:

```bash
uv run python -m multi_agent_research_assistant.smoke.smoke_pipeline
```

The script enforces one subquestion, one search, 12,000 total tokens, and a 120-second
deadline. Record its status, source links, elapsed time, and usage here after inspecting
whether the evidence supports the resulting claims.

Synthetic completion and a live Planner success do not establish live end-to-end
answer quality or latency. No result is described as independently verified truth.
