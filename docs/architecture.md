# Architecture and failure semantics

## Typed specialists

A specialist is a prompt, allowed inputs/tools, and an output schema. Planner returns
a ResearchPlan. Researcher extracts FindingDraft objects from supplied pages; the
application validates excerpts and attaches trusted source metadata. Reviewer returns
accepted IDs and explicit gaps. Writer returns a ResearchReport whose claims cite
accepted finding IDs. All use the StructuredModel interface; deployed calls use
BudgetedModel, while isolated smoke tests may use the unbudgeted LangChain adapter.

## Budget admission

1. Reject a blocked ledger or an outstanding reservation.
2. Count the complete structured request through OpenAI's input-count endpoint.
3. Reserve input count + maximum output tokens before generation.
4. Persist that reservation through the node journal.
5. Invoke with the same model/messages/schema/reasoning profile and enforced output cap.
6. Settle actual usage before output validation or optional observers.
7. Block further calls on missing usage or reservation overrun.

A counting failure happens before generation and leaves the ledger unchanged.
A generation timeout may have consumed tokens; it cannot be recorded as zero.
Reported input and output must sum to reported total tokens. Failed searches count
as attempts. A reviewer revision is charged when scheduled, even if its pass fails.

The caller profile is stateless text, strict JSON output, and reasoning effort
`none`. Extending it with tools/history/multimodal input requires extending the
counter and request-equivalence tests.

The graph checks the persisted UTC deadline before nodes, and HTTP timeouts use the
remaining allowance. A parent process separately enforces the deadline and terminates
a hung child. Downtime and queue time count toward the same deadline.

## State and storage

LangGraph's `run` channel holds the serializable RunState. It contains the plan,
position, findings, accepted IDs, reviews, gaps, source cache, search/revision counts,
budget, deadline, and report. The API's Redis state document is a materialized view
of this run, updated on node completion and during budget transitions.

The Redis checkpointer implements the synchronous BaseCheckpointSaver interface using
hashes, strings, pending writes, and typed serialization. It stores complete bounded
snapshots, prioritizing simplicity over storage compression. It requires ordinary Redis,
not Redis Search modules. Namespaces separate independent deployments.

The worker holds an expiring ownership token. Every journal and checkpoint write
watches that lease and verifies the token before committing. A heartbeat renews it.
An old worker cannot commit after a replacement has acquired ownership.

Node completion atomically writes:
- the cached node result;
- the API state view;
- its trace event;
- terminal queue removal when appropriate.

This handles the gap between an external operation completing and LangGraph saving
its next checkpoint. A replayed node reads its journal cache.

## Crash cases

| Crash point | Recovery |
| --- | --- |
| Before any node work | Resume the last graph checkpoint |
| After node result is journaled, before graph checkpoint | Return cached node state; no provider replay |
| While a reservation is open | Preserve known tokens, mark usage unknown, stop generation |
| During search or after settlement but before result is journaled | Do not guess the lost result or blindly repeat the operation; finalize incomplete |
| After terminal result is saved | Status/report remain available; queue entry is already removed |
| Lease lost while an old worker is alive | Fenced writes fail; parent stops its child |

This is not exactly-once execution of third-party APIs. Without a provider-supported
idempotency/reconciliation protocol, a crash after remote execution but before local
acknowledgment leaves an unavoidable uncertainty window. Conservative termination
protects the spending limit. Previously accepted findings remain usable.

Redis AOF `appendfsync always` is enabled in Compose to support durable acknowledgments
on the local filesystem. A single Redis instance is not a high-availability system.
Host/storage loss and restore policy require separate operational planning.

## Retrieval and reports

Tavily Search supplies candidates, followed by Tavily Extract for page content.
The adapter validates HTTP(S) URLs, strips tracking parameters and fragments,
deduplicates pages, and groups domains using an offline public-suffix list.
A cache avoids extracting the same page twice across subquestions. Site limits count
unique extracted pages. Context is capped at 16,000 characters per page, and this
bounded text is what snippet-occurrence validation uses.

No-results, rate-limited, timed-out, paywalled, and unavailable outcomes are explicit.
A generic extraction error is labeled unavailable, not guessed to be a paywall.
Page text is treated as untrusted evidence in prompts.

The Writer only receives reviewer-accepted findings. Citation validation is repeated
when the report is returned by the API. If Writer execution is unavailable, a
deterministic fallback presents accepted claims with citations and marks the result
partial. Unresolved criteria remain visible.

## Trace and API

Each trace records the node input, produced result, structured-model prompt and schema,
parsed output, budget transitions, tools, usage, and latency. Failed operations have
safe error categories; credentials and authorization headers are never included.
A killed in-flight request cannot supply a final response or exact usage; recovery
records that uncertainty instead of fabricating them.

POST enqueues work in a Redis sorted set and returns immediately. Separate workers
execute jobs, allowing API restarts without losing work. SSE events read persisted
trace entries and support Last-Event-ID, so reconnecting clients can replay missed steps.

The optional bearer token protects a single user's deployment. There are no user
accounts or tenant isolation claims. An authenticated reverse proxy and tenant-scoped
authorization would be required for a shared service.

## Primary references

- [LangGraph persistence](https://docs.langchain.com/oss/python/langgraph/persistence)
- [OpenAI token counting](https://developers.openai.com/api/docs/guides/token-counting)
- [Tavily Search](https://docs.tavily.com/documentation/api-reference/endpoint/search)
- [Tavily Extract](https://docs.tavily.com/documentation/api-reference/endpoint/extract)

