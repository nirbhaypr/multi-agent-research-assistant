"""Deterministic synthetic providers for demos, CI, and recovery experiments."""

import json
from datetime import UTC, datetime
from uuid import uuid4

from multi_agent_research_assistant.domain.models import SourceDocument, TokenUsage
from multi_agent_research_assistant.domain.runs import RetrievalResult
from multi_agent_research_assistant.orchestration.budgets import (
    reserve_tokens,
    settle_tokens,
)

COMPLETED = "Completed runs are normally retained for seven days."
FAILED = "Failed runs are retained for thirty days unless an administrator deletes them earlier."


class DemoModel:
    def __init__(
        self, budget, deadline, on_budget, *, calls=None, partial=False, output_cap=2000
    ):
        self.budget, self.on_budget = budget, on_budget
        self.calls = calls if calls is not None else []
        self.partial, self.output_cap = partial, output_cap

    def invoke(self, *, schema, messages, record_usage=None):
        self.budget = reserve_tokens(self.budget, tokens=100 + self.output_cap)
        self.on_budget(self.budget)
        self.calls.append(schema.__name__)
        usage = TokenUsage(input_tokens=100, output_tokens=40, total_tokens=140)
        self.budget = settle_tokens(self.budget, usage)
        self.on_budget(self.budget)
        if record_usage:
            record_usage(usage)
        text = messages[-1].content
        if schema.__name__ == "ResearchPlan":
            payload = {
                "question": text,
                "subquestions": [
                    {
                        "id": "sq_retention",
                        "question": "How long are completed and failed runs retained?",
                        "completion_criteria": "Identify both retention periods and stated exceptions.",
                    }
                ],
            }
        else:
            data = json.loads(text)
            if schema.__name__ == "ResearchDraft":
                source = data["sources"][0]
                claims = [COMPLETED] if self.partial else [COMPLETED, FAILED]
                payload = {
                    "findings": [
                        {"source_id": source["id"], "claim": claim, "snippet": claim}
                        for claim in claims
                    ]
                }
            elif schema.__name__ == "ResearchReview":
                payload = {
                    "status": "insufficient" if self.partial else "sufficient",
                    "accepted_finding_ids": [f["id"] for f in data["findings"]],
                    "gaps": ["Failed-run retention is missing."]
                    if self.partial
                    else [],
                    "reason": "Synthetic reviewer result.",
                }
            else:
                payload = {
                    "title": "Synthetic run retention policy",
                    "claims": [
                        {"text": f["claim"], "finding_ids": [f["id"]]}
                        for f in data["findings"]
                    ],
                }
        return schema.model_validate(payload)


class DemoRetriever:
    def __init__(self, *, partial=False):
        self.partial = partial

    def retrieve(self, query, *, state):
        source = SourceDocument(
            id=f"source_{uuid4().hex}",
            url="https://example.com/fixtures/retention",
            text=COMPLETED if self.partial else COMPLETED + " " + FAILED,
            retrieved_at=datetime.now(UTC),
        )
        return RetrievalResult(
            sources=[source],
            tool_calls=[
                {
                    "tool": "synthetic_search_extract",
                    "input": {"query": query},
                    "output": {"synthetic": True, "urls": [str(source.url)]},
                }
            ],
        )
