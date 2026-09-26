import json

import httpx
import pytest

from multi_agent_research_assistant.domain.runs import RunLimits, RunState
from multi_agent_research_assistant.retrieval.tavily import (
    TavilyRetriever,
    canonical_url,
    site,
)


def run():
    return RunState.new("retention?", RunLimits())


def test_extraction_provenance_dedup_site_cap_and_cache():
    calls = []

    def respond(request):
        data = json.loads(request.content)
        calls.append((request.url.path, data))
        if request.url.path == "/search":
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "url": "https://docs.example.com/a?utm_source=x#top",
                            "content": "UNTRUSTED SNIPPET",
                        },
                        {"url": "https://docs.example.com/a"},
                        {"url": "https://blog.example.com/b"},
                        {"url": "https://news.example.com/c"},
                        {"url": "https://other.org/a"},
                    ]
                },
            )
        assert data["urls"] == [
            "https://docs.example.com/a",
            "https://blog.example.com/b",
            "https://other.org/a",
        ]
        return httpx.Response(
            200,
            json={
                "results": [
                    {"url": u, "raw_content": "Fetched full page text."}
                    for u in data["urls"]
                ]
            },
        )

    state = run()
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        retriever = TavilyRetriever(client, "secret-test")
        first = retriever.retrieve("retention", state=state)
        second = retriever.retrieve("retention", state=state)
    assert len(first.sources) == 3
    assert len(second.sources) == 3
    assert len(calls) == 3
    assert all(
        s.text == "Fetched full page text." and s.retrieved_at.tzinfo
        for s in first.sources
    )
    assert state.domain_counts == {"example.com": 2, "other.org": 1}
    assert "secret-test" not in first.model_dump_json()


@pytest.mark.parametrize(
    ("status", "expected"),
    [(429, "rate_limited"), (500, "unavailable"), (401, "unavailable")],
)
def test_search_failures_are_structured(status, expected):
    with httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(status))
    ) as client:
        result = TavilyRetriever(client, "key").retrieve("query", state=run())
    assert not result.sources
    assert result.issues[0].status == expected
    assert result.tool_calls[0]["output"]["status"] == expected


def test_timeout_is_structured():
    def fail(request):
        raise httpx.ReadTimeout("secret provider URL")

    with httpx.Client(transport=httpx.MockTransport(fail)) as client:
        result = TavilyRetriever(client, "key").retrieve("query", state=run())
    assert result.issues[0].status == "timed_out"


def test_paywall_and_empty_results():
    def respond(request):
        if request.url.path == "/search":
            return httpx.Response(
                200, json={"results": [{"url": "https://example.com/pay"}]}
            )
        return httpx.Response(
            200,
            json={
                "results": [],
                "failed_results": [
                    {"url": "https://example.com/pay", "error": "Paywall"}
                ],
            },
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        result = TavilyRetriever(client, "key").retrieve("query", state=run())
    assert result.issues[0].status == "paywalled"
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda r: httpx.Response(200, json={"results": []})
        )
    ) as client:
        result = TavilyRetriever(client, "key").retrieve("query", state=run())
    assert result.issues[0].status == "no_results"


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "http://127.0.0.1/a",
        "http://localhost/a",
        "https://user:pass@example.com",
        "http://10.0.0.1",
        "http://example.com:8080",
    ],
)
def test_rejects_nonpublic_sources(url):
    with pytest.raises(ValueError):
        canonical_url(url)


def test_domain_grouping_and_meaningful_query_preserved():
    assert site("https://docs.service.co.uk/a") == "service.co.uk"
    assert (
        canonical_url("https://Example.com/a?v=2&utm_source=x#z")
        == "https://example.com/a?v=2"
    )
