import ipaddress
import time
from datetime import UTC, datetime
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import uuid4

import httpx
import tldextract

from multi_agent_research_assistant.domain.models import SourceDocument
from multi_agent_research_assistant.domain.runs import (
    RetrievalIssue,
    RetrievalResult,
    RunState,
)

_SUFFIX = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None)


def canonical_url(value: str) -> str:
    parts = urlsplit(value)
    host = (parts.hostname or "").lower().rstrip(".")
    if (
        parts.scheme not in ("http", "https")
        or not host
        or parts.username
        or parts.password
    ):
        raise ValueError("Only public HTTP(S) URLs are supported")
    if (
        parts.port not in (None, 80, 443)
        or host == "localhost"
        or host.endswith((".local", ".internal", ".localhost"))
    ):
        raise ValueError("Nonpublic source")
    try:
        if not ipaddress.ip_address(host).is_global:
            raise ValueError("Nonpublic source")
    except ValueError as exc:
        if str(exc) == "Nonpublic source":
            raise
        if "." not in host:
            raise ValueError("Nonpublic source") from exc
    query = [
        (k, v)
        for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if not k.lower().startswith("utm_") and k.lower() not in ("fbclid", "gclid")
    ]
    return urlunsplit(
        (parts.scheme, host, parts.path or "/", urlencode(sorted(query)), "")
    )


def site(url: str) -> str:
    host = urlsplit(url).hostname
    return _SUFFIX(host).top_domain_under_public_suffix or host


class TavilyRetriever:
    def __init__(self, client: httpx.Client, api_key: str):
        self.client, self.api_key = client, api_key

    def _post(self, endpoint, payload, run, result):
        started = time.monotonic()
        trace = {"tool": f"tavily_{endpoint}", "input": payload}
        try:
            remaining = run.deadline - time.time()
            if remaining <= 0:
                raise httpx.TimeoutException("deadline")
            response = self.client.post(
                f"https://api.tavily.com/{endpoint}",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
                timeout=min(20, remaining),
            )
            response.raise_for_status()
            data = response.json()
            if (
                not isinstance(data, dict)
                or not isinstance(data.get("results", []), list)
                or not all(isinstance(item, dict) for item in data.get("results", []))
                or not isinstance(data.get("failed_results", []), list)
            ):
                raise ValueError("Invalid provider response")
            trace["output"] = {
                "status": "completed",
                "request_id": data.get("request_id"),
                "result_count": len(data.get("results", [])),
            }
            return data
        except httpx.TimeoutException:
            status = "timed_out"
        except httpx.HTTPStatusError as exc:
            status = (
                "rate_limited" if exc.response.status_code == 429 else "unavailable"
            )
        except httpx.RequestError, ValueError:
            status = "unavailable"
        finally:
            trace["latency_ms"] = round((time.monotonic() - started) * 1000, 3)
            result.tool_calls.append(trace)
        trace["output"] = {"status": status}
        result.issues.append(
            RetrievalIssue(status=status, detail=f"Tavily {endpoint} did not complete.")
        )
        return None

    def retrieve(self, query: str, *, state: RunState) -> RetrievalResult:
        result = RetrievalResult()
        search = self._post(
            "search",
            {
                "query": query,
                "search_depth": "basic",
                "max_results": min(10, state.limits.max_sources_per_search * 3),
                "include_answer": False,
                "include_raw_content": False,
            },
            state,
            result,
        )
        if search is None:
            return result
        selected, seen = [], set()
        counts = state.domain_counts.copy()
        for item in search.get("results", []):
            try:
                url = canonical_url(item["url"])
            except KeyError, ValueError, TypeError:
                result.issues.append(
                    RetrievalIssue(
                        status="invalid_source",
                        detail="Rejected invalid candidate URL.",
                    )
                )
                continue
            if url in seen:
                continue
            seen.add(url)
            if url in state.source_cache:
                result.sources.append(state.source_cache[url])
            elif counts.get(site(url), 0) < state.limits.max_sources_per_domain:
                selected.append(url)
                counts[site(url)] = counts.get(site(url), 0) + 1
            if (
                len(selected) + len(result.sources)
                >= state.limits.max_sources_per_search
            ):
                break
        if selected:
            extracted = self._post(
                "extract",
                {
                    "urls": selected,
                    "extract_depth": "basic",
                    "format": "text",
                    "include_images": False,
                },
                state,
                result,
            )
            if extracted is not None:
                handled = set()
                for item in extracted.get("results", []):
                    try:
                        url = canonical_url(item["url"])
                        text = item["raw_content"]
                        if (
                            url not in selected
                            or url in handled
                            or not isinstance(text, str)
                            or not text.strip()
                        ):
                            continue
                        handled.add(url)
                        source = SourceDocument(
                            id=f"source_{uuid4().hex}",
                            url=url,
                            text=text[:16000],
                            retrieved_at=datetime.now(UTC),
                        )
                        state.source_cache[url] = source
                        state.seen_urls.append(url)
                        state.domain_counts[site(url)] = (
                            state.domain_counts.get(site(url), 0) + 1
                        )
                        result.sources.append(source)
                    except KeyError, ValueError, TypeError:
                        continue
                failures = {
                    item.get("url"): str(item.get("error", "")).lower()
                    for item in extracted.get("failed_results", [])
                    if isinstance(item, dict)
                }
                for url in selected:
                    if url in handled:
                        continue
                    message = failures.get(url, "")
                    status = (
                        "paywalled"
                        if any(
                            x in message
                            for x in ("paywall", "subscription required", "402")
                        )
                        else "timed_out"
                        if "timeout" in message or "timed out" in message
                        else "unavailable"
                    )
                    result.issues.append(
                        RetrievalIssue(
                            status=status, url=url, detail="Page extraction failed."
                        )
                    )
        if not result.sources and not result.issues:
            result.issues.append(
                RetrievalIssue(
                    status="no_results", detail="No eligible extracted sources."
                )
            )
        return result
