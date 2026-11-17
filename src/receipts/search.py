"""Tavily wrapper: search + extract with retries, credit tracking and clear failure modes."""

from __future__ import annotations

import math
import time
from typing import Any

from .config import Settings
from .trace import Tracer

# Errors where retrying is pointless (bad key, plan limit, bad request).
_FATAL = {"InvalidAPIKeyError", "MissingAPIKeyError", "UsageLimitExceededError", "ForbiddenError", "BadRequestError"}


class ToolError(RuntimeError):
    pass


class WebSearch:
    def __init__(self, settings: Settings, tracer: Tracer, client: Any = None):
        self.settings = settings
        self.tracer = tracer
        if client is None:
            from tavily import TavilyClient

            client = TavilyClient(api_key=settings.tavily_api_key)
        self.client = client

    def _call(self, label: str, fn, kwargs: dict, optional: tuple[str, ...] = ()) -> dict:
        delays = [0.0, 1.5, 4.0]
        for attempt, delay in enumerate(delays):
            if delay:
                time.sleep(delay)
            try:
                try:
                    return fn(**kwargs)
                except TypeError:
                    # Older tavily-python versions reject newer optional params.
                    trimmed = {k: v for k, v in kwargs.items() if k not in optional}
                    if trimmed == kwargs:
                        raise
                    return fn(**trimmed)
            except Exception as e:  # noqa: BLE001 — classify any SDK/network error
                name = type(e).__name__
                last = attempt == len(delays) - 1
                fatal = name in _FATAL
                self.tracer.emit(
                    "tool_error", tool=label, error=f"{name}: {str(e)[:160]}",
                    action="give_up" if (fatal or last) else "retry",
                )
                if fatal or last:
                    raise ToolError(f"Tavily {label} failed: {name}: {str(e)[:200]}") from e
        raise ToolError(f"Tavily {label} failed")  # unreachable

    def search(
        self,
        query: str,
        *,
        time_range: str | None = None,
        topic: str = "general",
        max_results: int | None = None,
    ) -> list[dict]:
        depth = self.settings.search_depth
        kwargs: dict[str, Any] = dict(
            query=query,
            search_depth=depth,
            topic=topic,
            max_results=max_results or self.settings.results_per_query,
            include_published_date=True,
            include_usage=True,
        )
        if time_range:
            kwargs["time_range"] = time_range
        t0 = time.monotonic()
        resp = self._call("search", self.client.search, kwargs, optional=("include_published_date", "include_usage"))
        results = resp.get("results", []) or []
        credits = (resp.get("usage") or {}).get("credits")
        if credits is None:
            credits = 2 if depth == "advanced" else 1
        self.tracer.record_tavily(float(credits))
        self.tracer.emit(
            "search", query=query, depth=depth, topic=topic, time_range=time_range,
            results=len(results), credits=credits, ms=int((time.monotonic() - t0) * 1000),
            top=[r.get("url") for r in results[:5]],
        )
        return results

    def extract(self, urls: list[str]) -> tuple[dict[str, str], dict[str, str]]:
        """Full page text for each URL. Returns ({url: text}, {url: error})."""
        pages: dict[str, str] = {}
        failed: dict[str, str] = {}
        for i in range(0, len(urls), 20):  # API limit: 20 URLs per call
            batch = urls[i : i + 20]
            t0 = time.monotonic()
            kwargs = dict(urls=batch, extract_depth="basic", format="text", include_usage=True)
            resp = self._call("extract", self.client.extract, kwargs, optional=("format", "include_usage"))
            for r in resp.get("results", []) or []:
                if r.get("raw_content"):
                    pages[r["url"]] = r["raw_content"]
            for r in resp.get("failed_results", []) or []:
                failed[r.get("url", "?")] = str(r.get("error", "failed"))
            credits = (resp.get("usage") or {}).get("credits")
            if credits is None:
                credits = math.ceil(len([u for u in batch if u in pages]) / 5)
            self.tracer.record_tavily(float(credits))
            self.tracer.emit(
                "extract", urls=len(batch), ok=len([u for u in batch if u in pages]),
                failed=len([u for u in batch if u in failed]), credits=credits,
                ms=int((time.monotonic() - t0) * 1000),
            )
        return pages, failed
