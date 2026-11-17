"""Research trace: every tool call and decision, with timing, tokens and credits.

Events are plain dicts so they can be streamed to a UI or saved as JSON.
API keys never enter the trace.
"""

from __future__ import annotations

import threading
import time
from typing import Callable

from .schema import Usage

Listener = Callable[[dict], None]


class Tracer:
    def __init__(self, listener: Listener | None = None):
        self.events: list[dict] = []
        self.usage = Usage()
        self.prices: dict[str, tuple[float, float]] = {}  # model -> ($/token in, $/token out)
        self._listener = listener
        self._lock = threading.Lock()
        self._t0 = time.monotonic()

    def elapsed(self) -> float:
        return time.monotonic() - self._t0

    def emit(self, kind: str, **data) -> dict:
        event = {"t": round(self.elapsed(), 2), "type": kind, **data}
        with self._lock:
            self.events.append(event)
        if self._listener:
            try:
                self._listener(event)
            except Exception:  # a broken listener must never break research
                pass
        return event

    def record_llm(self, model: str, prompt_tokens: int, completion_tokens: int) -> None:
        with self._lock:
            u = self.usage
            u.llm_calls += 1
            u.prompt_tokens += prompt_tokens
            u.completion_tokens += completion_tokens
            m = u.by_model.setdefault(model, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0})
            m["calls"] += 1
            m["prompt_tokens"] += prompt_tokens
            m["completion_tokens"] += completion_tokens
            if model in self.prices:
                pin, pout = self.prices[model]
                u.est_llm_cost_usd = (u.est_llm_cost_usd or 0.0) + prompt_tokens * pin + completion_tokens * pout

    def record_tavily(self, credits: float) -> None:
        with self._lock:
            self.usage.tavily_calls += 1
            self.usage.tavily_credits += credits
