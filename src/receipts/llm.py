"""Nebius Token Factory client (OpenAI-compatible) with fallback, JSON output and usage tracking."""

from __future__ import annotations

import json
import re
import time
from typing import Any

import httpx
import openai
from openai import OpenAI

from .config import Settings
from .trace import Tracer

_THINK_RE = re.compile(r"<think>.*?</think>", re.S | re.I)
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)

# Failures where trying another model can help.
_RETRYABLE = (
    openai.RateLimitError,  # 429: our project hit its rate limit
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.NotFoundError,  # model renamed or deprecated
    openai.InternalServerError,  # 5xx, including 529 "overloaded"
)


class LLMError(RuntimeError):
    pass


def parse_json(text: str) -> dict:
    """Pull one JSON object out of a model reply (handles <think> blocks and code fences)."""
    text = _THINK_RE.sub("", text or "").strip()
    fenced = _FENCE_RE.search(text)
    if fenced:
        text = fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object in model reply")
    return json.loads(text[start : end + 1])


def _short(err: Exception) -> str:
    msg = str(err).replace("\n", " ")
    return f"{type(err).__name__}: {msg[:160]}"


class LLM:
    def __init__(self, settings: Settings, tracer: Tracer, client: Any = None):
        self.settings = settings
        self.tracer = tracer
        self.client = client or OpenAI(
            base_url=settings.base_url,
            api_key=settings.nebius_api_key,
            max_retries=2,  # SDK backs off on 429/5xx before we fall back
            timeout=120,
        )
        # Remembers which structured-output mode each model accepted.
        self._json_mode: dict[str, str] = {}

    def model_for(self, tier: str) -> str:
        return self.settings.model_smart if tier == "smart" else self.settings.model_fast

    # ------------------------------------------------------------------ core call
    def complete(
        self,
        messages: list[dict],
        *,
        tier: str = "fast",
        role: str = "llm",
        temperature: float = 0.2,
        max_tokens: int = 2500,
        response_format: dict | None = None,
    ) -> str:
        primary = self.model_for(tier)
        models = [primary]
        if self.settings.model_fallback and self.settings.model_fallback != primary:
            models.append(self.settings.model_fallback)

        last_err: Exception | None = None
        for i, model in enumerate(models):
            kwargs: dict[str, Any] = dict(
                model=model, messages=messages, temperature=temperature, max_tokens=max_tokens
            )
            if response_format:
                kwargs["response_format"] = response_format
            t0 = time.monotonic()
            try:
                resp = self.client.chat.completions.create(**kwargs)
            except openai.AuthenticationError as e:
                raise LLMError("Token Factory rejected the API key (401). Check NEBIUS_API_KEY.") from e
            except openai.BadRequestError:
                raise  # caller may retry with a different response_format
            except _RETRYABLE as e:
                last_err = e
                more = i + 1 < len(models)
                self.tracer.emit(
                    "llm_error", role=role, model=model, error=_short(e),
                    action="fallback" if more else "give_up",
                )
                continue
            usage = getattr(resp, "usage", None)
            pt = int(getattr(usage, "prompt_tokens", 0) or 0)
            ct = int(getattr(usage, "completion_tokens", 0) or 0)
            self.tracer.record_llm(model, pt, ct)
            self.tracer.emit(
                "llm", role=role, model=model, ms=int((time.monotonic() - t0) * 1000),
                prompt_tokens=pt, completion_tokens=ct, fallback=i > 0,
            )
            return resp.choices[0].message.content or ""
        raise LLMError(f"All models failed for '{role}': {_short(last_err) if last_err else 'unknown'}")

    # ------------------------------------------------------------ structured call
    def json(
        self,
        *,
        role: str,
        system: str,
        user: str,
        schema: dict,
        tier: str = "fast",
        max_tokens: int = 2500,
        temperature: float = 0.1,
    ) -> dict:
        """Ask for JSON. Tries json_schema, then json_object, then plain prompting."""
        model = self.model_for(tier)
        modes = ["json_schema", "json_object", "none"]
        start = modes.index(self._json_mode.get(model, "json_schema"))
        system_full = (
            f"{system}\n\nReply with ONE JSON object that matches this JSON Schema. No prose.\n"
            f"{json.dumps(schema)}"
        )
        messages = [{"role": "system", "content": system_full}, {"role": "user", "content": user}]

        for mode in modes[start:]:
            rf = None
            if mode == "json_schema":
                rf = {"type": "json_schema", "json_schema": {"name": role, "schema": schema, "strict": True}}
            elif mode == "json_object":
                rf = {"type": "json_object"}
            try:
                text = self.complete(
                    messages, tier=tier, role=role, temperature=temperature,
                    max_tokens=max_tokens, response_format=rf,
                )
            except openai.BadRequestError as e:
                self.tracer.emit("llm_note", role=role, model=model, note=f"{mode} not accepted, downgrading", error=_short(e))
                self._json_mode[model] = modes[min(modes.index(mode) + 1, 2)]
                continue
            try:
                return parse_json(text)
            except (ValueError, json.JSONDecodeError):
                repair = messages + [
                    {"role": "assistant", "content": text[:4000]},
                    {"role": "user", "content": "That was not valid JSON. Reply with only the JSON object."},
                ]
                try:
                    return parse_json(
                        self.complete(repair, tier=tier, role=f"{role}_repair", temperature=0.0,
                                      max_tokens=max_tokens, response_format=rf)
                    )
                except (ValueError, json.JSONDecodeError, openai.BadRequestError):
                    continue
        raise LLMError(f"Model never returned valid JSON for '{role}'")

    # ------------------------------------------------------------------ catalog
    def fetch_catalog(self) -> list[dict]:
        """GET /v1/models?verbose=true — live model list with pricing when available."""
        url = self.settings.base_url.rstrip("/") + "/models"
        r = httpx.get(
            url, params={"verbose": "true"},
            headers={"Authorization": f"Bearer {self.settings.nebius_api_key}"}, timeout=20,
        )
        r.raise_for_status()
        return r.json().get("data", [])

    def load_prices(self) -> None:
        """Best effort: lets the trace show an estimated $ cost. Never fails the run."""
        try:
            for m in self.fetch_catalog():
                p = m.get("pricing") or {}
                if "prompt" in p and "completion" in p:
                    self.tracer.prices[m["id"]] = (float(p["prompt"]), float(p["completion"]))
        except Exception:
            pass
