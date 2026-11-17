"""Settings loaded from environment variables (.env supported)."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

DEFAULT_BASE_URL = "https://api.tokenfactory.us-north1.nebius.com/v1/"
DEFAULT_MODEL_FAST = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
DEFAULT_MODEL_SMART = "moonshotai/Kimi-K3"
DEFAULT_MODEL_FALLBACK = "Qwen/Qwen3-30B-A3B-Instruct-2507"

PLACEHOLDER_KEYS = {"", "your-token-factory-key", "your-tavily-key"}


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


@dataclass
class Settings:
    nebius_api_key: str | None = None
    tavily_api_key: str | None = None
    base_url: str = DEFAULT_BASE_URL
    model_fast: str = DEFAULT_MODEL_FAST
    model_smart: str = DEFAULT_MODEL_SMART
    model_fallback: str | None = DEFAULT_MODEL_FALLBACK

    # Research loop budget. Enforced in code, not just in prompts.
    max_rounds: int = 2
    max_subquestions: int = 4
    queries_per_round: int = 5
    results_per_query: int = 5
    max_sources: int = 8
    max_per_domain: int = 2
    max_claims_to_verify: int = 12
    search_depth: str = "advanced"
    tavily_credit_budget: int = 40
    time_budget_s: float = 180.0
    workers: int = 4

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        return cls(
            nebius_api_key=os.getenv("NEBIUS_API_KEY"),
            tavily_api_key=os.getenv("TAVILY_API_KEY"),
            base_url=os.getenv("NEBIUS_BASE_URL", DEFAULT_BASE_URL),
            model_fast=os.getenv("NEBIUS_MODEL_FAST", DEFAULT_MODEL_FAST),
            model_smart=os.getenv("NEBIUS_MODEL_SMART", DEFAULT_MODEL_SMART),
            model_fallback=os.getenv("NEBIUS_MODEL_FALLBACK", DEFAULT_MODEL_FALLBACK) or None,
            max_rounds=_int("RECEIPTS_MAX_ROUNDS", 2),
            max_sources=_int("RECEIPTS_MAX_SOURCES", 8),
            tavily_credit_budget=_int("RECEIPTS_TAVILY_CREDIT_BUDGET", 40),
            time_budget_s=_float("RECEIPTS_TIME_BUDGET_S", 180.0),
        )

    def missing_keys(self) -> list[str]:
        missing = []
        if (self.nebius_api_key or "").strip() in PLACEHOLDER_KEYS:
            missing.append("NEBIUS_API_KEY")
        if (self.tavily_api_key or "").strip() in PLACEHOLDER_KEYS:
            missing.append("TAVILY_API_KEY")
        return missing
