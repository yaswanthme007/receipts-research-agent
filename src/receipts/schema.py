"""Data model shared by every stage of the pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

Verdict = Literal["supports", "partially_supports", "does_not_support", "contradicts"]
ClaimStatus = Literal[
    "supported", "partial", "unsupported", "contradicted", "conflicting", "inference", "pending"
]
RunStatus = Literal["answered", "partial", "insufficient_evidence", "tool_failure"]


@dataclass
class SubQuestion:
    id: str
    text: str
    queries: list[str] = field(default_factory=list)
    covered: bool = False


@dataclass
class Plan:
    subquestions: list[SubQuestion]
    time_range: str | None = None  # day | week | month | year
    topic: str = "general"  # general | news | finance
    notes: str = ""


@dataclass
class Source:
    id: str  # S1, S2, ...
    url: str
    title: str
    domain: str
    snippet: str
    score: float = 0.0
    published_date: str | None = None
    query: str = ""
    page_text: str | None = None  # full page text from Tavily Extract
    fetch_error: str | None = None
    injection_flags: list[str] = field(default_factory=list)

    def best_text(self) -> str:
        return self.page_text or self.snippet


@dataclass
class Evidence:
    source_id: str
    verdict: Verdict
    quote: str | None  # exact text found on the page, or None
    quote_match: Literal["exact", "near", "none"] = "none"
    explanation: str = ""
    highlight_url: str | None = None


@dataclass
class Claim:
    id: str
    text: str
    kind: Literal["fact", "inference"] = "fact"
    source_ids: list[str] = field(default_factory=list)
    subquestion_id: str | None = None
    status: ClaimStatus = "pending"
    evidence: list[Evidence] = field(default_factory=list)
    confidence: Literal["high", "medium", "low", "none"] = "none"
    independent_domains: int = 0


@dataclass
class Conflict:
    claim_id: str
    claim_text: str
    supporting: list[str]
    contradicting: list[str]
    note: str = ""


@dataclass
class Usage:
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tavily_calls: int = 0
    tavily_credits: float = 0.0
    est_llm_cost_usd: float | None = None
    by_model: dict[str, dict[str, int]] = field(default_factory=dict)


@dataclass
class ResearchResult:
    question: str
    mode: Literal["research", "audit"] = "research"
    status: RunStatus = "insufficient_evidence"
    tldr: str = ""
    claims: list[Claim] = field(default_factory=list)
    dropped_claims: list[Claim] = field(default_factory=list)
    conflicts: list[Conflict] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    sources: list[Source] = field(default_factory=list)
    plan: Plan | None = None
    queries_run: list[str] = field(default_factory=list)
    rounds: int = 0
    stop_reason: str = ""
    errors: list[str] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    elapsed_s: float = 0.0
    trace: list[dict] = field(default_factory=list)

    def source(self, source_id: str) -> Source | None:
        return next((s for s in self.sources if s.id == source_id), None)

    def verified_count(self) -> tuple[int, int]:
        facts = [c for c in self.claims if c.kind == "fact"]
        ok = [c for c in facts if c.status in ("supported", "partial", "conflicting")]
        return len(ok), len(facts) + len([c for c in self.dropped_claims if c.kind == "fact"])

    def to_dict(self) -> dict:
        data = asdict(self)
        for s in data["sources"]:
            s.pop("page_text", None)  # large; not needed by clients
        return data
