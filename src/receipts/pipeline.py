"""The research loop.

plan -> search -> select & dedupe -> draft claims -> verify each claim against page text
     -> (gaps? refine queries and repeat, within budget) -> summarize only verified claims
"""

from __future__ import annotations

import datetime as dt
import time
from concurrent.futures import ThreadPoolExecutor

from . import prompts as P
from .config import Settings
from .evidence import (
    canonical_url,
    domain_of,
    fence_untrusted,
    find_quote,
    iso_date,
    scan_injection,
    select_sources,
    text_fragment_url,
    top_passages,
)
from .llm import LLMError
from .schema import Claim, Conflict, Evidence, Plan, ResearchResult, Source, SubQuestion
from .search import ToolError
from .trace import Tracer

KEPT = ("supported", "partial", "conflicting")


def _today() -> str:
    return dt.date.today().isoformat()


class Researcher:
    def __init__(self, settings: Settings, llm, web, tracer: Tracer):
        self.s = settings
        self.llm = llm
        self.web = web
        self.tracer = tracer
        self._verify_cache: dict[tuple[str, tuple[str, ...]], Claim] = {}
        self._next_source = 1

    # ================================================================ public API
    def run(self, question: str) -> ResearchResult:
        result = ResearchResult(question=question, mode="research")
        self.tracer.emit("start", mode="research", question=question)
        self._load_prices()

        plan = self._plan(question)
        result.plan = plan
        sources: list[Source] = []
        tried: list[str] = []
        pending = [(sq.id, q) for sq in plan.subquestions for q in sq.queries][: self.s.queries_per_round]
        claims: list[Claim] = []
        draft: dict = {}

        for round_no in range(1, self.s.max_rounds + 1):
            result.rounds = round_no
            self.tracer.emit("round", n=round_no, queries=[q for _, q in pending])
            candidates, errors = self._search_all(pending, plan)
            tried += [q for _, q in pending]
            result.errors += errors

            if errors and len(errors) == len(pending) and not sources:
                result.status = "tool_failure"
                result.stop_reason = "Every search call failed; nothing was answered from memory."
                break

            sources = self._add_sources(candidates, sources)
            result.sources = sources
            if not sources:
                if round_no < self.s.max_rounds:
                    pending = self._refine(question, plan, tried, "searches returned no usable results")
                    if pending:
                        continue
                result.stop_reason = "No usable search results."
                break

            draft = self._draft(question, plan, sources)
            claims = self._claims_from_draft(draft, plan, sources)
            self._verify(claims, sources)
            self._update_coverage(plan, claims)

            uncovered = [sq for sq in plan.subquestions if not sq.covered]
            if not uncovered:
                result.stop_reason = "Every sub-question has at least one verified claim."
                break
            if round_no == self.s.max_rounds:
                result.stop_reason = f"Reached max research rounds ({self.s.max_rounds})."
                break
            budget = self._budget_exceeded()
            if budget:
                result.stop_reason = budget
                break
            pending = self._refine(question, plan, tried, "no verified evidence yet", uncovered, claims)
            if not pending:
                result.stop_reason = "Planner found no new angles worth searching."
                break

        result.queries_run = tried
        if result.status != "tool_failure":
            self._finalize(result, claims, draft, sources)
        return self._close(result)

    def audit(self, text: str, max_claims: int = 8) -> ResearchResult:
        """Check someone else's text (an article, an AI answer) claim by claim."""
        result = ResearchResult(question=text[:300], mode="audit")
        self.tracer.emit("start", mode="audit", chars=len(text))
        self._load_prices()
        try:
            data = self.llm.json(
                role="audit_extract",
                system=P.AUDIT_SYSTEM.format(n=max_claims),
                user=f"<text_to_audit>\n{fence_untrusted(text, 12000)}\n</text_to_audit>",
                schema=P.AUDIT_SCHEMA,
                tier="fast",
            )
        except LLMError as e:
            result.errors.append(str(e))
            result.status = "tool_failure"
            result.stop_reason = "Could not extract claims."
            return self._close(result)

        items = [c for c in data.get("claims", []) if c.get("text")][:max_claims]
        plan = Plan(subquestions=[SubQuestion(id=f"Q{i+1}", text=c["text"], queries=[c.get("query") or c["text"]])
                                  for i, c in enumerate(items)])
        result.plan = plan
        sources: list[Source] = []
        claims: list[Claim] = []
        pending = [(sq.id, sq.queries[0]) for sq in plan.subquestions]
        hits_by_sq: dict[str, list[Source]] = {}
        candidates, errors = self._search_all(pending, plan, keep_query_tag=True)
        result.errors += errors
        result.queries_run = [q for _, q in pending]
        if pending and len(errors) == len(pending):
            result.status = "tool_failure"
            result.stop_reason = "Every search call failed."
            return self._close(result)
        for c in candidates:
            hits_by_sq.setdefault(c.query, []).append(c)

        seen: dict[str, Source] = {}
        for i, sq in enumerate(plan.subquestions):
            ids = []
            for cand in sorted(hits_by_sq.get(sq.id, []), key=lambda s: s.score, reverse=True)[:3]:
                key = canonical_url(cand.url)
                if key not in seen:
                    cand.id = f"S{self._next_source}"
                    self._next_source += 1
                    cand.injection_flags = scan_injection(cand.snippet)
                    seen[key] = cand
                    sources.append(cand)
                ids.append(seen[key].id)
            claims.append(Claim(id=f"C{i+1}", text=sq.text, kind="fact", source_ids=ids, subquestion_id=sq.id))

        result.sources = sources
        self._verify(claims, sources)
        result.claims = claims
        result.conflicts = self._conflicts(claims)
        n_ok = len([c for c in claims if c.status in KEPT])
        n_bad = len([c for c in claims if c.status == "contradicted"])
        result.tldr = (
            f"{n_ok} of {len(claims)} claims are backed by sources, {n_bad} are contradicted, "
            f"{len(claims) - n_ok - n_bad} could not be verified."
        )
        result.status = "answered" if claims else "insufficient_evidence"
        result.rounds = 1
        result.stop_reason = "Audit checks each claim once."
        return self._close(result)

    # ================================================================== stages
    def _load_prices(self) -> None:
        loader = getattr(self.llm, "load_prices", None)
        if loader:
            loader()

    def _plan(self, question: str) -> Plan:
        try:
            data = self.llm.json(
                role="plan",
                system=P.PLAN_SYSTEM.format(today=_today()),
                user=f"Question: {question}",
                schema=P.PLAN_SCHEMA,
                tier="fast",
            )
            sqs = []
            for i, sq in enumerate(data.get("subquestions", [])[: self.s.max_subquestions]):
                queries = [q.strip() for q in sq.get("queries", []) if q and q.strip()][:2]
                if sq.get("text") and queries:
                    sqs.append(SubQuestion(id=f"Q{i+1}", text=sq["text"].strip(), queries=queries))
            if not sqs:
                raise ValueError("empty plan")
            tr = data.get("time_range")
            plan = Plan(
                subquestions=sqs,
                time_range=None if tr in (None, "none") else tr,
                topic=data.get("topic") if data.get("topic") in ("general", "news", "finance") else "general",
                notes=data.get("notes", ""),
            )
        except (LLMError, ValueError) as e:
            self.tracer.emit("plan_fallback", error=str(e)[:160])
            plan = Plan(subquestions=[SubQuestion(id="Q1", text=question, queries=[question])])
        self.tracer.emit(
            "plan",
            subquestions=[{"id": s.id, "text": s.text, "queries": s.queries} for s in plan.subquestions],
            time_range=plan.time_range, topic=plan.topic,
        )
        return plan

    def _search_all(
        self, pending: list[tuple[str, str]], plan: Plan, keep_query_tag: bool = False
    ) -> tuple[list[Source], list[str]]:
        def one(item):
            sq_id, query = item
            try:
                rows = self.web.search(query, time_range=plan.time_range, topic=plan.topic)
                return sq_id, query, rows, None
            except ToolError as e:
                return sq_id, query, [], str(e)

        candidates, errors = [], []
        with ThreadPoolExecutor(max_workers=self.s.workers) as pool:
            for sq_id, query, rows, err in pool.map(one, pending):
                if err:
                    errors.append(err)
                    continue
                for r in rows:
                    url = r.get("url")
                    if not url:
                        continue
                    candidates.append(
                        Source(
                            id="",
                            url=url,
                            title=(r.get("title") or url)[:200],
                            domain=domain_of(url),
                            snippet=r.get("content") or "",
                            score=float(r.get("score") or 0.0),
                            published_date=iso_date(r.get("published_date")),
                            query=sq_id if keep_query_tag else query,
                        )
                    )
        if not candidates and not errors:
            self.tracer.emit("empty_results", queries=[q for _, q in pending])
        return candidates, errors

    def _add_sources(self, candidates: list[Source], existing: list[Source]) -> list[Source]:
        chosen = select_sources(candidates, existing, self.s.max_sources, self.s.max_per_domain)
        new = [s for s in chosen if not s.id]
        for s in new:
            s.id = f"S{self._next_source}"
            self._next_source += 1
            s.injection_flags = scan_injection(s.snippet)
            if s.injection_flags:
                self.tracer.emit("injection_flag", source=s.id, url=s.url, matches=s.injection_flags)
        self.tracer.emit(
            "select", candidates=len(candidates), kept_new=len(new), total=len(chosen),
            dropped=len(candidates) - len(new),
        )
        return chosen

    def _sources_block(self, sources: list[Source], per_source: int = 2400) -> str:
        parts = []
        for s in sources:
            parts.append(
                f'<source id="{s.id}" title="{fence_untrusted(s.title, 150)}" domain="{s.domain}" '
                f'published="{s.published_date or "unknown"}">\n{fence_untrusted(s.snippet, per_source)}\n</source>'
            )
        return "\n\n".join(parts)

    def _draft(self, question: str, plan: Plan, sources: list[Source]) -> dict:
        subq = "\n".join(f"{sq.id}: {sq.text}" for sq in plan.subquestions)
        user = (
            f"Question: {question}\n\nSub-questions:\n{subq}\n\n"
            f"Sources:\n{self._sources_block(sources)}"
        )
        try:
            return self.llm.json(
                role="draft",
                system=P.DRAFT_SYSTEM.format(today=_today(), trust=P.TRUST_BOUNDARY),
                user=user,
                schema=P.DRAFT_SCHEMA,
                tier="smart",
                max_tokens=3500,
                temperature=0.2,
            )
        except LLMError as e:
            self.tracer.emit("draft_failed", error=str(e)[:160])
            return {"tldr": "", "claims": [], "disagreements": [], "gaps": [str(e)]}

    def _claims_from_draft(self, draft: dict, plan: Plan, sources: list[Source]) -> list[Claim]:
        valid_ids = {s.id for s in sources}
        valid_sq = {sq.id for sq in plan.subquestions}
        claims = []
        for i, c in enumerate(draft.get("claims", [])):
            text = (c.get("text") or "").strip()
            if not text:
                continue
            ids = [sid for sid in dict.fromkeys(c.get("source_ids") or []) if sid in valid_ids]
            invented = [sid for sid in (c.get("source_ids") or []) if sid not in valid_ids]
            if invented:
                self.tracer.emit("invalid_citation", claim=text[:80], ids=invented)
            sq = c.get("subquestion_id") if c.get("subquestion_id") in valid_sq else None
            claims.append(
                Claim(
                    id=f"C{i+1}", text=text, kind="inference" if c.get("kind") == "inference" else "fact",
                    source_ids=ids, subquestion_id=sq,
                )
            )
        return claims

    # ------------------------------------------------------------------ verify
    def _verify(self, claims: list[Claim], sources: list[Source]) -> None:
        by_id = {s.id: s for s in sources}
        facts = [c for c in claims if c.kind == "fact"]
        to_check, over_budget = facts[: self.s.max_claims_to_verify], facts[self.s.max_claims_to_verify :]
        for c in over_budget:
            c.status = "unsupported"
            self.tracer.emit("skip_claim", claim=c.id, reason="verification budget")

        # Fetch full page text for every cited source we haven't read yet (one batched call).
        need = []
        for c in to_check:
            for sid in c.source_ids:
                src = by_id.get(sid)
                if src and src.page_text is None and src.fetch_error is None and src not in need:
                    need.append(src)
        if need:
            if self._credits_left() <= 0:
                for src in need:
                    src.fetch_error = "skipped: credit budget"
            else:
                try:
                    pages, failed = self.web.extract([s.url for s in need])
                    by_canon = {canonical_url(u): t for u, t in pages.items()}
                    for src in need:
                        text = pages.get(src.url) or by_canon.get(canonical_url(src.url))
                        if text:
                            src.page_text = text
                            flags = scan_injection(text)
                            if flags and not src.injection_flags:
                                src.injection_flags = flags
                                self.tracer.emit("injection_flag", source=src.id, url=src.url, matches=flags)
                        else:
                            src.fetch_error = failed.get(src.url, "no content returned")
                except ToolError as e:
                    for src in need:
                        src.fetch_error = str(e)[:120]

        with ThreadPoolExecutor(max_workers=self.s.workers) as pool:
            list(pool.map(lambda c: self._check_claim(c, by_id), to_check))

        for c in claims:
            if c.kind == "inference":
                c.status = "inference"

    def _check_claim(self, claim: Claim, by_id: dict[str, Source]) -> None:
        cited = [by_id[sid] for sid in claim.source_ids if sid in by_id]
        if not cited:
            claim.status = "unsupported"
            self.tracer.emit("verify", claim=claim.id, status="unsupported", reason="no valid citation")
            return

        key = (" ".join(claim.text.lower().split()), tuple(sorted(s.id for s in cited)))
        cached = self._verify_cache.get(key)
        if cached:
            claim.evidence, claim.status = cached.evidence, cached.status
            claim.confidence, claim.independent_domains = cached.confidence, cached.independent_domains
            return

        blocks = []
        for s in cited:
            passages = top_passages(s.best_text(), claim.text, k=3)
            body = "\n[...]\n".join(fence_untrusted(p, 1500) for p in passages)
            blocks.append(f'<source id="{s.id}" domain="{s.domain}">\n{body}\n</source>')
        user = f"Claim: {claim.text}\n\nPassages:\n" + "\n\n".join(blocks)

        try:
            data = self.llm.json(
                role="verify",
                system=P.VERIFY_SYSTEM.format(trust=P.TRUST_BOUNDARY),
                user=user,
                schema=P.VERIFY_SCHEMA,
                tier="fast",
                max_tokens=1200,
                temperature=0.0,
            )
        except LLMError as e:
            claim.status = "unsupported"
            self.tracer.emit("verify", claim=claim.id, status="unsupported", reason=f"checker failed: {e}"[:160])
            return

        evidence = []
        for j in data.get("judgments", []):
            src = by_id.get(j.get("source_id"))
            if not src or src not in cited:
                continue
            verdict = j.get("verdict")
            if verdict not in ("supports", "partially_supports", "does_not_support", "contradicts"):
                continue
            exact, match = find_quote(j.get("quote") or "", src.best_text())
            evidence.append(
                Evidence(
                    source_id=src.id,
                    verdict=verdict,
                    quote=exact,
                    quote_match=match,
                    explanation=(j.get("explanation") or "")[:300],
                    highlight_url=text_fragment_url(src.url, exact) if exact else None,
                )
            )
        claim.evidence = evidence

        # A verdict only counts if its quote was found on the actual page.
        sup = [e for e in evidence if e.verdict == "supports" and e.quote]
        par = [e for e in evidence if e.verdict == "partially_supports" and e.quote]
        con = [e for e in evidence if e.verdict == "contradicts" and e.quote]
        if (sup or par) and con:
            claim.status = "conflicting"
        elif sup:
            claim.status = "supported"
        elif par:
            claim.status = "partial"
        elif con:
            claim.status = "contradicted"
        else:
            claim.status = "unsupported"

        domains = {by_id[e.source_id].domain for e in sup + par}
        claim.independent_domains = len(domains)
        if claim.status == "supported":
            claim.confidence = "high" if len({by_id[e.source_id].domain for e in sup}) >= 2 else "medium"
        elif claim.status in ("partial", "conflicting"):
            claim.confidence = "low"
        else:
            claim.confidence = "none"

        self._verify_cache[key] = claim
        self.tracer.emit(
            "verify", claim=claim.id, text=claim.text[:120], status=claim.status,
            confidence=claim.confidence,
            evidence=[{"source": e.source_id, "verdict": e.verdict, "quote_match": e.quote_match} for e in evidence],
        )

    # ------------------------------------------------------------------ refine
    def _update_coverage(self, plan: Plan, claims: list[Claim]) -> None:
        for sq in plan.subquestions:
            sq.covered = any(c.subquestion_id == sq.id and c.status in KEPT for c in claims)
        # Single-question plans: any verified claim covers it, even if the model skipped the id.
        if len(plan.subquestions) == 1 and any(c.status in KEPT for c in claims):
            plan.subquestions[0].covered = True

    def _refine(self, question, plan, tried, reason, uncovered=None, claims=None) -> list[tuple[str, str]]:
        uncovered = uncovered or plan.subquestions
        failed = [c.text for c in (claims or []) if c.kind == "fact" and c.status not in KEPT][:6]
        user = (
            f"Question: {question}\nWhy we need more: {reason}\n\n"
            "Sub-questions still missing verified evidence:\n"
            + "\n".join(f"{sq.id}: {sq.text}" for sq in uncovered)
            + "\n\nQueries already tried:\n" + "\n".join(f"- {q}" for q in tried)
            + ("\n\nClaims we could not verify:\n" + "\n".join(f"- {t}" for t in failed) if failed else "")
        )
        try:
            data = self.llm.json(
                role="refine",
                system=P.REFINE_SYSTEM.format(today=_today(), n=self.s.queries_per_round),
                user=user,
                schema=P.REFINE_SCHEMA,
                tier="fast",
            )
        except LLMError as e:
            self.tracer.emit("refine_failed", error=str(e)[:160])
            return []
        valid = {sq.id for sq in plan.subquestions}
        seen = {q.lower().strip() for q in tried}
        out = []
        for item in data.get("queries", []):
            q = (item.get("query") or "").strip()
            if q and q.lower() not in seen:
                seen.add(q.lower())
                sq_id = item.get("subquestion_id") if item.get("subquestion_id") in valid else uncovered[0].id
                out.append((sq_id, q))
        out = out[: self.s.queries_per_round]
        self.tracer.emit("refine", reason=reason, new_queries=[q for _, q in out])
        return out

    # ---------------------------------------------------------------- finalize
    def _finalize(self, result: ResearchResult, claims: list[Claim], draft: dict, sources: list[Source]) -> None:
        kept_facts = [c for c in claims if c.kind == "fact" and c.status in KEPT]
        result.dropped_claims = [c for c in claims if c.kind == "fact" and c.status not in KEPT]
        inferences = [c for c in claims if c.kind == "inference"] if kept_facts else []
        result.claims = [c for c in claims if c in kept_facts or c in inferences]
        result.conflicts = self._conflicts(claims)
        for d in draft.get("disagreements", []) or []:
            ids = [sid for sid in d.get("source_ids", []) if result.source(sid)]
            if d.get("description") and len(ids) >= 2:
                result.conflicts.append(
                    Conflict(claim_id="", claim_text=d["description"], supporting=ids, contradicting=[],
                             note="Flagged by the writer model; open both sources to compare.")
                )

        gaps = [g for g in (draft.get("gaps") or []) if g]
        if result.plan:
            gaps += [f"No verified evidence found for: {sq.text}" for sq in result.plan.subquestions if not sq.covered]
        result.gaps = gaps

        if not kept_facts:
            result.status = "insufficient_evidence"
            searched = "; ".join(result.queries_run[:6]) or "nothing"
            result.tldr = (
                "Insufficient evidence: I could not verify any claim against the sources found, so I am "
                f"not giving an answer. Searched: {searched}."
            )
            if not result.stop_reason:
                result.stop_reason = "No claim survived verification."
            return

        all_covered = result.plan is None or all(sq.covered for sq in result.plan.subquestions)
        result.status = "answered" if all_covered else "partial"
        result.tldr = self._summarize(result.question, result.claims) or draft.get("tldr", "")

    def _conflicts(self, claims: list[Claim]) -> list[Conflict]:
        out = []
        for c in claims:
            if c.status == "conflicting":
                out.append(
                    Conflict(
                        claim_id=c.id, claim_text=c.text,
                        supporting=[e.source_id for e in c.evidence if e.verdict in ("supports", "partially_supports") and e.quote],
                        contradicting=[e.source_id for e in c.evidence if e.verdict == "contradicts" and e.quote],
                        note="Sources disagree; both quotes were found on the pages.",
                    )
                )
        return out

    def _summarize(self, question: str, claims: list[Claim]) -> str:
        listing = "\n".join(
            f"[{c.id}] ({'inference' if c.kind == 'inference' else c.status}) {c.text}" for c in claims
        )
        try:
            data = self.llm.json(
                role="summarize",
                system=P.SUMMARY_SYSTEM,
                user=f"Question: {question}\n\nVerified claims:\n{listing}",
                schema=P.SUMMARY_SCHEMA,
                tier="fast",
                max_tokens=500,
            )
            return (data.get("tldr") or "").strip()
        except LLMError:
            return " ".join(c.text for c in claims[:2])

    # ------------------------------------------------------------------ budget
    def _credits_left(self) -> float:
        return self.s.tavily_credit_budget - self.tracer.usage.tavily_credits

    def _budget_exceeded(self) -> str | None:
        if self._credits_left() <= 0:
            return f"Tavily credit budget reached ({self.s.tavily_credit_budget})."
        if self.tracer.elapsed() >= self.s.time_budget_s:
            return f"Time budget reached ({int(self.s.time_budget_s)} s)."
        return None

    def _close(self, result: ResearchResult) -> ResearchResult:
        result.usage = self.tracer.usage
        result.elapsed_s = round(self.tracer.elapsed(), 1)
        result.trace = list(self.tracer.events)
        ok, total = result.verified_count()
        self.tracer.emit(
            "done", status=result.status, verified=ok, total=total, conflicts=len(result.conflicts),
            seconds=result.elapsed_s, tavily_credits=result.usage.tavily_credits,
            llm_calls=result.usage.llm_calls,
        )
        result.trace = list(self.tracer.events)
        return result


def build(settings: Settings | None = None, listener=None) -> Researcher:
    """Wire up real clients from environment settings."""
    from .llm import LLM
    from .search import WebSearch

    settings = settings or Settings.from_env()
    tracer = Tracer(listener)
    return Researcher(settings, LLM(settings, tracer), WebSearch(settings, tracer), tracer)
