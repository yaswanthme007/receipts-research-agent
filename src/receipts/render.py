"""Render a ResearchResult as Markdown (CLI output, saved reports, README examples)."""

from __future__ import annotations

from .schema import Claim, ResearchResult

STATUS_LABEL = {
    "answered": "ANSWERED",
    "partial": "PARTIAL ANSWER",
    "insufficient_evidence": "INSUFFICIENT EVIDENCE",
    "tool_failure": "TOOL FAILURE",
}
CLAIM_LABEL = {
    "supported": "verified",
    "partial": "partly verified",
    "conflicting": "sources disagree",
    "contradicted": "contradicted",
    "unsupported": "not verified",
    "inference": "inference",
}


def _numbering(result: ResearchResult) -> dict[str, int]:
    """Display numbers [1], [2]... in order of first citation."""
    order: dict[str, int] = {}
    for c in result.claims + result.dropped_claims:
        for sid in c.source_ids:
            if sid not in order and result.source(sid):
                order[sid] = len(order) + 1
    for s in result.sources:
        order.setdefault(s.id, len(order) + 1)
    return order


def _cites(c: Claim, result: ResearchResult, num: dict[str, int]) -> str:
    links = []
    ev = {e.source_id: e for e in c.evidence}
    for sid in c.source_ids:
        src = result.source(sid)
        if not src:
            continue
        e = ev.get(sid)
        url = e.highlight_url if e and e.highlight_url else src.url
        links.append(f"[[{num[sid]}]]({url})")
    return " ".join(links)


def to_markdown(result: ResearchResult) -> str:
    num = _numbering(result)
    ok, total = result.verified_count()
    out = []
    title = "Audit" if result.mode == "audit" else "Question"
    out.append(f"# {title}: {result.question}\n")
    badge = f"**{STATUS_LABEL.get(result.status, result.status)}**"
    if total:
        badge += f" · {ok}/{total} factual claims verified against source text"
    if result.conflicts:
        badge += f" · {len(result.conflicts)} disagreement(s)"
    if result.gaps:
        badge += f" · {len(result.gaps)} gap(s)"
    out.append(badge + "\n")

    out.append("## Answer\n")
    out.append((result.tldr or "_No answer._") + "\n")

    if result.claims:
        out.append("## Findings\n")
        for c in result.claims:
            tag = CLAIM_LABEL.get(c.status, c.status)
            if c.kind == "fact" and c.confidence in ("high", "medium"):
                tag += f", {c.confidence} confidence"
            if c.independent_domains >= 2:
                tag += f", {c.independent_domains} independent sources"
            prefix = "*(Inference)* " if c.kind == "inference" else ""
            out.append(f"- **[{c.id}]** {prefix}{c.text} {_cites(c, result, num)} — _{tag}_")
            for e in c.evidence:
                if e.quote and e.verdict != "does_not_support":
                    src = result.source(e.source_id)
                    verb = {"supports": "says", "partially_supports": "partly says", "contradicts": "contradicts"}.get(e.verdict, "")
                    out.append(f'  > [{num[e.source_id]}] {src.domain} {verb}: "{e.quote.strip()}"')
        out.append("")

    if result.conflicts:
        out.append("## Where sources disagree\n")
        for cf in result.conflicts:
            side_a = ", ".join(f"[{num[s]}]" for s in cf.supporting if s in num)
            side_b = ", ".join(f"[{num[s]}]" for s in cf.contradicting if s in num)
            line = f"- {cf.claim_text} — for: {side_a or '–'}"
            if side_b:
                line += f"; against: {side_b}"
            out.append(line + (f" _({cf.note})_" if cf.note else ""))
        out.append("")

    if result.dropped_claims:
        out.append("## Removed: could not be verified\n")
        for c in result.dropped_claims:
            out.append(f"- ~~{c.text}~~ — _{CLAIM_LABEL.get(c.status, c.status)}_ {_cites(c, result, num)}")
        out.append("")

    if result.gaps:
        out.append("## Gaps and open questions\n")
        out += [f"- {g}" for g in result.gaps]
        out.append("")

    cited = [s for s in result.sources if any(s.id in c.source_ids for c in result.claims + result.dropped_claims)]
    if result.sources:
        out.append("## Sources\n")
        for s in sorted(result.sources, key=lambda s: num[s.id]):
            notes = []
            if s.published_date:
                notes.append(f"published ~{s.published_date[:10]}")
            if s not in cited:
                notes.append("consulted, not cited")
            if s.fetch_error:
                notes.append("page could not be read; checked against search snippet")
            if s.injection_flags:
                notes.append("WARNING: page contains instructions aimed at AI agents (ignored)")
            extra = f" — {'; '.join(notes)}" if notes else ""
            out.append(f"{num[s.id]}. [{s.title}]({s.url}) · {s.domain}{extra}")
        out.append("")

    u = result.usage
    out.append("## How this was researched\n")
    if result.plan:
        out.append("Plan: " + " | ".join(f"{sq.id} {sq.text}{' ✓' if sq.covered else ''}" for sq in result.plan.subquestions))
    out.append(f"- Rounds: {result.rounds} · stop reason: {result.stop_reason or '–'}")
    out.append(f"- Searches: {len(result.queries_run)} queries · Tavily calls: {u.tavily_calls} · credits: {u.tavily_credits:g}")
    models = ", ".join(f"`{m}` ×{v['calls']}" for m, v in u.by_model.items()) or "–"
    out.append(f"- Token Factory: {u.llm_calls} calls ({models}) · {u.prompt_tokens + u.completion_tokens:,} tokens")
    if u.est_llm_cost_usd is not None:
        out.append(f"- Estimated inference cost: ${u.est_llm_cost_usd:.4f}")
    out.append(f"- Time: {result.elapsed_s:.1f} s")
    if result.errors:
        out.append("- Errors: " + " | ".join(e[:120] for e in result.errors))
    return "\n".join(out) + "\n"
