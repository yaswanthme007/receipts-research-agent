"""Receipts CLI.

    python src/agent.py "Your research question"
    python src/agent.py --audit path/to/text.txt
    python src/agent.py --audit-text "Claim to check..."

Prints the answer as Markdown, streams the research trace to stderr, and saves
runs/<timestamp>.md and .json.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from receipts.config import Settings  # noqa: E402
from receipts.pipeline import build  # noqa: E402
from receipts.render import to_markdown  # noqa: E402


def _fmt(event: dict) -> str | None:
    t, kind = event["t"], event["type"]
    if kind == "plan":
        lines = [f"[{t:6.1f}s] plan: {len(event['subquestions'])} sub-question(s)"]
        lines += [f"           {s['id']}: {s['text']}  <- {s['queries']}" for s in event["subquestions"]]
        return "\n".join(lines)
    if kind == "search":
        return f"[{t:6.1f}s] tavily.search  \"{event['query']}\" -> {event['results']} results ({event['credits']} cr, {event['ms']} ms)"
    if kind == "extract":
        return f"[{t:6.1f}s] tavily.extract {event['urls']} pages -> {event['ok']} read, {event['failed']} failed"
    if kind == "llm":
        fb = " (fallback)" if event.get("fallback") else ""
        return f"[{t:6.1f}s] token_factory  {event['role']:<10} {event['model']}{fb} {event['prompt_tokens']}+{event['completion_tokens']} tok, {event['ms']} ms"
    if kind == "verify":
        return f"[{t:6.1f}s] verify {event['claim']}: {event['status']}"
    if kind in ("select", "refine", "round", "injection_flag", "tool_error", "llm_error", "empty_results",
                "invalid_citation", "plan_fallback", "draft_failed", "done"):
        data = {k: v for k, v in event.items() if k not in ("t", "type")}
        return f"[{t:6.1f}s] {kind}: {json.dumps(data, default=str)[:220]}"
    return None


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

    ap = argparse.ArgumentParser(description="Research agent that checks every claim against its source.")
    ap.add_argument("question", nargs="*", help="research question")
    ap.add_argument("--audit", metavar="FILE", help="audit the claims in a text file")
    ap.add_argument("--audit-text", metavar="TEXT", help="audit the claims in this text")
    ap.add_argument("--rounds", type=int, help="max research rounds (default 2)")
    ap.add_argument("--quiet", action="store_true", help="don't stream the trace")
    ap.add_argument("--out", default="runs", help="folder for saved reports")
    args = ap.parse_args()

    settings = Settings.from_env()
    missing = settings.missing_keys()
    if missing:
        print(f"Missing {', '.join(missing)}. Copy .env.example to .env and add your keys.", file=sys.stderr)
        return 2
    if args.rounds:
        settings.max_rounds = args.rounds

    listener = None if args.quiet else (lambda e: (line := _fmt(e)) and print(line, file=sys.stderr, flush=True))
    agent = build(settings, listener)

    if args.audit or args.audit_text:
        text = Path(args.audit).read_text(encoding="utf-8") if args.audit else args.audit_text
        result = agent.audit(text)
    else:
        question = " ".join(args.question).strip() or "What is Nebius Token Factory and which open models does it serve?"
        result = agent.run(question)

    md = to_markdown(result)
    print(md)

    os.makedirs(args.out, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    Path(args.out, f"{stamp}.md").write_text(md, encoding="utf-8")
    Path(args.out, f"{stamp}.json").write_text(json.dumps(result.to_dict(), indent=2, default=str), encoding="utf-8")
    print(f"Saved {args.out}/{stamp}.md and .json", file=sys.stderr)
    return 0 if result.status in ("answered", "partial") else 1


if __name__ == "__main__":
    raise SystemExit(main())
