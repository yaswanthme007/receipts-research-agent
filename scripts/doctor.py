"""Check your setup before running the agent.

    python scripts/doctor.py            # keys, model list, configured models, 1 tiny chat + 1 search
    python scripts/doctor.py --models   # just list the models your key can use

Uses ~1 Tavily credit and a few hundred tokens. Never prints your keys.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from receipts.config import Settings  # noqa: E402
from receipts.llm import LLM  # noqa: E402
from receipts.search import WebSearch  # noqa: E402
from receipts.trace import Tracer  # noqa: E402

OK, BAD, WARN = "[ok]  ", "[FAIL]", "[warn]"


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", action="store_true", help="only list available models")
    args = ap.parse_args()

    s = Settings.from_env()
    failures = 0
    missing = s.missing_keys()
    for key in ("NEBIUS_API_KEY", "TAVILY_API_KEY"):
        print(f"{BAD if key in missing else OK} {key} {'missing' if key in missing else 'set'}")

    tracer = Tracer()
    if "TAVILY_API_KEY" in missing:
        failures += 1
    elif not args.models:
        try:
            # Basic depth = 1 credit.
            rows = WebSearch(replace(s, search_depth="basic"), tracer).search("Nebius Token Factory", max_results=2)
            print(f"{OK} Tavily search works -> {len(rows)} results, e.g. {rows[0]['url'] if rows else '-'}")
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"{BAD} Tavily search failed: {str(e)[:200]}")

    if "NEBIUS_API_KEY" in missing:
        print("\nNEBIUS_API_KEY missing: Token Factory checks skipped. Add it to .env and re-run.")
        return 1
    print(f"{OK} base URL: {s.base_url}")

    llm = LLM(s, tracer)
    try:
        catalog = llm.fetch_catalog()
    except Exception as e:  # noqa: BLE001
        print(f"{BAD} could not list models: {type(e).__name__}: {str(e)[:200]}")
        return 1
    ids = sorted(m["id"] for m in catalog)
    print(f"{OK} {len(ids)} models available")
    if args.models:
        for m in sorted(catalog, key=lambda m: m["id"]):
            p = m.get("pricing") or {}
            price = ""
            if "prompt" in p and "completion" in p:
                price = f"  ${float(p['prompt']) * 1e6:.2f} in / ${float(p['completion']) * 1e6:.2f} out per 1M"
            ctx = f"  ctx {m['context_length']}" if m.get("context_length") else ""
            print(f"   {m['id']}{ctx}{price}")
        return 0

    for label, model in (("fast", s.model_fast), ("smart", s.model_smart), ("fallback", s.model_fallback)):
        if not model:
            continue
        found = model in ids
        failures += 0 if found or label == "fallback" else 1
        print(f"{OK if found else (WARN if label == 'fallback' else BAD)} {label} model {model} {'available' if found else 'NOT in catalog'}")
        if not found:
            close = [i for i in ids if model.split("/")[-1].split("-")[0].lower() in i.lower()][:5]
            if close:
                print(f"        similar: {', '.join(close)}")

    for tier in ("fast", "smart"):
        try:
            data = llm.json(
                role=f"doctor_{tier}",
                system="Answer briefly.",
                user='Return {"ok": true, "model": "<your name>"}',
                schema={"type": "object", "additionalProperties": False, "required": ["ok", "model"],
                        "properties": {"ok": {"type": "boolean"}, "model": {"type": "string"}}},
                tier=tier,
                max_tokens=200,
            )
            mode = llm._json_mode.get(llm.model_for(tier), "json_schema")
            print(f"{OK} {tier} model JSON call works (mode: {mode}) -> {data}")
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"{BAD} {tier} model call failed: {type(e).__name__}: {str(e)[:200]}")

    u = tracer.usage
    print(f"\nUsed {u.prompt_tokens + u.completion_tokens} tokens, {u.tavily_credits:g} Tavily credits.")
    print("All good." if not failures else f"{failures} problem(s) found.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
