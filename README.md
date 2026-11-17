# Receipts — a research agent that checks its own citations

> ODSC × Nebius Hackathon submission · Challenge: *Build a research agent that brings its sources*

**One-line pitch:** Ask a research question (or paste any text to audit) and get an answer where every factual claim has been checked against the actual page it cites. Each citation opens the source **highlighted on the exact sentence**, and anything that can't be verified is removed and listed openly.

- **Demo video (2–3 min):** _coming soon_
- **Team:** Yaswanth K B ([@yaswanthme007](https://github.com/yaswanthme007))

## The problem

AI research tools are good at producing links and bad at producing links that actually back up what they say. A 2026 study of deep-research agents (*"Cited but Not Verified"*, arXiv 2605.06635) found links worked over 94% of the time, but the cited page actually supported the claim only about 39–77% of the time. So the reader still has to open every source and check by hand.

**Who it's for:** engineers and tech leads making decisions that depend on current facts ("Which provider serves model X, at what price?", "What changed in library Y's latest release?"), plus anyone who has to check text they were handed, whether from a colleague, an article or another AI.

**What it saves:** the manual verification pass. Receipts does that pass itself and shows its work.

## What makes it different

| Typical research agent | Receipts |
|---|---|
| Lists links under the answer | Every fact is tied to a **verbatim quote** found on the cited page |
| Trusts the model's citations | A separate checker judges each claim, and the quote must match the real page text, so a made-up quote can't pass |
| Links open the top of the page | Links use [Text Fragments](https://developer.mozilla.org/en-US/docs/Web/URI/Reference/Fragment/Text_fragments) and open **scrolled to and highlighting** the quote |
| Unsupported claims slip through | Unverified claims are **removed and listed** under "Removed" |
| Picks one side when sources disagree | **Disagreements** are shown with both sources |
| Always produces an answer | Says **"Insufficient evidence"** and shows what it searched |
| Web text mixed into instructions | Web text is fenced as untrusted data; pages that try **prompt injection are flagged** |
| Black box | A **live trace** of every Tavily and Token Factory call, with tokens, credits, time and stop reason |

## How it works

```
question
  └─ 1. PLAN       fast model → 1-4 sub-questions, 1-2 targeted queries each, freshness + topic
  └─ 2. SEARCH     Tavily search (advanced depth, parallel, published dates)
  └─ 3. SELECT     canonical-URL dedupe · syndicated-copy (shingle) dedupe · ≤2 per domain · ranked by relevance
  └─ 4. DRAFT      smart model → atomic claims (fact / inference), each citing source IDs
  └─ 5. READ       Tavily extract → full page text for every cited source (one batched call)
  └─ 6. VERIFY     per claim: best passages from the page → fast model judges supports / partial / contradicts / no
                   → its quote must be found in the page text (exact or ≥85% word overlap on a real sentence)
  └─ 7. GAPS?      sub-questions without a verified claim → refine queries → back to 2 (budget permitting)
  └─ 8. ANSWER     summary written ONLY from verified claims · conflicts · removed claims · gaps · sources · trace
```

- **Query planning:** the planner splits the question into sub-questions and picks time range and topic. After verification, any sub-question without a verified claim triggers new queries from a different angle (official sources, exact names, broader or narrower scope). Already-tried queries are never repeated.
- **When it stops (enforced in code):** every sub-question has a verified claim, **or** max rounds is reached (default 2), **or** the Tavily credit budget or time budget is used up, **or** the planner has no new angle. The stop reason is shown in every answer.
- **Evidence handling:** URLs are canonicalized (tracking parameters removed), near-duplicate syndicated articles are dropped, and each domain is capped at 2 sources so one site can't dominate. Confidence is **high** only when 2+ independent domains support a claim.
- **Safety:** web content is wrapped in `<source>` blocks and every prompt says to treat it as data, never instructions. Look-alike tags inside page text are neutralized, and pages containing injection phrases are flagged in the output. The checker only judges support, so an injected page can't change what gets written.
- **Trade-offs:** the fast model does planning, checking and summaries, and the smart model only writes the draft. Page text is fetched once per source and checks are cached across rounds. Verification adds roughly one fast-model call per claim, which costs some time and tokens. We think that's the right price for an answer you can trust.
- **Reliability:** Tavily calls retry with backoff, but give up immediately on key or plan errors. Token Factory 429/529, timeouts or a missing model fall back automatically to `NEBIUS_MODEL_FALLBACK`. If structured JSON output isn't supported, it downgrades from `json_schema` to `json_object` to plain prompting. If every search fails, the result is `TOOL FAILURE`, never an answer made up from memory.

## Required services

| Requirement | How we use it |
|---|---|
| **Nebius Token Factory** (inference) | OpenAI-compatible API. `NEBIUS_MODEL_FAST` handles planning, claim checking, query refinement and the summary. `NEBIUS_MODEL_SMART` writes the claim-structured draft. `NEBIUS_MODEL_FALLBACK` takes over on overload or rate limits. Uses `response_format` JSON schema output, and `/v1/models?verbose=true` for live model availability and cost estimates. |
| **Tavily** (real-time web search) | `search` (advanced depth, published dates, time range and topic from the planner) finds evidence. `extract` gets full page text, which is the ground truth the checker quotes from. Source URLs carry through to the final answer as highlight links. |

## Setup

```bash
git clone https://github.com/yaswanthme007/<repo>.git
cd <repo>
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # add your keys. .env is git-ignored
python scripts/doctor.py           # checks keys, lists models, makes 1 test call each
python src/app.py                  # web app at http://127.0.0.1:8000
python src/agent.py "Your research question here"   # or use the CLI
```

**Web app:** type a question, or switch to *Check someone's text* to audit a paragraph. The research log streams every Tavily and Token Factory call live. In the answer, verified claims are highlighted, and each citation number opens the source page **scrolled to and highlighting the quoted sentence**. *Replay the last run* re-streams a saved run without spending credits. With no saved runs, it replays a clearly labelled fictional sample (`tests/fixtures/sample_run.json`) that is only used to develop the UI.

Audit mode:

```bash
python src/agent.py --audit-text "Paste a paragraph or an AI answer to fact-check"
python src/agent.py --audit path/to/article.txt
```

Every run saves `runs/<timestamp>.md` (the report) and `runs/<timestamp>.json` (the full result and trace).

Tests (offline, no keys needed):

```bash
python -m pytest
```

## Example

_Live example coming after first real run._

## Failure / insufficient-evidence case

- **No usable results:** the agent tries a refined query, then reports **INSUFFICIENT EVIDENCE** with the queries it ran. It never writes an answer.
- **All claims fail verification:** same result. Nothing unverified is presented as an answer.
- **Tool failure** (Tavily down or bad key): **TOOL FAILURE** with the error. Token Factory overload switches to the fallback model, and the trace shows it.

## Project layout

```
src/app.py                web app entry (FastAPI + Uvicorn)
src/receipts/server.py    streaming API: /api/run (NDJSON trace + result), /api/replay, /api/health
web/                      single-page UI (no build step): index.html, styles.css, app.js
src/agent.py              CLI
src/receipts/pipeline.py  research loop: plan → search → select → draft → verify → refine → finalize
src/receipts/evidence.py  dedupe, ranking, injection scan, passage picking, quote matching, highlight links
src/receipts/llm.py       Token Factory client: fallback, JSON modes, usage and cost tracking
src/receipts/search.py    Tavily client: search + extract, retries, credit tracking
src/receipts/prompts.py   prompts + JSON schemas (trust boundary in every web-facing prompt)
src/receipts/render.py    Markdown report
scripts/doctor.py         setup checker
tests/                    offline tests with fake Token Factory + Tavily
```

## Judging criteria map

| Criterion | Weight | Where to look |
|---|---|---|
| Sourced and verifiable answers | 30% | "What makes it different", Example, verify step in `pipeline.py` |
| Works end to end | 25% | Setup, `scripts/doctor.py`, failure cases, demo |
| Usefulness | 20% | The problem, audit mode |
| Agent design | 15% | How it works (planning, stop rules, evidence handling, safety, trade-offs) |
| Use of Token Factory and Tavily | 10% | Required services, trace output |

## License

MIT
