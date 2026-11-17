"""Compare the hackathon template agent with Receipts on the same questions.

    python eval/run_eval.py                    # all questions, both systems
    python eval/run_eval.py --limit 5          # first 5 questions
    python eval/run_eval.py --only q01,u02     # specific questions
    python eval/run_eval.py --systems receipts

Writes eval/results/<timestamp>/: summary.md, results.json, judged.csv (one row per judged
claim, for checking the judge by hand). Progress is saved after every question.

Budget: roughly 15-25 Tavily credits and 30-60k tokens per question for both systems plus judging.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from receipts.config import Settings  # noqa: E402
from receipts.evaluation import (  # noqa: E402
    Judge,
    run_baseline,
    run_receipts,
    summarize,
    summary_markdown,
    to_jsonable,
)


def load_questions(path: Path, limit: int | None, only: str | None) -> list[dict]:
    qs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if only:
        wanted = {q.strip() for q in only.split(",")}
        qs = [q for q in qs if q["id"] in wanted]
    return qs[:limit] if limit else qs


def save(outdir: Path, answers, judged, judge_model: str, n: int) -> None:
    (outdir / "results.json").write_text(json.dumps(to_jsonable(answers, judged), indent=2), encoding="utf-8")
    summary = summarize(answers, judged)
    (outdir / "summary.md").write_text(summary_markdown(summary, judge_model, n), encoding="utf-8")
    with open(outdir / "judged.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["system", "question_id", "claim", "judge_grade", "quote_found_on_page", "page_url", "cited_urls", "your_check"])
        for j in judged:
            w.writerow([j.system, j.qid, j.text, j.grade, j.quote or "", j.url or "", " ".join(j.urls), ""])


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default=str(ROOT / "eval" / "questions.jsonl"))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--only")
    ap.add_argument("--systems", default="baseline,receipts")
    args = ap.parse_args()

    settings = Settings.from_env()
    if settings.missing_keys():
        print(f"Missing {', '.join(settings.missing_keys())} in .env", file=sys.stderr)
        return 2
    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    questions = load_questions(Path(args.questions), args.limit, args.only)
    outdir = ROOT / "eval" / "results" / dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir.mkdir(parents=True, exist_ok=True)

    judge = Judge(settings)
    print(f"{len(questions)} questions, systems: {', '.join(systems)}, judge: {judge.model}")
    print(f"Results: {outdir}\n")

    answers, judged = [], []
    for i, q in enumerate(questions, 1):
        for system in systems:
            runner = run_baseline if system == "baseline" else run_receipts
            ans = runner(q["question"], q["id"], q["type"], settings)
            graded = judge.judge_answer(ans)
            answers.append(ans)
            judged += graded
            ok = sum(1 for g in graded if g.grade == "supported")
            print(f"[{i}/{len(questions)}] {q['id']} {system:<8} {ans.status:<22} "
                  f"claims {len(graded):>2}, supported {ok:>2}, declined {ans.abstained}, "
                  f"{ans.seconds:>5.1f}s" + (f"  ERROR {ans.error[:80]}" if ans.error else ""))
        save(outdir, answers, judged, judge.model, len(questions))

    print("\n" + (outdir / "summary.md").read_text(encoding="utf-8"))
    ju = judge.tracer.usage
    print(f"Judge used {ju.prompt_tokens + ju.completion_tokens:,} tokens and {ju.tavily_credits:g} Tavily credits.")
    print(f"Check the judge by hand: {outdir / 'judged.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
