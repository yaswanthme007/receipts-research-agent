import re

from fakes import PAGE_A, PAGE_A_URL, PAGE_B, PAGE_B_URL, FakeWeb, row
from receipts.config import Settings
from receipts.evaluation import (
    Judge,
    JudgedClaim,
    SystemAnswer,
    parse_cited_sentences,
    run_baseline,
    run_receipts,
    summarize,
    summary_markdown,
)
from receipts.schema import Claim, ResearchResult, Source
from receipts.trace import Tracer

S = Settings(nebius_api_key="x", tavily_api_key="y")


class ScriptedLLM:
    def __init__(self, text="", json_fn=None):
        self.text, self.json_fn, self.tracer = text, json_fn, Tracer()

    def complete(self, messages, **kw):
        self.prompt = messages[0]["content"]
        return self.text

    def json(self, *, role, user, **kw):
        return self.json_fn(role, user)


def test_parse_cited_sentences_handles_formats_and_sources_list():
    text = (
        "<think>hidden</think>Turbo launched in March 2026 [1]. It costs $20 per month [1, 2]. "
        "Some say it includes 500 GPU hours [2-3].\n"
        "This sentence has no citation but states a fact about pricing.\n"
        "The sources do not mention student discounts.\n\n"
        "Sources:\n[1] https://a\n[2] https://b"
    )
    claims, uncited = parse_cited_sentences(text, {1: "https://a", 2: "https://b", 3: "https://c"})
    assert claims == [
        ("Turbo launched in March 2026 .", ["https://a"]),
        ("It costs $20 per month .", ["https://a", "https://b"]),
        ("Some say it includes 500 GPU hours .", ["https://b", "https://c"]),
    ]
    assert uncited == 1  # the "do not mention" sentence is not counted as an uncited fact


def test_baseline_replicates_template_prompt_and_parses_citations():
    web = FakeWeb(results={"*": [row(PAGE_A_URL, PAGE_A), row(PAGE_B_URL, PAGE_B)]})
    llm = ScriptedLLM("The Turbo plan costs $20 per month for developers [1]. Analysts report $25 [2].")
    ans = run_baseline("Turbo price?", "q1", "current", S, llm=llm, web=web)
    assert "Cite claims as [n]" in llm.prompt
    assert f"[1] {PAGE_A_URL} — {PAGE_A_URL}" in llm.prompt  # template's "[i] title — url" format
    assert [u for _t, u in ans.claims] == [[PAGE_A_URL], [PAGE_B_URL]]
    assert ans.status == "answered" and ans.error is None


def test_baseline_empty_search_matches_template_message():
    ans = run_baseline("x?", "q1", "current", S, llm=ScriptedLLM(), web=FakeWeb(results={}))
    assert ans.answer_text.startswith("Insufficient evidence") and ans.claims == []


def judge_with(grades_fn, pages):
    llm = ScriptedLLM(json_fn=grades_fn)
    return Judge(S, llm=llm, web=FakeWeb(pages=pages))


def excerpt_ids(user):
    return re.findall(r'<excerpt id="(E\d+)"', user)


def test_judge_requires_quote_on_page():
    def grades(role, user):
        ids = excerpt_ids(user)
        if "$20" in user.split("\n", 1)[0]:
            return {"grades": [{"excerpt_id": ids[0], "grade": "supported",
                                "evidence": "The Turbo plan costs $20 per month for individual developers."}]}
        return {"grades": [{"excerpt_id": ids[0], "grade": "supported", "evidence": "Turbo is free for every student."}]}

    j = judge_with(grades, {PAGE_A_URL: PAGE_A})
    j.fetch([PAGE_A_URL])
    good = j.grade("receipts", "q1", "Turbo costs $20 per month.", [PAGE_A_URL])
    bad = j.grade("baseline", "q1", "Turbo is free for students.", [PAGE_A_URL])
    assert good.grade == "supported" and "$20" in good.quote
    assert bad.grade == "unsupported"  # judge's made-up evidence is not on the page


def test_judge_marks_unreadable_pages():
    j = judge_with(lambda r, u: {"grades": []}, {})
    j.fetch(["https://gone.example"])
    assert j.grade("baseline", "q1", "Anything at all here.", ["https://gone.example"]).grade == "unreadable"


def test_judge_answer_sets_abstention():
    j = judge_with(lambda role, user: {"declined": "Insufficient" in user}, {})
    a = SystemAnswer("baseline", "u1", "Fake?", "unanswerable", answer_text="Insufficient evidence to say.")
    j.judge_answer(a)
    assert a.abstained is True
    r = SystemAnswer("receipts", "u1", "Fake?", "unanswerable", status="insufficient_evidence")
    j.judge_answer(r)
    assert r.abstained is True


def test_run_receipts_uses_only_kept_fact_claims():
    class FakeResearcher:
        def run(self, q):
            res = ResearchResult(question=q, status="answered", tldr="Answer.")
            res.sources = [Source(id="S1", url=PAGE_A_URL, title="", domain="docs.example.com", snippet="x")]
            res.claims = [Claim(id="C1", text="Fact.", source_ids=["S1"], status="supported"),
                          Claim(id="C2", text="Inference.", kind="inference", source_ids=["S1"], status="inference")]
            return res

    ans = run_receipts("q?", "q1", "current", S, researcher=FakeResearcher())
    assert ans.claims == [("Fact.", [PAGE_A_URL])]


def test_summary_numbers():
    answers = [
        SystemAnswer("baseline", "q1", "", "current", abstained=False),
        SystemAnswer("baseline", "u1", "", "unanswerable", abstained=False, uncited_sentences=2),
        SystemAnswer("receipts", "q1", "", "current", abstained=False),
        SystemAnswer("receipts", "u1", "", "unanswerable", abstained=True),
    ]
    judged = [
        JudgedClaim("baseline", "q1", "a", [], "supported"),
        JudgedClaim("baseline", "q1", "b", [], "unsupported"),
        JudgedClaim("baseline", "q1", "c", [], "unreadable"),
        JudgedClaim("receipts", "q1", "a", [], "supported"),
        JudgedClaim("receipts", "q1", "b", [], "partial"),
    ]
    s = summarize(answers, judged)
    assert s["baseline"]["support_rate"] == 0.5 and s["baseline"]["claims_unreadable"] == 1
    assert s["receipts"]["support_or_partial_rate"] == 1.0
    assert s["baseline"]["declined_unanswerable"] == "0/1" and s["receipts"]["declined_unanswerable"] == "1/1"
    md = summary_markdown(s, "judge-model", 2)
    assert "| Fully supported by the cited page | 50% | 50% |" in md
