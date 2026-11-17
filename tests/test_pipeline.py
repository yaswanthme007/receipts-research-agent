import pytest

from fakes import PAGE_A, PAGE_A_URL, PAGE_B, PAGE_B_URL, FakeLLM, FakeWeb, ids_by_domain, plan, row
from receipts.config import Settings
from receipts.pipeline import Researcher
from receipts.render import to_markdown
from receipts.trace import Tracer


def make(llm, web, **overrides):
    s = Settings(nebius_api_key="x", tavily_api_key="y", workers=1, **overrides)
    return Researcher(s, llm, web, Tracer())


def draft_citing(*claims):
    """claims: (text, [domains], kind)"""
    def handler(user):
        ids = ids_by_domain(user)
        return {
            "tldr": "draft tldr",
            "claims": [
                {"text": t, "kind": k, "source_ids": [ids[d] for d in doms if d in ids], "subquestion_id": "Q1"}
                for t, doms, k in claims
            ],
            "disagreements": [],
            "gaps": [],
        }
    return handler


def verifier(table):
    """table: {claim substring: {domain: (verdict, quote)}}"""
    def handler(user):
        ids = ids_by_domain(user)
        claim_line = user.split("\n", 1)[0]
        for key, per_domain in table.items():
            if key in claim_line:
                return {"judgments": [
                    {"source_id": ids[d], "verdict": v, "quote": q, "explanation": "test"}
                    for d, (v, q) in per_domain.items() if d in ids
                ]}
        return {"judgments": []}
    return handler


SUMMARY = {"tldr": "Turbo costs $20/month [C1]."}


def test_supported_claim_gets_quote_and_highlight_link():
    web = FakeWeb(results={"turbo price": [row(PAGE_A_URL, PAGE_A[:120])]}, pages={PAGE_A_URL: PAGE_A})
    llm = FakeLLM(
        plan=plan("turbo price"),
        draft=draft_citing(("The Turbo plan costs $20 per month.", ["docs.example.com"], "fact")),
        verify=verifier({"$20": {"docs.example.com": ("supports", "The Turbo plan costs $20 per month for individual developers.")}}),
        summarize=SUMMARY,
    )
    r = make(llm, web).run("How much is Turbo?")
    assert r.status == "answered"
    c = r.claims[0]
    assert c.status == "supported" and c.confidence == "medium"
    assert c.evidence[0].quote_match == "exact"
    assert "#:~:text=" in c.evidence[0].highlight_url
    assert r.tldr == SUMMARY["tldr"]
    assert web.extracts == [[PAGE_A_URL]]  # page text fetched once for verification
    md = to_markdown(r)
    assert "1/1 factual claims verified" in md and "#:~:text=" in md


def test_fabricated_quote_is_not_accepted_and_claim_is_removed():
    web = FakeWeb(results={"*": [row(PAGE_A_URL, PAGE_A)]}, pages={PAGE_A_URL: PAGE_A})
    llm = FakeLLM(
        plan=plan("turbo price"),
        draft=draft_citing(
            ("The Turbo plan costs $20 per month.", ["docs.example.com"], "fact"),
            ("Turbo is free for students.", ["docs.example.com"], "fact"),
        ),
        verify=verifier({
            "$20": {"docs.example.com": ("supports", "The Turbo plan costs $20 per month for individual developers.")},
            "students": {"docs.example.com": ("supports", "Turbo is free forever for all students worldwide.")},
        }),
        summarize=SUMMARY,
    )
    r = make(llm, web, max_rounds=1).run("Turbo pricing?")
    assert [c.text for c in r.claims] == ["The Turbo plan costs $20 per month."]
    assert [c.text for c in r.dropped_claims] == ["Turbo is free for students."]
    assert r.dropped_claims[0].status == "unsupported"


def test_conflicting_sources_are_surfaced_and_injection_flagged():
    web = FakeWeb(
        results={"*": [row(PAGE_A_URL, PAGE_A, 0.9), row(PAGE_B_URL, PAGE_B, 0.8)]},
        pages={PAGE_A_URL: PAGE_A, PAGE_B_URL: PAGE_B},
    )
    llm = FakeLLM(
        plan=plan("turbo price"),
        draft=draft_citing(("The Turbo plan costs $20 per month.", ["docs.example.com", "news.other.org"], "fact")),
        verify=verifier({"$20": {
            "docs.example.com": ("supports", "The Turbo plan costs $20 per month for individual developers."),
            "news.other.org": ("contradicts", "Analysts say the Turbo plan costs $25 per month after the April price change."),
        }}),
        summarize=SUMMARY,
    )
    r = make(llm, web).run("Turbo pricing?")
    assert r.claims[0].status == "conflicting"
    assert len(r.conflicts) == 1 and r.conflicts[0].contradicting
    flagged = [s for s in r.sources if s.injection_flags]
    assert [s.url for s in flagged] == [PAGE_B_URL]
    assert "instructions aimed at AI agents" in to_markdown(r)


def test_empty_search_results_give_insufficient_evidence_without_guessing():
    web = FakeWeb(results={})
    llm = FakeLLM(plan=plan("obscure thing"), refine={"queries": [{"subquestion_id": "Q1", "query": "obscure thing official"}]})
    r = make(llm, web).run("What is the obscure thing?")
    assert r.status == "insufficient_evidence"
    assert r.claims == []
    assert "Insufficient evidence" in r.tldr
    assert "draft" not in llm.calls  # never wrote an answer from nothing
    assert web.searches == ["obscure thing", "obscure thing official"]


def test_tool_failure_is_reported_not_hidden():
    web = FakeWeb(fail=True)
    llm = FakeLLM(plan=plan("q1", "q2"))
    r = make(llm, web).run("Anything?")
    assert r.status == "tool_failure"
    assert len(r.errors) == 2
    assert "draft" not in llm.calls


def test_gap_triggers_second_round_with_new_query():
    web = FakeWeb(
        results={
            "turbo price": [row("https://blog.random.net/post", "A blog post that mentions Turbo but not its price at all.")],
            "turbo pricing page official": [row(PAGE_A_URL, PAGE_A)],
        },
        pages={PAGE_A_URL: PAGE_A, "https://blog.random.net/post": "A blog post that mentions Turbo but not its price at all."},
    )
    llm = FakeLLM(
        plan=plan("turbo price"),
        draft=draft_citing(
            ("The Turbo plan costs $20 per month.", ["docs.example.com", "blog.random.net"], "fact"),
        ),
        verify=verifier({"$20": {
            "docs.example.com": ("supports", "The Turbo plan costs $20 per month for individual developers."),
            "blog.random.net": ("does_not_support", ""),
        }}),
        refine={"queries": [{"subquestion_id": "Q1", "query": "turbo pricing page official"}]},
        summarize=SUMMARY,
    )
    r = make(llm, web, max_rounds=2).run("Turbo price?")
    assert r.rounds == 2
    assert web.searches == ["turbo price", "turbo pricing page official"]
    assert r.status == "answered" and r.claims[0].status == "supported"
    assert "Every sub-question" in r.stop_reason


def test_invented_source_ids_are_ignored():
    web = FakeWeb(results={"*": [row(PAGE_A_URL, PAGE_A)]}, pages={PAGE_A_URL: PAGE_A})

    def bad_draft(user):
        return {"tldr": "", "claims": [{"text": "Claim citing a ghost.", "kind": "fact", "source_ids": ["S99"], "subquestion_id": "Q1"}],
                "disagreements": [], "gaps": []}

    llm = FakeLLM(plan=plan("x"), draft=bad_draft, refine={"queries": []})
    r = make(llm, web).run("x?")
    assert r.status == "insufficient_evidence"
    assert r.dropped_claims[0].source_ids == []


def test_audit_mode_checks_each_claim():
    web = FakeWeb(
        results={"turbo price": [row(PAGE_A_URL, PAGE_A)], "turbo launch": [row(PAGE_A_URL, PAGE_A)]},
        pages={PAGE_A_URL: PAGE_A},
    )
    llm = FakeLLM(
        audit_extract={"claims": [
            {"text": "Turbo costs $20 per month.", "query": "turbo price"},
            {"text": "Turbo launched in 2019.", "query": "turbo launch"},
        ]},
        verify=verifier({
            "$20": {"docs.example.com": ("supports", "The Turbo plan costs $20 per month for individual developers.")},
            "2019": {"docs.example.com": ("contradicts", "Example Cloud launched its Turbo plan in March 2026.")},
        }),
    )
    r = make(llm, web).audit("Turbo costs $20 per month. It launched in 2019.")
    assert [c.status for c in r.claims] == ["supported", "contradicted"]
    assert len(r.sources) == 1  # same URL found twice is stored once
    assert "1 of 2 claims are backed" in r.tldr


@pytest.mark.parametrize("budget,expected", [(0, "credit budget")])
def test_budget_stops_research(budget, expected):
    web = FakeWeb(results={"*": [row("https://blog.random.net/post", "Turbo mention without price details here.")]},
                  pages={"https://blog.random.net/post": "Turbo mention without price details here."})
    llm = FakeLLM(
        plan=plan("q"),
        draft=draft_citing(("Turbo costs $20.", ["blog.random.net"], "fact")),
        verify=verifier({}),
        refine={"queries": [{"subquestion_id": "Q1", "query": "q2"}]},
    )
    researcher = make(llm, web, max_rounds=3, tavily_credit_budget=budget)
    researcher.tracer.usage.tavily_credits = 1  # already over budget
    r = researcher.run("q?")
    assert expected in r.stop_reason
    assert r.status == "insufficient_evidence"
