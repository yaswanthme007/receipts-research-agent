"""Evidence handling: URL canonicalization, dedupe, ranking, injection scanning,
passage selection and verbatim quote matching. Pure functions, no network."""

from __future__ import annotations

import re
import unicodedata
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit

from .schema import Source

# ----------------------------------------------------------------------------- URLs
_TRACKING = re.compile(r"^(utm_|fbclid$|gclid$|mc_cid$|mc_eid$|ref$|ref_src$|_ga$|_gl$)")


def canonical_url(url: str) -> str:
    parts = urlsplit(url.strip())
    host = (parts.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    query = urlencode([(k, v) for k, v in parse_qsl(parts.query) if not _TRACKING.match(k.lower())])
    path = parts.path.rstrip("/") or "/"
    return urlunsplit(("https", host, path, query, ""))


def iso_date(value: str | None) -> str | None:
    """Tavily dates arrive as RFC 2822 ('Tue, 31 Mar 2026 00:00:00 GMT') or ISO. Return YYYY-MM-DD."""
    if not value:
        return None
    from datetime import datetime
    from email.utils import parsedate_to_datetime

    for parse in (parsedate_to_datetime, lambda v: datetime.fromisoformat(v.replace("Z", "+00:00"))):
        try:
            return parse(value).date().isoformat()
        except (TypeError, ValueError, IndexError):
            continue
    return None


def domain_of(url: str) -> str:
    host = (urlsplit(url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


# ------------------------------------------------------------------------ text utils
_STOP = set(
    "a an the and or of to in on for with by from at as is are was were be been being this that these "
    "those it its which who whom what when where how why not no than then there their they them we you "
    "he she his her our your can could would should may might will shall do does did has have had into "
    "about over under also more most less such only any all each per via".split()
)
_WORD = re.compile(r"[a-z0-9][a-z0-9.%$\-]*")


def _fold(text: str) -> str:
    """Lowercase, unify quotes/dashes/whitespace so small formatting differences don't matter."""
    text = unicodedata.normalize("NFKC", text)
    text = (
        text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
        .replace("–", "-").replace("—", "-").replace(" ", " ")
    )
    return re.sub(r"\s+", " ", text).strip().lower()


def tokens(text: str) -> list[str]:
    return [w.strip(".-") for w in _WORD.findall(_fold(text)) if w.strip(".-") and w.strip(".-") not in _STOP]


def _shingles(text: str, n: int = 5) -> set[tuple[str, ...]]:
    t = tokens(text)
    return {tuple(t[i : i + n]) for i in range(max(0, len(t) - n + 1))}


def near_duplicate(a: str, b: str, threshold: float = 0.6) -> bool:
    sa, sb = _shingles(a), _shingles(b)
    if not sa or not sb:
        return False
    return len(sa & sb) / min(len(sa), len(sb)) >= threshold


# ------------------------------------------------------------------- selection
def select_sources(
    candidates: list[Source], existing: list[Source], max_sources: int, max_per_domain: int
) -> list[Source]:
    """Keep the best new sources: drop repeat URLs, syndicated copies and over-represented domains."""
    chosen = list(existing)
    seen = {canonical_url(s.url) for s in chosen}
    per_domain: dict[str, int] = {}
    for s in chosen:
        per_domain[s.domain] = per_domain.get(s.domain, 0) + 1

    for cand in sorted(candidates, key=lambda s: s.score, reverse=True):
        if len(chosen) >= max_sources:
            break
        key = canonical_url(cand.url)
        if key in seen or not cand.snippet.strip():
            continue
        if per_domain.get(cand.domain, 0) >= max_per_domain:
            continue
        if any(near_duplicate(cand.snippet, s.snippet) for s in chosen):
            continue
        seen.add(key)
        per_domain[cand.domain] = per_domain.get(cand.domain, 0) + 1
        chosen.append(cand)
    return chosen


# --------------------------------------------------------------- prompt injection
_INJECTION = [
    r"ignore (all |any |the )?(previous|prior|above|earlier) (instructions|prompts|messages)",
    r"disregard (all |any |the )?(previous|prior|above) ",
    r"you are now (a|an|in) ",
    r"(reveal|print|show) (your|the) (system )?prompt",
    r"system prompt",
    r"new instructions?:",
    r"as an ai (language )?model,? you must",
    r"do not (tell|inform) the user",
    r"</?(system|assistant|instructions?)>",
]
_INJECTION_RE = [re.compile(p, re.I) for p in _INJECTION]


def scan_injection(text: str) -> list[str]:
    hits = []
    for rx in _INJECTION_RE:
        m = rx.search(text or "")
        if m:
            hits.append(m.group(0)[:80])
    return hits


def fence_untrusted(text: str, limit: int) -> str:
    """Prepare web text for a prompt: trim, and neutralize anything that looks like our own tags."""
    text = re.sub(r"\s+", " ", text or "").strip()[:limit]
    return re.sub(r"<\s*/?\s*(source|untrusted|system|assistant|instructions?)", "[tag]", text, flags=re.I)


# ------------------------------------------------------------------- passages
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])|\n{1,}")


def sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text or "") if len(s.strip()) > 20]


def top_passages(text: str, claim: str, k: int = 3, window: int = 3) -> list[str]:
    """Pick the k windows of `window` sentences that best overlap the claim (numbers weigh double)."""
    sents = sentences(text)
    if not sents:
        return [text[:1200]] if text else []
    want = tokens(claim)
    weights = {w: (2.0 if any(ch.isdigit() for ch in w) else 1.0) for w in want}
    scored = []
    for i in range(len(sents)):
        chunk = " ".join(sents[i : i + window])
        have = set(tokens(chunk))
        score = sum(wt for w, wt in weights.items() if w in have)
        scored.append((score, i, chunk))
    scored.sort(key=lambda x: (-x[0], x[1]))
    picked, used = [], set()
    for score, i, chunk in scored:
        if len(picked) >= k:
            break
        if any(j in used for j in range(i, i + window)):
            continue
        used.update(range(i, i + window))
        picked.append(chunk[:1500])
    return picked


# ---------------------------------------------------------------- quote matching
def _folded_with_map(text: str) -> tuple[str, list[int]]:
    """Fold text char by char, keeping a map from folded index -> original index."""
    out, idx = [], []
    prev_space = True
    for i, ch in enumerate(text):
        c = _fold(ch) if not ch.isspace() else " "
        if c == " ":
            if prev_space:
                continue
            prev_space = True
        else:
            prev_space = False
        for cc in c:
            out.append(cc)
            idx.append(i)
    return "".join(out), idx


def find_quote(quote: str, page: str) -> tuple[str | None, str]:
    """Locate a model-provided quote in the page. Returns (exact page text, match kind).

    'exact'  – the quote appears verbatim (ignoring case, whitespace, curly quotes).
    'near'   – the quote was slightly paraphrased; we return the real page sentence that
               contains >= 85% of its words, so what we show is always genuine page text.
    'none'   – not found: the claim cannot count as verified.
    """
    if not quote or not page:
        return None, "none"
    q = _fold(quote).strip(" .\"'")
    if len(q) < 12:
        return None, "none"
    folded, mapping = _folded_with_map(page)
    pos = folded.find(q)
    if pos >= 0:
        start, end = mapping[pos], mapping[pos + len(q) - 1] + 1
        return page[start:end], "exact"

    want = set(tokens(quote))
    if len(want) < 4:
        return None, "none"
    best, best_score = None, 0.0
    sents = sentences(page)
    for i in range(len(sents)):
        for w in (1, 2):
            chunk = " ".join(sents[i : i + w])
            if len(chunk) > 3 * len(quote) + 200:
                continue
            score = len(want & set(tokens(chunk))) / len(want)
            # Higher overlap wins; on a tie the shorter (tighter) passage wins.
            if score > best_score or (score == best_score and best and len(chunk) < len(best)):
                best, best_score = chunk, score
    if best and best_score >= 0.85:
        return best, "near"
    return None, "none"


# ----------------------------------------------------------------- text fragments
def _enc(s: str) -> str:
    return quote(s, safe="").replace("-", "%2D")


def text_fragment_url(url: str, quote_text: str) -> str:
    """URL that opens the page scrolled to, and highlighting, the quote (Text Fragments spec)."""
    base = url.split("#", 1)[0]
    words = re.sub(r"\s+", " ", quote_text).strip().split(" ")
    if len(words) <= 10:
        frag = _enc(" ".join(words))
    else:
        frag = f"{_enc(' '.join(words[:5]))},{_enc(' '.join(words[-5:]))}"
    return f"{base}#:~:text={frag}"
