"""Filing-edge screen: SEC primary-document fetch + "Lazy Prices" text similarity.

Core anomaly (Cohen, Malloy & Nguyen 2020): firms that materially CHANGE the
language of their 10-K/10-Q (esp. Risk Factors / MD&A) vs the prior comparable
filing subsequently underperform; stable-language "non-changers" outperform.
The market underreacts because the filings are long and tedious.

This module is the deterministic core: fetch filings, extract sections, and
compute a text-stability score (cosine similarity vs the prior comparable).
No LLM is used here — that lives in filing_analysis.py and only runs on the
small subset whose language actually changed.
"""
import re
import math
import time
import logging
from collections import Counter
from typing import Optional

import requests
from bs4 import BeautifulSoup

from src.config import get_env
from src.cache import (
    get_submissions, put_submissions,
    get_filing_doc, put_filing_doc,
    get_filing_similarity, put_filing_similarity,
)

logger = logging.getLogger(__name__)

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_nodash}/{doc}"

# SEC fair-access: stay well under 10 req/s.
_REQUEST_SLEEP = 0.12

# Word-token pattern: alphabetic words only (drop numbers/symbols so that
# similarity reflects *language* change, not boilerplate numeric churn).
_WORD_RE = re.compile(r"[a-z]+")

# Common English stopwords. Dropping these before scoring widens the dynamic
# range of the cosine / Jaccard similarity: filings are dominated by function
# words, so leaving them in compresses all scores toward 1.0 and hides the
# language change that the "Lazy Prices" signal is trying to detect.
_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "is", "are",
    "was", "were", "be", "been", "being", "as", "at", "by", "with", "that",
    "this", "these", "those", "it", "its", "on", "from", "we", "our", "us",
    "their", "them", "they", "which", "such", "any", "all", "may", "will",
    "shall", "would", "could", "should", "also", "than", "then", "other",
    "including", "etc", "but", "not", "no", "if", "into", "out", "up", "down",
    "over", "under", "about", "after", "before", "between", "during", "each",
    "more", "most", "some", "so", "only", "own", "same", "very", "can", "do",
    "does", "did", "has", "have", "had", "he", "she", "his", "her", "you",
    "your", "i", "me", "my", "who", "whom", "whose", "what", "when", "where",
    "why", "how",
})

# Section boundary markers in 10-K filings. Anchored to section TITLES (not bare
# item numbers) so that in-prose cross-references like "...see Part II, Item 8 of
# this Form 10-K..." don't prematurely terminate a section. The real body span is
# then recovered as the longest candidate slice (see _slice_section).
_RISK_START = re.compile(r"item\s*1a[.\s]+risk\s+factors", re.IGNORECASE)
_RISK_END = re.compile(r"item\s*1b[.\s]+unresolved|item\s*2[.\s]+propert", re.IGNORECASE)
_MDA_START = re.compile(r"item\s*7[.\s]+management", re.IGNORECASE)
_MDA_END = re.compile(r"item\s*7a[.\s]+quantitative|item\s*8[.\s]+financial", re.IGNORECASE)


def _sec_headers() -> dict:
    return {"User-Agent": get_env("SEC_USER_AGENT")}


# --------------------------------------------------------------------------
# Pure functions (no network) — unit tested
# --------------------------------------------------------------------------

def normalize_tokens(text: str, use_stopwords: bool = True) -> list[str]:
    """Lowercase, keep alphabetic words only.

    When `use_stopwords` is True (default) common English stopwords are dropped
    so that similarity reflects substantive language change, not function-word
    boilerplate.
    """
    toks = _WORD_RE.findall(text.lower())
    if use_stopwords:
        toks = [t for t in toks if t not in _STOPWORDS]
    return toks


def cosine_similarity(text_a: str, text_b: str, use_stopwords: bool = True) -> float:
    """Term-frequency cosine similarity between two documents in [0, 1]."""
    a = Counter(normalize_tokens(text_a, use_stopwords=use_stopwords))
    b = Counter(normalize_tokens(text_b, use_stopwords=use_stopwords))
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    dot = sum(a[w] * b[w] for w in common)
    if dot == 0:
        return 0.0
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    return dot / (norm_a * norm_b)


def bigram_jaccard(text_a: str, text_b: str, use_stopwords: bool = True) -> float:
    """Jaccard similarity of adjacent word-bigram sets in [0, 1].

    Bigrams capture local phrasing / word ORDER that a bag-of-words cosine
    misses, so a blend of the two is more sensitive to real edits (e.g.
    reordered or reworded risk language). Returns 0.0 if either side has no
    bigrams.
    """
    ta = normalize_tokens(text_a, use_stopwords=use_stopwords)
    tb = normalize_tokens(text_b, use_stopwords=use_stopwords)
    bg_a = {(ta[i], ta[i + 1]) for i in range(len(ta) - 1)}
    bg_b = {(tb[i], tb[i + 1]) for i in range(len(tb) - 1)}
    if not bg_a or not bg_b:
        return 0.0
    inter = len(bg_a & bg_b)
    union = len(bg_a | bg_b)
    if union == 0:
        return 0.0
    return inter / union


def section_similarity(a: str, b: str, jaccard_blend: float = 0.5,
                       use_stopwords: bool = True) -> float:
    """Blended section similarity: blend*bigram_jaccard + (1-blend)*cosine.

    Used for the discriminating Risk-Factors / MD&A sections; document-level
    similarity stays pure cosine as a robust floor.
    """
    jac = bigram_jaccard(a, b, use_stopwords=use_stopwords)
    cos = cosine_similarity(a, b, use_stopwords=use_stopwords)
    return jaccard_blend * jac + (1.0 - jaccard_blend) * cos


def _slice_section(text: str, start_re: re.Pattern, end_re: re.Pattern) -> str:
    """Return the LONGEST text span between a start marker and the next end marker.

    Filings repeat item headers in the table of contents and in cross-references;
    those produce tiny spans (the markers sit adjacent). The real section body is
    by far the longest candidate, so we take the max-length slice.
    """
    spans = []
    for m in start_re.finditer(text):
        start = m.end()
        end_match = end_re.search(text, start)
        end = end_match.start() if end_match else len(text)
        spans.append(text[start:end].strip())
    if not spans:
        return ""
    return max(spans, key=len)


# A section must reach this length to be trusted; shorter "extractions" are
# table-of-contents fragments or styling artifacts (e.g. "RIS K FACTORS"), not
# the real body. Below this we fall back to document-level similarity.
MIN_SECTION_CHARS = 1000


def plain_text(html: str) -> str:
    """Strip HTML to whitespace-collapsed plain text."""
    text = BeautifulSoup(html, "html.parser").get_text(" ")
    return re.sub(r"\s+", " ", text)


def extract_sections(html: str) -> dict:
    """Extract Risk Factors (Item 1A) and MD&A (Item 7) plain text.

    Returns {"risk_factors": str, "mda": str}; missing sections are "".
    Section parsing of raw SEC HTML is inherently fragile across filers, so this
    is treated as a refinement on top of document-level similarity, not the
    primary signal (see compute_filing_similarity).
    """
    text = plain_text(html)
    return {
        "risk_factors": _slice_section(text, _RISK_START, _RISK_END),
        "mda": _slice_section(text, _MDA_START, _MDA_END),
    }


def parse_submissions(data: dict) -> list[dict]:
    """Flatten the EDGAR submissions 'recent' arrays into per-filing records,
    newest filing first."""
    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accs = recent.get("accessionNumber", [])
    fdates = recent.get("filingDate", [])
    rdates = recent.get("reportDate", [])
    docs = recent.get("primaryDocument", [])
    recs = []
    for i in range(len(forms)):
        recs.append({
            "form": forms[i],
            "accession": accs[i] if i < len(accs) else "",
            "filing_date": fdates[i] if i < len(fdates) else "",
            "report_date": rdates[i] if i < len(rdates) else "",
            "primary_doc": docs[i] if i < len(docs) else "",
        })
    recs.sort(key=lambda r: r["filing_date"], reverse=True)
    return recs


def _report_month(rec: dict) -> str:
    rd = rec.get("report_date", "")
    return rd[5:7] if len(rd) >= 7 else ""


def select_comparable(recs: list[dict], form: str) -> tuple[Optional[dict], Optional[dict]]:
    """Pick the most recent filing of `form` (current) and its prior comparable.

    10-K → previous 10-K. 10-Q → prior 10-Q in the SAME fiscal quarter
    (matched on report-date month), falling back to the 4th-previous 10-Q.
    """
    matches = [r for r in recs if r["form"] == form]
    if not matches:
        return None, None
    current = matches[0]
    rest = matches[1:]
    if not rest:
        return current, None
    if form == "10-K":
        return current, rest[0]
    # 10-Q: match same fiscal quarter by report month
    cur_month = _report_month(current)
    same_q = [r for r in rest if _report_month(r) == cur_month and cur_month]
    if same_q:
        return current, same_q[0]
    # fallback: 4 quarters back if available, else the immediately prior 10-Q
    return current, (rest[3] if len(rest) >= 4 else rest[0])


# --------------------------------------------------------------------------
# Network + cache layer
# --------------------------------------------------------------------------

def _cik10(cik) -> str:
    return str(cik).zfill(10)


def fetch_submissions(cik, db_path: str, ttl_hours: int = 24) -> Optional[dict]:
    """Fetch (and cache) the EDGAR submissions index for a CIK. Re-polled daily."""
    cik10 = _cik10(cik)
    cached = get_submissions(db_path, cik10, ttl_hours=ttl_hours)
    if cached is not None:
        return cached
    url = SUBMISSIONS_URL.format(cik=cik10)
    try:
        resp = requests.get(url, headers=_sec_headers(), timeout=30)
        resp.raise_for_status()
        data = resp.json()
        put_submissions(db_path, cik10, data)
        time.sleep(_REQUEST_SLEEP)
        return data
    except Exception as e:
        logger.warning(f"[filings] submissions failed for CIK {cik10}: {e}")
        return None


def fetch_filing_doc(cik, accession: str, primary_doc: str, db_path: str) -> Optional[str]:
    """Fetch (and permanently cache) a filing's primary document HTML.

    Filings are immutable once filed, so this is cached forever by accession.
    """
    cached = get_filing_doc(db_path, accession)
    if cached is not None:
        return cached
    acc_nodash = accession.replace("-", "")
    cik_int = int(_cik10(cik))
    url = ARCHIVES_URL.format(cik_int=cik_int, acc_nodash=acc_nodash, doc=primary_doc)
    try:
        resp = requests.get(url, headers=_sec_headers(), timeout=30)
        resp.raise_for_status()
        html = resp.text
        put_filing_doc(db_path, accession, cik=_cik10(cik), form="", html=html)
        time.sleep(_REQUEST_SLEEP)
        return html
    except Exception as e:
        logger.warning(f"[filings] doc fetch failed for {accession}: {e}")
        return None


def _risk_metrics(risk_text: str, prior_risk_text: str,
                  use_stopwords: bool = True) -> tuple[int, int, float]:
    """Word counts of the current / prior Risk-Factors sections and their growth.

    Positive `risk_growth` = added risk language vs the prior comparable, which
    the "Lazy Prices" reading treats as bearish.
    """
    risk_len = len(normalize_tokens(risk_text or "", use_stopwords=use_stopwords))
    prior_risk_len = len(normalize_tokens(prior_risk_text or "", use_stopwords=use_stopwords))
    if prior_risk_len == 0:
        risk_growth = 0.0
    else:
        risk_growth = (risk_len - prior_risk_len) / max(prior_risk_len, 1)
    return risk_len, prior_risk_len, risk_growth


def compute_filing_similarity(cik, db_path: str, form: str = "10-K",
                              ttl_hours: int = 24,
                              parse_fail_section_sim: float = 0.30,
                              parse_fail_doc_floor: float = 0.98,
                              jaccard_blend: float = 0.5,
                              use_stopwords: bool = True) -> Optional[dict]:
    """End-to-end text-stability for one company's latest `form` filing.

    Returns {text_stability, doc_sim, risk_sim, mda_sim, sections_used,
    accession, prior_accession, report_date, form, filing_date, risk_len,
    prior_risk_len, risk_growth, risk_text, prior_risk_text, mda_text,
    prior_mda_text} or None if no comparable pair. The section texts carry the
    current/prior sections so the (conditional) Claude layer can diff them
    without re-fetching.
    """
    data = fetch_submissions(cik, db_path, ttl_hours=ttl_hours)
    if data is None:
        return None
    recs = parse_submissions(data)
    current, prior = select_comparable(recs, form)
    if current is None or prior is None:
        return None

    # Memoized by the current accession (immutable filing pair).
    cached = get_filing_similarity(db_path, current["accession"])
    if cached is not None:
        # Backfill keys added after this payload was first cached, without
        # forcing a full re-fetch of the (immutable) filing pair.
        if "filing_date" not in cached:
            cached["filing_date"] = current.get("filing_date", "")
        if not {"risk_len", "prior_risk_len", "risk_growth"}.issubset(cached):
            rl, prl, rg = _risk_metrics(
                cached.get("risk_text", ""), cached.get("prior_risk_text", ""),
                use_stopwords=use_stopwords,
            )
            cached["risk_len"] = rl
            cached["prior_risk_len"] = prl
            cached["risk_growth"] = rg
        return cached

    cur_html = fetch_filing_doc(cik, current["accession"], current["primary_doc"], db_path)
    prior_html = fetch_filing_doc(cik, prior["accession"], prior["primary_doc"], db_path)
    if cur_html is None or prior_html is None:
        return None

    cur_text = plain_text(cur_html)
    prior_text = plain_text(prior_html)
    cur_sec = extract_sections(cur_html)
    prior_sec = extract_sections(prior_html)

    # Document-level similarity: robust, paper-faithful, pure cosine floor.
    doc_sim = cosine_similarity(cur_text, prior_text, use_stopwords=use_stopwords)
    # Sections: blended bigram-Jaccard + cosine (more discriminating).
    risk_sim = section_similarity(cur_sec["risk_factors"], prior_sec["risk_factors"],
                                  jaccard_blend=jaccard_blend, use_stopwords=use_stopwords)
    mda_sim = section_similarity(cur_sec["mda"], prior_sec["mda"],
                                 jaccard_blend=jaccard_blend, use_stopwords=use_stopwords)

    # Use a section's similarity only when BOTH sides cleanly extracted (length
    # guard rejects TOC fragments / styling artifacts). Sections are more
    # discriminating; document similarity is the robust fallback floor.
    def _valid(a, b):
        return len(a) >= MIN_SECTION_CHARS and len(b) >= MIN_SECTION_CHARS

    # Parse-failure guard: an implausibly low section sim alongside a near-
    # identical document sim signals an extraction failure, not a real edit.
    # Such a section is ignored so text_stability falls back toward doc_sim.
    def _failed(section_sim):
        return section_sim < parse_fail_section_sim and doc_sim > parse_fail_doc_floor

    sims = []
    if _valid(cur_sec["risk_factors"], prior_sec["risk_factors"]) and not _failed(risk_sim):
        sims.append(risk_sim)
    if _valid(cur_sec["mda"], prior_sec["mda"]) and not _failed(mda_sim):
        sims.append(mda_sim)
    text_stability = sum(sims) / len(sims) if sims else doc_sim

    risk_len, prior_risk_len, risk_growth = _risk_metrics(
        cur_sec["risk_factors"], prior_sec["risk_factors"], use_stopwords=use_stopwords,
    )

    result = {
        "text_stability": text_stability,
        "doc_sim": doc_sim,
        "risk_sim": risk_sim,
        "mda_sim": mda_sim,
        "sections_used": len(sims),
        "accession": current["accession"],
        "prior_accession": prior["accession"],
        "report_date": current.get("report_date", ""),
        "form": form,
        "filing_date": current.get("filing_date", ""),
        "risk_len": risk_len,
        "prior_risk_len": prior_risk_len,
        "risk_growth": risk_growth,
        "risk_text": cur_sec["risk_factors"],
        "prior_risk_text": prior_sec["risk_factors"],
        "mda_text": cur_sec["mda"],
        "prior_mda_text": prior_sec["mda"],
    }
    put_filing_similarity(db_path, cik=_cik10(cik), result=result)
    return result
