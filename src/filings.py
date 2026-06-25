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

def normalize_tokens(text: str) -> list[str]:
    """Lowercase, keep alphabetic words only."""
    return _WORD_RE.findall(text.lower())


def cosine_similarity(text_a: str, text_b: str) -> float:
    """Term-frequency cosine similarity between two documents in [0, 1]."""
    a = Counter(normalize_tokens(text_a))
    b = Counter(normalize_tokens(text_b))
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    dot = sum(a[w] * b[w] for w in common)
    if dot == 0:
        return 0.0
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    return dot / (norm_a * norm_b)


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


def compute_filing_similarity(cik, db_path: str, form: str = "10-K",
                              ttl_hours: int = 24) -> Optional[dict]:
    """End-to-end text-stability for one company's latest `form` filing.

    Returns {text_stability, risk_sim, mda_sim, accession, prior_accession,
    report_date, form, risk_text, mda_text} or None if no comparable pair.
    `risk_text`/`mda_text` carry the current sections so the (conditional)
    Claude layer can diff them without re-fetching.
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
        return cached

    cur_html = fetch_filing_doc(cik, current["accession"], current["primary_doc"], db_path)
    prior_html = fetch_filing_doc(cik, prior["accession"], prior["primary_doc"], db_path)
    if cur_html is None or prior_html is None:
        return None

    cur_text = plain_text(cur_html)
    prior_text = plain_text(prior_html)
    cur_sec = extract_sections(cur_html)
    prior_sec = extract_sections(prior_html)

    # Document-level similarity: robust, paper-faithful, computed for every filer.
    doc_sim = cosine_similarity(cur_text, prior_text)
    risk_sim = cosine_similarity(cur_sec["risk_factors"], prior_sec["risk_factors"])
    mda_sim = cosine_similarity(cur_sec["mda"], prior_sec["mda"])

    # Use a section's similarity only when BOTH sides cleanly extracted (length
    # guard rejects TOC fragments / styling artifacts). Sections are more
    # discriminating; document similarity is the robust fallback floor.
    def _valid(a, b):
        return len(a) >= MIN_SECTION_CHARS and len(b) >= MIN_SECTION_CHARS

    sims = []
    if _valid(cur_sec["risk_factors"], prior_sec["risk_factors"]):
        sims.append(risk_sim)
    if _valid(cur_sec["mda"], prior_sec["mda"]):
        sims.append(mda_sim)
    text_stability = sum(sims) / len(sims) if sims else doc_sim

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
        "risk_text": cur_sec["risk_factors"],
        "prior_risk_text": prior_sec["risk_factors"],
        "mda_text": cur_sec["mda"],
        "prior_mda_text": prior_sec["mda"],
    }
    put_filing_similarity(db_path, cik=_cik10(cik), result=result)
    return result
