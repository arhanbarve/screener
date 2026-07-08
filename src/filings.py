"""SEC EDGAR primary-document fetch + cache. Shared by the idea-bank backtest
modules (asr_backtest, divinit_backtest, event_backtest, sc13d_backtest) for
pulling filing submissions indices and raw filing HTML.
"""
import time
import logging
from typing import Optional

import requests

from src.config import get_env
from src.cache import (
    get_submissions, put_submissions,
    get_filing_doc, put_filing_doc,
)

logger = logging.getLogger(__name__)

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_nodash}/{doc}"

# SEC fair-access: stay well under 10 req/s.
_REQUEST_SLEEP = 0.12


def _sec_headers() -> dict:
    return {"User-Agent": get_env("SEC_USER_AGENT")}


def plain_text(html: str) -> str:
    """Strip HTML to whitespace-collapsed plain text."""
    import re
    from bs4 import BeautifulSoup
    text = BeautifulSoup(html, "html.parser").get_text(" ")
    return re.sub(r"\s+", " ", text)


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
