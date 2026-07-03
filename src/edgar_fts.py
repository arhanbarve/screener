"""Shared EDGAR full-text search plumbing (docs/strategy-specs.md "Shared
data plumbing"). Used by idea #1 (Schedule 13D) and every later idea in the
bank that needs FTS.

Verified live against efts.sec.gov 2026-07-03 — two corrections to the spec's
assumed API shape, both confirmed by hitting the endpoint directly:

1. The `forms` filter must be the ROOT form label as returned in each hit's
   `_source.root_forms` / `_source.form` (e.g. "SCHEDULE 13D"), not the short
   SEC form-type code ("SC 13D") the spec guessed. `forms=SC 13D` returns 0
   hits; `forms=SCHEDULE 13D` returns the real set (910 hits for a test
   month window). This filter matches BOTH the initial filing and /A
   amendments (root form is shared) — callers must filter on the exact
   `form` field afterward to isolate initials.
2. No per-accession header-text fetch is needed to find the subject
   company's CIK. Each hit's `ciks` / `display_names` arrays already list
   the subject company first (index 0) and filer(s) after (index 1+); the
   subject's ticker (when public) is inline in `display_names[0]`, e.g.
   "PHIBRO ANIMAL HEALTH CORP  (PAHC)  (CIK 0001069899)". This is a large
   simplification over the spec's assumed workflow — no extra HTTP round
   trip per hit.
"""

import json
import logging
import random
import re
import sqlite3
import time

import pandas as pd
import requests

from src.config import get_env

logger = logging.getLogger(__name__)

FTS_URL = "https://efts.sec.gov/LATEST/search-index"
REQUEST_SLEEP_S = 0.2  # well under SEC's 10 req/s cap
MAX_FROM = 10_000       # SEC hard cap on `from` pagination per query

_TICKER_RE = re.compile(r"\(([^)]*)\)\s*\(CIK")


def _headers() -> dict:
    return {"User-Agent": get_env("SEC_USER_AGENT")}


def parse_ticker_hint(display_name: str) -> str | None:
    """Extract the first ticker from a subject's FTS display_name, e.g.
    "SIM Acquisition Corp. I  (SIMA, SIMAU, SIMAW)  (CIK 0002014982)" -> "SIMA".
    Returns None for private entities (no parenthetical before "(CIK...")."""
    m = _TICKER_RE.search(display_name or "")
    if not m:
        return None
    first = m.group(1).split(",")[0].strip()
    return first or None


def month_windows(start: str, end: str) -> list[tuple[str, str]]:
    """Split [start, end] into calendar-month windows (FTS caps at 10k hits
    per query; month windows keep any single phrase well under that)."""
    windows = []
    cur = pd.Timestamp(start).replace(day=1)
    end_ts = pd.Timestamp(end)
    while cur <= end_ts:
        win_end = cur + pd.offsets.MonthEnd(0)
        win_start = max(cur, pd.Timestamp(start))
        win_end = min(win_end, end_ts)
        windows.append((win_start.date().isoformat(), win_end.date().isoformat()))
        cur = cur + pd.offsets.MonthBegin(1)
    return windows


def _init_cache_table(db_path: str):
    con = sqlite3.connect(db_path)
    con.execute("""
        CREATE TABLE IF NOT EXISTS edgar_fts_cache (
            query_key TEXT,
            from_offset INTEGER,
            fetched_at TEXT,
            payload TEXT,
            PRIMARY KEY (query_key, from_offset)
        )""")
    con.commit()
    con.close()


def _fetch_page(phrase: str, forms: str, start: str, end: str,
                from_offset: int, db_path: str) -> dict | None:
    """One FTS page, cached forever by (query, window, offset) — historical
    EDGAR filings are immutable, same pattern as filings.fetch_filing_doc."""
    query_key = f"{phrase}|{forms}|{start}|{end}"
    con = sqlite3.connect(db_path)
    row = con.execute(
        "SELECT payload FROM edgar_fts_cache WHERE query_key = ? AND from_offset = ?",
        (query_key, from_offset)).fetchone()
    if row is not None:
        con.close()
        return json.loads(row[0])

    params = {"q": f'"{phrase}"', "forms": forms, "startdt": start, "enddt": end}
    if from_offset:
        params["from"] = from_offset

    payload = None
    max_attempts = 5
    for attempt in range(max_attempts):
        try:
            resp = requests.get(FTS_URL, headers=_headers(), params=params, timeout=30)
            resp.raise_for_status()
            payload = resp.json()
            break
        except Exception as e:
            if attempt == max_attempts - 1:
                logger.warning(f"[edgar_fts] fetch failed for {query_key} from={from_offset} "
                               f"after {max_attempts} attempts: {e}")
                con.close()
                return None
            backoff = REQUEST_SLEEP_S * (2 ** attempt)
            logger.info(f"[edgar_fts] retry {attempt + 1}/{max_attempts} for {query_key} "
                       f"from={from_offset} after {e} (sleep {backoff:.1f}s)")
            time.sleep(backoff)

    con.execute(
        "INSERT OR REPLACE INTO edgar_fts_cache VALUES (?,?,?,?)",
        (query_key, from_offset, pd.Timestamp.now().isoformat(), json.dumps(payload)))
    con.commit()
    con.close()
    time.sleep(REQUEST_SLEEP_S)
    return payload


def search(phrase: str, forms: str, start: str, end: str,
          db_path: str = "data/cache.db", max_raw_hits: int | None = None,
          sample_months: int | None = None) -> pd.DataFrame:
    """FTS hits -> DataFrame[cik, ticker_hint, file_date, adsh, form].
    `cik` is the SUBJECT company (zero-padded 10-digit), not the filer.
    `form` is the hit's exact form string (e.g. "SCHEDULE 13D" vs
    "SCHEDULE 13D/A") — callers filter this for initial-vs-amendment.

    `max_raw_hits` bounds total hits collected (stops once reached, still
    iterating months in chronological order so there's no outcome-based
    selection bias). Some phrases (e.g. "Schedule 13G" — filed on every
    passive >=5% stake, refiled on every holdings change) return tens of
    thousands of hits across a 3-year window; without a cap a control-only
    category can burn many minutes of paginated requests for no gate benefit
    (registry idea #5 precedent: cap control N at ~3x signal N).

    `sample_months` randomly selects that many months (seed=0, deterministic)
    from the range instead of all of them, spreading a max_raw_hits cap
    across the full window rather than concentrating it in the first 1-2
    months of high-volume phrases."""
    _init_cache_table(db_path)
    windows = month_windows(start, end)
    if sample_months is not None and sample_months < len(windows):
        windows = sorted(random.Random(0).sample(windows, sample_months))
    # Spread max_raw_hits evenly across windows so one high-volume month
    # can't alone exhaust the whole cap (defeats the point of sample_months).
    per_window_cap = (max_raw_hits // len(windows)) if (max_raw_hits and windows) else None
    rows = []
    for win_start, win_end in windows:
        window_rows_start = len(rows)
        from_offset = 0
        while True:
            payload = _fetch_page(phrase, forms, win_start, win_end, from_offset, db_path)
            if payload is None:
                break
            hits = payload.get("hits", {}).get("hits", [])
            total = payload.get("hits", {}).get("total", {}).get("value", 0)
            for h in hits:
                s = h.get("_source", {})
                ciks = s.get("ciks") or []
                names = s.get("display_names") or []
                if not ciks:
                    continue
                # hit["_id"] is "{accession}:{primary_doc_filename}" — the
                # only place the primary document's filename is exposed;
                # needed to fetch actual filing text (no snippet field
                # exists on the hit itself, see module docstring point 2).
                _id = h.get("_id", "")
                primary_doc = _id.split(":", 1)[1] if ":" in _id else None
                rows.append({
                    "cik": str(ciks[0]).zfill(10),
                    "ticker_hint": parse_ticker_hint(names[0]) if names else None,
                    "file_date": s.get("file_date"),
                    "adsh": s.get("adsh"),
                    "form": s.get("form"),
                    "primary_doc": primary_doc,
                })
            from_offset += len(hits)
            if len(hits) == 0 or from_offset >= min(total, MAX_FROM):
                break
            if per_window_cap is not None and len(rows) - window_rows_start >= per_window_cap:
                break
            if max_raw_hits is not None and len(rows) >= max_raw_hits:
                break
        if max_raw_hits is not None and len(rows) >= max_raw_hits:
            logger.info(f"[edgar_fts] {phrase!r}/{forms}: stopped at {len(rows)} hits "
                       f"(max_raw_hits={max_raw_hits}), window ended at {win_end}")
            break
    return pd.DataFrame(rows)
