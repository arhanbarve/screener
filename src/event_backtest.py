"""Event-driven filing backtest: two phases, near-zero cost.

Confirms or kills each thesis-A/B event category (see
handoff-thesis-pivot-2026-07-02.md) before any live watcher, UI, or
meaningful Claude spend. Full design:
docs/superpowers/specs/2026-07-02-event-backtest-design.md

Phase 1 — pure event study, $0, no Claude calls. Pulls 8-K Item 1.01/3.01,
NT 10-K/10-Q → real-filing pairs (delinquent filer regains status), SC TO-I
(odd-lot tenders), and going-concern-removed 10-K pairs from EDGAR's free
submissions API over the liquidity-gated universe, computes 5/20/60-day
abnormal returns vs. IWM (a size-matched proxy), and gates each category on
a one-sample t-test.

Phase 2 — materiality validation on a hard-capped sample (<=300 events),
only for categories that passed Phase 1. Default path is a CSV export for
hand-labeling (zero marginal cost). --claude-batch opts into a Haiku
Message-Batches-API submission, capped and pre-flight cost-gated at $5.

Usage:
    python -m src.event_backtest phase1 [--days 730] [--limit N] [--dry-run]
    python -m src.event_backtest phase2 --events-csv PATH --summary-csv PATH [--claude-batch] [--dry-run]
    python -m src.event_backtest phase2-fetch --batch-id ID --sample-csv PATH
"""
import argparse
import json
import logging
import os
import re
import time
from datetime import date, timedelta

import pandas as pd
import requests
from scipy.stats import ttest_1samp

from src.config import load_config
from src.cache import (
    init_db, get_submissions, put_submissions,
    get_backtest_prices, put_backtest_prices,
)
from src.universe import build_universe
from src.prices import fetch_all_prices, _fetch_batch_yfinance
from src.lazy_run import _apply_neglect_gate
from src.filings import fetch_submissions, fetch_filing_doc, plain_text, _sec_headers, ARCHIVES_URL, _cik10
from src.news import _parse_json

logger = logging.getLogger(__name__)

DB_PATH = "data/cache.db"
UNIVERSE_PATH = "data/universe.parquet"

BENCHMARK = "IWM"
HORIZONS = (5, 20, 60)

# 1.01 and 3.01 are both handled separately (classify_1_01_events,
# classify_3_01_events) since neither item code alone captures the
# distinction that matters — see those functions' docstrings.
EVENT_8K_ITEMS = {}
NT_FORMS = {"NT 10-K", "NT 10-Q"}
TENDER_FORM = "SC TO-I"

# Plain-string going-concern detector. Deliberately simple per spec — no
# diffing library or model, just presence vs. absence of the standard
# audit-opinion phrasing across consecutive 10-Ks.
GOING_CONCERN_RE = re.compile(
    r"substantial doubt.{0,120}(going concern|ability to continue)", re.IGNORECASE
)
ODD_LOT_RE = re.compile(r"odd[\s-]?lot", re.IGNORECASE)

# Item 3.01 ("Notice of Delisting or Failure to Satisfy a Continued Listing
# Rule...; Transfer of Listing") is filed both for the initial deficiency
# notice (negative) AND the follow-up compliance-regained filing (positive) —
# the item code alone can't distinguish them, so pooling all 3.01s averages
# two opposite-direction events into a signal that looks like noise.
_3_01_REGAINED_RE = re.compile(
    r"regained compliance|back in(?:to)? compliance|now (?:in|meets?) compliance|"
    r"compliance has been regained", re.IGNORECASE,
)
_3_01_NOTICE_RE = re.compile(
    r"notice of (?:delisting|non-?compliance|deficiency)|"
    r"fail(?:ed|s|ure)? to (?:satisfy|maintain|meet)|deficiency letter|"
    r"below the (?:minimum|required)", re.IGNORECASE,
)

# Item 1.01 ("Entry into a Material Definitive Agreement") pools everything
# from multi-hundred-million-dollar credit facilities to routine small-dollar
# vendor contracts under one item code — Phase 1's first full run found a
# consistent negative drift signal (all 3 horizons, N=3385) that missed the
# gate's magnitude floor, plausibly diluted by routine agreements. Splitting
# by disclosed deal size tests whether restricting to real capital-moving
# deals sharpens that near-miss into a clean signal.
_DOLLAR_AMOUNT_RE = re.compile(
    r"\$\s?([\d,]+(?:\.\d+)?)\s*(billion|million|thousand)?", re.IGNORECASE,
)
_DOLLAR_MULTIPLIERS = {"thousand": 1e3, "million": 1e6, "billion": 1e9}
MATERIALITY_THRESHOLD_USD = 10_000_000


def _extract_max_dollar_amount(text: str) -> float | None:
    amounts = []
    for m in _DOLLAR_AMOUNT_RE.finditer(text):
        num = float(m.group(1).replace(",", ""))
        mult = _DOLLAR_MULTIPLIERS.get((m.group(2) or "").lower(), 1)
        amounts.append(num * mult)
    return max(amounts) if amounts else None


def _classify_1_01_materiality(text: str, threshold: float = MATERIALITY_THRESHOLD_USD) -> str:
    amount = _extract_max_dollar_amount(text)
    if amount is None:
        return "8k_1_01_unclassified"
    return "8k_1_01_material" if amount >= threshold else "8k_1_01_immaterial"

# Phase 1 gate thresholds (advisory — printed, not auto-enforced on code).
GATE_MIN_ABS_ABN_RETURN = 0.015
GATE_MAX_PVALUE = 0.10
GATE_HORIZONS = (20, 60)

# Phase 2 hard caps — do not raise without editing this constant deliberately.
MAX_PHASE2_SAMPLE = 300
PHASE2_ABORT_USD = 5.0
HAIKU_BATCH_IN_PER_MTOK = 0.50   # ~50% off the $1/MTok sync rate
HAIKU_BATCH_OUT_PER_MTOK = 2.50  # ~50% off the $5/MTok sync rate
PHASE2_MAX_INPUT_TOKENS_PER_FILING = 8000
PHASE2_MAX_OUTPUT_TOKENS_PER_FILING = 300


# ── EDGAR submissions flattening ─────────────────────────────────────────────

def _flatten_recent_dict(d: dict) -> list[dict]:
    """Flatten one EDGAR submissions-shaped dict (either the top-level
    `filings.recent` object or a paginated `filings.files` page, which has
    the identical array schema) into per-filing records."""
    forms = d.get("form", [])
    accs = d.get("accessionNumber", [])
    fdates = d.get("filingDate", [])
    docs = d.get("primaryDocument", [])
    items_arr = d.get("items", [])
    out = []
    for i in range(len(forms)):
        out.append({
            "form": forms[i],
            "accession": accs[i] if i < len(accs) else "",
            "filing_date": fdates[i] if i < len(fdates) else "",
            "primary_doc": docs[i] if i < len(docs) else "",
            "items": items_arr[i] if i < len(items_arr) else "",
        })
    return out


def _fetch_submissions_page(name: str, db_path: str) -> dict | None:
    """Fetch (and cache, since these older pages are immutable) one paginated
    submissions file. Reuses the `submissions` cache table with a non-CIK key."""
    cache_key = f"pg:{name}"
    cached = get_submissions(db_path, cache_key, ttl_hours=24 * 365)
    if cached is not None:
        return cached
    url = f"https://data.sec.gov/submissions/{name}"
    try:
        resp = requests.get(url, headers=_sec_headers(), timeout=30)
        resp.raise_for_status()
        data = resp.json()
        put_submissions(db_path, cache_key, data)
        time.sleep(0.12)
        return data
    except Exception as e:
        logger.warning(f"[event_backtest] page fetch failed for {name}: {e}")
        return None


def list_filings(cik: str, db_path: str, lookback_days: int) -> list[dict]:
    """All filings for a CIK within the lookback window, item codes included.

    `fetch_submissions`'s `recent` array usually covers 2 years for the
    low-filing-frequency small caps this backtest targets; if it doesn't,
    older paginated pages (`filings.files`) are pulled to fill the gap.
    """
    data = fetch_submissions(cik, db_path, ttl_hours=24 * 7)
    if not data:
        return []
    cutoff = (date.today() - timedelta(days=lookback_days)).isoformat()
    recs = _flatten_recent_dict(data.get("filings", {}).get("recent", {}))
    oldest = min((r["filing_date"] for r in recs if r["filing_date"]), default="9999-99-99")
    if oldest > cutoff:
        for f in data.get("filings", {}).get("files", []):
            filing_to = f.get("filingTo", "9999-99-99")
            if filing_to < cutoff:
                continue
            page = _fetch_submissions_page(f["name"], db_path)
            if page:
                recs.extend(_flatten_recent_dict(page))
    return [r for r in recs if r["filing_date"] >= cutoff]


# ── Event extraction ─────────────────────────────────────────────────────────

def extract_category_events(ticker: str, cik: str, recs: list[dict]) -> list[dict]:
    """8-K item events, odd-lot tenders, and delinquent-filer-regains events.
    Going-concern events are handled separately (need a doc fetch)."""
    events = []
    for r in recs:
        if r["form"] == "8-K":
            codes = [c.strip() for c in str(r.get("items", "")).split(",") if c.strip()]
            for code in codes:
                if code in EVENT_8K_ITEMS:
                    events.append({
                        "ticker": ticker, "cik": cik, "category": EVENT_8K_ITEMS[code],
                        "event_date": r["filing_date"], "accession": r["accession"],
                        "primary_doc": r["primary_doc"],
                    })
        elif r["form"] == TENDER_FORM:
            events.append({
                "ticker": ticker, "cik": cik, "category": "odd_lot_tender",
                "event_date": r["filing_date"], "accession": r["accession"],
                "primary_doc": r["primary_doc"],
            })

    nts = sorted([r for r in recs if r["form"] in NT_FORMS], key=lambda r: r["filing_date"])
    reals = sorted([r for r in recs if r["form"] in {"10-K", "10-Q"}], key=lambda r: r["filing_date"])
    for nt in nts:
        base = "10-K" if nt["form"] == "NT 10-K" else "10-Q"
        after = [r for r in reals if r["form"] == base and r["filing_date"] > nt["filing_date"]]
        if after:
            real = min(after, key=lambda r: r["filing_date"])
            events.append({
                "ticker": ticker, "cik": cik, "category": "delinquent_filer_regains",
                "event_date": real["filing_date"], "accession": real["accession"],
                "primary_doc": real["primary_doc"],
            })
    return events


def _classify_3_01_text(text: str) -> str:
    if _3_01_REGAINED_RE.search(text):
        return "8k_3_01_regained"
    if _3_01_NOTICE_RE.search(text):
        return "8k_3_01_notice"
    return "8k_3_01_unclassified"


def classify_3_01_events(ticker: str, cik: str, recs: list[dict], db_path: str) -> list[dict]:
    """Item 3.01 8-Ks, split by direction (regained compliance vs. deficiency
    notice vs. neither pattern matched) — see `_3_01_REGAINED_RE` docstring."""
    events = []
    for r in recs:
        if r["form"] != "8-K":
            continue
        codes = [c.strip() for c in str(r.get("items", "")).split(",") if c.strip()]
        if "3.01" not in codes:
            continue
        category = "8k_3_01_unclassified"
        if r.get("accession") and r.get("primary_doc"):
            html = fetch_filing_doc(cik, r["accession"], r["primary_doc"], db_path)
            if html:
                category = _classify_3_01_text(plain_text(html))
        events.append({
            "ticker": ticker, "cik": cik, "category": category,
            "event_date": r["filing_date"], "accession": r["accession"],
            "primary_doc": r["primary_doc"],
        })
    return events


def classify_1_01_events(ticker: str, cik: str, recs: list[dict], db_path: str) -> list[dict]:
    """Item 1.01 8-Ks, split by disclosed deal size (see MATERIALITY_THRESHOLD_USD
    docstring) — tests whether restricting to large, capital-moving agreements
    reveals drift that's diluted when 1.01 is treated as one bucket."""
    events = []
    for r in recs:
        if r["form"] != "8-K":
            continue
        codes = [c.strip() for c in str(r.get("items", "")).split(",") if c.strip()]
        if "1.01" not in codes:
            continue
        category = "8k_1_01_unclassified"
        if r.get("accession") and r.get("primary_doc"):
            html = fetch_filing_doc(cik, r["accession"], r["primary_doc"], db_path)
            if html:
                section = _extract_item_section(html, "1.01", 8000)
                category = _classify_1_01_materiality(section)
        events.append({
            "ticker": ticker, "cik": cik, "category": category,
            "event_date": r["filing_date"], "accession": r["accession"],
            "primary_doc": r["primary_doc"],
        })
    return events


def detect_going_concern_events(ticker: str, cik: str, recs: list[dict], db_path: str) -> list[dict]:
    """Consecutive 10-K pairs where going-concern language is present in the
    prior filing and absent in the current one."""
    tenks = sorted(
        [r for r in recs if r["form"] == "10-K" and r["accession"] and r["primary_doc"]],
        key=lambda r: r["filing_date"],
    )
    flagged = []
    for r in tenks:
        html = fetch_filing_doc(cik, r["accession"], r["primary_doc"], db_path)
        has_gc = bool(html and GOING_CONCERN_RE.search(plain_text(html)))
        flagged.append((r, has_gc))

    events = []
    for (prior, prior_gc), (cur, cur_gc) in zip(flagged, flagged[1:]):
        if prior_gc and not cur_gc:
            events.append({
                "ticker": ticker, "cik": cik, "category": "going_concern_removed",
                "event_date": cur["filing_date"], "accession": cur["accession"],
                "primary_doc": cur["primary_doc"],
            })
    return events


def collect_events(universe_df: pd.DataFrame, db_path: str, lookback_days: int) -> pd.DataFrame:
    rows = []
    total = len(universe_df)
    for i, (_, u) in enumerate(universe_df.iterrows()):
        ticker, cik = str(u["ticker"]), str(u["cik"])
        recs = list_filings(cik, db_path, lookback_days)
        if recs:
            rows.extend(extract_category_events(ticker, cik, recs))
            rows.extend(classify_1_01_events(ticker, cik, recs, db_path))
            rows.extend(classify_3_01_events(ticker, cik, recs, db_path))
            rows.extend(detect_going_concern_events(ticker, cik, recs, db_path))
        if (i + 1) % 200 == 0:
            logger.info(f"[event_backtest] scanned {i+1}/{total} tickers, {len(rows)} events so far")
    return pd.DataFrame(rows)


# ── Prices + forward returns ─────────────────────────────────────────────────

def _covers_range(df: pd.DataFrame | None, start: str, end: str, slack_days: int = 5) -> bool:
    """Whether a cached price frame reaches back to `start` AND forward to `end`.
    Checking only the start (as an earlier version of this did) lets a re-run
    silently reuse a cache that doesn't extend far enough forward for newly
    added events — _forward_return then quietly returns None for them."""
    if df is None or df.empty:
        return False
    return (
        df.index.min() <= pd.Timestamp(start) + pd.Timedelta(days=slack_days)
        and df.index.max() >= pd.Timestamp(end) - pd.Timedelta(days=slack_days)
    )


def get_history(ticker: str, db_path: str, start: str, end: str, ttl_days: int = 30) -> pd.DataFrame | None:
    cached = get_backtest_prices(db_path, ticker, ttl_days=ttl_days)
    if _covers_range(cached, start, end):
        return cached
    fetched = _fetch_batch_yfinance([ticker], start=start, end=end).get(ticker)
    if fetched is not None and not fetched.empty:
        put_backtest_prices(db_path, ticker, fetched)
        return fetched
    return cached


def get_history_bulk(tickers: list[str], db_path: str, start: str, end: str,
                     ttl_days: int = 30, batch_size: int = 200) -> dict[str, pd.DataFrame]:
    """Batched yfinance fetch for tickers whose cache is missing or doesn't
    cover [start, end] — matches src.prices's rate-limit-conscious batching
    (chunked + 1s inter-batch sleep) instead of one HTTP call per ticker."""
    result: dict[str, pd.DataFrame] = {}
    to_fetch = []
    for t in tickers:
        cached = get_backtest_prices(db_path, t, ttl_days=ttl_days)
        if _covers_range(cached, start, end):
            result[t] = cached
        else:
            to_fetch.append(t)

    batches = [to_fetch[i:i + batch_size] for i in range(0, len(to_fetch), batch_size)]
    for i, batch in enumerate(batches):
        fetched = _fetch_batch_yfinance(batch, start=start, end=end)
        for t, df in fetched.items():
            put_backtest_prices(db_path, t, df)
            result[t] = df
        if i < len(batches) - 1:
            time.sleep(1.0)
    return result


def _forward_return(prices: pd.DataFrame, event_date: str, horizon: int) -> float | None:
    if prices is None or prices.empty:
        return None
    ts = pd.Timestamp(event_date)
    on_or_after = prices.index[prices.index >= ts]
    if len(on_or_after) == 0:
        return None
    t0 = on_or_after[0]
    future = prices.index[prices.index > t0]
    if len(future) < horizon:
        return None
    t_h = future[horizon - 1]
    p0 = float(prices.loc[t0, "close"])
    ph = float(prices.loc[t_h, "close"])
    if p0 == 0:
        return None
    return (ph - p0) / p0


def compute_abnormal_returns(events_df: pd.DataFrame, db_path: str,
                             horizons: tuple = HORIZONS,
                             benchmark: str = BENCHMARK) -> pd.DataFrame:
    if events_df.empty:
        return events_df
    start = (pd.Timestamp(events_df["event_date"].min()) - pd.Timedelta(days=10)).date().isoformat()
    end = (pd.Timestamp(events_df["event_date"].max()) + pd.Timedelta(days=120)).date().isoformat()

    tickers = list(events_df["ticker"].unique())
    price_cache = get_history_bulk([*tickers, benchmark], db_path, start, end)
    bench = price_cache.get(benchmark)
    if bench is None:
        raise RuntimeError(f"Failed to fetch benchmark {benchmark} history — cannot compute abnormal returns")

    rows = []
    for _, ev in events_df.iterrows():
        p = price_cache.get(ev["ticker"])
        row = dict(ev)
        for h in horizons:
            r = _forward_return(p, ev["event_date"], h)
            b = _forward_return(bench, ev["event_date"], h)
            row[f"ret_{h}d"] = r
            row[f"abn_ret_{h}d"] = (r - b) if (r is not None and b is not None) else None
        rows.append(row)
    return pd.DataFrame(rows)


# ── Phase 1 summary + gate ───────────────────────────────────────────────────

# Microcap event studies are dominated by fat-tail outliers (reverse splits,
# halts, thin-float pumps) — a handful of 100x-return filings can make a raw
# mean positive and "significant" while the typical event shows nothing.
# Winsorizing per category×horizon before computing mean/t-test, and requiring
# the (winsorized) mean and median to agree in sign, keeps the gate from
# PASSing on outlier artifacts instead of real drift.
GATE_WINSOR_PCT = 0.01


def _winsorize(vals: pd.Series, pct: float) -> pd.Series:
    if pct <= 0 or len(vals) < 5:
        return vals
    lo, hi = vals.quantile(pct), vals.quantile(1 - pct)
    return vals.clip(lo, hi)


def summarize(events_df: pd.DataFrame, horizons: tuple = HORIZONS,
             min_abs_return: float = GATE_MIN_ABS_ABN_RETURN,
             max_p: float = GATE_MAX_PVALUE,
             gate_horizons: tuple = GATE_HORIZONS,
             winsorize: bool = True,
             winsor_pct: float = GATE_WINSOR_PCT) -> pd.DataFrame:
    if events_df.empty:
        return pd.DataFrame()

    span_days = (pd.Timestamp(events_df["event_date"].max()) - pd.Timestamp(events_df["event_date"].min())).days
    months = max(span_days / 30.44, 1e-9)

    rows = []
    for category, g in events_df.groupby("category"):
        n = len(g)
        stats_by_h = {}
        passed = False
        for h in horizons:
            raw_vals = g[f"abn_ret_{h}d"].dropna().astype(float)
            if len(raw_vals) < 5:
                stats_by_h[h] = {"n": len(raw_vals), "mean": None, "median": None, "p": None, "sign_agree": None}
                continue
            vals = _winsorize(raw_vals, winsor_pct) if winsorize else raw_vals
            _, p_val = ttest_1samp(vals, 0.0)
            mean_ar = float(vals.mean())
            median_ar = float(vals.median())
            sign_agree = (mean_ar > 0) == (median_ar > 0)
            stats_by_h[h] = {"n": len(raw_vals), "mean": mean_ar, "median": median_ar,
                             "p": float(p_val), "sign_agree": sign_agree}
            if (h in gate_horizons and abs(mean_ar) >= min_abs_return
                    and p_val < max_p and sign_agree):
                passed = True

        row = {"category": category, "n_events": n, "events_per_month": round(n / months, 2)}
        for h in horizons:
            row[f"h{h}_n"] = stats_by_h[h]["n"]
            row[f"h{h}_mean_abn_ret"] = stats_by_h[h]["mean"]
            row[f"h{h}_median_abn_ret"] = stats_by_h[h]["median"]
            row[f"h{h}_pvalue"] = stats_by_h[h]["p"]
            row[f"h{h}_sign_agree"] = stats_by_h[h]["sign_agree"]
        row["gate"] = "PASS" if passed else "FAIL"
        rows.append(row)

    return pd.DataFrame(rows).sort_values("n_events", ascending=False).reset_index(drop=True)


# ── Phase 2: sampling, cost estimate, Claude batch ───────────────────────────

def select_phase2_sample(events_df: pd.DataFrame, summary_df: pd.DataFrame,
                         horizon: int = 20, max_total: int = MAX_PHASE2_SAMPLE,
                         n_deciles: int = 10, include_categories: list | None = None,
                         seed: int = 0) -> pd.DataFrame:
    """Stratify across Phase-1 return deciles within PASS categories (plus any
    explicitly forced categories). Hard-capped at MAX_PHASE2_SAMPLE."""
    passing = set(summary_df[summary_df["gate"] == "PASS"]["category"])
    if include_categories:
        passing |= set(include_categories)
    if not passing:
        return pd.DataFrame()

    col = f"abn_ret_{horizon}d"
    pool = events_df[events_df["category"].isin(passing) & events_df[col].notna()].copy()
    if pool.empty:
        return pool

    try:
        pool["decile"] = pd.qcut(pool[col], min(n_deciles, pool[col].nunique()),
                                 labels=False, duplicates="drop")
    except ValueError:
        pool["decile"] = 0

    n_groups = max(pool["decile"].nunique(), 1)
    per_decile = max(1, max_total // n_groups)
    sampled = (
        pool.groupby("decile", group_keys=False)
        .apply(lambda g: g.sample(n=min(len(g), per_decile), random_state=seed))
    )
    sampled = sampled.sample(frac=1, random_state=seed).head(max_total).reset_index(drop=True)
    assert len(sampled) <= MAX_PHASE2_SAMPLE, "Phase 2 sample exceeded hard cap"
    return sampled


def export_hand_label_csv(sample_df: pd.DataFrame, path: str, horizon: int = 20, top_n: int = 50):
    """Sort by abnormal return so the largest winners lead and largest losers
    trail; flags the top/bottom `top_n` (100 rows total by default) as the
    recommended zero-cost hand-label starting point, per spec ("hand-label
    50-100 of the largest and smallest apparent-return events")."""
    col = f"abn_ret_{horizon}d"
    df = sample_df.sort_values(col, ascending=False).reset_index(drop=True) if col in sample_df else sample_df
    df["hand_label_priority"] = False
    if {"cik", "accession", "primary_doc"}.issubset(df.columns):
        df["filing_url"] = df.apply(
            lambda r: ARCHIVES_URL.format(
                cik_int=int(_cik10(r["cik"])), acc_nodash=str(r["accession"]).replace("-", ""),
                doc=r["primary_doc"],
            ),
            axis=1,
        )
    if col in df:
        df.loc[df.index[:top_n], "hand_label_priority"] = True
        df.loc[df.index[-top_n:], "hand_label_priority"] = True
    df.to_csv(path, index=False)


def estimate_phase2_cost(sample_df: pd.DataFrame,
                         tokens_in_per_filing: int = PHASE2_MAX_INPUT_TOKENS_PER_FILING,
                         tokens_out_per_filing: int = PHASE2_MAX_OUTPUT_TOKENS_PER_FILING) -> dict:
    n = len(sample_df)
    total_in = n * tokens_in_per_filing
    total_out = n * tokens_out_per_filing
    cost = (total_in / 1e6) * HAIKU_BATCH_IN_PER_MTOK + (total_out / 1e6) * HAIKU_BATCH_OUT_PER_MTOK
    return {"n_events": n, "input_tokens": total_in, "output_tokens": total_out,
            "estimated_cost_usd": round(cost, 4)}


_ITEM_HEADER_RE = re.compile(r"item\s*\d+\.\d+\b", re.IGNORECASE)


def _extract_item_section(html: str, item_code: str, max_chars: int) -> str:
    """Text from the item's header up to the next item header (or `max_chars`,
    whichever comes first) — bounds extraction to a single item's disclosure so
    multi-item 8-Ks (e.g. 1.01 filed alongside 2.03, 9.01) don't leak unrelated
    dollar figures or context into a section meant for one item."""
    text = plain_text(html)
    m = re.search(rf"item\s*{re.escape(item_code)}\b", text, re.IGNORECASE)
    if not m:
        return text[:max_chars]
    next_m = _ITEM_HEADER_RE.search(text, m.end())
    end = next_m.start() if next_m else len(text)
    return text[m.start():min(end, m.start() + max_chars)]


def _extract_around(text: str, pattern: re.Pattern, max_chars: int, before_frac: float = 1 / 3) -> str | None:
    """Section around the first regex match, or None if no header/anchor found."""
    m = pattern.search(text)
    if not m:
        return None
    start = max(0, m.start() - int(max_chars * before_frac))
    return text[start:start + max_chars]


_EXPLANATORY_NOTE_RE = re.compile(r"explanatory note", re.IGNORECASE)
_OFFER_TO_PURCHASE_RE = re.compile(r"offer to purchase", re.IGNORECASE)


def _section_for_category(html: str, category: str, max_chars: int) -> str:
    """Extract the header-anchored section relevant to each event category, so a
    multi-hundred-page filing never gets sent whole. Falls back to a head
    truncation only when no relevant header/anchor is found (rare: most 8-Ks,
    10-Ks, and SC TO-Is include the anchor text near the disclosed event)."""
    text = plain_text(html)
    item_map = {
        "8k_1_01_material": "1.01", "8k_1_01_immaterial": "1.01", "8k_1_01_unclassified": "1.01",
        "8k_3_01_regained": "3.01", "8k_3_01_notice": "3.01", "8k_3_01_unclassified": "3.01",
    }
    if category in item_map:
        return _extract_item_section(html, item_map[category], max_chars)
    if category == "going_concern_removed":
        return (
            _extract_around(text, GOING_CONCERN_RE, max_chars)
            or _extract_around(text, _EXPLANATORY_NOTE_RE, max_chars)
            or text[:max_chars]
        )
    if category == "odd_lot_tender":
        return (
            _extract_around(text, ODD_LOT_RE, max_chars)
            or _extract_around(text, _OFFER_TO_PURCHASE_RE, max_chars)
            or text[:max_chars]
        )
    if category == "delinquent_filer_regains":
        return _extract_around(text, _EXPLANATORY_NOTE_RE, max_chars) or text[:max_chars]
    return text[:max_chars]


def submit_phase2_batch(sample_df: pd.DataFrame, db_path: str,
                        dry_run: bool = False, abort_usd: float = PHASE2_ABORT_USD) -> dict | None:
    """Pre-flight cost check, then submit a Haiku Message-Batches-API job.

    Batch API is asynchronous and gives no mid-run usage signal, so the $5
    ceiling is enforced BEFORE submission (projected cost), not as a
    per-filing running counter — that pattern only works for synchronous
    per-call loops, which this deliberately avoids per the spec.
    """
    assert len(sample_df) <= MAX_PHASE2_SAMPLE, (
        f"Phase 2 sample ({len(sample_df)}) exceeds hard cap {MAX_PHASE2_SAMPLE}"
    )
    est = estimate_phase2_cost(sample_df)
    print(f"[phase2] {est['n_events']} events, ~{est['input_tokens']:,} in / "
          f"{est['output_tokens']:,} out tokens, est. ${est['estimated_cost_usd']:.2f}")
    if est["estimated_cost_usd"] > abort_usd:
        raise RuntimeError(
            f"[phase2] projected cost ${est['estimated_cost_usd']:.2f} exceeds "
            f"${abort_usd} abort threshold — refusing to submit"
        )
    if dry_run:
        print("[phase2] --dry-run: not submitting")
        return None

    import anthropic
    client = anthropic.Anthropic()

    batch_requests = []
    for _, ev in sample_df.iterrows():
        if not ev.get("accession") or not ev.get("primary_doc"):
            continue
        html = fetch_filing_doc(ev["cik"], ev["accession"], ev["primary_doc"], db_path)
        if not html:
            continue
        section = _section_for_category(html, ev["category"], PHASE2_MAX_INPUT_TOKENS_PER_FILING * 4)
        prompt = (
            f"Filing excerpt for {ev['ticker']} ({ev['category']}):\n{section}\n\n"
            "Extract the disclosed or estimable dollar value of this event and any "
            "named counterparty. Return ONLY JSON: "
            '{"dollar_value": <number or null>, "counterparty": "<string or null>", '
            '"materiality_note": "<1 sentence>"}'
        )
        batch_requests.append({
            "custom_id": f"{ev['ticker']}-{ev['accession']}",
            "params": {
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": PHASE2_MAX_OUTPUT_TOKENS_PER_FILING,
                "messages": [{"role": "user", "content": prompt}],
            },
        })

    if not batch_requests:
        print("[phase2] no requests to submit (no fetchable filing docs)")
        return None

    batch = client.messages.batches.create(requests=batch_requests)
    print(f"[phase2] submitted batch {batch.id} ({len(batch_requests)} requests)")
    return {"batch_id": batch.id, "custom_ids": [r["custom_id"] for r in batch_requests]}


def fetch_phase2_results(batch_id: str) -> pd.DataFrame:
    import anthropic
    client = anthropic.Anthropic()
    batch = client.messages.batches.retrieve(batch_id)
    if batch.processing_status != "ended":
        print(f"[phase2] batch {batch_id} still '{batch.processing_status}' — try again later")
        return pd.DataFrame()

    rows = []
    for result in client.messages.batches.results(batch_id):
        row = {"custom_id": result.custom_id, "status": result.result.type}
        if result.result.type == "succeeded":
            text = next((b.text for b in reversed(result.result.message.content) if b.type == "text"), "")
            parsed = _parse_json(text)
            if parsed:
                row.update(parsed)
        rows.append(row)
    return pd.DataFrame(rows)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _load_universe(cfg: dict) -> pd.DataFrame:
    if not os.path.exists(UNIVERSE_PATH):
        return build_universe(cfg, UNIVERSE_PATH)
    return pd.read_parquet(UNIVERSE_PATH)


def run_phase1(args):
    cfg = load_config()
    os.makedirs("data", exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    init_db(DB_PATH)

    universe_df = _load_universe(cfg)
    lp = cfg["lazy_prices"]
    price_cfg = {
        **cfg,
        "liquidity_gate": {
            "min_market_cap": lp["min_market_cap"],
            "min_avg_dollar_vol_20d": lp["min_avg_dollar_vol_20d"],
        },
    }
    _, factors_df = fetch_all_prices(universe_df, price_cfg, DB_PATH)
    gated = _apply_neglect_gate(factors_df, lp, min_price=cfg["universe"].get("min_price", 0.0))
    if args.limit:
        gated = gated.head(args.limit)
    print(f"[phase1] {len(gated)} tickers in gated universe, {args.days}d lookback")

    # Phase 1 never calls the Claude API, so --dry-run has nothing to cost-gate.
    # It still runs the (free) EDGAR scan to report real per-category event
    # counts, per spec, but skips the yfinance forward-return pass and file
    # writes so a quick check doesn't also fetch years of price history.
    events_df = collect_events(gated[["ticker", "cik"]], DB_PATH, args.days)
    print(f"[phase1] {len(events_df)} raw events collected")
    if events_df.empty:
        print("[phase1] no events found — nothing to summarize")
        return

    if args.dry_run:
        span_days = (pd.Timestamp(events_df["event_date"].max()) - pd.Timestamp(events_df["event_date"].min())).days
        months = max(span_days / 30.44, 1e-9)
        counts = events_df.groupby("category").size().rename("n_events").reset_index()
        counts["events_per_month"] = (counts["n_events"] / months).round(2)
        print(counts.to_string(index=False))
        print("[phase1] dry-run: event counts above are real (Phase 1 is $0 regardless of --dry-run). "
              "Skipped forward-return computation and file writes. Re-run without --dry-run for the full summary.")
        return

    events_df = compute_abnormal_returns(events_df, DB_PATH)
    today = date.today().isoformat()
    events_path = os.path.join(args.out_dir, f"event_backtest_events_{today}.csv")
    summary_path = os.path.join(args.out_dir, f"event_backtest_summary_{today}.csv")
    events_df.to_csv(events_path, index=False)

    summary_df = summarize(events_df)
    summary_df.to_csv(summary_path, index=False)
    print(summary_df.to_string(index=False))
    print(f"[phase1] wrote {events_path}")
    print(f"[phase1] wrote {summary_path}")


def run_phase2(args):
    events_df = pd.read_csv(args.events_csv)
    summary_df = pd.read_csv(args.summary_csv)

    sample_df = select_phase2_sample(
        events_df, summary_df, include_categories=args.include_category or None,
    )
    if sample_df.empty:
        print("[phase2] no PASS categories (and none forced via --include-category) — nothing to sample")
        return

    today = date.today().isoformat()
    sample_path = os.path.join(args.out_dir, f"phase2_sample_{today}.csv")
    export_hand_label_csv(sample_df, sample_path)
    print(f"[phase2] {len(sample_df)} events sampled → {sample_path}")

    if not args.claude_batch:
        print("[phase2] default zero-cost path: hand-label the CSV above. Rows are sorted by "
              "20-day abnormal return; the top/bottom 50 are flagged hand_label_priority=True — "
              "start there (largest/smallest apparent-return events) before doing the rest. "
              "Pass --claude-batch to use the Haiku Batches API instead.")
        return

    result = submit_phase2_batch(sample_df, DB_PATH, dry_run=args.dry_run)
    if result:
        mapping_path = os.path.join(args.out_dir, f"phase2_batch_{today}.json")
        with open(mapping_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[phase2] batch mapping saved to {mapping_path} — retrieve later with phase2-fetch")


def run_phase2_fetch(args):
    results_df = fetch_phase2_results(args.batch_id)
    if results_df.empty:
        return
    out_path = args.out or f"output/phase2_results_{date.today().isoformat()}.csv"
    results_df.to_csv(out_path, index=False)
    print(f"[phase2-fetch] wrote {out_path}")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Event-driven filing backtest (Phase 1 + Phase 2)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("phase1", help="Pure event study, $0")
    p1.add_argument("--days", type=int, default=730)
    p1.add_argument("--limit", type=int, default=None, help="Limit universe size (dev/test)")
    p1.add_argument("--dry-run", action="store_true")
    p1.add_argument("--out-dir", default="output")
    p1.set_defaults(func=run_phase1)

    p2 = sub.add_parser("phase2", help="Capped materiality validation sample")
    p2.add_argument("--events-csv", required=True)
    p2.add_argument("--summary-csv", required=True)
    p2.add_argument("--include-category", action="append", default=[])
    p2.add_argument("--claude-batch", action="store_true")
    p2.add_argument("--dry-run", action="store_true")
    p2.add_argument("--out-dir", default="output")
    p2.set_defaults(func=run_phase2)

    p2f = sub.add_parser("phase2-fetch", help="Poll + retrieve a submitted Phase 2 batch")
    p2f.add_argument("--batch-id", required=True)
    p2f.add_argument("--out", default=None)
    p2f.set_defaults(func=run_phase2_fetch)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
