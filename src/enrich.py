"""Finalist enrichment: the per-ticker lookups that are too slow or too
rate-limited for the whole universe but cheap for ~60 names.

  eps_rev_30d / eps_rev_90d   true estimate-revision momentum from yfinance
                              `eps_trend` (current vs 30/90 days ago, current
                              fiscal year row, current quarter as fallback)
  short_float, days_to_cover  yfinance get_info (was a per-survivor call)
  analyst_target_pct          mean analyst target / price − 1
  days_to_earnings            Finnhub earnings calendar (no yfinance)

Everything is best-effort with a wall-clock budget: Yahoo rate-limits, and a
run must finish before the trader session at 17:15 ET. Results are cached per
(ticker, date) so a rerun the same day costs nothing.
"""
import json
import logging
import math
import time
from datetime import date, datetime

import numpy as np
import pandas as pd

from src.cache import _get_conn, _now_iso
from src.compose import rank_normalize
from src.finnhub_data import days_to_earnings

logger = logging.getLogger(__name__)

ENRICH_COLUMNS = ["eps_rev_30d", "eps_rev_90d", "short_float", "days_to_cover",
                  "analyst_target_pct", "days_to_earnings"]
MIN_ABS_EPS = 0.02   # below this a percent change is meaningless
DEFAULT_BUDGET_SECS = 240.0
RETRY_SLEEP = 1.5


def init_enrich_table(db_path: str) -> None:
    _get_conn(db_path).execute("""
        CREATE TABLE IF NOT EXISTS finalist_enrichment (
            ticker TEXT, as_of_date TEXT, payload TEXT, fetched_at TEXT,
            PRIMARY KEY(ticker, as_of_date)
        )
    """)
    _get_conn(db_path).commit()


def _get_cached(db_path: str, ticker: str, as_of: str) -> dict | None:
    row = _get_conn(db_path).execute(
        "SELECT payload FROM finalist_enrichment WHERE ticker=? AND as_of_date=?",
        (ticker.upper(), as_of)).fetchone()
    return json.loads(row[0]) if row else None


def _put_cached(db_path: str, ticker: str, as_of: str, payload: dict) -> None:
    conn = _get_conn(db_path)
    conn.execute("INSERT OR REPLACE INTO finalist_enrichment VALUES (?,?,?,?)",
                 (ticker.upper(), as_of, json.dumps(payload), _now_iso()))
    conn.commit()


# ── pure helpers ──────────────────────────────────────────────────────────────

def _pct_change(now, then) -> float:
    try:
        now, then = float(now), float(then)
    except (TypeError, ValueError):
        return float("nan")
    if math.isnan(now) or math.isnan(then) or abs(then) < MIN_ABS_EPS:
        return float("nan")
    return (now - then) / abs(then)


def eps_revision(eps_trend: pd.DataFrame | None) -> tuple[float, float]:
    """(rev_30d, rev_90d) from a yfinance eps_trend frame.

    Index is the period ('0q', '+1q', '0y', '+1y'); columns include 'current',
    '30daysAgo', '90daysAgo'. Current fiscal year ('0y') is used first because
    quarterly consensus is noisier; '0q' is the fallback. Negative EPS is
    handled by dividing by |then|, so −0.50 → −0.40 is a +20% revision.
    """
    if eps_trend is None or getattr(eps_trend, "empty", True):
        return float("nan"), float("nan")
    for period in ("0y", "0q"):
        if period not in eps_trend.index:
            continue
        row = eps_trend.loc[period]
        r30 = _pct_change(row.get("current"), row.get("30daysAgo"))
        r90 = _pct_change(row.get("current"), row.get("90daysAgo"))
        if not (math.isnan(r30) and math.isnan(r90)):
            return r30, r90
    return float("nan"), float("nan")


def target_pct(targets: dict | None, price: float | None) -> float:
    mean = (targets or {}).get("mean")
    try:
        mean, price = float(mean), float(price)
    except (TypeError, ValueError):
        return float("nan")
    if not price or math.isnan(mean) or math.isnan(price):
        return float("nan")
    return mean / price - 1.0


def short_interest(info: dict | None) -> tuple[float, float]:
    from src.fundamentals import parse_short_interest
    if not info:
        return float("nan"), float("nan")
    try:
        return parse_short_interest(info)
    except Exception:  # noqa: BLE001
        return float("nan"), float("nan")


def composite_final(df: pd.DataFrame, weight: float = 0.05) -> pd.DataFrame:
    """composite + weight × rank-normalised eps_rev_30d (NaN → 0, i.e. neutral).
    Applied within finalists only, so a name is never dropped from the pool for
    lacking estimate coverage — it just gets no lift."""
    out = df.copy()
    if "eps_rev_30d" in out.columns and out["eps_rev_30d"].notna().sum() >= 3:
        z = rank_normalize(out["eps_rev_30d"])
        z = z.where(out["eps_rev_30d"].notna(), 0.0)
    else:
        z = pd.Series(0.0, index=out.index)
    out["z_eps_rev_30d"] = z
    out["composite_final"] = out["composite"] + weight * z
    return out


# ── yfinance IO ───────────────────────────────────────────────────────────────

def _yf_fetch(ticker: str) -> dict:
    """One ticker's raw yfinance payloads. Raises on failure; caller retries."""
    import yfinance as yf
    t = yf.Ticker(ticker)
    out = {"eps_trend": None, "targets": None, "info": None}
    out["eps_trend"] = t.eps_trend
    try:
        out["targets"] = t.analyst_price_targets
    except Exception:  # noqa: BLE001 — optional
        out["targets"] = None
    try:
        out["info"] = t.get_info()
    except Exception:  # noqa: BLE001 — optional
        out["info"] = None
    return out


def enrich_finalists(
    df: pd.DataFrame,
    db_path: str,
    calendar: dict[str, str],
    today: date | None = None,
    fetch=_yf_fetch,
    budget_secs: float = DEFAULT_BUDGET_SECS,
    sleep=time.sleep,
    clock=time.monotonic,
) -> tuple[pd.DataFrame, dict]:
    """Attach ENRICH_COLUMNS to the finalist frame. Returns (df, stats)."""
    today = today or date.today()
    as_of = today.isoformat()
    init_enrich_table(db_path)
    out = df.copy()
    for c in ENRICH_COLUMNS:
        if c not in out.columns:
            out[c] = np.nan
    stats = {"finalists": len(out), "cached": 0, "fetched": 0, "failed": 0, "budget_exhausted": 0}
    start = clock()

    for idx, row in out.iterrows():
        ticker = str(row["ticker"]).upper()
        dte = days_to_earnings(ticker, calendar, today)
        out.at[idx, "days_to_earnings"] = np.nan if dte is None else float(dte)

        cached = _get_cached(db_path, ticker, as_of)
        if cached is not None:
            stats["cached"] += 1
            for c in ("eps_rev_30d", "eps_rev_90d", "short_float", "days_to_cover", "analyst_target_pct"):
                v = cached.get(c)
                out.at[idx, c] = np.nan if v is None else float(v)
            continue

        if clock() - start > budget_secs:
            stats["budget_exhausted"] += 1
            continue

        raw = None
        for attempt in range(2):
            try:
                raw = fetch(ticker)
                break
            except Exception as e:  # noqa: BLE001
                if attempt == 0:
                    sleep(RETRY_SLEEP)
                else:
                    logger.warning(f"[enrich] {ticker}: {type(e).__name__}: {e}")
        if raw is None:
            stats["failed"] += 1
            continue

        r30, r90 = eps_revision(raw.get("eps_trend"))
        sf, dtc = short_interest(raw.get("info"))
        tp = target_pct(raw.get("targets"), row.get("price"))
        payload = {"eps_rev_30d": _none_if_nan(r30), "eps_rev_90d": _none_if_nan(r90),
                   "short_float": _none_if_nan(sf), "days_to_cover": _none_if_nan(dtc),
                   "analyst_target_pct": _none_if_nan(tp)}
        _put_cached(db_path, ticker, as_of, payload)
        stats["fetched"] += 1
        out.at[idx, "eps_rev_30d"] = r30
        out.at[idx, "eps_rev_90d"] = r90
        out.at[idx, "short_float"] = sf
        out.at[idx, "days_to_cover"] = dtc
        out.at[idx, "analyst_target_pct"] = tp

    print(f"[enrich] finalists={stats['finalists']} cached={stats['cached']} fetched={stats['fetched']} "
          f"failed={stats['failed']} budget_exhausted={stats['budget_exhausted']}")
    return out, stats


def _none_if_nan(v):
    try:
        return None if v is None or math.isnan(float(v)) else float(v)
    except (TypeError, ValueError):
        return None
