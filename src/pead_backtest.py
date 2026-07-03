"""PEAD backtest — post-earnings-announcement drift, positive EPS surprises.

Fit constraints this is built against (docs/handoffs/FIT-2026-07-03.md):
- Universe: liquid US equities >= $2.5B market cap, from the cached market_cap
  table — NOT the $50M-$2B microcap gate in config.yaml's lazy_prices section.
- Horizon: days to weeks -> horizons (5, 10, 20) trading days, gate on (5, 20).
- Long-only buy signal: only positive-surprise buckets are actionable; the
  miss bucket is kept purely as a directional control.
- $0 API cost: yfinance earnings dates + prices, cached market caps, no Claude.

Entry timing: event_date is set to announcement date + 1 calendar day, so
_forward_return's t0 resolves to the first trading close AFTER the announcement.
Measured drift therefore excludes the announcement-day jump (which a next-day
buyer cannot capture) for both before-open and after-close announcers.

Benchmark is SPY, not IWM — the universe is mid/large cap.
"""

import argparse
import logging
import os
import sqlite3
import time
from datetime import date, datetime, timedelta

import pandas as pd
import yfinance as yf

from src.cache import init_db
from src.event_backtest import compute_abnormal_returns, summarize

logger = logging.getLogger(__name__)

DB_PATH = "data/cache.db"
BENCHMARK = "SPY"
HORIZONS = (5, 10, 20)
GATE_HORIZONS = (5, 20)
MIN_MARKET_CAP = 2_500_000_000
MIN_ABS_ESTIMATE = 0.02   # |EPS estimate| floor; below this surprise% explodes
EARNINGS_LIMIT = 12       # quarters requested per ticker from yfinance
FETCH_SLEEP_S = 0.4       # politeness delay between per-ticker earnings fetches

# Surprise(%) buckets. Yahoo's Surprise(%) = (actual - estimate)/|estimate|*100.
SURPRISE_BUCKETS = (
    ("pead_beat_large", 15.0, float("inf")),
    ("pead_beat_mid", 5.0, 15.0),
    ("pead_beat_small", 0.0, 5.0),
    ("pead_miss", float("-inf"), 0.0),
)


def classify_surprise(surprise_pct: float) -> str | None:
    if surprise_pct is None or pd.isna(surprise_pct):
        return None
    for name, lo, hi in SURPRISE_BUCKETS:
        if lo <= surprise_pct < hi:
            return name
    return None


def event_date_from_announcement(announce_date: str) -> str:
    """Shift +1 calendar day so t0 = first trading close after the announcement."""
    return (pd.Timestamp(announce_date) + pd.Timedelta(days=1)).date().isoformat()


# ── universe ──────────────────────────────────────────────────────────────────

def load_universe(db_path: str = DB_PATH, min_cap: float = MIN_MARKET_CAP,
                  cap_fresh_since: str = "2026-06-01") -> pd.DataFrame:
    """Tickers >= min_cap from the cached market_cap table, restricted to
    SEC-registered names in universe.parquet (drops OTC/foreign-only listings)."""
    con = sqlite3.connect(db_path)
    caps = pd.read_sql_query(
        "SELECT ticker, value AS market_cap FROM market_cap "
        "WHERE value >= ? AND fetched_at >= ?",
        con, params=(min_cap, cap_fresh_since))
    con.close()
    uni = pd.read_parquet("data/universe.parquet")
    merged = caps.merge(uni[["ticker"]], on="ticker", how="inner")
    return merged.sort_values("market_cap", ascending=False).reset_index(drop=True)


# ── earnings-event collection (checkpointed in sqlite, resumable) ─────────────

def _init_earnings_table(db_path: str):
    con = sqlite3.connect(db_path)
    con.execute("""
        CREATE TABLE IF NOT EXISTS pead_earnings (
            ticker TEXT,
            announce_date TEXT,
            eps_estimate REAL,
            eps_actual REAL,
            surprise_pct REAL,
            fetched_at TEXT,
            PRIMARY KEY (ticker, announce_date)
        )""")
    con.execute("""
        CREATE TABLE IF NOT EXISTS pead_fetch_log (
            ticker TEXT PRIMARY KEY,
            fetched_at TEXT,
            n_rows INTEGER
        )""")
    con.commit()
    con.close()


def _already_fetched(db_path: str, ttl_days: int = 7) -> set:
    con = sqlite3.connect(db_path)
    cutoff = (datetime.now() - timedelta(days=ttl_days)).isoformat()
    rows = con.execute(
        "SELECT ticker FROM pead_fetch_log WHERE fetched_at >= ?", (cutoff,)).fetchall()
    con.close()
    return {r[0] for r in rows}


def fetch_earnings_one(ticker: str) -> list[dict]:
    """One yfinance earnings-dates call. Returns [] on any failure (logged)."""
    try:
        df = yf.Ticker(ticker).get_earnings_dates(limit=EARNINGS_LIMIT)
    except Exception as e:
        logger.warning(f"[pead] earnings fetch failed for {ticker}: {e}")
        return []
    if df is None or df.empty:
        return []
    out = []
    for ts, row in df.iterrows():
        est = row.get("EPS Estimate")
        act = row.get("Reported EPS")
        spr = row.get("Surprise(%)")
        if pd.isna(est) or pd.isna(act) or pd.isna(spr):
            continue  # future or unreported quarters
        out.append({
            "ticker": ticker,
            "announce_date": pd.Timestamp(ts).date().isoformat(),
            "eps_estimate": float(est),
            "eps_actual": float(act),
            "surprise_pct": float(spr),
        })
    return out


def collect_earnings_events(tickers: list[str], db_path: str = DB_PATH,
                            sleep_s: float = FETCH_SLEEP_S) -> None:
    """Fetch + checkpoint earnings rows for every ticker not fetched recently.
    Safe to interrupt and re-run; progress persists in sqlite."""
    _init_earnings_table(db_path)
    done = _already_fetched(db_path)
    todo = [t for t in tickers if t not in done]
    logger.info(f"[pead] {len(todo)} tickers to fetch ({len(done)} cached)")
    con = sqlite3.connect(db_path)
    for i, t in enumerate(todo):
        rows = fetch_earnings_one(t)
        now = datetime.now().isoformat()
        for r in rows:
            con.execute(
                "INSERT OR REPLACE INTO pead_earnings VALUES (?,?,?,?,?,?)",
                (r["ticker"], r["announce_date"], r["eps_estimate"],
                 r["eps_actual"], r["surprise_pct"], now))
        con.execute("INSERT OR REPLACE INTO pead_fetch_log VALUES (?,?,?)",
                    (t, now, len(rows)))
        con.commit()
        if (i + 1) % 100 == 0:
            print(f"[pead] earnings fetch {i + 1}/{len(todo)}", flush=True)
        time.sleep(sleep_s)
    con.close()


def load_events(db_path: str = DB_PATH, lookback_days: int = 730,
                min_abs_estimate: float = MIN_ABS_ESTIMATE) -> pd.DataFrame:
    """Checkpointed earnings rows -> event DataFrame for the return harness."""
    con = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM pead_earnings", con)
    con.close()
    if df.empty:
        return df
    cutoff = (date.today() - timedelta(days=lookback_days)).isoformat()
    df = df[(df["announce_date"] >= cutoff)
            & (df["announce_date"] < date.today().isoformat())
            & (df["eps_estimate"].abs() >= min_abs_estimate)].copy()
    df["category"] = df["surprise_pct"].map(classify_surprise)
    df = df.dropna(subset=["category"])
    df["event_date"] = df["announce_date"].map(event_date_from_announcement)
    return df.reset_index(drop=True)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="PEAD backtest (fit: FIT-2026-07-03.md)")
    ap.add_argument("--days", type=int, default=730, help="event lookback window")
    ap.add_argument("--limit", type=int, default=0, help="cap universe size (debug)")
    ap.add_argument("--skip-fetch", action="store_true",
                    help="use already-checkpointed earnings rows only")
    ap.add_argument("--out-dir", default="output")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    os.makedirs(args.out_dir, exist_ok=True)
    init_db(DB_PATH)

    uni = load_universe()
    if args.limit:
        uni = uni.head(args.limit)
    print(f"[pead] universe: {len(uni)} tickers >= ${MIN_MARKET_CAP/1e9:.1f}B", flush=True)

    if not args.skip_fetch:
        collect_earnings_events(list(uni["ticker"]))

    events = load_events(lookback_days=args.days)
    events = events[events["ticker"].isin(set(uni["ticker"]))]
    print(f"[pead] {len(events)} events in last {args.days}d", flush=True)
    if events.empty:
        print("[pead] no events — nothing to summarize")
        return

    events = compute_abnormal_returns(events, DB_PATH, horizons=HORIZONS,
                                      benchmark=BENCHMARK)
    today = date.today().isoformat()
    events_path = os.path.join(args.out_dir, f"pead_events_{today}.csv")
    summary_path = os.path.join(args.out_dir, f"pead_summary_{today}.csv")
    events.to_csv(events_path, index=False)

    summary = summarize(events, horizons=HORIZONS, gate_horizons=GATE_HORIZONS)
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))
    print(f"[pead] wrote {events_path}")
    print(f"[pead] wrote {summary_path}")


if __name__ == "__main__":
    main()
