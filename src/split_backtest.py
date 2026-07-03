"""Stock-split announcement backtest — idea #2 in the ranked bank.

Registry: docs/strategy-registry.md. Spec: docs/strategy-specs.md #2.
Fit doc: docs/handoffs/FIT-2026-07-03.md.

Hypothesis: forward stock split (>=2:1) announcement -> positive drift
(retail attention + index-option accessibility channel; run-up from
announcement to effective date falls inside a days-weeks window). Control:
reverse splits (expect null/negative — distress signal, not attention).

Real deviation from the spec: EDGAR FTS hits carry no text snippet (verified
live 2026-07-03 — see src/edgar_fts.py docstring), so forward/reverse
classification can't come from regex on a snippet as the spec assumed.
Classification instead comes straight from the spec's own cross-validation
step, promoted to the PRIMARY signal: yfinance's real recorded split factor
(Ticker.splits) within 90 days after the EDGAR hit. This is strictly more
reliable than regex (ground truth, not text-pattern guessing) and needs no
per-accession document fetch. EDGAR hits with no matching real split in the
window are dropped (rumor/boilerplate/unrelated "stock split" mentions in an
8-K, e.g. warrant anti-dilution language) — this kills most of the raw
22,855 "stock split" 8-K hits before any universe filter is even applied.

Entry timing: event_date = matched announcement's file_date + 1 calendar day
(same convention as every other idea — first close strictly after public
disclosure of the announcement, not the split's effective date).
"""

import argparse
import logging
import os
import sqlite3
import time
from datetime import date, datetime, timedelta

import pandas as pd
import yfinance as yf

from src.backtest_recipe import (
    drop_earnings_contamination,
    filter_to_universe,
    load_earnings_for_tickers,
)
from src.cache import init_db
from src.edgar_fts import search
from src.event_backtest import compute_abnormal_returns, summarize
from src.insider_backtest import split_half_summary
from src.pead_backtest import event_date_from_announcement, load_universe

logger = logging.getLogger(__name__)

DB_PATH = "data/cache.db"
BENCHMARK = "SPY"
HORIZONS = (5, 10, 20)
GATE_HORIZONS = (5, 20)
MATCH_WINDOW_DAYS = 90      # announcement-to-effective-date gap ceiling
LOOKBACK_YEARS = 3
SIGNAL_CATEGORY = "split_fwd_announce"
CONTROL_CATEGORY = "split_reverse_announce"
FTS_PHRASE = "stock split"
FTS_FORM = "8-K"


def fetch_raw_hits(start: str, end: str, db_path: str = DB_PATH) -> pd.DataFrame:
    hits = search(FTS_PHRASE, FTS_FORM, start, end, db_path=db_path)
    if hits.empty:
        return hits
    hits["file_date"] = pd.to_datetime(hits["file_date"])
    return hits.reset_index(drop=True)


def _init_splits_table(db_path: str):
    con = sqlite3.connect(db_path)
    con.execute("""
        CREATE TABLE IF NOT EXISTS ticker_splits_cache (
            ticker TEXT,
            split_date TEXT,
            factor REAL,
            fetched_at TEXT,
            PRIMARY KEY (ticker, split_date)
        )""")
    con.execute("""
        CREATE TABLE IF NOT EXISTS ticker_splits_fetch_log (
            ticker TEXT PRIMARY KEY,
            fetched_at TEXT
        )""")
    con.commit()
    con.close()


def fetch_splits_for_tickers(tickers: list[str], db_path: str = DB_PATH,
                             sleep_s: float = 0.3, ttl_days: int = 7) -> None:
    """Checkpointed like pead_backtest.collect_earnings_events — safe to
    interrupt and re-run, only fetches tickers not already cached."""
    _init_splits_table(db_path)
    con = sqlite3.connect(db_path)
    cutoff = (datetime.now() - timedelta(days=ttl_days)).isoformat()
    done = {r[0] for r in con.execute(
        "SELECT ticker FROM ticker_splits_fetch_log WHERE fetched_at >= ?", (cutoff,)).fetchall()}
    todo = [t for t in tickers if t not in done]
    logger.info(f"[split] {len(todo)} tickers to fetch splits for ({len(done)} cached)")
    for i, t in enumerate(todo):
        try:
            s = yf.Ticker(t).splits
        except Exception as e:
            logger.warning(f"[split] splits fetch failed for {t}: {e}")
            s = None
        now = datetime.now().isoformat()
        if s is not None:
            for ts, factor in s.items():
                con.execute(
                    "INSERT OR REPLACE INTO ticker_splits_cache VALUES (?,?,?,?)",
                    (t, pd.Timestamp(ts).date().isoformat(), float(factor), now))
        con.execute("INSERT OR REPLACE INTO ticker_splits_fetch_log VALUES (?,?)", (t, now))
        con.commit()
        if (i + 1) % 100 == 0:
            print(f"[split] splits fetch {i + 1}/{len(todo)}", flush=True)
        time.sleep(sleep_s)
    con.close()


def load_splits_by_ticker(tickers: set, db_path: str = DB_PATH) -> dict:
    if not tickers:
        return {}
    con = sqlite3.connect(db_path)
    placeholders = ",".join("?" for _ in tickers)
    df = pd.read_sql_query(
        f"SELECT ticker, split_date, factor FROM ticker_splits_cache WHERE ticker IN ({placeholders})",
        con, params=list(tickers))
    con.close()
    if df.empty:
        return {}
    df["split_date"] = pd.to_datetime(df["split_date"])
    out = {}
    for ticker, g in df.groupby("ticker"):
        out[ticker] = list(zip(g["split_date"], g["factor"]))
    return out


def match_hits_to_splits(hits: pd.DataFrame, splits_by_ticker: dict,
                         window_days: int = MATCH_WINDOW_DAYS) -> pd.DataFrame:
    """For each real yfinance split, find the earliest EDGAR 8-K hit for
    that ticker filed on/before the split date and within `window_days`
    before it — that's the announcement. Real splits with no matching hit
    (yfinance split but nothing found mentioning "stock split") and hits
    with no matching real split (rumor/boilerplate/unrelated mention) are
    both dropped — only confirmed announcement-to-execution pairs remain."""
    if hits.empty:
        return pd.DataFrame()
    rows = []
    for ticker, g in hits.groupby("ticker"):
        real_splits = splits_by_ticker.get(ticker, [])
        g = g.sort_values("file_date")
        for split_date, factor in real_splits:
            if factor == 1:
                continue
            candidates = g[(g["file_date"] <= split_date)
                          & (g["file_date"] >= split_date - pd.Timedelta(days=window_days))]
            if candidates.empty:
                continue
            first = candidates.iloc[0]
            category = SIGNAL_CATEGORY if factor > 1 else CONTROL_CATEGORY
            rows.append({
                "ticker": ticker, "cik": first["cik"], "file_date": first["file_date"],
                "adsh": first["adsh"], "split_date": split_date, "factor": float(factor),
                "category": category,
            })
    return pd.DataFrame(rows)


def build_events(raw_hits: pd.DataFrame, uni: pd.DataFrame, splits_by_ticker: dict,
                 earnings: pd.DataFrame) -> pd.DataFrame:
    hits_uni = filter_to_universe(raw_hits, uni)
    matched = match_hits_to_splits(hits_uni, splits_by_ticker)
    if matched.empty:
        return matched
    matched["event_date"] = matched["file_date"].dt.date.astype(str).map(event_date_from_announcement)
    matched = drop_earnings_contamination(matched, earnings)
    return matched.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description="Stock-split announcement backtest (registry idea #2)")
    ap.add_argument("--years", type=int, default=LOOKBACK_YEARS)
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--tag", default="")
    ap.add_argument("--dry-run-only", action="store_true",
                    help="only count in-universe confirmed events, do not fetch abnormal returns")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    os.makedirs(args.out_dir, exist_ok=True)
    init_db(DB_PATH)

    end = date.today().isoformat()
    start = (date.today() - timedelta(days=365 * args.years)).isoformat()

    uni = load_universe()
    uni_cik = uni.merge(pd.read_parquet("data/universe.parquet")[["ticker", "cik"]],
                        on="ticker", how="left")
    print(f"[split] universe: {len(uni_cik)} tickers >= $2.5B", flush=True)

    print("[split] fetching 'stock split' 8-K hits from EDGAR FTS...", flush=True)
    raw_hits = fetch_raw_hits(start, end)
    hits_uni = filter_to_universe(raw_hits, uni_cik)
    print(f"[split] raw hits: {len(raw_hits)}, in-universe (pre-classification): {len(hits_uni)} "
          f"across {hits_uni['ticker'].nunique() if not hits_uni.empty else 0} tickers", flush=True)

    tickers = sorted(hits_uni["ticker"].unique()) if not hits_uni.empty else []
    print(f"[split] fetching yfinance split history for {len(tickers)} tickers...", flush=True)
    fetch_splits_for_tickers(tickers)
    splits_by_ticker = load_splits_by_ticker(set(tickers))

    matched = match_hits_to_splits(hits_uni, splits_by_ticker)
    years = max(args.years, 1)
    n_signal = int((matched["category"] == SIGNAL_CATEGORY).sum()) if not matched.empty else 0
    n_control = int((matched["category"] == CONTROL_CATEGORY).sum()) if not matched.empty else 0
    print(f"[split] dry-run confirmed counts: signal={n_signal} ({n_signal / years:.1f}/yr), "
          f"control={n_control} ({n_control / years:.1f}/yr)", flush=True)

    if n_signal < 10 * years or n_signal < 50:
        print(f"[split] KILLED at dry-run: signal count too thin "
              f"(need >=10/yr and >=50 total, got {n_signal} total over {years}y)")
        return

    if args.dry_run_only:
        print("[split] dry-run-only requested, stopping before price fetch")
        return

    earnings = load_earnings_for_tickers(set(tickers), DB_PATH)
    events = build_events(raw_hits, uni_cik, splits_by_ticker, earnings)
    print(f"[split] {len(events)} events after contamination filter "
          f"({events['category'].value_counts().to_dict() if not events.empty else {}})", flush=True)
    if events.empty:
        return

    events = compute_abnormal_returns(events, DB_PATH, horizons=HORIZONS, benchmark=BENCHMARK)
    stem = date.today().isoformat() + (f"_{args.tag}" if args.tag else "")
    events.to_csv(os.path.join(args.out_dir, f"split_events_{stem}.csv"), index=False)

    summary = summarize(events, horizons=HORIZONS, gate_horizons=GATE_HORIZONS)
    summary.to_csv(os.path.join(args.out_dir, f"split_summary_{stem}.csv"), index=False)
    print(summary.to_string(index=False))

    halves = split_half_summary(events.rename(columns={"file_date": "filing_date"}))
    halves.to_csv(os.path.join(args.out_dir, f"split_halves_{stem}.csv"), index=False)
    print(halves.to_string(index=False))


if __name__ == "__main__":
    main()
