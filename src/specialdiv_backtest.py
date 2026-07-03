"""Special cash dividend announcement backtest — idea #3 in the ranked bank.

Registry: docs/strategy-registry.md. Spec: docs/strategy-specs.md #3.
Fit doc: docs/handoffs/FIT-2026-07-03.md.

Hypothesis: special/one-time cash dividend announcement -> positive drift
(committed cash return, unambiguous positive surprise, no repeat-schedule
anticipation). Control: `regular_div_increase` — a routine, expected raise
(ratio 1.0-1.10x vs the prior payment), sourced independently from yfinance
dividend history rather than EDGAR (per spec: companies don't press-release
routine raises, so there's no comparable 8-K population to sample from).

Classification follows the same principle established in ideas #1-2: real
market data over text regex. EDGAR FTS hits carry no snippet field (verified
live), so a candidate "special dividend" 8-K hit is confirmed against the
ticker's real yfinance dividend payment history — a payment within 90 days
after the hit that's >=1.5x the ticker's trailing regular-dividend baseline
(median of the prior 4 payments) confirms a genuine one-off special
dividend, not routine noise or an unrelated 8-K mention of the phrase.
Tickers with insufficient prior dividend history to baseline against are
dropped — "special" is only measurable relative to an established routine.
"""

import argparse
import logging
import os
import random
import sqlite3
import time
from datetime import date, datetime, timedelta

import pandas as pd
import yfinance as yf

from src.backtest_recipe import (
    dedupe_events,
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
MATCH_WINDOW_DAYS = 90
BASELINE_LOOKBACK_PAYMENTS = 4
SPECIAL_JUMP_MULTIPLE = 1.5      # payment must be >=1.5x trailing baseline to count as "special"
CONTROL_RATIO_LO, CONTROL_RATIO_HI = 1.0, 1.10
LOOKBACK_YEARS = 3
CONTROL_CAP_MULTIPLE = 3
CONTROL_SCAN_TICKERS = 300        # cap the universe scan for control discovery
SIGNAL_CATEGORY = "special_div"
CONTROL_CATEGORY = "regular_div_increase"
FTS_PHRASES = ("special dividend", "special cash dividend")
FTS_FORM = "8-K"


def fetch_raw_hits(start: str, end: str, db_path: str = DB_PATH) -> pd.DataFrame:
    frames = [search(p, FTS_FORM, start, end, db_path=db_path) for p in FTS_PHRASES]
    hits = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if hits.empty:
        return hits
    hits = hits.drop_duplicates(subset=["adsh"]).reset_index(drop=True)
    hits["file_date"] = pd.to_datetime(hits["file_date"])
    return hits


def _init_dividends_table(db_path: str):
    con = sqlite3.connect(db_path)
    con.execute("""
        CREATE TABLE IF NOT EXISTS ticker_dividends_cache (
            ticker TEXT,
            ex_date TEXT,
            amount REAL,
            fetched_at TEXT,
            PRIMARY KEY (ticker, ex_date)
        )""")
    con.execute("""
        CREATE TABLE IF NOT EXISTS ticker_dividends_fetch_log (
            ticker TEXT PRIMARY KEY,
            fetched_at TEXT
        )""")
    con.commit()
    con.close()


def fetch_dividends_for_tickers(tickers: list[str], db_path: str = DB_PATH,
                                sleep_s: float = 0.3, ttl_days: int = 7) -> None:
    """Checkpointed like split_backtest.fetch_splits_for_tickers — safe to
    interrupt and re-run, only fetches tickers not already cached."""
    _init_dividends_table(db_path)
    con = sqlite3.connect(db_path)
    cutoff = (datetime.now() - timedelta(days=ttl_days)).isoformat()
    done = {r[0] for r in con.execute(
        "SELECT ticker FROM ticker_dividends_fetch_log WHERE fetched_at >= ?", (cutoff,)).fetchall()}
    todo = [t for t in tickers if t not in done]
    logger.info(f"[specialdiv] {len(todo)} tickers to fetch dividends for ({len(done)} cached)")
    for i, t in enumerate(todo):
        try:
            s = yf.Ticker(t).dividends
        except Exception as e:
            logger.warning(f"[specialdiv] dividends fetch failed for {t}: {e}")
            s = None
        now = datetime.now().isoformat()
        if s is not None:
            for ts, amount in s.items():
                con.execute(
                    "INSERT OR REPLACE INTO ticker_dividends_cache VALUES (?,?,?,?)",
                    (t, pd.Timestamp(ts).date().isoformat(), float(amount), now))
        con.execute("INSERT OR REPLACE INTO ticker_dividends_fetch_log VALUES (?,?)", (t, now))
        con.commit()
        if (i + 1) % 100 == 0:
            print(f"[specialdiv] dividends fetch {i + 1}/{len(todo)}", flush=True)
        time.sleep(sleep_s)
    con.close()


def load_dividends_by_ticker(tickers: set, db_path: str = DB_PATH) -> dict:
    if not tickers:
        return {}
    con = sqlite3.connect(db_path)
    placeholders = ",".join("?" for _ in tickers)
    df = pd.read_sql_query(
        f"SELECT ticker, ex_date, amount FROM ticker_dividends_cache WHERE ticker IN ({placeholders})",
        con, params=list(tickers))
    con.close()
    if df.empty:
        return {}
    df["ex_date"] = pd.to_datetime(df["ex_date"])
    out = {}
    for ticker, g in df.groupby("ticker"):
        g = g.sort_values("ex_date")
        out[ticker] = list(zip(g["ex_date"], g["amount"]))
    return out


def _median(vals: list[float]) -> float:
    s = sorted(vals)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 == 1 else (s[mid - 1] + s[mid]) / 2


def confirm_special_events(hits: pd.DataFrame, dividends_by_ticker: dict,
                           window_days: int = MATCH_WINDOW_DAYS,
                           baseline_n: int = BASELINE_LOOKBACK_PAYMENTS,
                           jump_multiple: float = SPECIAL_JUMP_MULTIPLE) -> pd.DataFrame:
    """For each ticker's EDGAR "special dividend" hits, confirm against a
    real payment within `window_days` after the hit that's >=`jump_multiple`
    x the median of the `baseline_n` payments immediately preceding it.
    Unconfirmed hits (no matching outsized payment) are dropped."""
    if hits.empty:
        return pd.DataFrame()
    rows = []
    for ticker, g in hits.groupby("ticker"):
        payments = dividends_by_ticker.get(ticker, [])
        if len(payments) < baseline_n + 1:
            continue
        g = g.sort_values("file_date")
        for i in range(baseline_n, len(payments)):
            pay_date, amount = payments[i]
            baseline_vals = [a for _, a in payments[i - baseline_n:i]]
            baseline = _median(baseline_vals)
            if baseline <= 0 or amount < jump_multiple * baseline:
                continue
            candidates = g[(g["file_date"] <= pay_date)
                          & (g["file_date"] >= pay_date - pd.Timedelta(days=window_days))]
            if candidates.empty:
                continue
            first = candidates.iloc[0]
            rows.append({
                "ticker": ticker, "cik": first["cik"], "file_date": first["file_date"],
                "adsh": first["adsh"], "pay_date": pay_date, "amount": float(amount),
                "baseline": float(baseline), "category": SIGNAL_CATEGORY,
            })
    return pd.DataFrame(rows)


def find_regular_increase_events(dividends_by_ticker: dict, start: str, end: str,
                                 ratio_lo: float = CONTROL_RATIO_LO,
                                 ratio_hi: float = CONTROL_RATIO_HI) -> pd.DataFrame:
    """Payments that are a modest (ratio_lo, ratio_hi] raise over the
    immediately preceding payment — a routine increase, not a special
    dividend. No EDGAR hit needed (companies don't press-release routine
    raises); event_date approximates off the payment's ex_date itself,
    understating true announcement lead time — acceptable for a null-
    expectation control, not the gated signal category."""
    rows = []
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    for ticker, payments in dividends_by_ticker.items():
        for i in range(1, len(payments)):
            date_i, amt_i = payments[i]
            _, amt_prev = payments[i - 1]
            if not (start_ts <= date_i <= end_ts) or amt_prev <= 0:
                continue
            ratio = amt_i / amt_prev
            if ratio_lo < ratio <= ratio_hi:
                rows.append({
                    "ticker": ticker, "cik": None, "file_date": date_i,
                    "adsh": None, "pay_date": date_i, "amount": float(amt_i),
                    "baseline": float(amt_prev), "category": CONTROL_CATEGORY,
                })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="Special dividend backtest (registry idea #3)")
    ap.add_argument("--years", type=int, default=LOOKBACK_YEARS)
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--tag", default="")
    ap.add_argument("--dry-run-only", action="store_true",
                    help="only count confirmed signal events, do not fetch control or price data")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    os.makedirs(args.out_dir, exist_ok=True)
    init_db(DB_PATH)

    end = date.today().isoformat()
    start = (date.today() - timedelta(days=365 * args.years)).isoformat()

    uni = load_universe()
    uni_cik = uni.merge(pd.read_parquet("data/universe.parquet")[["ticker", "cik"]],
                        on="ticker", how="left")
    print(f"[specialdiv] universe: {len(uni_cik)} tickers >= $2.5B", flush=True)

    print("[specialdiv] fetching 'special dividend' 8-K hits from EDGAR FTS...", flush=True)
    raw_hits = fetch_raw_hits(start, end)
    hits_uni = filter_to_universe(raw_hits, uni_cik)
    print(f"[specialdiv] raw hits: {len(raw_hits)}, in-universe (pre-classification): {len(hits_uni)} "
          f"across {hits_uni['ticker'].nunique() if not hits_uni.empty else 0} tickers", flush=True)

    signal_tickers = sorted(hits_uni["ticker"].unique()) if not hits_uni.empty else []
    print(f"[specialdiv] fetching yfinance dividend history for {len(signal_tickers)} candidate tickers...",
          flush=True)
    fetch_dividends_for_tickers(signal_tickers)
    div_by_ticker = load_dividends_by_ticker(set(signal_tickers))

    confirmed = confirm_special_events(hits_uni, div_by_ticker)
    confirmed = dedupe_events(confirmed) if not confirmed.empty else confirmed
    years = max(args.years, 1)
    n_signal = len(confirmed)
    print(f"[specialdiv] dry-run confirmed signal count: {n_signal} ({n_signal / years:.1f}/yr)", flush=True)

    if n_signal < 10 * years or n_signal < 50:
        print(f"[specialdiv] KILLED at dry-run: signal count too thin "
              f"(need >=10/yr and >=50 total, got {n_signal} total over {years}y)")
        return

    if args.dry_run_only:
        print("[specialdiv] dry-run-only requested, stopping before control/price fetch")
        return

    all_uni_tickers = list(uni_cik["ticker"].unique())
    control_scan = sorted(random.Random(0).sample(
        all_uni_tickers, min(CONTROL_SCAN_TICKERS, len(all_uni_tickers))))
    print(f"[specialdiv] fetching yfinance dividend history for {len(control_scan)} "
          f"randomly sampled control-candidate tickers...", flush=True)
    fetch_dividends_for_tickers(control_scan)
    control_div_by_ticker = load_dividends_by_ticker(set(control_scan))
    control_raw = find_regular_increase_events(control_div_by_ticker, start, end)
    control_cap = n_signal * CONTROL_CAP_MULTIPLE
    if len(control_raw) > control_cap:
        control_raw = control_raw.sample(n=control_cap, random_state=0).reset_index(drop=True)
    print(f"[specialdiv] control count: {len(control_raw)} (capped at {control_cap})", flush=True)

    events = pd.concat([confirmed, control_raw], ignore_index=True)
    events["event_date"] = events["file_date"].dt.date.astype(str).map(event_date_from_announcement)

    all_tickers = set(events["ticker"])
    earnings = load_earnings_for_tickers(all_tickers, DB_PATH)
    events = drop_earnings_contamination(events, earnings)
    print(f"[specialdiv] {len(events)} events after contamination filter "
          f"({events['category'].value_counts().to_dict() if not events.empty else {}})", flush=True)
    if events.empty:
        return

    events = compute_abnormal_returns(events, DB_PATH, horizons=HORIZONS, benchmark=BENCHMARK)
    stem = date.today().isoformat() + (f"_{args.tag}" if args.tag else "")
    events.to_csv(os.path.join(args.out_dir, f"specialdiv_events_{stem}.csv"), index=False)

    summary = summarize(events, horizons=HORIZONS, gate_horizons=GATE_HORIZONS)
    summary.to_csv(os.path.join(args.out_dir, f"specialdiv_summary_{stem}.csv"), index=False)
    print(summary.to_string(index=False))

    halves = split_half_summary(events.rename(columns={"file_date": "filing_date"}))
    halves.to_csv(os.path.join(args.out_dir, f"specialdiv_halves_{stem}.csv"), index=False)
    print(halves.to_string(index=False))


if __name__ == "__main__":
    main()
