"""Schedule 13D activist-stake backtest — idea #1 in the ranked bank.

Registry: docs/strategy-registry.md Q2. Spec: docs/strategy-specs.md #1.
Fit doc: docs/handoffs/FIT-2026-07-03.md.

Hypothesis: initial Schedule 13D (>=5% activist stake, active intent) on an
in-universe (>=$2.5B) subject company signals positive drift over the
following days-weeks, as the market prices in expected activist pressure.
Control: Schedule 13G (same 5% disclosure threshold, passive intent) —
expected ~null, isolates the "activist" channel from the "someone bought 5%"
channel.

Data: EDGAR full-text search via src.edgar_fts, $0 cost. See edgar_fts.py's
docstring for two live-verified corrections to the spec's assumed API shape
(forms filter value, no header-fetch needed for subject CIK).

Entry timing: event_date = file_date + 1 calendar day (same convention as
PEAD/insider — first close strictly after public disclosure).
"""

import argparse
import logging
import os
import sqlite3
from datetime import date, timedelta

import pandas as pd

from src.cache import init_db
from src.edgar_fts import search
from src.event_backtest import compute_abnormal_returns, summarize
from src.insider_backtest import split_half_summary
from src.pead_backtest import (
    collect_earnings_events,
    event_date_from_announcement,
    load_universe,
)

logger = logging.getLogger(__name__)

DB_PATH = "data/cache.db"
BENCHMARK = "SPY"
HORIZONS = (5, 10, 20)
GATE_HORIZONS = (5, 20)
DEDUPE_DAYS = 20            # min gap between successive events for one issuer
EARNINGS_CONTAMINATION_DAYS = 3
LOOKBACK_YEARS = 3
CONTROL_CAP_MULTIPLE = 3    # cap control raw hits at ~3x signal N (idea #5 precedent)
CONTROL_SAMPLE_MONTHS = 12  # spread the cap across the full lookback, not clustered

SIGNAL_PHRASE = "Schedule 13D"
SIGNAL_FORM_FILTER = "SCHEDULE 13D"    # root_forms value (matches / and /A)
SIGNAL_FORM_EXACT = "SCHEDULE 13D"     # exact form string: initial only
SIGNAL_CATEGORY = "sc13d_new"

CONTROL_PHRASE = "Schedule 13G"
CONTROL_FORM_FILTER = "SCHEDULE 13G"
CONTROL_FORM_EXACT = "SCHEDULE 13G"
CONTROL_CATEGORY = "sc13g_new"


def fetch_raw_events(phrase: str, form_filter: str, form_exact: str,
                     start: str, end: str, category: str,
                     db_path: str = DB_PATH, max_raw_hits: int | None = None,
                     sample_months: int | None = None) -> pd.DataFrame:
    """FTS hits -> initial-filing-only events with event_date set. Does not
    yet apply universe/dedup/contamination filters (dry-run count needs the
    raw hit count before those, per spec step 8)."""
    hits = search(phrase, form_filter, start, end, db_path=db_path,
                 max_raw_hits=max_raw_hits, sample_months=sample_months)
    if hits.empty:
        return hits
    hits = hits[hits["form"] == form_exact].copy()
    hits["category"] = category
    hits["event_date"] = hits["file_date"].map(event_date_from_announcement)
    hits["file_date"] = pd.to_datetime(hits["file_date"])
    return hits.reset_index(drop=True)


def filter_to_universe(events: pd.DataFrame, uni: pd.DataFrame) -> pd.DataFrame:
    """CIK-join to the universe (this IS the >=$2.5B universe filter, per
    shared recipe step 2). `uni` must have `cik` (zero-padded 10) and
    `ticker` columns."""
    if events.empty:
        return events
    merged = events.merge(uni[["cik", "ticker"]], on="cik", how="inner")
    return merged.reset_index(drop=True)


def dedupe_events(events: pd.DataFrame, dedupe_days: int = DEDUPE_DAYS) -> pd.DataFrame:
    """One event per ticker per `dedupe_days`; first (earliest) wins."""
    if events.empty:
        return events
    kept = []
    for ticker, g in events.sort_values("file_date").groupby("ticker"):
        last_kept = None
        for _, row in g.iterrows():
            if last_kept is not None and (row["file_date"] - last_kept).days < dedupe_days:
                continue
            kept.append(row)
            last_kept = row["file_date"]
    return pd.DataFrame(kept).reset_index(drop=True) if kept else events.iloc[0:0]


def drop_earnings_contamination(events: pd.DataFrame, earnings: pd.DataFrame,
                                window_days: int = EARNINGS_CONTAMINATION_DAYS) -> pd.DataFrame:
    """Drop events within `window_days` calendar days of that ticker's
    nearest earnings announcement — otherwise the category measures earnings
    reaction, not the 13D/13G disclosure itself."""
    if events.empty or earnings.empty:
        return events
    ann_by_ticker = {}
    for ticker, g in earnings.groupby("ticker"):
        ann_by_ticker[ticker] = pd.to_datetime(g["announce_date"]).tolist()

    keep = []
    for _, row in events.iterrows():
        dates = ann_by_ticker.get(row["ticker"])
        if not dates:
            keep.append(True)
            continue
        ev_ts = pd.Timestamp(row["event_date"])
        contaminated = any(abs((ev_ts - d).days) <= window_days for d in dates)
        keep.append(not contaminated)
    return events[pd.Series(keep, index=events.index)].reset_index(drop=True)


def build_events(raw: pd.DataFrame, uni: pd.DataFrame, earnings: pd.DataFrame) -> pd.DataFrame:
    events = filter_to_universe(raw, uni)
    events = dedupe_events(events)
    events = drop_earnings_contamination(events, earnings)
    return events


def _load_earnings_for_tickers(tickers: set, db_path: str = DB_PATH) -> pd.DataFrame:
    """Targeted earnings-date checkpoint: fetch only tickers that actually
    appear as 13D/13G subjects (cheap) rather than the whole universe."""
    if not tickers:
        return pd.DataFrame(columns=["ticker", "announce_date"])
    collect_earnings_events(sorted(tickers), db_path=db_path)
    con = sqlite3.connect(db_path)
    placeholders = ",".join("?" for _ in tickers)
    df = pd.read_sql_query(
        f"SELECT ticker, announce_date FROM pead_earnings WHERE ticker IN ({placeholders})",
        con, params=list(tickers))
    con.close()
    return df


def main():
    ap = argparse.ArgumentParser(description="Schedule 13D activist backtest (registry idea #1)")
    ap.add_argument("--years", type=int, default=LOOKBACK_YEARS)
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--tag", default="")
    ap.add_argument("--dry-run-only", action="store_true",
                    help="only count in-universe events, do not fetch prices")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    os.makedirs(args.out_dir, exist_ok=True)
    init_db(DB_PATH)

    end = date.today().isoformat()
    start = (date.today() - timedelta(days=365 * args.years)).isoformat()

    uni = load_universe()
    uni_cik = uni.merge(pd.read_parquet("data/universe.parquet")[["ticker", "cik"]],
                        on="ticker", how="left")
    print(f"[sc13d] universe: {len(uni_cik)} tickers >= $2.5B", flush=True)

    print("[sc13d] fetching Schedule 13D hits from EDGAR FTS...", flush=True)
    raw_signal = fetch_raw_events(SIGNAL_PHRASE, SIGNAL_FORM_FILTER, SIGNAL_FORM_EXACT,
                                  start, end, SIGNAL_CATEGORY)
    sig_uni = filter_to_universe(raw_signal, uni_cik)
    years = max(args.years, 1)
    n_signal = len(sig_uni)
    print(f"[sc13d] dry-run signal count: {n_signal} in-universe "
          f"({n_signal / years:.1f}/yr)", flush=True)

    # Gate on signal only (spec step 8) — the control fetch (Schedule 13G) is
    # far larger (unrestricted phrase match, ~5-8k hits/month) and is only
    # needed if signal survives; fetching it unconditionally wastes minutes
    # of EDGAR requests on a dry-run that dies on signal alone.
    if n_signal < 10 * years or n_signal < 50:
        print(f"[sc13d] KILLED at dry-run: signal count too thin "
              f"(need >=10/yr and >=50 total, got {n_signal} total over {years}y)")
        return

    if args.dry_run_only:
        print("[sc13d] dry-run-only requested, stopping before price fetch")
        return

    control_cap = n_signal * CONTROL_CAP_MULTIPLE
    print(f"[sc13d] fetching Schedule 13G hits from EDGAR FTS "
          f"(capped ~{control_cap} raw hits across {CONTROL_SAMPLE_MONTHS} sampled months)...",
          flush=True)
    raw_control = fetch_raw_events(CONTROL_PHRASE, CONTROL_FORM_FILTER, CONTROL_FORM_EXACT,
                                   start, end, CONTROL_CATEGORY,
                                   max_raw_hits=control_cap, sample_months=CONTROL_SAMPLE_MONTHS)
    ctl_uni = filter_to_universe(raw_control, uni_cik)
    n_control = len(ctl_uni)
    print(f"[sc13d] control count: {n_control} in-universe (sampled window)", flush=True)

    all_tickers = set(sig_uni["ticker"]) | set(ctl_uni["ticker"])
    earnings = _load_earnings_for_tickers(all_tickers)

    events_signal = build_events(raw_signal, uni_cik, earnings)
    events_control = build_events(raw_control, uni_cik, earnings)
    events = pd.concat([events_signal, events_control], ignore_index=True)
    print(f"[sc13d] {len(events)} events after dedup+contamination filter "
          f"({events['category'].value_counts().to_dict() if not events.empty else {}})",
          flush=True)
    if events.empty:
        return

    events = compute_abnormal_returns(events, DB_PATH, horizons=HORIZONS, benchmark=BENCHMARK)
    stem = date.today().isoformat() + (f"_{args.tag}" if args.tag else "")
    events.to_csv(os.path.join(args.out_dir, f"sc13d_events_{stem}.csv"), index=False)

    summary = summarize(events, horizons=HORIZONS, gate_horizons=GATE_HORIZONS)
    summary.to_csv(os.path.join(args.out_dir, f"sc13d_summary_{stem}.csv"), index=False)
    print(summary.to_string(index=False))

    halves = split_half_summary(events.rename(columns={"file_date": "filing_date"}))
    halves.to_csv(os.path.join(args.out_dir, f"sc13d_halves_{stem}.csv"), index=False)
    print(halves.to_string(index=False))


if __name__ == "__main__":
    main()
