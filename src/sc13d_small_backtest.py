"""Schedule 13D activist-stake backtest, $500M-$2.5B band — registry Q11, IDEA BANK 3.

Registry: docs/strategy-registry.md Q11. Fit: docs/handoffs/FIT-2026-07-03-v2.md.

Q2 (same detection, $2.5B+ universe) was a clean gate FAIL with a
split-half sign flip (ledger #5). Activists concentrate in $500M-$10B and
the published announcement drift is strongest in small/mid caps — the v2
band is the favorable slice the $2.5B floor excluded. Detection (EDGAR FTS,
initial filings only), 13G control design, control cap (3x signal, sampled
across 12 months), dedupe and contamination handling are IDENTICAL to
src/sc13d_backtest.py — only universe, band/liquidity filters, horizons
and the net-of-cost gate differ, all pre-registered in registry Q11.
"""

import argparse
import logging
import os
from datetime import date, timedelta

import pandas as pd

from src.backtest_recipe import (
    as_net_frame,
    attach_cap_proxy,
    attach_dollar_vol,
    attach_net_returns,
    filter_cap_proxy,
    filter_dollar_vol,
)
from src.cache import init_db
from src.event_backtest import compute_abnormal_returns, get_history_bulk, summarize
from src.pead_backtest import load_universe
from src.sc13d_backtest import (
    CONTROL_CATEGORY,
    CONTROL_FORM_EXACT,
    CONTROL_FORM_FILTER,
    CONTROL_PHRASE,
    CONTROL_SAMPLE_MONTHS,
    SIGNAL_CATEGORY,
    SIGNAL_FORM_EXACT,
    SIGNAL_FORM_FILTER,
    SIGNAL_PHRASE,
    _load_earnings_for_tickers,
    build_events,
    fetch_raw_events,
    filter_to_universe,
)

logger = logging.getLogger(__name__)

DB_PATH = "data/cache.db"
BENCHMARK = "SPY"
HORIZONS = (5, 20, 40)
GATE_HORIZONS = (5, 20, 40)   # pre-declared: activist drift is weeks-scale
LOOKBACK_YEARS = 3
CONTROL_CAP_MULTIPLE = 3

# Pre-registered in docs/strategy-registry.md Q11 — do not tune after runs.
MIN_CAP = 5e8
MAX_CAP = 2.5e9
MIN_DOLLAR_VOL = 2e6
NET_GATE_MIN = 0.01


def control_cap_for(n_signal: int) -> int:
    return n_signal * CONTROL_CAP_MULTIPLE


def split_half(events: pd.DataFrame) -> pd.DataFrame:
    """PASS requirement #3, computed on the NET frame (gate basis)."""
    if events.empty:
        return pd.DataFrame()
    mid = events["file_date"].quantile(0.5)
    frames = []
    for name, half in (("first_half", events[events["file_date"] <= mid]),
                       ("second_half", events[events["file_date"] > mid])):
        s = summarize(as_net_frame(half, HORIZONS), horizons=HORIZONS,
                      gate_horizons=GATE_HORIZONS, min_abs_return=NET_GATE_MIN)
        if not s.empty:
            s.insert(0, "half", name)
            frames.append(s)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser(description="Schedule 13D $500M-$2.5B backtest (registry Q11)")
    ap.add_argument("--years", type=int, default=LOOKBACK_YEARS)
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--dry-run-only", action="store_true",
                    help="only count in-universe signal events, do not fetch control/prices")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    os.makedirs(args.out_dir, exist_ok=True)
    init_db(DB_PATH)

    end = date.today().isoformat()
    start = (date.today() - timedelta(days=365 * args.years)).isoformat()

    # Universe: everything currently >= $500M; cap_proxy decides band per event.
    uni = load_universe(min_cap=MIN_CAP)
    in_band_now = uni[uni["market_cap"] < MAX_CAP]
    uni_cik = uni.merge(pd.read_parquet("data/universe.parquet")[["ticker", "cik"]],
                        on="ticker", how="left")
    print(f"[sc13d_small] universe: {len(uni_cik)} tickers >= ${MIN_CAP/1e9:.1f}B "
          f"({len(in_band_now)} currently in band)", flush=True)

    print("[sc13d_small] fetching Schedule 13D hits from EDGAR FTS...", flush=True)
    raw_signal = fetch_raw_events(SIGNAL_PHRASE, SIGNAL_FORM_FILTER, SIGNAL_FORM_EXACT,
                                  start, end, SIGNAL_CATEGORY)
    sig_uni = filter_to_universe(raw_signal, uni_cik)
    sig_band_now = sig_uni[sig_uni["ticker"].isin(set(in_band_now["ticker"]))]
    years = max(args.years, 1)
    n_signal = len(sig_band_now)
    print(f"[sc13d_small] dry-run signal count (band approximated by CURRENT "
          f"cap): {n_signal} ({n_signal / years:.1f}/yr); "
          f"{len(sig_uni)} across the whole >=$500M scan", flush=True)

    if n_signal < 10 * years or n_signal < 50:
        print(f"[sc13d_small] KILLED at dry-run: signal count too thin "
              f"(need >=10/yr and >=50 total, got {n_signal} over {years}y)")
        return

    if args.dry_run_only:
        print("[sc13d_small] dry-run-only requested, stopping before control/price fetch")
        return

    control_cap = control_cap_for(len(sig_uni))
    print(f"[sc13d_small] fetching Schedule 13G hits (capped ~{control_cap} raw "
          f"hits across {CONTROL_SAMPLE_MONTHS} sampled months)...", flush=True)
    raw_control = fetch_raw_events(CONTROL_PHRASE, CONTROL_FORM_FILTER, CONTROL_FORM_EXACT,
                                   start, end, CONTROL_CATEGORY,
                                   max_raw_hits=control_cap,
                                   sample_months=CONTROL_SAMPLE_MONTHS)
    ctl_uni = filter_to_universe(raw_control, uni_cik)
    print(f"[sc13d_small] control count: {len(ctl_uni)} in >=$500M scan", flush=True)

    all_tickers = set(sig_uni["ticker"]) | set(ctl_uni["ticker"])
    earnings = _load_earnings_for_tickers(all_tickers)

    events_signal = build_events(raw_signal, uni_cik, earnings)
    events_control = build_events(raw_control, uni_cik, earnings)
    events = pd.concat([events_signal, events_control], ignore_index=True)
    print(f"[sc13d_small] {len(events)} events after dedup+contamination "
          f"({events['category'].value_counts().to_dict() if not events.empty else {}})",
          flush=True)
    if events.empty:
        return

    start_px = (events["file_date"].min() - pd.Timedelta(days=60)).date().isoformat()
    end_px = (events["file_date"].max() + pd.Timedelta(days=120)).date().isoformat()
    price_cache = get_history_bulk([*events["ticker"].unique(), BENCHMARK],
                                   DB_PATH, start_px, end_px)

    n0 = len(events)
    events = attach_cap_proxy(events, uni, price_cache)
    events = filter_cap_proxy(events, min_cap=MIN_CAP, max_cap=MAX_CAP)
    n1 = len(events)
    events = attach_dollar_vol(events, price_cache)
    events = filter_dollar_vol(events, min_dollar_vol=MIN_DOLLAR_VOL)
    n2 = len(events)
    print(f"[sc13d_small] band filter {n0} -> {n1}; dollar-vol filter -> {n2}", flush=True)
    if events.empty:
        print("[sc13d_small] no in-band events — nothing to summarize")
        return

    span_yrs = max((events["file_date"].max()
                    - events["file_date"].min()).days / 365.25, 1e-9)
    n_sig_final = int((events["category"] == SIGNAL_CATEGORY).sum())
    print(f"[sc13d_small] exact post-filter counts: signal={n_sig_final} "
          f"({n_sig_final/span_yrs:.1f}/yr)", flush=True)

    events = compute_abnormal_returns(events, DB_PATH, horizons=HORIZONS,
                                      benchmark=BENCHMARK)
    events = attach_net_returns(events, HORIZONS)

    today = date.today().isoformat()
    events.to_csv(os.path.join(args.out_dir, f"sc13d_small_events_{today}.csv"), index=False)

    print("\n=== GROSS abnormal returns (comparability with bank 1-2) ===")
    gross = summarize(events, horizons=HORIZONS, gate_horizons=GATE_HORIZONS)
    print(gross.to_string(index=False))

    print(f"\n=== NET of cost model — THE GATE (>= {NET_GATE_MIN:.1%} net) ===")
    net = summarize(as_net_frame(events, HORIZONS), horizons=HORIZONS,
                    gate_horizons=GATE_HORIZONS, min_abs_return=NET_GATE_MIN)
    print(net.to_string(index=False))
    net.to_csv(os.path.join(args.out_dir, f"sc13d_small_summary_{today}.csv"), index=False)

    print("\n=== Split-half stability (net basis) ===")
    halves = split_half(events)
    print(halves.to_string(index=False))
    halves.to_csv(os.path.join(args.out_dir, f"sc13d_small_halves_{today}.csv"), index=False)


if __name__ == "__main__":
    main()
