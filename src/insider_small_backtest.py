"""Insider cluster-buying backtest, $500M-$2.5B band — registry Q10, IDEA BANK 3.

Registry: docs/strategy-registry.md Q10. Fit: docs/handoffs/FIT-2026-07-03-v2.md.

Q1 (same detection, $2.5B+ universe) was killed by an out-of-sample
confirmation FAIL (ledger #3-4). The published cluster-buying effect
concentrates in small caps; the v2 band change is a universe change, not a
same-universe variant. Detection, cluster window, dedupe and 10b5-1
exclusion are IDENTICAL to src/insider_backtest.py — only the universe,
band/liquidity filters, horizons and net-of-cost gate differ, all
pre-registered in the registry Q10 section before this file was written.

Data: SEC Form 345 quarterly zips already on disk (data/insider/ has
2024q3-2026q1, data/insider/confirm/ has 2020q3-2024q2). The Q10 run reads
a dedicated directory (data/insider_q10/) holding copies of the quarters
inside the 3-year lookback, so Q1's directories stay untouched.
"""

import argparse
import logging
import os
from datetime import date

import pandas as pd

from src.backtest_recipe import (
    as_net_frame,
    attach_cap_proxy,
    attach_dollar_vol,
    attach_net_returns,
    drop_earnings_contamination,
    filter_cap_proxy,
    filter_dollar_vol,
    load_earnings_for_tickers,
)
from src.cache import init_db
from src.event_backtest import compute_abnormal_returns, get_history_bulk, summarize
from src.insider_backtest import build_events, load_all_quarters
from src.pead_backtest import load_universe

logger = logging.getLogger(__name__)

DB_PATH = "data/cache.db"
DATA_DIR = "data/insider_q10"
BENCHMARK = "SPY"
HORIZONS = (5, 20, 40)
GATE_HORIZONS = (20, 40)      # pre-declared: cluster effect front-loaded ~1 month
LOOKBACK_DAYS = 1095          # 3 years

# Pre-registered in docs/strategy-registry.md Q10 — do not tune after runs.
MIN_CAP = 5e8
MAX_CAP = 2.5e9
MIN_DOLLAR_VOL = 2e6
NET_GATE_MIN = 0.01

SIGNAL_CATEGORY = "insider_cluster_buy"
CONTROL_CATEGORIES = ("insider_cluster_sell", "insider_single_buy")


def filter_lookback(events: pd.DataFrame, today: pd.Timestamp | None = None,
                    lookback_days: int = LOOKBACK_DAYS) -> pd.DataFrame:
    """Quarterly zips cover whole quarters; enforce the exact 3y window."""
    if events.empty:
        return events
    today = today or pd.Timestamp(date.today())
    cutoff = today - pd.Timedelta(days=lookback_days)
    return events[events["filing_date"] >= cutoff].reset_index(drop=True)


def split_half(events: pd.DataFrame) -> pd.DataFrame:
    """PASS requirement #3, computed on the NET frame (gate basis)."""
    mid = events["filing_date"].quantile(0.5)
    frames = []
    for name, half in (("first_half", events[events["filing_date"] <= mid]),
                       ("second_half", events[events["filing_date"] > mid])):
        s = summarize(as_net_frame(half, HORIZONS), horizons=HORIZONS,
                      gate_horizons=GATE_HORIZONS, min_abs_return=NET_GATE_MIN)
        if not s.empty:
            s.insert(0, "half", name)
            frames.append(s)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser(description="Insider cluster $500M-$2.5B backtest (registry Q10)")
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--dry-run-only", action="store_true",
                    help="stop after post-contamination counts (band approximated by CURRENT cap)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    os.makedirs(args.out_dir, exist_ok=True)
    init_db(DB_PATH)

    # Universe: everything currently >= $500M; cap_proxy decides band per event.
    uni = load_universe(min_cap=MIN_CAP)
    in_band_now = uni[uni["market_cap"] < MAX_CAP]
    print(f"[insider_small] universe: {len(uni)} tickers >= ${MIN_CAP/1e9:.1f}B "
          f"({len(in_band_now)} currently in band)", flush=True)

    records = load_all_quarters(args.data_dir)
    events = build_events(records, set(uni["ticker"]))
    events = filter_lookback(events)
    print(f"[insider_small] {len(events)} events in lookback "
          f"({events['category'].value_counts().to_dict() if not events.empty else {}})", flush=True)
    if events.empty:
        print("[insider_small] no events — nothing to summarize")
        return

    # Earnings contamination folded into the dry-run count (ledger #7 fix).
    earnings = load_earnings_for_tickers(set(events["ticker"]), DB_PATH)
    events = drop_earnings_contamination(events, earnings)
    print(f"[insider_small] post-contamination: {len(events)} "
          f"({events['category'].value_counts().to_dict()})", flush=True)

    if args.dry_run_only:
        approx = events[events["ticker"].isin(set(in_band_now["ticker"]))]
        span_yrs = max((approx["filing_date"].max()
                        - approx["filing_date"].min()).days / 365.25, 1e-9)
        counts = approx.groupby("category").size().rename("n").reset_index()
        counts["per_year"] = (counts["n"] / span_yrs).round(1)
        print(counts.to_string(index=False))
        print("[insider_small] dry-run: band approximated by CURRENT cap (true "
              "gate uses cap_proxy at event; exact counts printed in the full "
              "run). Kill bar: signal < 10/yr or < 50 total.", flush=True)
        return

    start = (events["filing_date"].min() - pd.Timedelta(days=60)).date().isoformat()
    end = (events["filing_date"].max() + pd.Timedelta(days=120)).date().isoformat()
    tickers = list(events["ticker"].unique())
    price_cache = get_history_bulk([*tickers, BENCHMARK], DB_PATH, start, end)

    n0 = len(events)
    events = attach_cap_proxy(events, uni, price_cache)
    events = filter_cap_proxy(events, min_cap=MIN_CAP, max_cap=MAX_CAP)
    n1 = len(events)
    events = attach_dollar_vol(events, price_cache)
    events = filter_dollar_vol(events, min_dollar_vol=MIN_DOLLAR_VOL)
    n2 = len(events)
    print(f"[insider_small] band filter {n0} -> {n1}; dollar-vol filter -> {n2}", flush=True)
    if events.empty:
        print("[insider_small] no in-band events — nothing to summarize")
        return

    span_yrs = max((events["filing_date"].max()
                    - events["filing_date"].min()).days / 365.25, 1e-9)
    n_signal = int((events["category"] == SIGNAL_CATEGORY).sum())
    print(f"[insider_small] exact post-filter counts: signal={n_signal} "
          f"({n_signal/span_yrs:.1f}/yr)", flush=True)

    events = compute_abnormal_returns(events, DB_PATH, horizons=HORIZONS,
                                      benchmark=BENCHMARK)
    events = attach_net_returns(events, HORIZONS)

    today = date.today().isoformat()
    events.to_csv(os.path.join(args.out_dir, f"insider_small_events_{today}.csv"), index=False)

    print("\n=== GROSS abnormal returns (comparability with bank 1-2) ===")
    gross = summarize(events, horizons=HORIZONS, gate_horizons=GATE_HORIZONS)
    print(gross.to_string(index=False))

    print(f"\n=== NET of cost model — THE GATE (>= {NET_GATE_MIN:.1%} net) ===")
    net = summarize(as_net_frame(events, HORIZONS), horizons=HORIZONS,
                    gate_horizons=GATE_HORIZONS, min_abs_return=NET_GATE_MIN)
    print(net.to_string(index=False))
    net.to_csv(os.path.join(args.out_dir, f"insider_small_summary_{today}.csv"), index=False)

    print("\n=== Split-half stability (net basis) ===")
    halves = split_half(events)
    print(halves.to_string(index=False))
    halves.to_csv(os.path.join(args.out_dir, f"insider_small_halves_{today}.csv"), index=False)


if __name__ == "__main__":
    main()
