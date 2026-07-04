"""PEAD backtest, $500M-$2.5B band — registry Q9, IDEA BANK 3.

Registry: docs/strategy-registry.md Q9. Fit: docs/handoffs/FIT-2026-07-03-v2.md.

First idea after the v2 constraint relaxation (cap floor $500M + $2M ADV
liquidity leg, horizon to ~2 months, gate at 1.0% NET of the pre-registered
cost model). Original PEAD (src/pead_backtest.py, ledger #2) died in the
$2.5B+ universe when the miss-bucket control exposed a universe-wide drift
(beat-vs-miss p=0.70). Hypothesis here is the published one: PEAD persists
where analyst coverage is thin and arbitrage is capacity-constrained —
exactly the band the $2.5B floor excluded.

Differences from the ledger #2 run, all pre-registered before this file
was written (registry Q9 section):
- Band membership is judged at EVENT time via cap_proxy in [$500M, $2.5B),
  not current cap alone (survivorship mitigation, IDEA BANK 2 preamble).
- Liquidity: 20-day median dollar volume >= $2M strictly before the event.
- Horizons (5, 20, 40) with h40 declared (published PEAD tail ~60 trading
  days; the 2-month horizon is now in fit). Gate at all three.
- Gate runs on abn_ret NET of the band cost model (>= 1.0% net, p < 0.10,
  sign agreement); gross columns are reported alongside for comparability.
- PASS additionally requires the beat-vs-miss spread to be significant —
  the exact discriminator that killed ledger #2 — plus split-half
  stability, per the registry PASS requirements.

Surprise buckets, |estimate| floor, entry timing (+1 calendar day) and
benchmark (SPY) are unchanged from src/pead_backtest.py for comparability.
"""

import argparse
import logging
import os
from datetime import date

import pandas as pd
from scipy.stats import ttest_ind

from src.backtest_recipe import (
    attach_cap_proxy,
    attach_dollar_vol,
    attach_net_returns,
    filter_cap_proxy,
    filter_dollar_vol,
)
from src.cache import init_db
from src.event_backtest import compute_abnormal_returns, get_history_bulk, summarize
from src.pead_backtest import collect_earnings_events, load_events, load_universe

logger = logging.getLogger(__name__)

DB_PATH = "data/cache.db"
BENCHMARK = "SPY"
HORIZONS = (5, 20, 40)
GATE_HORIZONS = (5, 20, 40)   # h40 pre-declared for Q9 (registry)
LOOKBACK_DAYS = 1095          # 3 years

# Pre-registered in docs/strategy-registry.md Q9 — do not tune after runs.
MIN_CAP = 5e8
MAX_CAP = 2.5e9
MIN_DOLLAR_VOL = 2e6
NET_GATE_MIN = 0.01           # 1.0% net of band cost

SIGNAL_CATEGORIES = ("pead_beat_large", "pead_beat_mid", "pead_beat_small")
CONTROL_CATEGORY = "pead_miss"


def _as_net_frame(events: pd.DataFrame, horizons: tuple = HORIZONS) -> pd.DataFrame:
    """Frame whose abn_ret_{h}d columns hold NET values, so summarize()
    gates on net without modification."""
    net = events.copy()
    for h in horizons:
        net[f"abn_ret_{h}d"] = net[f"abn_ret_net_{h}d"]
    return net


def beat_vs_miss_spread(events: pd.DataFrame, horizons: tuple = HORIZONS) -> pd.DataFrame:
    """Welch t-test of beat buckets (pooled) vs the miss bucket per horizon,
    on GROSS abnormal returns (a spread nets out costs common to both legs).
    This discriminator is what exposed ledger #2's universe-wide drift."""
    beats = events[events["category"].isin(SIGNAL_CATEGORIES)]
    misses = events[events["category"] == CONTROL_CATEGORY]
    rows = []
    for h in horizons:
        b = beats[f"abn_ret_{h}d"].dropna().astype(float)
        m = misses[f"abn_ret_{h}d"].dropna().astype(float)
        if len(b) < 5 or len(m) < 5:
            rows.append({"horizon": h, "n_beat": len(b), "n_miss": len(m),
                         "beat_mean": None, "miss_mean": None,
                         "spread": None, "p_spread": None})
            continue
        _, p = ttest_ind(b, m, equal_var=False)
        rows.append({"horizon": h, "n_beat": len(b), "n_miss": len(m),
                     "beat_mean": float(b.mean()), "miss_mean": float(m.mean()),
                     "spread": float(b.mean() - m.mean()), "p_spread": float(p)})
    return pd.DataFrame(rows)


def split_half(events: pd.DataFrame) -> pd.DataFrame:
    """PASS requirement #3: sign/magnitude stability across sample halves,
    computed on the NET frame (same basis as the gate)."""
    mid = events["announce_date"].quantile(0.5)
    frames = []
    for name, half in (("first_half", events[events["announce_date"] <= mid]),
                       ("second_half", events[events["announce_date"] > mid])):
        s = summarize(_as_net_frame(half), horizons=HORIZONS,
                      gate_horizons=GATE_HORIZONS, min_abs_return=NET_GATE_MIN)
        if not s.empty:
            s.insert(0, "half", name)
            frames.append(s)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser(description="PEAD $500M-$2.5B backtest (registry Q9)")
    ap.add_argument("--days", type=int, default=LOOKBACK_DAYS)
    ap.add_argument("--limit", type=int, default=0, help="cap universe size (debug)")
    ap.add_argument("--skip-fetch", action="store_true",
                    help="use already-checkpointed earnings rows only")
    ap.add_argument("--dry-run-only", action="store_true",
                    help="stop after event counts (band approximated by CURRENT cap)")
    ap.add_argument("--out-dir", default="output")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    os.makedirs(args.out_dir, exist_ok=True)
    init_db(DB_PATH)

    # Universe: everything currently >= $500M. Names now ABOVE $2.5B stay in
    # the scan — a name at $4B today may have been in-band at event time;
    # cap_proxy (not current cap) decides band membership per event.
    uni = load_universe(min_cap=MIN_CAP)
    if args.limit:
        uni = uni.head(args.limit)
    in_band_now = uni[uni["market_cap"] < MAX_CAP]
    print(f"[pead_small] universe: {len(uni)} tickers >= ${MIN_CAP/1e9:.1f}B "
          f"({len(in_band_now)} currently in band)", flush=True)

    if not args.skip_fetch:
        collect_earnings_events(list(uni["ticker"]))

    events = load_events(lookback_days=args.days)
    events = events[events["ticker"].isin(set(uni["ticker"]))].reset_index(drop=True)
    print(f"[pead_small] {len(events)} events in last {args.days}d (pre-band)", flush=True)
    if events.empty:
        print("[pead_small] no events — nothing to summarize")
        return

    if args.dry_run_only:
        approx = events[events["ticker"].isin(set(in_band_now["ticker"]))]
        span_yrs = max((pd.Timestamp(approx["announce_date"].max())
                        - pd.Timestamp(approx["announce_date"].min())).days / 365.25, 1e-9)
        counts = approx.groupby("category").size().rename("n").reset_index()
        counts["per_year"] = (counts["n"] / span_yrs).round(1)
        print(counts.to_string(index=False))
        print("[pead_small] dry-run: band approximated by CURRENT cap (true gate "
              "uses cap_proxy at event; exact counts printed in the full run). "
              "Kill bar: signal < 10/yr or < 50 total.", flush=True)
        return

    # Price histories once, reused by cap_proxy, dollar-vol and returns.
    start = (pd.Timestamp(events["event_date"].min()) - pd.Timedelta(days=60)).date().isoformat()
    end = (pd.Timestamp(events["event_date"].max()) + pd.Timedelta(days=120)).date().isoformat()
    tickers = list(events["ticker"].unique())
    price_cache = get_history_bulk([*tickers, BENCHMARK], DB_PATH, start, end)

    n0 = len(events)
    events = attach_cap_proxy(events, uni, price_cache)
    events = filter_cap_proxy(events, min_cap=MIN_CAP, max_cap=MAX_CAP)
    n1 = len(events)
    events = attach_dollar_vol(events, price_cache)
    events = filter_dollar_vol(events, min_dollar_vol=MIN_DOLLAR_VOL)
    n2 = len(events)
    print(f"[pead_small] band filter {n0} -> {n1}; dollar-vol filter -> {n2}", flush=True)
    if events.empty:
        print("[pead_small] no in-band events — nothing to summarize")
        return

    span_yrs = max((pd.Timestamp(events["announce_date"].max())
                    - pd.Timestamp(events["announce_date"].min())).days / 365.25, 1e-9)
    counts = events.groupby("category").size()
    n_signal = int(counts.reindex(SIGNAL_CATEGORIES).fillna(0).sum())
    print(f"[pead_small] exact post-filter counts: signal={n_signal} "
          f"({n_signal/span_yrs:.1f}/yr), miss={int(counts.get(CONTROL_CATEGORY, 0))}", flush=True)

    events = compute_abnormal_returns(events, DB_PATH, horizons=HORIZONS,
                                      benchmark=BENCHMARK)
    events = attach_net_returns(events, HORIZONS)

    today = date.today().isoformat()
    events.to_csv(os.path.join(args.out_dir, f"pead_small_events_{today}.csv"), index=False)

    print("\n=== GROSS abnormal returns (comparability with bank 1-2) ===")
    gross = summarize(events, horizons=HORIZONS, gate_horizons=GATE_HORIZONS)
    print(gross.to_string(index=False))

    print(f"\n=== NET of cost model — THE GATE (>= {NET_GATE_MIN:.1%} net) ===")
    net = summarize(_as_net_frame(events), horizons=HORIZONS,
                    gate_horizons=GATE_HORIZONS, min_abs_return=NET_GATE_MIN)
    print(net.to_string(index=False))
    net.to_csv(os.path.join(args.out_dir, f"pead_small_summary_{today}.csv"), index=False)

    print("\n=== Beat-vs-miss spread (required discriminator, gross) ===")
    spread = beat_vs_miss_spread(events)
    print(spread.to_string(index=False))
    spread.to_csv(os.path.join(args.out_dir, f"pead_small_spread_{today}.csv"), index=False)

    print("\n=== Split-half stability (net basis) ===")
    halves = split_half(events)
    print(halves.to_string(index=False))
    halves.to_csv(os.path.join(args.out_dir, f"pead_small_halves_{today}.csv"), index=False)


if __name__ == "__main__":
    main()
