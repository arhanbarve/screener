"""Intra-industry lead-lag spillover — idea C (registry Q7, IDEA BANK 2).

Registry: docs/strategy-registry.md Q7. Fit: docs/handoffs/FIT-2026-07-03.md.

Hypothesis: information diffuses gradually across economically linked
names (Cohen-Frazzini; Hou). When a leader in a yfinance industry group
moves >= +7% vs SPY in one day, the largest same-industry peer that did
NOT move (|1d abnormal| <= 2%) and has no news of its own drifts up over
the following days-weeks. Control: identical construction off leaders that
moved <= -7% (leadlag_down) — long-only cannot trade it, it exists to show
the effect is directional rather than "any industry excitement".

Honest prior (flagged in the pivot analysis): the published effect
concentrates in SMALL laggards; both legs >= $2.5B is the least favorable
slice. This is idea C, run only after A and B resolve.
"""

import argparse
import logging
import os
import time
from datetime import date, timedelta

import pandas as pd
import yfinance as yf

from src.backtest_recipe import (
    attach_cap_proxy,
    dedupe_events,
    filter_cap_proxy,
    load_earnings_for_tickers,
    split_by_earnings,
    split_by_news,
)
from src.cache import get_ticker_profile, init_db, put_ticker_profile
from src.event_backtest import compute_abnormal_returns, get_history_bulk, summarize
from src.insider_backtest import split_half_summary
from src.pead_backtest import event_date_from_announcement, load_universe
from src.reversal_backtest import cap_per_day, load_filing_dates

logger = logging.getLogger(__name__)

DB_PATH = "data/cache.db"
BENCHMARK = "SPY"
HORIZONS = (5, 10, 20)
GATE_HORIZONS = (5, 20)
LOOKBACK_YEARS = 3

# Pre-registered in docs/strategy-registry.md Q7 — do not tune after runs.
LEADER_MOVE = 0.07
LAGGARD_MAX_MOVE = 0.02
MIN_INDUSTRY_PEERS = 3
NEWS_DAYS_BEFORE = 3
NEWS_DAYS_AFTER = 1
MIN_CAP_PROXY = 2.5e9
PROFILE_SLEEP_S = 0.3

CATEGORY_SIGNAL = "leadlag_up"
CATEGORY_CONTROL = "leadlag_down"


def collect_profiles(tickers: list[str], db_path: str = DB_PATH,
                     sleep_s: float = PROFILE_SLEEP_S) -> dict:
    """{ticker: industry} via yfinance .get_info(), checkpointed in sqlite.
    One-time ~0.3s/ticker cost for the universe; instant afterwards.
    Failed lookups cache as empty so they are not refetched every run."""
    out = {}
    todo = []
    for t in tickers:
        prof = get_ticker_profile(db_path, t)
        if prof is None:
            todo.append(t)
        elif prof["industry"]:
            out[t] = prof["industry"]
    print(f"[leadlag] profiles: {len(out)} cached, {len(todo)} to fetch", flush=True)
    for i, t in enumerate(todo):
        sector, industry = None, None
        try:
            info = yf.Ticker(t).get_info()
            sector = info.get("sector")
            industry = info.get("industry")
        except Exception as e:
            logger.warning(f"[leadlag] profile fetch failed for {t}: {e}")
        put_ticker_profile(db_path, t, sector, industry)
        if industry:
            out[t] = industry
        if (i + 1) % 100 == 0:
            print(f"[leadlag] profile fetch {i + 1}/{len(todo)}", flush=True)
        time.sleep(sleep_s)
    return out


def detect_leadlag(prices_by_ticker: dict, bench: pd.DataFrame,
                   industry_by_ticker: dict, caps_by_ticker: dict,
                   start, end,
                   leader_move: float = LEADER_MOVE,
                   laggard_max_move: float = LAGGARD_MAX_MOVE,
                   min_peers: int = MIN_INDUSTRY_PEERS) -> pd.DataFrame:
    """Pure. One laggard event max per (industry, day): the largest-cap
    same-industry peer with |1d abnormal| <= laggard_max_move on a day when
    some peer moved beyond +/-leader_move. Industry-days with leaders in
    BOTH directions are skipped as ambiguous. Returns columns: ticker,
    trigger_date, leader, leader_ret, direction ('up'|'down'), z (leader
    abnormal ret, kept under the name z so cap_per_day/dedupe reuse works)."""
    b_ret1 = bench["close"].pct_change()
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)

    abn = {}
    for t, df in prices_by_ticker.items():
        if df is None or df.empty or t not in industry_by_ticker:
            continue
        idx = df.index.intersection(b_ret1.index)
        if len(idx) < 10:
            continue
        abn[t] = df.loc[idx, "close"].pct_change() - b_ret1.loc[idx]
    if not abn:
        return pd.DataFrame(columns=["ticker", "trigger_date", "leader",
                                     "leader_ret", "direction", "z"])
    abn_df = pd.DataFrame(abn)

    members_by_industry = {}
    for t in abn_df.columns:
        members_by_industry.setdefault(industry_by_ticker[t], []).append(t)

    rows = []
    for day in abn_df.index:
        if not (start_ts <= day <= end_ts):
            continue
        day_ret = abn_df.loc[day].dropna()
        movers = day_ret[day_ret.abs() >= leader_move]
        if movers.empty:
            continue
        for industry, members in members_by_industry.items():
            if len(members) < min_peers:
                continue
            leaders = [t for t in movers.index if industry_by_ticker[t] == industry]
            if not leaders:
                continue
            has_up = any(day_ret[t] >= leader_move for t in leaders)
            has_down = any(day_ret[t] <= -leader_move for t in leaders)
            if has_up and has_down:
                continue  # ambiguous industry-day, pre-registered skip
            direction = "up" if has_up else "down"
            lead = max(leaders, key=lambda t: abs(day_ret[t]))
            laggards = [t for t in members
                        if t in day_ret.index and t not in leaders
                        and abs(day_ret[t]) <= laggard_max_move]
            if not laggards:
                continue
            lag = max(laggards, key=lambda t: caps_by_ticker.get(t, 0))
            rows.append({"ticker": lag, "trigger_date": day, "leader": lead,
                         "leader_ret": float(day_ret[lead]),
                         "direction": direction,
                         "z": float(day_ret[lead])})
    cols = ["ticker", "trigger_date", "leader", "leader_ret", "direction", "z"]
    return pd.DataFrame(rows, columns=cols)


def main():
    ap = argparse.ArgumentParser(description="Intra-industry lead-lag backtest (registry Q7)")
    ap.add_argument("--years", type=int, default=LOOKBACK_YEARS)
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--tag", default="")
    ap.add_argument("--dry-run-only", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    os.makedirs(args.out_dir, exist_ok=True)
    init_db(DB_PATH)

    end = date.today().isoformat()
    start = (date.today() - timedelta(days=365 * args.years)).isoformat()
    fetch_start = (date.today() - timedelta(days=365 * args.years + 30)).isoformat()

    uni = load_universe()
    uni_cik = uni.merge(pd.read_parquet("data/universe.parquet")[["ticker", "cik"]],
                        on="ticker", how="left")
    tickers = sorted(uni_cik["ticker"].unique())
    print(f"[leadlag] universe: {len(tickers)} tickers >= $2.5B", flush=True)

    industry_by_ticker = collect_profiles(tickers)
    n_ind = len(set(industry_by_ticker.values()))
    print(f"[leadlag] {len(industry_by_ticker)} tickers mapped to {n_ind} industries",
          flush=True)

    price_cache = get_history_bulk([*tickers, BENCHMARK], DB_PATH, fetch_start, end)
    bench = price_cache.get(BENCHMARK)
    if bench is None or bench.empty:
        raise RuntimeError("no benchmark history")

    caps_by_ticker = dict(zip(uni["ticker"], uni["market_cap"]))
    candidates = detect_leadlag({t: price_cache.get(t) for t in tickers}, bench,
                                industry_by_ticker, caps_by_ticker, start, end)
    print(f"[leadlag] raw laggard events: "
          f"{candidates['direction'].value_counts().to_dict() if not candidates.empty else {}}",
          flush=True)
    if candidates.empty:
        print("[leadlag] KILLED at dry-run: no events")
        return

    candidates = cap_per_day(candidates)
    parts = []
    for direction, g in candidates.groupby("direction"):
        g = g.assign(file_date=g["trigger_date"])
        parts.append(dedupe_events(g))
    candidates = pd.concat(parts, ignore_index=True)

    candidates["event_date"] = candidates["trigger_date"].dt.date.astype(str).map(
        event_date_from_announcement)
    candidates = attach_cap_proxy(candidates, uni, price_cache)
    candidates = filter_cap_proxy(candidates, MIN_CAP_PROXY)

    # laggard must have no news of its own — newsy laggards are DISCARDED
    # here (unlike idea A there is no news control; the control is the
    # leader's direction)
    earnings = load_earnings_for_tickers(set(candidates["ticker"]), DB_PATH)
    filing_dates = load_filing_dates(candidates, uni_cik, 365 * args.years + 30)
    clean_earn, _ = split_by_earnings(candidates, earnings)
    clean, _ = split_by_news(clean_earn, filing_dates,
                             days_before=NEWS_DAYS_BEFORE, days_after=NEWS_DAYS_AFTER)

    clean = clean.assign(
        category=clean["direction"].map({"up": CATEGORY_SIGNAL,
                                         "down": CATEGORY_CONTROL}))
    n_signal = len(clean[clean["category"] == CATEGORY_SIGNAL])
    years = max(args.years, 1)
    print(f"[leadlag] dry-run signal count ({CATEGORY_SIGNAL}): {n_signal} "
          f"({n_signal / years:.1f}/yr); control "
          f"{CATEGORY_CONTROL}={len(clean) - n_signal}", flush=True)

    if n_signal < 10 * years or n_signal < 50:
        print(f"[leadlag] KILLED at dry-run: signal count too thin "
              f"(need >=10/yr and >=50 total, got {n_signal} over {years}y)")
        return
    if args.dry_run_only:
        print("[leadlag] dry-run-only requested, stopping before price gate")
        return

    events = compute_abnormal_returns(clean, DB_PATH, horizons=HORIZONS,
                                      benchmark=BENCHMARK)
    stem = date.today().isoformat() + (f"_{args.tag}" if args.tag else "")
    events.to_csv(os.path.join(args.out_dir, f"leadlag_events_{stem}.csv"), index=False)

    summary = summarize(events, horizons=HORIZONS, gate_horizons=GATE_HORIZONS)
    summary.to_csv(os.path.join(args.out_dir, f"leadlag_summary_{stem}.csv"), index=False)
    print(summary.to_string(index=False))

    halves = split_half_summary(events.rename(columns={"trigger_date": "filing_date"}))
    halves.to_csv(os.path.join(args.out_dir, f"leadlag_halves_{stem}.csv"), index=False)
    print(halves.to_string(index=False))


if __name__ == "__main__":
    main()
