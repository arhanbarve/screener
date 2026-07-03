"""Idiosyncratic no-news drawdown reversal — idea A (registry Q5, IDEA BANK 2).

Registry: docs/strategy-registry.md Q5. Fit: docs/handoffs/FIT-2026-07-03.md.

First non-disclosure-event idea after the corporate-event bank went 0-for-8.
Hypothesis is a liquidity risk premium, not an information edge: a 5-day
residual (vs SPY) drop that is both large (<= -7%) and abnormal for the name
(<= -2.5 sigma of its own daily residual vol) WITHOUT any news event
(earnings +/-3d, 8-K/6-K in [-3, +1]) is flow-driven and mean-reverts.
Drops WITH news are the discriminating control (rev_drop_news): if they
bounce just as hard, the "no-news" conditioning is doing nothing and the
signal is not the liquidity mechanism. Symmetric no-news up-moves
(rev_spike_nonews) are the second control (expect nothing).

Simple SPY subtraction (beta=1 assumption) is deliberate: the z-score is
taken against the name's OWN residual vol, so persistently-high-beta names
have persistently wide residual vol and do not trigger spuriously.

Vol window ends RET_WINDOW days before the trigger so the move being tested
does not inflate its own denominator.

Entry follows the harness convention (event_date = trigger + 1 calendar
day; t0 = first close on/after that): the measured window starts at the
close of the day AFTER the trigger, so any same/next-day bounce a scanner
could not have captured is excluded.
"""

import argparse
import logging
import os
from datetime import date, timedelta

import pandas as pd

from src.backtest_recipe import (
    attach_cap_proxy,
    dedupe_events,
    filter_cap_proxy,
    load_earnings_for_tickers,
    split_by_earnings,
    split_by_news,
)
from src.cache import init_db
from src.event_backtest import (
    compute_abnormal_returns,
    get_history_bulk,
    list_filings,
    summarize,
)
from src.insider_backtest import split_half_summary
from src.pead_backtest import event_date_from_announcement, load_universe

logger = logging.getLogger(__name__)

DB_PATH = "data/cache.db"
BENCHMARK = "SPY"
HORIZONS = (5, 10, 20)
GATE_HORIZONS = (5, 20)
LOOKBACK_YEARS = 3

# Pre-registered in docs/strategy-registry.md Q5 — do not tune after runs.
RET_WINDOW = 5          # trading days for the dislocation move
VOL_WINDOW = 60         # trading days of daily residuals for the z denominator
Z_TRIGGER = 2.5
RESID_FLOOR = 0.07
MAX_PER_DAY = 5         # per calendar day per direction, most-extreme |z| kept
NEWS_DAYS_BEFORE = 3
NEWS_DAYS_AFTER = 1
NEWS_FORMS = ("8-K", "6-K")
MIN_CAP_PROXY = 2.5e9

CATEGORY_SIGNAL = "rev_drop_nonews"
CATEGORY_NEWS = "rev_drop_news"
CATEGORY_SPIKE = "rev_spike_nonews"


def detect_dislocations(prices_by_ticker: dict, bench: pd.DataFrame,
                        start, end,
                        ret_window: int = RET_WINDOW,
                        vol_window: int = VOL_WINDOW,
                        z_trigger: float = Z_TRIGGER,
                        resid_floor: float = RESID_FLOOR) -> pd.DataFrame:
    """Pure. Scan every ticker for trading days in [start, end] where the
    ret_window-day residual return vs the benchmark is beyond +/-resid_floor
    AND beyond +/-z_trigger residual sigmas. Returns columns: ticker,
    trigger_date (Timestamp), resid_5d, z, direction ('drop'|'spike')."""
    b_close = bench["close"]
    b_ret1 = b_close.pct_change()
    b_retw = b_close.pct_change(ret_window)
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    min_len = vol_window + 2 * ret_window + 5
    rows = []
    for ticker, df in prices_by_ticker.items():
        if df is None or df.empty:
            continue
        idx = df.index.intersection(b_close.index)
        if len(idx) < min_len:
            continue
        c = df.loc[idx, "close"]
        resid1 = c.pct_change() - b_ret1.loc[idx]
        residw = c.pct_change(ret_window) - b_retw.loc[idx]
        # sigma of daily residuals over vol_window, ending ret_window days
        # BEFORE each date, scaled to the ret_window horizon
        sigma_w = resid1.rolling(vol_window).std().shift(ret_window) * (ret_window ** 0.5)
        z = residw / sigma_w
        in_window = (idx >= start_ts) & (idx <= end_ts)
        valid = z.notna() & residw.notna() & in_window
        drop_mask = valid & (z <= -z_trigger) & (residw <= -resid_floor)
        spike_mask = valid & (z >= z_trigger) & (residw >= resid_floor)
        for mask, direction in ((drop_mask, "drop"), (spike_mask, "spike")):
            for t in idx[mask]:
                rows.append({"ticker": ticker, "trigger_date": t,
                             "resid_5d": float(residw.loc[t]),
                             "z": float(z.loc[t]), "direction": direction})
    cols = ["ticker", "trigger_date", "resid_5d", "z", "direction"]
    return pd.DataFrame(rows, columns=cols)


def cap_per_day(events: pd.DataFrame, max_per_day: int = MAX_PER_DAY) -> pd.DataFrame:
    """Clustered-trigger cap (pre-registered): at most max_per_day events per
    (calendar day, direction), keeping the most extreme |z|. Limits the
    cross-sectional-correlation distortion of the t-test on crash days."""
    if events.empty:
        return events
    return (events.assign(_absz=events["z"].abs())
            .sort_values("_absz", ascending=False)
            .groupby(["trigger_date", "direction"], group_keys=False)
            .head(max_per_day)
            .drop(columns="_absz")
            .sort_values("trigger_date")
            .reset_index(drop=True))


def load_filing_dates(events: pd.DataFrame, uni_cik: pd.DataFrame,
                      lookback_days: int, db_path: str = DB_PATH,
                      forms: tuple = NEWS_FORMS) -> dict:
    """One submissions-JSON scan per distinct CIK among event tickers
    (cached). Returns {ticker: [Timestamp, ...]} of 8-K/6-K filing dates."""
    cik_by_ticker = dict(zip(uni_cik["ticker"], uni_cik["cik"]))
    out = {}
    tickers = sorted(set(events["ticker"]))
    for i, ticker in enumerate(tickers):
        cik = cik_by_ticker.get(ticker)
        if cik is None or pd.isna(cik):
            out[ticker] = []
            continue
        recs = list_filings(str(cik), db_path, lookback_days)
        out[ticker] = [pd.Timestamp(r["filing_date"]) for r in recs
                       if r["form"] in forms and r["filing_date"]]
        if (i + 1) % 100 == 0:
            print(f"[reversal] filings scanned {i + 1}/{len(tickers)}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description="No-news drawdown reversal backtest (registry Q5)")
    ap.add_argument("--years", type=int, default=LOOKBACK_YEARS)
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--tag", default="")
    ap.add_argument("--dry-run-only", action="store_true",
                    help="stop after the signal count kill decision")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    os.makedirs(args.out_dir, exist_ok=True)
    init_db(DB_PATH)

    end = date.today().isoformat()
    start = (date.today() - timedelta(days=365 * args.years)).isoformat()
    # price history must reach back far enough for the vol window
    fetch_start = (date.today() - timedelta(days=365 * args.years + 150)).isoformat()

    uni = load_universe()
    uni_cik = uni.merge(pd.read_parquet("data/universe.parquet")[["ticker", "cik"]],
                        on="ticker", how="left")
    tickers = sorted(uni_cik["ticker"].unique())
    print(f"[reversal] universe: {len(tickers)} tickers >= $2.5B", flush=True)

    print(f"[reversal] loading price history {fetch_start} .. {end} "
          f"(bulk, sqlite-cached)...", flush=True)
    price_cache = get_history_bulk([*tickers, BENCHMARK], DB_PATH, fetch_start, end)
    bench = price_cache.get(BENCHMARK)
    if bench is None or bench.empty:
        raise RuntimeError("no benchmark history")

    candidates = detect_dislocations(
        {t: price_cache.get(t) for t in tickers}, bench, start, end)
    print(f"[reversal] raw dislocations: "
          f"{candidates['direction'].value_counts().to_dict() if not candidates.empty else {}}",
          flush=True)
    if candidates.empty:
        print("[reversal] KILLED at dry-run: no dislocations detected")
        return

    candidates = cap_per_day(candidates)
    # dedupe 20d per ticker per direction, earliest wins
    parts = []
    for direction, g in candidates.groupby("direction"):
        g = g.assign(file_date=g["trigger_date"])
        parts.append(dedupe_events(g))
    candidates = pd.concat(parts, ignore_index=True)
    print(f"[reversal] after per-day cap + dedupe: "
          f"{candidates['direction'].value_counts().to_dict()}", flush=True)

    candidates["event_date"] = candidates["trigger_date"].dt.date.astype(str).map(
        event_date_from_announcement)
    candidates = attach_cap_proxy(candidates, uni, price_cache)
    n_unpriced = int(candidates["cap_proxy"].isna().sum())
    candidates = filter_cap_proxy(candidates, MIN_CAP_PROXY)
    print(f"[reversal] after cap-at-event proxy filter (>= $2.5B): "
          f"{candidates['direction'].value_counts().to_dict() if not candidates.empty else {}} "
          f"({n_unpriced} dropped for missing price data)", flush=True)

    # classify: earnings first, then filings — either one makes a drop "news"
    earnings = load_earnings_for_tickers(set(candidates["ticker"]), DB_PATH)
    lookback_days = 365 * args.years + 30
    filing_dates = load_filing_dates(candidates, uni_cik, lookback_days)

    drops = candidates[candidates["direction"] == "drop"]
    spikes = candidates[candidates["direction"] == "spike"]

    drops_earn_clean, drops_earn = split_by_earnings(drops, earnings)
    drops_clean, drops_filing = split_by_news(
        drops_earn_clean, filing_dates,
        days_before=NEWS_DAYS_BEFORE, days_after=NEWS_DAYS_AFTER)
    drops_news = pd.concat([drops_earn, drops_filing], ignore_index=True)

    spikes_earn_clean, _ = split_by_earnings(spikes, earnings)
    spikes_clean, _ = split_by_news(
        spikes_earn_clean, filing_dates,
        days_before=NEWS_DAYS_BEFORE, days_after=NEWS_DAYS_AFTER)

    drops_clean = drops_clean.assign(category=CATEGORY_SIGNAL)
    drops_news = drops_news.assign(category=CATEGORY_NEWS)
    spikes_clean = spikes_clean.assign(category=CATEGORY_SPIKE)

    years = max(args.years, 1)
    n_signal = len(drops_clean)
    print(f"[reversal] dry-run signal count ({CATEGORY_SIGNAL}): "
          f"{n_signal} ({n_signal / years:.1f}/yr); "
          f"controls: {CATEGORY_NEWS}={len(drops_news)}, "
          f"{CATEGORY_SPIKE}={len(spikes_clean)}", flush=True)

    if n_signal < 10 * years or n_signal < 50:
        print(f"[reversal] KILLED at dry-run: signal count too thin "
              f"(need >=10/yr and >=50 total, got {n_signal} over {years}y)")
        return
    if args.dry_run_only:
        print("[reversal] dry-run-only requested, stopping before price gate")
        return

    events = pd.concat([drops_clean, drops_news, spikes_clean], ignore_index=True)
    events = compute_abnormal_returns(events, DB_PATH, horizons=HORIZONS,
                                      benchmark=BENCHMARK)
    stem = date.today().isoformat() + (f"_{args.tag}" if args.tag else "")
    events.to_csv(os.path.join(args.out_dir, f"reversal_events_{stem}.csv"), index=False)

    summary = summarize(events, horizons=HORIZONS, gate_horizons=GATE_HORIZONS)
    summary.to_csv(os.path.join(args.out_dir, f"reversal_summary_{stem}.csv"), index=False)
    print(summary.to_string(index=False))

    halves = split_half_summary(events.rename(columns={"trigger_date": "filing_date"}))
    halves.to_csv(os.path.join(args.out_dir, f"reversal_halves_{stem}.csv"), index=False)
    print(halves.to_string(index=False))


if __name__ == "__main__":
    main()
