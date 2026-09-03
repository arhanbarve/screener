"""S&P 500 deletion-overshoot bounce — idea B (registry Q6, IDEA BANK 2).

Registry: docs/strategy-registry.md Q6. Fit: docs/handoffs/FIT-2026-07-03.md.

Hypothesis: index deletion forces indexed money to sell regardless of
fundamentals — the purest identifiable non-fundamental seller — and the
resulting overshoot reverts over the following weeks. Non-M&A deletions
only (an acquired name leaving the index is not a forced-selling event on
a still-trading stock). Control: additions in the same window (expect
null/negative; the inclusion premium is documented as decayed).

Event source is Wikipedia's S&P 500 constituent-change table (free). The
"Date" column is the EFFECTIVE date; entry is effective date + 1 calendar
day, i.e. after the forced selling into the effective close is done.

Pre-registered data-integrity kill (registry Q6): if > 20% of non-M&A
deletions have no usable yfinance price history (delisted since), the
priceable remainder is survivorship-selected in exactly the direction that
fakes a bounce — KILL regardless of gate output.
"""

import argparse
import io
import logging
import os
import re

from src.config import get_env
from datetime import date, timedelta

import pandas as pd
import requests

from src.backtest_recipe import (
    attach_cap_proxy,
    drop_earnings_contamination,
    filter_cap_proxy,
    load_earnings_for_tickers,
)
from src.cache import init_db
from src.event_backtest import compute_abnormal_returns, get_history_bulk, summarize
from src.insider_backtest import split_half_summary
from src.pead_backtest import load_universe

logger = logging.getLogger(__name__)

DB_PATH = "data/cache.db"
BENCHMARK = "SPY"
HORIZONS = (5, 10, 20)
GATE_HORIZONS = (5, 20)
LOOKBACK_YEARS = 6          # pre-registered: deletions are rare, 3y is too thin
MIN_CAP_PROXY = 2.5e9
MAX_UNPRICED_FRAC = 0.20    # pre-registered data-integrity kill threshold

CATEGORY_DELETION = "sp500_deletion"
CATEGORY_ADDITION = "sp500_addition"

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
UA = {"User-Agent": get_env("SEC_USER_AGENT")}

MA_REASON_RE = re.compile(
    r"acquir|merg|taken private|purchas|bought|combin|bankrupt|chapter 11|delist",
    re.IGNORECASE)


def is_ma_reason(reason: str) -> bool:
    """True if the removal reason is M&A/bankruptcy — excluded from the
    signal because the stock either stops trading or the removal is not a
    forced-selling event on a going concern."""
    return bool(MA_REASON_RE.search(reason or ""))


def fetch_changes_table() -> pd.DataFrame:
    """The constituent-changes table from Wikipedia (the one whose columns
    include an 'Added'/'Removed' MultiIndex level)."""
    resp = requests.get(WIKI_URL, headers=UA, timeout=30)
    resp.raise_for_status()
    for t in pd.read_html(io.StringIO(resp.text)):
        if isinstance(t.columns, pd.MultiIndex) and "Added" in t.columns.get_level_values(0):
            return t
    raise RuntimeError("S&P 500 changes table not found on Wikipedia page")


def parse_changes(raw: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Flatten the Wikipedia table into event rows: columns ticker,
    effective_date (Timestamp), event_date (iso str, effective + 1 day),
    reason, category (sp500_deletion for non-M&A removals, sp500_addition
    for all additions)."""
    dates = pd.to_datetime(raw[("Effective Date", "Effective Date")],
                           format="%B %d, %Y", errors="coerce")
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    rows = []
    for i in range(len(raw)):
        d = dates.iloc[i]
        if pd.isna(d) or not (start_ts <= d <= end_ts):
            continue
        reason = str(raw[("Reason", "Reason")].iloc[i])
        removed = raw[("Removed", "Ticker")].iloc[i]
        added = raw[("Added", "Ticker")].iloc[i]
        event_date = (d + pd.Timedelta(days=1)).date().isoformat()
        if isinstance(removed, str) and removed.strip() and not is_ma_reason(reason):
            rows.append({"ticker": removed.strip(), "effective_date": d,
                         "event_date": event_date, "reason": reason,
                         "category": CATEGORY_DELETION})
        if isinstance(added, str) and added.strip():
            rows.append({"ticker": added.strip(), "effective_date": d,
                         "event_date": event_date, "reason": reason,
                         "category": CATEGORY_ADDITION})
    cols = ["ticker", "effective_date", "event_date", "reason", "category"]
    return pd.DataFrame(rows, columns=cols)


def main():
    ap = argparse.ArgumentParser(description="S&P 500 deletion bounce backtest (registry Q6)")
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

    raw = fetch_changes_table()
    events = parse_changes(raw, start, end)
    n_del = len(events[events["category"] == CATEGORY_DELETION])
    n_add = len(events[events["category"] == CATEGORY_ADDITION])
    print(f"[spdel] parsed changes in window: {n_del} non-M&A deletions, "
          f"{n_add} additions", flush=True)

    years = max(args.years, 1)
    if n_del < 10 * years or n_del < 50:
        print(f"[spdel] KILLED at dry-run: {n_del} non-M&A deletions over {years}y "
              f"(need >=10/yr and >=50 total)")
        return
    if args.dry_run_only:
        print("[spdel] dry-run-only requested, stopping before price fetch")
        return

    # price data + data-integrity check on the SIGNAL category
    tickers = sorted(set(events["ticker"]))
    fetch_start = (date.today() - timedelta(days=365 * args.years + 30)).isoformat()
    price_cache = get_history_bulk([*tickers, BENCHMARK], DB_PATH, fetch_start, end)

    dels = events[events["category"] == CATEGORY_DELETION]
    unpriced = [t for t in dels["ticker"]
                if price_cache.get(t) is None or price_cache.get(t).empty]
    frac = len(unpriced) / max(len(dels), 1)
    print(f"[spdel] deletions without price history: {len(unpriced)}/{len(dels)} "
          f"({frac:.0%}) {sorted(set(unpriced))}", flush=True)
    if frac > MAX_UNPRICED_FRAC:
        print(f"[spdel] KILLED (pre-registered data-integrity rule): "
              f"{frac:.0%} > {MAX_UNPRICED_FRAC:.0%} of deletions unpriceable — "
              f"remaining sample is survivorship-selected toward bounces")
        return

    uni = load_universe()
    events = attach_cap_proxy(events, uni, price_cache)
    events = filter_cap_proxy(events, MIN_CAP_PROXY)
    print(f"[spdel] after cap-at-event proxy filter: "
          f"{events['category'].value_counts().to_dict() if not events.empty else {}}",
          flush=True)

    earnings = load_earnings_for_tickers(set(events["ticker"]), DB_PATH)
    events = drop_earnings_contamination(events, earnings)
    print(f"[spdel] after earnings-contamination filter: "
          f"{events['category'].value_counts().to_dict() if not events.empty else {}}",
          flush=True)
    if events.empty:
        return

    events = compute_abnormal_returns(events, DB_PATH, horizons=HORIZONS,
                                      benchmark=BENCHMARK)
    stem = date.today().isoformat() + (f"_{args.tag}" if args.tag else "")
    events.to_csv(os.path.join(args.out_dir, f"spdel_events_{stem}.csv"), index=False)

    summary = summarize(events, horizons=HORIZONS, gate_horizons=GATE_HORIZONS)
    summary.to_csv(os.path.join(args.out_dir, f"spdel_summary_{stem}.csv"), index=False)
    print(summary.to_string(index=False))

    halves = split_half_summary(events.rename(columns={"effective_date": "filing_date"}))
    halves.to_csv(os.path.join(args.out_dir, f"spdel_halves_{stem}.csv"), index=False)
    print(halves.to_string(index=False))


if __name__ == "__main__":
    main()
