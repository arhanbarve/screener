"""Large dividend hike backtest — idea #6 in the ranked bank.

Registry: docs/strategy-registry.md. Spec: docs/strategy-specs.md #6.
Fit doc: docs/handoffs/FIT-2026-07-03.md.

Hypothesis: a large regular-dividend hike (>=25% vs the prior payment)
signals the same management-confidence channel as idea #4 (dividend
initiations), weaker per-event but with many more events. Controls:
`div_hike_small` (1.0-1.10x, routine, expect null) and `div_cut` (<0.9x,
expect negative) — two controls this time, not one.

Detection is primary-from-yfinance (whole-universe dividend series, already
fully cached from ideas #3/#4 — no re-fetch needed). Declaration date
recovery reuses idea #4's submissions-JSON regex fallback directly (spec:
"same submissions-JSON 8-K window trick as idea 4") — no EDGAR FTS phrase
tier here, since the spec doesn't propose one for routine hikes (unlike
"inaugural"/"initiates" language for a first-ever dividend, a routine
increase has no distinctive boilerplate phrase to search for).

Cost discipline (lesson from ideas #1/#4/#5): declaration-date resolution
is expensive — one submissions-JSON fetch plus potentially several document
fetches per candidate. Raw candidates split into 3 buckets (1000 large
hikes, ~2200 small hikes, ~925 cuts, verified against the fully-cached
dividend data) are counted first for free; only the SIGNAL category
(div_hike_large) gets declaration-resolved before the dry-run kill decision.
Controls are only resolved — each capped at 3x signal N, random sample
seed 0 — after signal clears the bar.
"""

import argparse
import logging
import os
from datetime import date, timedelta

import pandas as pd

from src.backtest_recipe import (
    dedupe_events,
    drop_earnings_contamination,
    filter_to_universe,
    load_earnings_for_tickers,
)
from src.cache import init_db
from src.divinit_backtest import find_declaration_via_submissions
from src.event_backtest import compute_abnormal_returns, summarize
from src.insider_backtest import split_half_summary
from src.pead_backtest import event_date_from_announcement, load_universe
from src.specialdiv_backtest import load_dividends_by_ticker

logger = logging.getLogger(__name__)

DB_PATH = "data/cache.db"
BENCHMARK = "SPY"
HORIZONS = (5, 10, 20)
GATE_HORIZONS = (5, 20)
LOOKBACK_YEARS = 3
HIKE_LARGE_RATIO = 1.25
HIKE_SMALL_LO, HIKE_SMALL_HI = 1.0, 1.10
CUT_RATIO = 0.9
CONTROL_CAP_MULTIPLE = 3
CATEGORY_LARGE = "div_hike_large"
CATEGORY_SMALL = "div_hike_small"
CATEGORY_CUT = "div_cut"


def detect_hike_candidates(dividends_by_ticker: dict, start: str, end: str) -> pd.DataFrame:
    """Pure: classify every payment (with a prior payment to compare
    against) whose ex_date falls in [start, end] into one of three buckets
    by ratio vs. the immediately preceding payment. Payments in the
    ambiguous 0.9-1.0 or 1.10-1.25 zones are not candidates for any
    category."""
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    rows = []
    for ticker, payments in dividends_by_ticker.items():
        payments = sorted(payments)
        for i in range(1, len(payments)):
            date_i, amt_i = payments[i]
            _, amt_prev = payments[i - 1]
            if not (start_ts <= date_i <= end_ts) or amt_prev <= 0:
                continue
            ratio = amt_i / amt_prev
            if ratio >= HIKE_LARGE_RATIO:
                category = CATEGORY_LARGE
            elif HIKE_SMALL_LO < ratio <= HIKE_SMALL_HI:
                category = CATEGORY_SMALL
            elif ratio < CUT_RATIO:
                category = CATEGORY_CUT
            else:
                continue
            rows.append({"ticker": ticker, "ex_date": date_i, "amount": float(amt_i),
                        "ratio": float(ratio), "category": category})
    return pd.DataFrame(rows)


def resolve_declarations(candidates: pd.DataFrame, uni_cik: pd.DataFrame,
                         db_path: str = DB_PATH) -> pd.DataFrame:
    """Attach a `file_date` (declaration date) to each candidate via
    find_declaration_via_submissions; drops candidates with no CIK mapping
    or no resolvable declaration."""
    if candidates.empty:
        return candidates
    candidates = candidates.merge(uni_cik[["ticker", "cik"]], on="ticker", how="left")
    rows = []
    for _, row in candidates.iterrows():
        if pd.isna(row.get("cik")):
            continue
        file_date, _confirmed = find_declaration_via_submissions(row["cik"], row["ex_date"], db_path)
        if file_date is None:
            continue
        out = row.to_dict()
        out["file_date"] = file_date
        rows.append(out)
    return pd.DataFrame(rows)


def _finalize(resolved: pd.DataFrame, earnings: pd.DataFrame) -> pd.DataFrame:
    if resolved.empty:
        return resolved
    resolved["event_date"] = resolved["file_date"].dt.date.astype(str).map(event_date_from_announcement)
    resolved = dedupe_events(resolved)
    return drop_earnings_contamination(resolved, earnings)


def main():
    ap = argparse.ArgumentParser(description="Large dividend hike backtest (registry idea #6)")
    ap.add_argument("--years", type=int, default=LOOKBACK_YEARS)
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--tag", default="")
    ap.add_argument("--dry-run-only", action="store_true",
                    help="only resolve+count the signal category, do not touch controls or prices")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    os.makedirs(args.out_dir, exist_ok=True)
    init_db(DB_PATH)

    end = date.today().isoformat()
    start = (date.today() - timedelta(days=365 * args.years)).isoformat()

    uni = load_universe()
    uni_cik = uni.merge(pd.read_parquet("data/universe.parquet")[["ticker", "cik"]],
                        on="ticker", how="left")
    print(f"[divhike] universe: {len(uni_cik)} tickers >= $2.5B", flush=True)

    tickers = sorted(uni_cik["ticker"].unique())
    div_by_ticker = load_dividends_by_ticker(set(tickers))
    print(f"[divhike] dividend history cached for {len(div_by_ticker)} tickers", flush=True)

    candidates = detect_hike_candidates(div_by_ticker, start, end)
    counts = candidates["category"].value_counts().to_dict() if not candidates.empty else {}
    print(f"[divhike] raw candidates by category: {counts}", flush=True)

    signal_raw = candidates[candidates["category"] == CATEGORY_LARGE] if not candidates.empty else candidates
    if signal_raw.empty:
        print("[divhike] KILLED at dry-run: no raw large-hike candidates")
        return

    print(f"[divhike] resolving declaration dates for {len(signal_raw)} signal candidates "
          f"(submissions-JSON scan, one call per distinct CIK, cached)...", flush=True)
    resolved_signal = resolve_declarations(signal_raw, uni_cik)
    print(f"[divhike] {len(resolved_signal)}/{len(signal_raw)} signal candidates resolved a declaration date",
          flush=True)

    signal_tickers = set(resolved_signal["ticker"]) if not resolved_signal.empty else set()
    earnings = load_earnings_for_tickers(signal_tickers, DB_PATH)
    resolved_signal = _finalize(resolved_signal, earnings)

    years = max(args.years, 1)
    n_signal = len(resolved_signal)
    print(f"[divhike] dry-run signal count (post-contamination): {n_signal} ({n_signal / years:.1f}/yr)",
          flush=True)

    if n_signal < 10 * years or n_signal < 50:
        print(f"[divhike] KILLED at dry-run: signal count too thin "
              f"(need >=10/yr and >=50 total, got {n_signal} total over {years}y)")
        return

    if args.dry_run_only:
        print("[divhike] dry-run-only requested, stopping before control resolution/price fetch")
        return

    control_cap = n_signal * CONTROL_CAP_MULTIPLE
    resolved_controls = []
    for cat in (CATEGORY_SMALL, CATEGORY_CUT):
        raw = candidates[candidates["category"] == cat]
        if len(raw) > control_cap:
            raw = raw.sample(n=control_cap, random_state=0)
        print(f"[divhike] resolving declaration dates for {len(raw)} {cat} candidates "
              f"(capped at {control_cap})...", flush=True)
        resolved = resolve_declarations(raw, uni_cik)
        ctl_tickers = set(resolved["ticker"]) if not resolved.empty else set()
        ctl_earnings = load_earnings_for_tickers(ctl_tickers, DB_PATH)
        resolved = _finalize(resolved, ctl_earnings)
        print(f"[divhike] {cat} count: {len(resolved)}", flush=True)
        resolved_controls.append(resolved)

    events = pd.concat([resolved_signal, *resolved_controls], ignore_index=True)
    print(f"[divhike] {len(events)} total events "
          f"({events['category'].value_counts().to_dict() if not events.empty else {}})", flush=True)
    if events.empty:
        return

    events = compute_abnormal_returns(events, DB_PATH, horizons=HORIZONS, benchmark=BENCHMARK)
    stem = date.today().isoformat() + (f"_{args.tag}" if args.tag else "")
    events.to_csv(os.path.join(args.out_dir, f"divhike_events_{stem}.csv"), index=False)

    summary = summarize(events, horizons=HORIZONS, gate_horizons=GATE_HORIZONS)
    summary.to_csv(os.path.join(args.out_dir, f"divhike_summary_{stem}.csv"), index=False)
    print(summary.to_string(index=False))

    halves = split_half_summary(events.rename(columns={"file_date": "filing_date"}))
    halves.to_csv(os.path.join(args.out_dir, f"divhike_halves_{stem}.csv"), index=False)
    print(halves.to_string(index=False))


if __name__ == "__main__":
    main()
