"""Dividend initiation backtest — idea #4 in the ranked bank.

Registry: docs/strategy-registry.md. Spec: docs/strategy-specs.md #4.
Fit doc: docs/handoffs/FIT-2026-07-03.md.

Hypothesis: a company's first-ever cash dividend, or first after a >=3-year
gap, signals management's confidence in a permanent payout commitment
(signaling channel — underreaction is documented). Control: `div_resumption`
(gap of 1-3 years) — a weaker version of the same signal, expected weaker
drift.

Detection is primary-from-yfinance (dividend payment history for the whole
universe, not just EDGAR-hit candidates), since the initiation/resumption
classification depends on the FULL prior payment history per ticker, not
just events in the lookback window.

Declaration date (yfinance doesn't provide it — only ex-dates) is recovered
in two tiers per spec: (1) EDGAR FTS for "initiates quarterly dividend" /
"declares inaugural" phrases matched to the candidate's CIK within 45 days
before the ex-date; (2) fallback — scan the ticker's own submissions JSON
for 8-Ks in that window and regex "declar.*dividend" on the fetched
document text. If an 8-K exists in the window but doesn't regex-confirm,
approximate declaration as ex_date - 14 days; if no 8-K exists in the
window at all, drop the event (can't establish a public announcement to
enter on).

Process fix from idea #3 (registry ledger #7 finding): the earnings-
contamination filter is folded into the dry-run count here, not deferred
to after it — contamination needs no price fetch, so there's no reason the
dry-run kill decision should run on a pre-contamination number.
"""

import argparse
import logging
import os
import re
from datetime import date, timedelta

import pandas as pd

from src.backtest_recipe import (
    dedupe_events,
    drop_earnings_contamination,
    filter_to_universe,
    load_earnings_for_tickers,
)
from src.cache import init_db
from src.edgar_fts import search
from src.event_backtest import compute_abnormal_returns, summarize
from src.filings import fetch_filing_doc, fetch_submissions, parse_submissions, plain_text
from src.insider_backtest import split_half_summary
from src.pead_backtest import event_date_from_announcement, load_universe
from src.specialdiv_backtest import fetch_dividends_for_tickers, load_dividends_by_ticker

logger = logging.getLogger(__name__)

DB_PATH = "data/cache.db"
BENCHMARK = "SPY"
HORIZONS = (5, 10, 20)
GATE_HORIZONS = (5, 20)
LOOKBACK_YEARS = 3
INITIATION_GAP_YEARS = 3.0     # first-ever, or gap >= this -> div_initiation
RESUMPTION_GAP_MIN_YEARS = 1.0  # gap in [1, 3) years -> div_resumption
DECL_WINDOW_DAYS = 45           # declaration-date lookup window before ex_date
SIGNAL_CATEGORY = "div_initiation"
CONTROL_CATEGORY = "div_resumption"
FTS_PHRASES = ("initiates quarterly dividend", "declares inaugural")
FTS_FORM = "8-K"
DECLARE_RE = re.compile(r"declar\w*.{0,80}dividend", re.IGNORECASE | re.DOTALL)


def detect_candidates(dividends_by_ticker: dict, start: str, end: str) -> pd.DataFrame:
    """Pure: for each ticker's full payment history (sorted by ex_date),
    classify a payment as `div_initiation` if it's the first payment ever
    seen or follows a gap >= INITIATION_GAP_YEARS, or `div_resumption` if
    the gap is in [RESUMPTION_GAP_MIN_YEARS, INITIATION_GAP_YEARS). Only
    payments whose ex_date falls in [start, end] are candidates (that's
    when the study window's entry would occur)."""
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    rows = []
    for ticker, payments in dividends_by_ticker.items():
        if not payments:
            continue
        payments = sorted(payments)
        for i, (ex_date, amount) in enumerate(payments):
            if not (start_ts <= ex_date <= end_ts):
                continue
            if i == 0:
                category, gap_years = SIGNAL_CATEGORY, None
            else:
                prev_date, _ = payments[i - 1]
                gap_years = (ex_date - prev_date).days / 365.25
                if gap_years >= INITIATION_GAP_YEARS:
                    category = SIGNAL_CATEGORY
                elif gap_years >= RESUMPTION_GAP_MIN_YEARS:
                    category = CONTROL_CATEGORY
                else:
                    continue
            rows.append({"ticker": ticker, "ex_date": ex_date, "amount": float(amount),
                        "gap_years": gap_years, "category": category})
    return pd.DataFrame(rows)


def match_fts_declaration(candidates: pd.DataFrame, fts_hits: pd.DataFrame,
                          window_days: int = DECL_WINDOW_DAYS) -> dict:
    """Tier 1: for each candidate (ticker/cik + ex_date), find the latest
    FTS hit for that CIK filed within `window_days` before the ex_date.
    Returns {(ticker, ex_date): file_date}."""
    if candidates.empty or fts_hits.empty:
        return {}
    out = {}
    hits_by_cik = {cik: g.sort_values("file_date") for cik, g in fts_hits.groupby("cik")}
    for _, row in candidates.iterrows():
        g = hits_by_cik.get(row["cik"])
        if g is None:
            continue
        window = g[(g["file_date"] <= row["ex_date"])
                  & (g["file_date"] >= row["ex_date"] - pd.Timedelta(days=window_days))]
        if not window.empty:
            out[(row["ticker"], row["ex_date"])] = window.iloc[-1]["file_date"]
    return out


def find_declaration_via_submissions(cik: str, ex_date: pd.Timestamp, db_path: str,
                                     window_days: int = DECL_WINDOW_DAYS):
    """Tier 2 fallback: scan the CIK's own 8-K submissions in the window
    before ex_date; regex-confirm "declar...dividend" in the fetched doc
    text. Returns (declaration_date, confirmed: bool) or (None, False) if
    no 8-K exists in the window at all."""
    data = fetch_submissions(cik, db_path)
    if not data:
        return None, False
    recs = parse_submissions(data)
    lo = (ex_date - pd.Timedelta(days=window_days)).date().isoformat()
    hi = ex_date.date().isoformat()
    candidates_8k = [r for r in recs if r["form"] == "8-K" and lo <= r["filing_date"] <= hi]
    if not candidates_8k:
        return None, False
    for rec in sorted(candidates_8k, key=lambda r: r["filing_date"]):
        html = fetch_filing_doc(cik, rec["accession"], rec["primary_doc"], db_path)
        if html and DECLARE_RE.search(plain_text(html)):
            return pd.Timestamp(rec["filing_date"]), True
    # An 8-K exists in the window but none regex-confirmed — spec's lenient
    # fallback: approximate declaration as ex_date - 14 days.
    return ex_date - pd.Timedelta(days=14), False


def resolve_declarations(candidates: pd.DataFrame, fts_hits: pd.DataFrame,
                         db_path: str = DB_PATH) -> pd.DataFrame:
    """Attach a `file_date` (declaration date) to each candidate via the
    two-tier lookup; drops candidates where neither tier finds anything."""
    if candidates.empty:
        return candidates
    fts_matches = match_fts_declaration(candidates, fts_hits)
    rows = []
    for _, row in candidates.iterrows():
        key = (row["ticker"], row["ex_date"])
        if key in fts_matches:
            file_date = fts_matches[key]
        else:
            file_date, _confirmed = find_declaration_via_submissions(row["cik"], row["ex_date"], db_path)
            if file_date is None:
                continue
        out = row.to_dict()
        out["file_date"] = file_date
        rows.append(out)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="Dividend initiation backtest (registry idea #4)")
    ap.add_argument("--years", type=int, default=LOOKBACK_YEARS)
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--tag", default="")
    ap.add_argument("--dry-run-only", action="store_true",
                    help="only count confirmed, contamination-filtered signal events")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    os.makedirs(args.out_dir, exist_ok=True)
    init_db(DB_PATH)

    end = date.today().isoformat()
    start = (date.today() - timedelta(days=365 * args.years)).isoformat()

    uni = load_universe()
    uni_cik = uni.merge(pd.read_parquet("data/universe.parquet")[["ticker", "cik"]],
                        on="ticker", how="left")
    print(f"[divinit] universe: {len(uni_cik)} tickers >= $2.5B", flush=True)

    tickers = sorted(uni_cik["ticker"].unique())
    print(f"[divinit] fetching yfinance dividend history for {len(tickers)} universe tickers...", flush=True)
    fetch_dividends_for_tickers(tickers)
    div_by_ticker = load_dividends_by_ticker(set(tickers))

    candidates = detect_candidates(div_by_ticker, start, end)
    print(f"[divinit] raw candidates from dividend series: {len(candidates)} "
          f"({candidates['category'].value_counts().to_dict() if not candidates.empty else {}})", flush=True)
    if candidates.empty:
        print("[divinit] KILLED at dry-run: no candidates")
        return

    candidates = candidates.merge(uni_cik[["ticker", "cik"]], on="ticker", how="left")

    print("[divinit] fetching declaration-phrase 8-K hits from EDGAR FTS...", flush=True)
    fts_hits = pd.concat([search(p, FTS_FORM, start, end, db_path=DB_PATH) for p in FTS_PHRASES],
                         ignore_index=True)
    if not fts_hits.empty:
        fts_hits["file_date"] = pd.to_datetime(fts_hits["file_date"])

    resolved = resolve_declarations(candidates, fts_hits)
    print(f"[divinit] {len(resolved)}/{len(candidates)} candidates resolved a declaration date", flush=True)
    if resolved.empty:
        print("[divinit] KILLED at dry-run: no candidates resolved a declaration date")
        return

    resolved["event_date"] = resolved["file_date"].dt.date.astype(str).map(event_date_from_announcement)
    resolved = dedupe_events(resolved)

    all_tickers = set(resolved["ticker"])
    earnings = load_earnings_for_tickers(all_tickers, DB_PATH)
    resolved = drop_earnings_contamination(resolved, earnings)

    years = max(args.years, 1)
    n_signal = int((resolved["category"] == SIGNAL_CATEGORY).sum())
    n_control = int((resolved["category"] == CONTROL_CATEGORY).sum())
    print(f"[divinit] dry-run counts (post-contamination): signal={n_signal} ({n_signal / years:.1f}/yr), "
          f"control={n_control} ({n_control / years:.1f}/yr)", flush=True)

    if n_signal < 10 * years or n_signal < 50:
        print(f"[divinit] KILLED at dry-run: signal count too thin "
              f"(need >=10/yr and >=50 total, got {n_signal} total over {years}y)")
        return

    if args.dry_run_only:
        print("[divinit] dry-run-only requested, stopping before price fetch")
        return

    events = resolved
    events = compute_abnormal_returns(events, DB_PATH, horizons=HORIZONS, benchmark=BENCHMARK)
    stem = date.today().isoformat() + (f"_{args.tag}" if args.tag else "")
    events.to_csv(os.path.join(args.out_dir, f"divinit_events_{stem}.csv"), index=False)

    summary = summarize(events, horizons=HORIZONS, gate_horizons=GATE_HORIZONS)
    summary.to_csv(os.path.join(args.out_dir, f"divinit_summary_{stem}.csv"), index=False)
    print(summary.to_string(index=False))

    halves = split_half_summary(events.rename(columns={"file_date": "filing_date"}))
    halves.to_csv(os.path.join(args.out_dir, f"divinit_halves_{stem}.csv"), index=False)
    print(halves.to_string(index=False))


if __name__ == "__main__":
    main()
