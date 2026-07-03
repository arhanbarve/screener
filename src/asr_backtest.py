"""Accelerated share repurchase (ASR) announcement backtest — idea #5.

Registry: docs/strategy-registry.md. Spec: docs/strategy-specs.md #5.
Fit doc: docs/handoffs/FIT-2026-07-03.md.

Hypothesis: ASR announcement (committed, immediately-executed buyback) ->
positive drift; committed capital deployment is a stronger signal than a
routine open-market buyback authorization (cheap talk, no committed
timeline or size). Control: `buyback_authorization` — a routine repurchase-
program announcement, expected weaker/null, sampled and capped the same way
Schedule 13G was for idea #1 (edgar_fts.search's max_raw_hits/sample_months).

Real deviation from the spec (same finding as ideas #1/#2): EDGAR FTS hits
carry no snippet/highlight text field (verified live), so the spec's
planned confound filter ("require 'enter/entered into' or 'announc' near
the phrase in the snippet") can't run on a snippet that doesn't exist.
Implemented instead against the actual filing document text (one fetch per
in-universe candidate, cached forever by accession like every other
document fetch in this codebase, primary-doc filename recovered from the
FTS hit's `_id` field per edgar_fts.py) — a candidate confirms only if
"enter(ed) into" or "announc" appears within a text window around the ASR
phrase. This is what filters ASR *progress updates* buried in routine
(often earnings) 8-Ks out from genuine new-announcement 8-Ks, per the
spec's stated trap.
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
from src.filings import fetch_filing_doc, plain_text
from src.insider_backtest import split_half_summary
from src.pead_backtest import event_date_from_announcement, load_universe

logger = logging.getLogger(__name__)

DB_PATH = "data/cache.db"
BENCHMARK = "SPY"
HORIZONS = (5, 10, 20)
GATE_HORIZONS = (5, 20)
LOOKBACK_YEARS = 3
CONTROL_CAP_MULTIPLE = 3
CONTROL_SAMPLE_MONTHS = 12
SIGNAL_CATEGORY = "asr_announce"
CONTROL_CATEGORY = "buyback_authorization"
SIGNAL_PHRASE = "accelerated share repurchase"
CONTROL_PHRASE = "share repurchase program"
FTS_FORM = "8-K"
CONFIRM_WINDOW_CHARS = 300
CONFIRM_RE = re.compile(r"enter(?:ed)?\s+into|announc", re.IGNORECASE)


def fetch_raw_hits(phrase: str, start: str, end: str, db_path: str = DB_PATH,
                   **search_kwargs) -> pd.DataFrame:
    hits = search(phrase, FTS_FORM, start, end, db_path=db_path, **search_kwargs)
    if hits.empty:
        return hits
    hits["file_date"] = pd.to_datetime(hits["file_date"])
    return hits.reset_index(drop=True)


def confirm_new_announcement_text(text: str, phrase: str = SIGNAL_PHRASE,
                                  window: int = CONFIRM_WINDOW_CHARS) -> bool:
    """True if "enter(ed) into" or "announc" appears within `window` chars
    of the phrase — distinguishes a genuine new-ASR-announcement 8-K from a
    routine progress-update mention of an existing ASR."""
    if not text:
        return False
    lower = text.lower()
    idx = lower.find(phrase.lower())
    if idx == -1:
        return False
    around = text[max(0, idx - window): idx + len(phrase) + window]
    return bool(CONFIRM_RE.search(around))


def confirm_hits(hits: pd.DataFrame, db_path: str = DB_PATH) -> pd.DataFrame:
    """Fetch each candidate's filing document and confirm via
    confirm_new_announcement_text; drops unconfirmed hits (and hits with no
    recoverable primary_doc filename or a failed fetch)."""
    if hits.empty:
        return hits
    keep = []
    for _, row in hits.iterrows():
        if not row.get("primary_doc"):
            keep.append(False)
            continue
        html = fetch_filing_doc(row["cik"], row["adsh"], row["primary_doc"], db_path)
        keep.append(confirm_new_announcement_text(plain_text(html)) if html else False)
    return hits[pd.Series(keep, index=hits.index)].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description="ASR announcement backtest (registry idea #5)")
    ap.add_argument("--years", type=int, default=LOOKBACK_YEARS)
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--tag", default="")
    ap.add_argument("--dry-run-only", action="store_true",
                    help="only count confirmed, contamination-filtered signal events")
    ap.add_argument("--hand-check", type=int, default=0,
                    help="print N random confirmed+N random rejected hits for manual precision review")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    os.makedirs(args.out_dir, exist_ok=True)
    init_db(DB_PATH)

    end = date.today().isoformat()
    start = (date.today() - timedelta(days=365 * args.years)).isoformat()

    uni = load_universe()
    uni_cik = uni.merge(pd.read_parquet("data/universe.parquet")[["ticker", "cik"]],
                        on="ticker", how="left")
    print(f"[asr] universe: {len(uni_cik)} tickers >= $2.5B", flush=True)

    print("[asr] fetching 'accelerated share repurchase' 8-K hits from EDGAR FTS...", flush=True)
    raw_hits = fetch_raw_hits(SIGNAL_PHRASE, start, end)
    hits_uni = filter_to_universe(raw_hits, uni_cik)
    print(f"[asr] raw hits: {len(raw_hits)}, in-universe (pre-confirmation): {len(hits_uni)} "
          f"across {hits_uni['ticker'].nunique() if not hits_uni.empty else 0} tickers", flush=True)

    print(f"[asr] fetching + regex-confirming filing text for {len(hits_uni)} candidates...", flush=True)
    confirmed = confirm_hits(hits_uni)
    print(f"[asr] confirmed new-announcement hits: {len(confirmed)} "
          f"({len(confirmed) / max(len(hits_uni), 1) * 100:.0f}% of in-universe raw hits)", flush=True)

    if args.hand_check:
        rejected = hits_uni[~hits_uni["adsh"].isin(confirmed["adsh"])] if not confirmed.empty else hits_uni
        print(f"\n[asr] hand-check sample (spec step: verify precision, record in RUNLOG):", flush=True)
        for _, row in confirmed.sample(min(args.hand_check, len(confirmed)), random_state=0).iterrows():
            print(f"  CONFIRMED {row['ticker']} {row['file_date'].date()} adsh={row['adsh']}", flush=True)
        for _, row in rejected.sample(min(args.hand_check, len(rejected)), random_state=0).iterrows():
            print(f"  REJECTED  {row['ticker']} {row['file_date'].date()} adsh={row['adsh']}", flush=True)

    confirmed = dedupe_events(confirmed) if not confirmed.empty else confirmed
    if not confirmed.empty:
        confirmed["category"] = SIGNAL_CATEGORY
        confirmed["event_date"] = confirmed["file_date"].dt.date.astype(str).map(event_date_from_announcement)

    all_tickers = set(confirmed["ticker"]) if not confirmed.empty else set()
    earnings = load_earnings_for_tickers(all_tickers, DB_PATH)
    confirmed = drop_earnings_contamination(confirmed, earnings) if not confirmed.empty else confirmed

    years = max(args.years, 1)
    n_signal = len(confirmed)
    print(f"[asr] dry-run confirmed signal count (post-contamination): {n_signal} ({n_signal / years:.1f}/yr)",
          flush=True)

    if n_signal < 10 * years or n_signal < 50:
        print(f"[asr] KILLED at dry-run: signal count too thin "
              f"(need >=10/yr and >=50 total, got {n_signal} total over {years}y)")
        return

    if args.dry_run_only:
        print("[asr] dry-run-only requested, stopping before control/price fetch")
        return

    control_cap = n_signal * CONTROL_CAP_MULTIPLE
    print(f"[asr] fetching 'share repurchase program' 8-K hits (control, capped ~{control_cap} raw "
          f"across {CONTROL_SAMPLE_MONTHS} sampled months)...", flush=True)
    raw_control = fetch_raw_hits(CONTROL_PHRASE, start, end,
                                 max_raw_hits=control_cap, sample_months=CONTROL_SAMPLE_MONTHS)
    control_uni = filter_to_universe(raw_control, uni_cik)
    control_uni = dedupe_events(control_uni) if not control_uni.empty else control_uni
    if not control_uni.empty:
        control_uni["category"] = CONTROL_CATEGORY
    print(f"[asr] control count: {len(control_uni)} in-universe (sampled window)", flush=True)

    events = pd.concat([confirmed, control_uni], ignore_index=True)
    events["event_date"] = events["file_date"].dt.date.astype(str).map(event_date_from_announcement)

    control_tickers = set(control_uni["ticker"]) if not control_uni.empty else set()
    control_earnings = load_earnings_for_tickers(control_tickers, DB_PATH)
    events = drop_earnings_contamination(events, pd.concat([earnings, control_earnings], ignore_index=True))
    print(f"[asr] {len(events)} events after full contamination filter "
          f"({events['category'].value_counts().to_dict() if not events.empty else {}})", flush=True)
    if events.empty:
        return

    events = compute_abnormal_returns(events, DB_PATH, horizons=HORIZONS, benchmark=BENCHMARK)
    stem = date.today().isoformat() + (f"_{args.tag}" if args.tag else "")
    events.to_csv(os.path.join(args.out_dir, f"asr_events_{stem}.csv"), index=False)

    summary = summarize(events, horizons=HORIZONS, gate_horizons=GATE_HORIZONS)
    summary.to_csv(os.path.join(args.out_dir, f"asr_summary_{stem}.csv"), index=False)
    print(summary.to_string(index=False))

    halves = split_half_summary(events.rename(columns={"file_date": "filing_date"}))
    halves.to_csv(os.path.join(args.out_dir, f"asr_halves_{stem}.csv"), index=False)
    print(halves.to_string(index=False))


if __name__ == "__main__":
    main()
