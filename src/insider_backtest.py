"""Insider cluster-buying backtest — Form 4 open-market purchases (code P).

Registry entry: docs/strategy-registry.md Q1. Fit doc: docs/handoffs/FIT-2026-07-03.md.

Hypothesis: >=2 distinct insiders buying open-market within a 10-day window
signals positive drift over the following days-weeks. Controls: single-insider
buys (weaker published effect) and insider SELL clusters (expected ~null in
large caps, where selling is routine compensation flow).

Data: SEC Insider Transactions bulk data sets (quarterly TSV zips in
data/insider/), NOT per-filing XML fetches. $0 cost.

Lookahead discipline: transactions become public at FILING_DATE (up to 2
business days after TRANS_DATE). Events are timed off the filing date that
completes the cluster, and event_date is filing_date + 1 calendar day so the
return harness enters at the first close AFTER the signal is public.
10b5-1-flagged filings are excluded (pre-scheduled, no information content).
"""

import argparse
import glob
import logging
import os
import zipfile
from datetime import date

import pandas as pd

from src.cache import init_db
from src.event_backtest import compute_abnormal_returns, summarize
from src.pead_backtest import load_universe, event_date_from_announcement

logger = logging.getLogger(__name__)

DB_PATH = "data/cache.db"
DATA_DIR = "data/insider"
BENCHMARK = "SPY"
HORIZONS = (5, 10, 20)
GATE_HORIZONS = (5, 20)
MIN_TXN_USD = 25_000       # per-owner total below this is token buying, ignored
CLUSTER_WINDOW_DAYS = 10   # distinct owners must file within this many days
DEDUPE_DAYS = 20           # min gap between successive events for one issuer


def _read_tsv(zf: zipfile.ZipFile, name: str, usecols: list[str]) -> pd.DataFrame:
    """Columns missing from older quarters (e.g. AFF10B5ONE, added with the
    2023 10b5-1 disclosure rule) come back filled with NA instead of erroring."""
    with zf.open(name) as f:
        header = pd.read_csv(f, sep="\t", nrows=0).columns
    present = [c for c in usecols if c in header]
    with zf.open(name) as f:
        df = pd.read_csv(f, sep="\t", usecols=present, low_memory=False)
    for c in usecols:
        if c not in df.columns:
            df[c] = pd.NA
    return df


def load_quarter(zip_path: str) -> pd.DataFrame:
    """One quarter's Form 4 non-derivative transactions joined with issuer +
    owner, reduced to per-(issuer, owner, filing_date, side) dollar totals."""
    zf = zipfile.ZipFile(zip_path)
    sub = _read_tsv(zf, "SUBMISSION.tsv",
                    ["ACCESSION_NUMBER", "FILING_DATE", "DOCUMENT_TYPE",
                     "ISSUERTRADINGSYMBOL", "AFF10B5ONE"])
    trans = _read_tsv(zf, "NONDERIV_TRANS.tsv",
                      ["ACCESSION_NUMBER", "TRANS_CODE", "TRANS_SHARES",
                       "TRANS_PRICEPERSHARE", "TRANS_ACQUIRED_DISP_CD"])
    owners = _read_tsv(zf, "REPORTINGOWNER.tsv",
                       ["ACCESSION_NUMBER", "RPTOWNERCIK"])
    zf.close()

    sub = sub[(sub["DOCUMENT_TYPE"].astype(str) == "4")
              & (sub["AFF10B5ONE"].astype(str).str.lower() != "true")]
    trans = trans[trans["TRANS_CODE"].isin(["P", "S"])
                  & (trans["TRANS_SHARES"] > 0)
                  & (trans["TRANS_PRICEPERSHARE"] > 0)].copy()
    trans["value_usd"] = trans["TRANS_SHARES"] * trans["TRANS_PRICEPERSHARE"]

    df = (trans.merge(sub, on="ACCESSION_NUMBER")
               .merge(owners.drop_duplicates("ACCESSION_NUMBER"), on="ACCESSION_NUMBER"))
    df["ticker"] = df["ISSUERTRADINGSYMBOL"].astype(str).str.strip().str.upper()
    df = df[df["ticker"].str.fullmatch(r"[A-Z]{1,5}")]
    df["filing_date"] = pd.to_datetime(df["FILING_DATE"], format="%d-%b-%Y")

    grouped = (df.groupby(["ticker", "RPTOWNERCIK", "filing_date", "TRANS_CODE"])
                 ["value_usd"].sum().reset_index())
    return grouped[grouped["value_usd"] >= MIN_TXN_USD]


def load_all_quarters(data_dir: str = DATA_DIR) -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(data_dir, "*_form345.zip")))
    if not paths:
        raise RuntimeError(f"No form345 zips in {data_dir}")
    frames = [load_quarter(p) for p in paths]
    logger.info(f"[insider] loaded {len(paths)} quarters")
    return pd.concat(frames, ignore_index=True)


def detect_events(buys: pd.DataFrame, code: str, cluster_label: str,
                  single_label: str | None) -> list[dict]:
    """Walk one side's (P or S) records chronologically per issuer. A cluster
    event fires on the filing date where a 2nd distinct owner appears within
    CLUSTER_WINDOW_DAYS; issuer events are then suppressed for DEDUPE_DAYS.
    Records never part of a cluster become single events (if labeled)."""
    events = []
    side = buys[buys["TRANS_CODE"] == code]
    for ticker, g in side.groupby("ticker"):
        g = g.sort_values("filing_date")
        last_event_date = None
        for _, row in g.iterrows():
            fd = row["filing_date"]
            if last_event_date is not None and (fd - last_event_date).days < DEDUPE_DAYS:
                continue
            window = g[(g["filing_date"] > fd - pd.Timedelta(days=CLUSTER_WINDOW_DAYS))
                       & (g["filing_date"] <= fd)]
            n_owners = window["RPTOWNERCIK"].nunique()
            if n_owners >= 2:
                events.append({"ticker": ticker, "category": cluster_label,
                               "filing_date": fd, "n_owners": n_owners,
                               "total_usd": float(window["value_usd"].sum())})
                last_event_date = fd
            elif single_label is not None:
                lookahead = g[(g["filing_date"] > fd)
                              & (g["filing_date"] < fd + pd.Timedelta(days=CLUSTER_WINDOW_DAYS))]
                if lookahead["RPTOWNERCIK"].nunique() == 0 or \
                        set(lookahead["RPTOWNERCIK"]) == {row["RPTOWNERCIK"]}:
                    events.append({"ticker": ticker, "category": single_label,
                                   "filing_date": fd, "n_owners": 1,
                                   "total_usd": float(row["value_usd"])})
                    last_event_date = fd
    return events


def build_events(records: pd.DataFrame, universe_tickers: set) -> pd.DataFrame:
    records = records[records["ticker"].isin(universe_tickers)]
    events = detect_events(records, "P", "insider_cluster_buy", "insider_single_buy")
    events += detect_events(records, "S", "insider_cluster_sell", None)
    df = pd.DataFrame(events)
    if df.empty:
        return df
    df["event_date"] = df["filing_date"].dt.date.astype(str).map(event_date_from_announcement)
    return df.sort_values("filing_date").reset_index(drop=True)


def split_half_summary(events: pd.DataFrame) -> pd.DataFrame:
    """PASS confirmation #3 in the registry: sign stability across halves."""
    mid = events["filing_date"].quantile(0.5)
    frames = []
    for name, half in (("first_half", events[events["filing_date"] <= mid]),
                       ("second_half", events[events["filing_date"] > mid])):
        s = summarize(half, horizons=HORIZONS, gate_horizons=GATE_HORIZONS)
        if not s.empty:
            s.insert(0, "half", name)
            frames.append(s)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser(description="Insider cluster backtest (registry Q1)")
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--gate-horizons", default="5,20",
                    help="comma-separated; pre-register BEFORE running, per registry rules")
    ap.add_argument("--tag", default="", help="suffix for output filenames")
    args = ap.parse_args()
    gate_horizons = tuple(int(h) for h in args.gate_horizons.split(","))

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    os.makedirs(args.out_dir, exist_ok=True)
    init_db(DB_PATH)

    uni = load_universe()
    print(f"[insider] universe: {len(uni)} tickers", flush=True)

    records = load_all_quarters(args.data_dir)
    events = build_events(records, set(uni["ticker"]))
    print(f"[insider] {len(events)} events "
          f"({events['category'].value_counts().to_dict() if not events.empty else {}})", flush=True)
    if events.empty:
        return

    events = compute_abnormal_returns(events, DB_PATH, horizons=HORIZONS,
                                      benchmark=BENCHMARK)
    stem = date.today().isoformat() + (f"_{args.tag}" if args.tag else "")
    events.to_csv(os.path.join(args.out_dir, f"insider_events_{stem}.csv"), index=False)

    summary = summarize(events, horizons=HORIZONS, gate_horizons=gate_horizons)
    summary.to_csv(os.path.join(args.out_dir, f"insider_summary_{stem}.csv"), index=False)
    print(summary.to_string(index=False))

    halves = split_half_summary(events)
    halves.to_csv(os.path.join(args.out_dir, f"insider_halves_{stem}.csv"), index=False)
    print(halves.to_string(index=False))


if __name__ == "__main__":
    main()
