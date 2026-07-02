"""Filing-edge veto overlay.

The strong leg of the "Lazy Prices" anomaly is avoiding deteriorating-language
filers. This module cross-references the filing-edge WATCH list (deteriorating
filers) against the main momentum screener's top picks and currently held
positions, so the user is warned when a stock they are about to buy or already
hold is a negative filer.

All loaders are defensive: missing files or columns yield empty results rather
than raising, so the caller can always render something.
"""

from __future__ import annotations

import glob
import json
import os

import pandas as pd

WATCH_COLUMNS = [
    "text_stability",
    "change_direction",
    "change_reason",
    "eight_k_penalty",
    "red_flags",
]

REPORT_COLUMNS = [
    "ticker",
    "in_screener",
    "in_positions",
    "text_stability",
    "change_direction",
    "change_reason",
    "eight_k_penalty",
    "red_flags",
]


def latest_file(pattern: str, output_dir: str = "output") -> str | None:
    """Return the newest file matching ``pattern`` in ``output_dir``, or None."""
    matches = glob.glob(os.path.join(output_dir, pattern))
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)


def load_watch(output_dir: str = "output") -> pd.DataFrame:
    """Load watch-list rows from the newest filing_edge csv.

    Returns an empty DataFrame if no file exists, it cannot be read, or it has
    no ``__list__`` column / no watch rows.
    """
    path = latest_file("filing_edge_*.csv", output_dir=output_dir)
    if path is None:
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    if "__list__" not in df.columns or "ticker" not in df.columns:
        return pd.DataFrame()
    watch = df[df["__list__"].astype(str) == "watch"].copy()
    return watch


def load_screener_tickers(output_dir: str = "output") -> set[str]:
    """Return the set of tickers from the newest screen_*.csv (all rows)."""
    path = latest_file("screen_*.csv", output_dir=output_dir)
    if path is None:
        return set()
    try:
        df = pd.read_csv(path)
    except Exception:
        return set()
    if "ticker" not in df.columns:
        return set()
    return {str(t).strip().upper() for t in df["ticker"].dropna()}


def load_position_tickers(path: str = "positions.json") -> set[str]:
    """Return the set of tickers held, from positions.json."""
    if not os.path.exists(path):
        return set()
    try:
        with open(path) as fh:
            data = json.load(fh)
    except Exception:
        return set()
    if not isinstance(data, list):
        return set()
    tickers = set()
    for row in data:
        if isinstance(row, dict) and row.get("ticker"):
            tickers.add(str(row["ticker"]).strip().upper())
    return tickers


def build_veto_report(
    output_dir: str = "output",
    positions_path: str = "positions.json",
) -> pd.DataFrame:
    """Build the actionable veto report.

    For each watch-list ticker, flag whether it also appears in the screener
    top picks (``in_screener``) and/or currently held positions
    (``in_positions``), and carry the filing-edge deterioration fields.

    Only rows where the ticker is actionable (in_screener OR in_positions) are
    returned, sorted with ``change_direction == -1`` first, then by ascending
    ``text_stability``.
    """
    watch = load_watch(output_dir=output_dir)
    if watch.empty:
        return pd.DataFrame(columns=REPORT_COLUMNS)

    screener = load_screener_tickers(output_dir=output_dir)
    positions = load_position_tickers(positions_path)

    rows = []
    for _, r in watch.iterrows():
        ticker = str(r.get("ticker", "")).strip().upper()
        if not ticker:
            continue
        in_screener = ticker in screener
        in_positions = ticker in positions
        if not (in_screener or in_positions):
            continue
        row = {
            "ticker": ticker,
            "in_screener": in_screener,
            "in_positions": in_positions,
        }
        for col in WATCH_COLUMNS:
            row[col] = r.get(col) if col in watch.columns else None
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=REPORT_COLUMNS)

    report = pd.DataFrame(rows, columns=REPORT_COLUMNS)

    # -1 (deteriorating) first, then least-stable text first.
    report["_dir_rank"] = report["change_direction"].apply(
        lambda v: 0 if pd.notna(v) and v == -1 else 1
    )
    report["_stability"] = pd.to_numeric(
        report["text_stability"], errors="coerce"
    )
    report = report.sort_values(
        by=["_dir_rank", "_stability"],
        ascending=[True, True],
        na_position="last",
    ).drop(columns=["_dir_rank", "_stability"])

    return report.reset_index(drop=True)
