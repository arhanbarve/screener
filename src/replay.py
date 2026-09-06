"""Replay harness: what did the daily screens actually do afterwards?

    python -m src.replay [--top 20] [--horizons 5,10,20] [--min-rows 15]

For every output/screen_YYYY-MM-DD.csv, take the top N tickers and measure the
average forward return over each horizon from that day's close, minus SPY over
the same window, using the OHLCV already in data/cache.db (the `prices` table
keeps every fetched bar, so two months of daily runs give two months of
forward returns for free). Also breaks results down by entry grade and by
conviction bucket, and flags screens whose row count is suspiciously small —
the 2026-08-31 → 09-03 screens came from a ~40-name universe and must not be
read as evidence about the strategy.

Pure functions take frames; `main` does the IO. This is a plumbing check and a
small-sample sanity read, not a backtest: ~30 daily screens with overlapping
20-day windows are a handful of independent observations.
"""
import argparse
import glob
import os
import sqlite3
from datetime import date

import numpy as np
import pandas as pd

DEFAULT_HORIZONS = (5, 10, 20)


def load_screens(output_dir: str) -> list[tuple[str, pd.DataFrame]]:
    out = []
    for path in sorted(glob.glob(os.path.join(output_dir, "screen_*.csv"))):
        d = os.path.basename(path)[len("screen_"):-len(".csv")]
        try:
            df = pd.read_csv(path)
        except Exception:  # noqa: BLE001 — a corrupt file is skipped, not fatal
            continue
        if "ticker" in df.columns:
            out.append((d, df))
    return out


def load_closes(db_path: str, tickers: set[str]) -> dict[str, pd.Series]:
    """ticker -> close series indexed by date (Timestamp), from cache.db."""
    if not tickers:
        return {}
    conn = sqlite3.connect(db_path)
    marks = ",".join("?" * len(tickers))
    rows = conn.execute(
        f"SELECT ticker, date, close FROM prices WHERE ticker IN ({marks}) ORDER BY ticker, date",
        sorted(tickers)).fetchall()
    conn.close()
    df = pd.DataFrame(rows, columns=["ticker", "date", "close"])
    if df.empty:
        return {}
    df["date"] = pd.to_datetime(df["date"])
    return {t: g.set_index("date")["close"] for t, g in df.groupby("ticker")}


def forward_return(closes: pd.Series, as_of: str, horizon: int) -> float:
    """Return from the close ON `as_of` (or the last bar before it) to the close
    `horizon` trading bars later. NaN if either end is missing."""
    if closes is None or closes.empty:
        return float("nan")
    ts = pd.Timestamp(as_of)
    idx = closes.index
    start_pos = idx.searchsorted(ts, side="right") - 1
    if start_pos < 0:
        return float("nan")
    end_pos = start_pos + horizon
    if end_pos >= len(idx):
        return float("nan")
    p0, p1 = float(closes.iloc[start_pos]), float(closes.iloc[end_pos])
    if p0 <= 0:
        return float("nan")
    return p1 / p0 - 1.0


def replay(screens: list[tuple[str, pd.DataFrame]], closes: dict[str, pd.Series],
           horizons=DEFAULT_HORIZONS, top_n: int = 20, min_rows: int = 15) -> pd.DataFrame:
    """One row per screen date per horizon: n, mean excess return vs SPY,
    hit rate (share of names beating SPY), plus a `degraded` flag."""
    spy = closes.get("SPY")
    records = []
    for d, df in screens:
        head = df.head(top_n)
        degraded = len(df) < min_rows
        for h in horizons:
            rets, excess = [], []
            spy_r = forward_return(spy, d, h) if spy is not None else float("nan")
            for t in head["ticker"].astype(str):
                r = forward_return(closes.get(t), d, h)
                if r == r:
                    rets.append(r)
                    if spy_r == spy_r:
                        excess.append(r - spy_r)
            if not rets:
                continue
            records.append({
                "date": d, "horizon": h, "n": len(rets), "degraded": degraded,
                "mean_ret": float(np.mean(rets)),
                "spy_ret": spy_r,
                "mean_excess": float(np.mean(excess)) if excess else float("nan"),
                "hit_rate": float(np.mean([e > 0 for e in excess])) if excess else float("nan"),
            })
    return pd.DataFrame(records)


def breakdown(screens: list[tuple[str, pd.DataFrame]], closes: dict[str, pd.Series],
              by: str, horizon: int = 20, top_n: int = 20, min_rows: int = 15) -> pd.DataFrame:
    """Mean excess return grouped by a screen column (entry grade, conviction
    bucket, entry_signal). Degraded screens are excluded."""
    spy = closes.get("SPY")
    rows = []
    for d, df in screens:
        if len(df) < min_rows or by not in df.columns:
            continue
        spy_r = forward_return(spy, d, horizon) if spy is not None else float("nan")
        for _, r in df.head(top_n).iterrows():
            fr = forward_return(closes.get(str(r["ticker"])), d, horizon)
            if fr != fr or spy_r != spy_r:
                continue
            key = r[by]
            if by == "conviction":
                key = "8-10" if key >= 8 else ("5-7" if key >= 5 else "1-4")
            rows.append({"group": str(key), "excess": fr - spy_r})
    if not rows:
        return pd.DataFrame(columns=["group", "n", "mean_excess", "hit_rate"])
    g = pd.DataFrame(rows).groupby("group")["excess"]
    return pd.DataFrame({"n": g.size(), "mean_excess": g.mean(),
                         "hit_rate": g.apply(lambda s: float((s > 0).mean()))}).reset_index()


def summary(table: pd.DataFrame) -> pd.DataFrame:
    """Per-horizon aggregate over non-degraded screens."""
    if table.empty:
        return table
    ok = table[~table["degraded"]]
    if ok.empty:
        return pd.DataFrame()
    g = ok.groupby("horizon")
    return pd.DataFrame({"screens": g.size(), "mean_excess": g["mean_excess"].mean(),
                         "median_excess": g["mean_excess"].median(),
                         "share_positive": g["mean_excess"].apply(lambda s: float((s > 0).mean()))}).reset_index()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="replay", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", default="output")
    p.add_argument("--db", default="data/cache.db")
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--horizons", default="5,10,20")
    p.add_argument("--min-rows", type=int, default=15)
    a = p.parse_args(argv)
    horizons = tuple(int(h) for h in a.horizons.split(","))

    screens = load_screens(a.output_dir)
    tickers = {"SPY"} | {str(t) for _, df in screens for t in df.head(a.top)["ticker"]}
    closes = load_closes(a.db, tickers)
    table = replay(screens, closes, horizons, a.top, a.min_rows)
    pd.set_option("display.width", 200)
    print(f"{len(screens)} screens, {sum(1 for _, d in screens if len(d) < a.min_rows)} flagged degraded (< {a.min_rows} rows)\n")
    print(table.to_string(index=False, float_format=lambda x: f"{x:+.2%}" if abs(x) < 5 else f"{x:.0f}"))
    print("\nSummary (non-degraded screens):")
    print(summary(table).to_string(index=False, float_format=lambda x: f"{x:+.2%}" if abs(x) < 5 else f"{x:.0f}"))
    for by in ("entry", "conviction", "entry_signal"):
        print(f"\nBy {by} (h={horizons[-1]}):")
        print(breakdown(screens, closes, by, horizons[-1], a.top, a.min_rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
