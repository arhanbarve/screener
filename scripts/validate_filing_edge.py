#!/usr/bin/env python3
"""Filing-edge backtest validation harness.

Reads point-in-time snapshots from filing_edge_snapshots, fetches forward
returns via yfinance, and reports whether high-stability longs outperform
low-stability watches over the stated holding period.

Usage:
    python scripts/validate_filing_edge.py --horizon 21   # 21-trading-day (1 month) forward return
    python scripts/validate_filing_edge.py --horizon 63   # 3 months

Requires: at least 2 weeks of filing_edge snapshots captured by lazy_run.py.
Output: printed report + validation_results_{date}.csv.
"""
import argparse
import sqlite3
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

DB_PATH = "data/cache.db"
OUTPUT_DIR = "output"


def load_snapshots(db_path: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        "SELECT * FROM filing_edge_snapshots ORDER BY snapshot_date, list_type, rank",
        conn,
    )
    conn.close()
    return df


def fetch_forward_return(ticker: str, from_date: str, horizon_days: int) -> float | None:
    """Return the ticker's forward return from from_date over horizon_days trading days.
    Uses yfinance with a 2× calendar buffer to find the right trading-day window.
    """
    try:
        start = pd.Timestamp(from_date)
        # Fetch 1.5× calendar days to ensure we get horizon_days *trading* days
        end = start + timedelta(days=int(horizon_days * 1.5) + 30)
        df = yf.download(ticker, start=start.date(), end=end.date(),
                         progress=False, auto_adjust=True)
        if df is None or len(df) < 2:
            return None
        closes = df["Close"].dropna()
        if len(closes) < horizon_days:
            return None
        entry_price = float(closes.iloc[0])
        exit_price  = float(closes.iloc[horizon_days - 1])
        if entry_price <= 0:
            return None
        return (exit_price - entry_price) / entry_price
    except Exception:
        return None


def run(horizon_days: int, min_snapshots: int):
    snapshots = load_snapshots(DB_PATH)
    if snapshots.empty:
        print("[validate] No snapshots found. Run lazy_run.py for several days first.")
        sys.exit(1)

    snap_dates = sorted(snapshots["snapshot_date"].unique())
    today_str = date.today().isoformat()

    # Only validate snapshot dates old enough that forward return is observable
    min_age = timedelta(days=int(horizon_days * 1.5) + 5)
    eligible = [d for d in snap_dates if (pd.Timestamp(today_str) - pd.Timestamp(d)).days >= min_age.days]
    if len(eligible) < min_snapshots:
        print(f"[validate] Only {len(eligible)} eligible snapshot dates "
              f"(need ≥{min_snapshots} with ≥{min_age.days} calendar days of history).")
        print("Dates available:", snap_dates)
        sys.exit(1)

    print(f"[validate] {len(eligible)} snapshot dates eligible for {horizon_days}-day forward return test")

    rows = []
    for snap_date in eligible:
        day_snaps = snapshots[snapshots["snapshot_date"] == snap_date]
        for _, row in day_snaps.iterrows():
            ticker = str(row["ticker"])
            list_type = str(row["list_type"])
            fwd = fetch_forward_return(ticker, snap_date, horizon_days)
            rows.append({
                "snapshot_date": snap_date,
                "ticker": ticker,
                "list_type": list_type,
                "rank": int(row["rank"]),
                "text_stability": float(row["text_stability"]),
                "composite": float(row["composite"]),
                "forward_return": fwd,
            })
        print(f"  {snap_date}: fetched {len(day_snaps)} tickers")

    results = pd.DataFrame(rows)
    results = results[results["forward_return"].notna()]

    if results.empty:
        print("[validate] No valid forward returns found.")
        sys.exit(1)

    # ── Summary statistics ──────────────────────────────────────────────────
    longs = results[results["list_type"] == "long"]
    watch = results[results["list_type"] == "watch"]

    print(f"\n{'='*60}")
    print(f"FILING EDGE VALIDATION — {horizon_days}-day forward return")
    print(f"{'='*60}")
    print(f"Snapshots:        {results['snapshot_date'].nunique()}")
    print(f"Long observations: {len(longs)}")
    print(f"Watch observations:{len(watch)}")
    print()

    if len(longs) > 0:
        print(f"LONGS (stable-filing):     mean={longs['forward_return'].mean():.2%}  "
              f"median={longs['forward_return'].median():.2%}  "
              f"win_rate={( longs['forward_return'] > 0).mean():.1%}")
    if len(watch) > 0:
        print(f"WATCH (deteriorating):     mean={watch['forward_return'].mean():.2%}  "
              f"median={watch['forward_return'].median():.2%}  "
              f"win_rate={(watch['forward_return'] > 0).mean():.1%}")

    if len(longs) > 0 and len(watch) > 0:
        spread = longs['forward_return'].mean() - watch['forward_return'].mean()
        print(f"\nLong-Watch spread:         {spread:+.2%}")
        print()
        if spread > 0.02:
            print("✅ POSITIVE SPREAD — consistent with Lazy Prices anomaly.")
        elif spread > 0:
            print("⚠️  Small positive spread — may need more data.")
        else:
            print("❌ NO SPREAD or negative — anomaly not detected in this sample.")
            print("   Consider: larger universe, longer history, or different holding period.")

    # Stability buckets
    if len(results) >= 10:
        med_stab = results["text_stability"].median()
        high = results[results["text_stability"] >= med_stab]
        low  = results[results["text_stability"] <  med_stab]
        print(f"\nStability bucket split (median={med_stab:.4f}):")
        print(f"  High stability (n={len(high)}): mean={high['forward_return'].mean():.2%}")
        print(f"  Low  stability (n={len(low)}):  mean={low['forward_return'].mean():.2%}")

    # Save detailed results
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    out_path = f"{OUTPUT_DIR}/validation_results_{today_str}.csv"
    results.to_csv(out_path, index=False, float_format="%.6f")
    print(f"\n[validate] Detailed results → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=21,
                        help="Trading days of forward return (default 21 ≈ 1 month)")
    parser.add_argument("--min-snapshots", type=int, default=3,
                        help="Minimum number of snapshot dates required")
    args = parser.parse_args()
    run(args.horizon, args.min_snapshots)
