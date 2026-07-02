import glob
import os
from datetime import date

import pandas as pd


def load_filing_streak(output_dir: str, lookback_days: int = 14) -> dict[tuple[str, str], dict]:
    """
    Read last N filing_edge_*.csv files (excluding today) and return per
    (ticker, list_type) appearance count and consecutive trading-day streak.
    """
    today_str = date.today().isoformat()
    pattern = os.path.join(output_dir, "filing_edge_*.csv")
    files = sorted(glob.glob(pattern), reverse=True)  # most-recent first

    files = [f for f in files if os.path.basename(f) != f"filing_edge_{today_str}.csv"]
    files = files[:lookback_days]

    if not files:
        return {}

    long_sets: list[set[str]] = []
    watch_sets: list[set[str]] = []
    for f in files:
        try:
            df = pd.read_csv(f, usecols=["ticker", "__list__"])
            long_sets.append(set(df.loc[df["__list__"] == "long", "ticker"].astype(str)))
            watch_sets.append(set(df.loc[df["__list__"] == "watch", "ticker"].astype(str)))
        except Exception:
            long_sets.append(set())
            watch_sets.append(set())

    all_long_tickers: set[str] = set().union(*long_sets)
    all_watch_tickers: set[str] = set().union(*watch_sets)

    result: dict[tuple[str, str], dict] = {}

    for ticker in all_long_tickers:
        count = sum(1 for s in long_sets if ticker in s)
        consecutive = 0
        for s in long_sets:  # index 0 = most recent
            if ticker in s:
                consecutive += 1
            else:
                break
        result[(ticker, "long")] = {"count": count, "consecutive": consecutive}

    for ticker in all_watch_tickers:
        count = sum(1 for s in watch_sets if ticker in s)
        consecutive = 0
        for s in watch_sets:  # index 0 = most recent
            if ticker in s:
                consecutive += 1
            else:
                break
        result[(ticker, "watch")] = {"count": count, "consecutive": consecutive}

    return result
