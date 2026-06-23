import glob
import os
from datetime import date

import pandas as pd


def load_streak_history(output_dir: str, lookback_days: int = 14) -> dict[str, dict]:
    """
    Read last N screen_*.csv files (excluding today) and return per-ticker
    appearance count and consecutive trading-day streak.
    """
    today_str = date.today().isoformat()
    pattern = os.path.join(output_dir, "screen_*.csv")
    files = sorted(glob.glob(pattern), reverse=True)  # most-recent first

    files = [f for f in files if os.path.basename(f) != f"screen_{today_str}.csv"]
    files = files[:lookback_days]

    if not files:
        return {}

    ticker_sets: list[set[str]] = []
    for f in files:
        try:
            df = pd.read_csv(f, usecols=["ticker"])
            ticker_sets.append(set(df["ticker"].astype(str)))
        except Exception:
            ticker_sets.append(set())

    all_tickers: set[str] = set().union(*ticker_sets)
    result: dict[str, dict] = {}

    for ticker in all_tickers:
        count = sum(1 for s in ticker_sets if ticker in s)
        consecutive = 0
        for s in ticker_sets:  # index 0 = most recent
            if ticker in s:
                consecutive += 1
            else:
                break
        result[ticker] = {"count": count, "consecutive": consecutive}

    return result
