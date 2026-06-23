import os
import tempfile
from datetime import date, timedelta

import pandas as pd
import pytest
from src.streak import load_streak_history


def _write_screen(dir_path: str, date_str: str, tickers: list[str]) -> None:
    df = pd.DataFrame({"ticker": tickers})
    df.to_csv(os.path.join(dir_path, f"screen_{date_str}.csv"), index=False)


def test_no_files_returns_empty_dict():
    with tempfile.TemporaryDirectory() as d:
        result = load_streak_history(d, lookback_days=14)
    assert result == {}


def test_single_file_gives_count_1():
    with tempfile.TemporaryDirectory() as d:
        yesterday = (date.today() - timedelta(days=1)).isoformat()
        _write_screen(d, yesterday, ["AAPL", "MSFT", "GOOG"])
        result = load_streak_history(d, lookback_days=14)
    assert result["AAPL"] == {"count": 1, "consecutive": 1}
    assert result["MSFT"] == {"count": 1, "consecutive": 1}
    assert result["GOOG"] == {"count": 1, "consecutive": 1}


def test_streak_consecutive_breaks_correctly():
    with tempfile.TemporaryDirectory() as d:
        # d0 = most recent (yesterday), d1 = 2 days ago, d2 = 3 days ago
        d0 = (date.today() - timedelta(days=1)).isoformat()
        d1 = (date.today() - timedelta(days=2)).isoformat()
        d2 = (date.today() - timedelta(days=3)).isoformat()
        _write_screen(d, d0, ["A", "B"])
        _write_screen(d, d1, ["A", "B"])
        _write_screen(d, d2, ["B", "C"])  # A absent here
        result = load_streak_history(d, lookback_days=14)

    assert result["A"]["count"] == 2
    assert result["A"]["consecutive"] == 2   # in d0 and d1 but not d2 → streak stops at 2
    assert result["B"]["count"] == 3
    assert result["B"]["consecutive"] == 3   # in all three
    assert result["C"]["count"] == 1
    assert result["C"]["consecutive"] == 0   # in d2 only (most-recent is d0, so first miss = 0)


def test_today_file_excluded():
    with tempfile.TemporaryDirectory() as d:
        today_str = date.today().isoformat()
        yesterday = (date.today() - timedelta(days=1)).isoformat()
        _write_screen(d, today_str, ["ONLY_TODAY"])
        _write_screen(d, yesterday, ["YESTERDAY"])
        result = load_streak_history(d, lookback_days=14)

    assert "ONLY_TODAY" not in result
    assert "YESTERDAY" in result


def test_lookback_limit():
    with tempfile.TemporaryDirectory() as d:
        # 10 files, but lookback=3 — only last 3 counted
        for i in range(1, 11):
            date_str = (date.today() - timedelta(days=i)).isoformat()
            _write_screen(d, date_str, ["TICK"])
        result = load_streak_history(d, lookback_days=3)

    assert result["TICK"]["count"] == 3
    assert result["TICK"]["consecutive"] == 3
