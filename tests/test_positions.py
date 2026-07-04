import json
import math
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch

from src.positions import (
    load_positions,
    save_positions,
    add_position,
    remove_position,
    compute_exit_signals,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_ohlcv(n: int = 60, trend: str = "up") -> pd.DataFrame:
    """Synthetic OHLCV. trend='up' → rising prices; 'down' → falling."""
    idx = pd.date_range("2026-01-01", periods=n, freq="B")
    if trend == "up":
        close = pd.Series([100 + i * 1.5 for i in range(n)], index=idx)
    else:
        close = pd.Series([200 - i * 1.5 for i in range(n)], index=idx)
    high = close * 1.01
    low = close * 0.99
    volume = pd.Series([1_000_000] * n, index=idx)
    return pd.DataFrame({"open": close, "high": high, "low": low, "close": close, "volume": volume})


# ── CRUD tests ────────────────────────────────────────────────────────────────

def test_load_positions_missing_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = load_positions()
    assert result == []


def test_save_and_load_roundtrip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    positions = [{"ticker": "AAPL", "entry_date": "2026-06-01", "entry_price": 150.0}]
    save_positions(positions)
    assert load_positions() == positions


def test_save_positions_no_tmp_file_left(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    positions = [{"ticker": "AAPL", "entry_date": "2026-06-01", "entry_price": 150.0}]
    save_positions(positions)
    assert not (tmp_path / "positions.tmp").exists()


def test_add_position(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    add_position("TSLA", "2026-06-01", 200.0)
    positions = load_positions()
    assert len(positions) == 1
    assert positions[0]["ticker"] == "TSLA"
    assert positions[0]["entry_price"] == 200.0


def test_add_position_duplicate_raises(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    add_position("AAPL", "2026-06-01", 150.0)
    with pytest.raises(ValueError, match="AAPL already in open positions"):
        add_position("aapl", "2026-06-02", 155.0)


def test_remove_position(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    add_position("AAPL", "2026-06-01", 150.0)
    add_position("MSFT", "2026-06-01", 300.0)
    remove_position("AAPL")
    positions = load_positions()
    assert len(positions) == 1
    assert positions[0]["ticker"] == "MSFT"


def test_remove_position_noop_if_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    add_position("AAPL", "2026-06-01", 150.0)
    remove_position("ZZZZ")  # should not raise
    assert len(load_positions()) == 1


# ── compute_exit_signals tests ────────────────────────────────────────────────

_SOFT_FLAGS = ("macd", "sma50_break", "sma20_break", "rsi", "stoch",
               "mfi", "obv_dist", "adx", "bb_blowoff", "rs_decay")
_GRADES = {"HOLD", "WATCH", "TRIM", "STRONG EXIT"}


def test_exit_signals_empty_df():
    result = compute_exit_signals(pd.DataFrame())
    assert result["score"] == 0
    assert result["soft_score"] == 0
    assert result["grade"] == "HOLD"
    assert result["hard_exit"] is False
    assert result["rsi"] is None
    assert result["macd"] is None


def test_exit_signals_too_short():
    df = _make_ohlcv(n=35)
    result = compute_exit_signals(df)
    assert result["score"] == 0
    assert result["grade"] == "HOLD"
    assert result["hard_exit"] is False


def test_exit_signals_returns_expected_keys():
    df = _make_ohlcv(n=60)
    result = compute_exit_signals(df)
    # Legacy display keys still present (UI backward-compat) plus new tiered keys.
    expected = {
        "rsi", "macd", "stoch", "adx", "mfi", "score",
        "rsi_val", "adx_val", "mfi_val", "stoch_k", "stoch_d", "macd_state",
        "hard_exit", "hard_reasons", "chandelier", "death_cross",
        "sma20_break", "sma50_break", "obv_dist", "bb_blowoff", "rs_decay",
        "soft_score", "soft_max", "grade", "days_to_earnings",
    }
    assert expected.issubset(set(result.keys()))


def test_exit_signals_score_is_weighted_soft_sum():
    df = _make_ohlcv(n=60)
    result = compute_exit_signals(df)
    # score is a legacy alias for the weighted soft score, not a count of flags.
    assert result["score"] == result["soft_score"]


def test_exit_signals_soft_score_bounded():
    df = _make_ohlcv(n=60)
    result = compute_exit_signals(df)
    assert 0 <= result["soft_score"] <= result["soft_max"]
    assert result["soft_max"] == 12


def test_exit_signals_grade_valid():
    for trend in ("up", "down"):
        result = compute_exit_signals(_make_ohlcv(n=260, trend=trend))
        assert result["grade"] in _GRADES


def test_exit_signals_hard_tier_needs_long_history():
    # < 210 bars: hard tier cannot be evaluated (no SMA200).
    result = compute_exit_signals(_make_ohlcv(n=60, trend="down"))
    assert result["hard_exit"] is False
    assert result["chandelier"] is None
    assert result["death_cross"] is None


def test_exit_signals_hard_exit_fires_in_downtrend():
    # Sustained downtrend with enough history: price below SMA200 + 50<200.
    result = compute_exit_signals(_make_ohlcv(n=260, trend="down"))
    assert result["hard_exit"] is True
    assert result["death_cross"] is True
    assert result["grade"] == "STRONG EXIT"


def test_exit_signals_hard_exit_quiet_in_uptrend():
    # Clean uptrend above the 200-day must never trigger a hard exit.
    result = compute_exit_signals(_make_ohlcv(n=260, trend="up"))
    assert result["hard_exit"] is False
    assert result["death_cross"] is False


def test_exit_signals_rsi_val_is_float_or_none():
    df = _make_ohlcv(n=60)
    result = compute_exit_signals(df)
    if result["rsi_val"] is not None:
        assert isinstance(result["rsi_val"], float)
        assert 0 <= result["rsi_val"] <= 100


def test_exit_signals_macd_state_valid_values():
    df = _make_ohlcv(n=60)
    result = compute_exit_signals(df)
    valid = {"bullish", "bullish_cross", "bearish", "bearish_cross", "unknown", None}
    assert result["macd_state"] in valid


def test_exit_signals_missing_columns():
    df = pd.DataFrame({"close": [100.0] * 50})  # missing high/low/volume
    result = compute_exit_signals(df)
    assert result["score"] == 0
    assert result["grade"] == "HOLD"
    assert result["rsi"] is None
