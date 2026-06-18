# tests/test_prices.py
import pandas as pd
import numpy as np
import tempfile, os
from unittest.mock import patch, MagicMock
from src.cache import init_db
from src.prices import compute_price_factors, apply_liquidity_gate

def make_ohlcv(n=252, start_price=50.0):
    prices = np.linspace(start_price, start_price * 1.1, n)
    idx = pd.date_range(end="2024-01-31", periods=n, freq="B")
    return pd.DataFrame({
        "open": prices * 0.99,
        "high": prices * 1.01,
        "low":  prices * 0.98,
        "close": prices,
        "volume": np.ones(n) * 2_000_000,
    }, index=idx)

def test_compute_price_factors_returns_required_columns():
    df = make_ohlcv(252)
    spy = make_ohlcv(252, start_price=450.0)
    result = compute_price_factors("AAPL", df, spy, market_cap=5e11)
    for col in ["mom_12_1", "rs_6m", "pct_from_high", "avg_dollar_vol_20d", "market_cap", "mom_1m"]:
        assert col in result, f"Missing column: {col}"

def test_compute_price_factors_skips_short_series():
    df = make_ohlcv(100)  # < 252
    spy = make_ohlcv(252, start_price=450.0)
    result = compute_price_factors("AAPL", df, spy, market_cap=5e11)
    assert result is None

def test_apply_liquidity_gate():
    rows = [
        {"ticker": "A", "market_cap": 400e6, "avg_dollar_vol_20d": 10e6},
        {"ticker": "B", "market_cap": 200e6, "avg_dollar_vol_20d": 10e6},  # mcap fail
        {"ticker": "C", "market_cap": 400e6, "avg_dollar_vol_20d": 2e6},   # vol fail
    ]
    df = pd.DataFrame(rows)
    cfg = {"liquidity_gate": {"min_market_cap": 300e6, "min_avg_dollar_vol_20d": 5e6}}
    result = apply_liquidity_gate(df, cfg)
    assert list(result["ticker"]) == ["A"]
