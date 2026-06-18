# tests/test_factors.py
import numpy as np
import pandas as pd
import pytest
from src.factors import (
    mom_12_1, mom_1m, compute_sue, rev_breadth_score,
    gp_assets_score, rs_vs_spy, pct_from_52w_high,
    breakout_flag, avg_dollar_vol, squeeze_flag,
)

def make_price_series(n: int = 252) -> pd.Series:
    """Linear ramp: 100 at t-252 → 110 at t-21 → 112 at t=0."""
    prices = np.linspace(100.0, 110.0, n - 21)
    tail = np.linspace(110.0, 112.0, 21 + 1)[1:]
    all_prices = np.concatenate([prices, tail])
    idx = pd.date_range(end="2024-01-31", periods=n, freq="B")
    return pd.Series(all_prices, index=idx)

def test_mom_12_1_known_value():
    close = make_price_series(252)
    result = mom_12_1(close)
    assert abs(result - 0.10) < 0.01

def test_mom_1m_known_value():
    close = make_price_series(252)
    result = mom_1m(close)
    expected = (112.0 / 110.0) - 1.0
    assert abs(result - expected) < 0.01

def test_mom_12_1_requires_252_bars():
    close = make_price_series(252).iloc[:200]
    with pytest.raises(ValueError, match="252"):
        mom_12_1(close)

def test_sue_with_four_quarters():
    actuals   = [1.0, 1.1, 1.2, 1.3]
    estimates = [0.9, 1.0, 1.1, 1.0]
    result = compute_sue(actuals, estimates)
    assert abs(result - 3.0) < 1e-6

def test_sue_fallback_fewer_than_four_quarters():
    actuals   = [1.3]
    estimates = [1.0]
    result = compute_sue(actuals, estimates)
    assert abs(result - 0.3) < 1e-6

def test_rev_breadth_score_basic():
    result = rev_breadth_score(n_up=3, n_down=1, n_total=5)
    assert abs(result - 0.4) < 1e-6

def test_rev_breadth_score_magnitude_fallback():
    result = rev_breadth_score(n_up=0, n_down=0, n_total=0,
                               consensus_now=1.1, consensus_90d_ago=1.0)
    assert abs(result - 0.1) < 1e-6

def test_gp_assets_score():
    result = gp_assets_score(revenue=100.0, cogs=60.0, assets=200.0)
    assert abs(result - 0.20) < 1e-6

def test_rs_vs_spy():
    stock = make_price_series(252)
    spy   = make_price_series(252) * 0.95
    result = rs_vs_spy(stock, spy, window=126)
    assert result > 0

def test_pct_from_52w_high():
    close = make_price_series(252)
    result = pct_from_52w_high(close)
    assert abs(result) < 1e-4

def test_breakout_flag_triggers():
    close  = make_price_series(252)
    volumes = pd.Series(np.ones(252) * 1e6, index=close.index)
    volumes.iloc[-1] = 2.5e6
    result = breakout_flag(close, volumes)
    assert result is True

def test_avg_dollar_vol():
    close   = pd.Series([10.0] * 20)
    volumes = pd.Series([1_000_000] * 20)
    result  = avg_dollar_vol(close, volumes, window=20)
    assert abs(result - 10_000_000.0) < 1.0

def test_squeeze_flag_triggers():
    result = squeeze_flag(short_float=0.20, days_to_cover=6.0, mom_1m_val=0.05)
    assert result is True

def test_squeeze_flag_no_trigger_low_short():
    result = squeeze_flag(short_float=0.05, days_to_cover=6.0, mom_1m_val=0.05)
    assert result is False
