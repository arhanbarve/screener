"""Portfolio simulator tests — synthetic panels, no network."""
import numpy as np
import pandas as pd
import pytest

from src.portfolio_sim import cagr, max_drawdown, sharpe, per_year


def test_cagr_known_curve():
    idx = pd.bdate_range("2020-01-01", periods=504)  # ~2 years
    equity = pd.Series(np.linspace(100.0, 144.0, 504), index=idx)
    years = (idx[-1] - idx[0]).days / 365.25
    assert cagr(equity) == pytest.approx((144 / 100) ** (1 / years) - 1, rel=1e-9)


def test_max_drawdown_known_curve():
    idx = pd.bdate_range("2020-01-01", periods=5)
    equity = pd.Series([100, 120, 90, 110, 130], index=idx, dtype=float)
    dd, dd_days = max_drawdown(equity)
    assert dd == pytest.approx(-0.25)  # 120 -> 90
    # convention: dd_days = consecutive trading days spent below the running
    # peak. 90 and 110 are below the 120 peak; 130 recovers -> 2 days.
    assert dd_days == 2


def test_sharpe_zero_vol_is_nan():
    idx = pd.bdate_range("2020-01-01", periods=10)
    flat = pd.Series(100.0, index=idx)
    assert np.isnan(sharpe(flat))


def test_per_year_table():
    idx = pd.bdate_range("2020-06-01", "2021-06-01")
    equity = pd.Series(np.linspace(100, 120, len(idx)), index=idx)
    table = per_year(equity)
    assert set(table.index) == {2020, 2021}
    assert (table["return"] > 0).all()
