"""Factor panel tests — synthetic prices, no network."""
import numpy as np
import pandas as pd
import pytest

from src.factor_panel import rebalance_dates, ticker_factor_frame
from src import factors


def _synthetic_prices(n=600, seed=7, drift=0.0005, vol=0.02, start="2021-01-04"):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    close = pd.Series(100 * np.cumprod(1 + rng.normal(drift, vol, n)), index=idx)
    volume = pd.Series(rng.integers(200_000, 2_000_000, n).astype(float), index=idx)
    return pd.DataFrame({"close": close, "volume": volume})


def test_rebalance_dates_last_trading_day_of_week():
    idx = pd.bdate_range("2024-01-01", "2024-01-31")
    dates = rebalance_dates(idx)
    # January 2024: Fridays are 5, 12, 19, 26; month ends Wed 31st
    got = [d.strftime("%Y-%m-%d") for d in dates]
    assert got == ["2024-01-05", "2024-01-12", "2024-01-19", "2024-01-26", "2024-01-31"]


def test_ticker_factor_frame_matches_scalar_functions():
    df = _synthetic_prices()
    spy = _synthetic_prices(seed=99, drift=0.0003, vol=0.01)
    dates = rebalance_dates(df.index)[-5:]

    frame = ticker_factor_frame(df, spy["close"], dates)
    assert list(frame.index) == list(dates)

    t = dates[-1]
    hist = df.loc[:t]
    spy_hist = spy.loc[:t, "close"]
    row = frame.loc[t]

    assert row["mom_12_1"] == pytest.approx(factors.mom_12_1(hist["close"]), rel=1e-9)
    assert row["rs_6m"] == pytest.approx(
        factors.rs_vs_spy(hist["close"], spy_hist, window=126), rel=1e-9)
    rs_3m = factors.rs_vs_spy(hist["close"], spy_hist, window=63)
    assert row["rs_accel"] == pytest.approx(2 * rs_3m - row["rs_6m"], rel=1e-9)
    assert row["rs_slope"] == pytest.approx(
        factors.rs_slope(hist["close"], spy_hist, window=63), rel=1e-6)
    assert row["residual_mom"] == pytest.approx(
        factors.residual_momentum(hist["close"], spy_hist), rel=1e-9)
    assert row["pct_from_high"] == pytest.approx(
        factors.pct_from_52w_high(hist["close"]), rel=1e-9)
    assert row["dollar_vol_20d"] == pytest.approx(
        factors.avg_dollar_vol(hist["close"], hist["volume"]), rel=1e-9)
    # gate ingredients
    assert row["above_sma200"] == (hist["close"].iloc[-1] >= hist["close"].rolling(200).mean().iloc[-1])
    assert row["close"] == pytest.approx(float(hist["close"].iloc[-1]))


def test_ticker_factor_frame_nan_when_insufficient_history():
    df = _synthetic_prices(n=100)
    spy = _synthetic_prices(n=100, seed=99)
    dates = rebalance_dates(df.index)[-2:]
    frame = ticker_factor_frame(df, spy["close"], dates)
    assert np.isnan(frame.loc[dates[-1], "mom_12_1"])
