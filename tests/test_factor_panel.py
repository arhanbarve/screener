"""Factor panel tests — synthetic prices, no network."""
import numpy as np
import pandas as pd
import pytest

from src.factor_panel import rebalance_dates, ticker_factor_frame, _monthly_asof
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


def test_ticker_factor_frame_above_sma200_is_nullable_boolean():
    # Mix an insufficient-history row (NaN placeholder) with computed rows
    # to reproduce the dtype-coercion landmine: bool(float('nan')) is True,
    # so a plain float/object column would make the NaN row look like it
    # passes the SMA200 gate under a naive `if row["above_sma200"]:` check.
    df = _synthetic_prices(n=300)
    spy = _synthetic_prices(n=300, seed=99)
    all_dates = rebalance_dates(df.index)
    dates = [all_dates[0], all_dates[-1]]  # first: insufficient history; last: computed
    frame = ticker_factor_frame(df, spy["close"], dates)

    assert np.isnan(frame.loc[dates[0], "close"])  # insufficient-history row
    assert str(frame["above_sma200"].dtype) == "boolean"
    assert pd.isna(frame["above_sma200"].iloc[0])
    assert not bool(frame["above_sma200"].iloc[0] is True)


def _monthly_series(idx, values):
    s = pd.Series(values, index=idx, dtype=float)
    monthly = s.resample("ME").last()
    return s, monthly, monthly.index.to_period("M")


def test_monthly_asof_mid_month_uses_daily_value_not_future_month_end():
    idx = pd.bdate_range("2024-01-02", "2024-03-29")
    daily = pd.Series(np.arange(len(idx), dtype=float) + 100.0, index=idx)
    monthly = daily.resample("ME").last()
    monthly_periods = monthly.index.to_period("M")

    t = pd.Timestamp("2024-02-15")  # mid-February, not a month end
    m = _monthly_asof(monthly, daily, t, monthly_periods)

    assert len(m) == 2  # Jan month-end + Feb (in-progress) bin only
    assert m.index[-1].to_period("M") == t.to_period("M")
    assert m.iloc[-1] == daily.loc[t]  # overwritten with t's value, not Feb's true month-end
    assert m.iloc[-1] != monthly.loc[monthly_periods == t.to_period("M")].iloc[0]
    assert m.iloc[0] == monthly.iloc[0]  # completed Jan bin untouched


def test_monthly_asof_on_month_end_matches_full_month_value():
    idx = pd.bdate_range("2024-01-02", "2024-03-29")
    daily = pd.Series(np.arange(len(idx), dtype=float) + 100.0, index=idx)
    monthly = daily.resample("ME").last()
    monthly_periods = monthly.index.to_period("M")

    t = monthly.index[monthly_periods == pd.Period("2024-02", "M")][0]
    m = _monthly_asof(monthly, daily, t, monthly_periods)

    assert len(m) == 2  # Jan, Feb only -- March excluded
    assert m.index[-1] == t
    assert m.iloc[-1] == daily.loc[t]


def test_monthly_asof_before_any_month_end_formed_returns_empty():
    # `monthly` here only has Feb/Mar bins (as if the ticker's monthly
    # resample hasn't produced a Jan bin yet); t sits in January, before
    # any formed month-end, so the result must be empty rather than
    # raising or fabricating a value.
    idx = pd.bdate_range("2024-02-01", "2024-03-29")
    daily = pd.Series(np.arange(len(idx), dtype=float) + 100.0, index=idx)
    monthly = daily.resample("ME").last()
    monthly_periods = monthly.index.to_period("M")

    t = pd.Timestamp("2024-01-10")
    m = _monthly_asof(monthly, daily, t, monthly_periods)

    assert len(m) == 0


from src.factor_panel import (
    apply_gates, composite_scores, breadth_series,
    PRICE_BLOCK_WEIGHTS, GATE_MIN_MCAP, GATE_MIN_DOLLAR_VOL, GATE_MAX_BELOW_HIGH,
)


def _panel_row(ticker, date, **overrides):
    row = {
        "ticker": ticker, "date": pd.Timestamp(date), "close": 50.0,
        "mom_12_1": 0.2, "residual_mom": 1.0, "rs_6m": 0.05, "rs_accel": 0.01,
        "rs_slope": 0.001, "pct_from_high": -0.05, "dollar_vol_20d": 10e6,
        "above_sma200": True, "mcap": 1e9,
    }
    row.update(overrides)
    return row


def test_apply_gates():
    d = "2023-06-02"
    panel = pd.DataFrame([
        _panel_row("PASS", d),
        _panel_row("SMLL", d, mcap=100e6),                    # below $300M
        _panel_row("THIN", d, dollar_vol_20d=1e6),            # below $5M ADV
        _panel_row("BELW", d, above_sma200=False),            # below SMA200
        _panel_row("DEEP", d, pct_from_high=-0.50),           # >35% off high
        _panel_row("NODA", d, mom_12_1=np.nan),               # missing factor
    ])
    gated = apply_gates(panel)
    assert list(gated.loc[gated["passes_gates"], "ticker"]) == ["PASS"]


def test_composite_scores_zscore_and_weights():
    d = pd.Timestamp("2023-06-02")
    # three passing tickers, factors constructed so only mom_12_1 differs
    base = dict(residual_mom=1.0, rs_6m=0.05, rs_accel=0.01,
                rs_slope=0.001, pct_from_high=-0.05)
    panel = pd.DataFrame([
        _panel_row("A", d, mom_12_1=0.30, **base),
        _panel_row("B", d, mom_12_1=0.20, **base),
        _panel_row("C", d, mom_12_1=0.10, **base),
    ])
    panel["passes_gates"] = True
    scored = composite_scores(panel)
    by_t = scored.set_index("ticker")
    # identical factors z to 0; mom_12_1 z-scores are +1.09.., 0, -1.09.. (ddof=1... pandas std default ddof=1)
    assert by_t.loc["A", "composite"] > by_t.loc["B", "composite"] > by_t.loc["C", "composite"]
    assert by_t.loc["B", "composite"] == pytest.approx(0.0, abs=1e-12)
    # composite = w_mom * z_mom exactly, since all other z are 0
    z_a = (0.30 - 0.20) / pd.Series([0.30, 0.20, 0.10]).std()
    assert by_t.loc["A", "composite"] == pytest.approx(
        PRICE_BLOCK_WEIGHTS["mom_12_1"] * z_a, rel=1e-9)


def test_breadth_series_fractions():
    idx = pd.bdate_range("2021-01-04", periods=300)
    up = pd.DataFrame({"close": np.linspace(10, 40, 300),
                       "volume": np.full(300, 1e6)}, index=idx)
    down = pd.DataFrame({"close": np.linspace(40, 10, 300),
                         "volume": np.full(300, 1e6)}, index=idx)
    b = breadth_series({"UP": up, "DOWN": down})
    last = b.iloc[-1]
    assert last["pct_above_200"] == pytest.approx(0.5)   # UP above, DOWN below
    assert last["pct_above_50"] == pytest.approx(0.5)
    # before any ticker has 200 bars, the 200d breadth is undefined
    assert np.isnan(b.iloc[100]["pct_above_200"])
