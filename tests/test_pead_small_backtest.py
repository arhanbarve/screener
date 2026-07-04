import numpy as np
import pandas as pd

from src.backtest_recipe import (
    attach_dollar_vol,
    attach_net_returns,
    band_cost,
    filter_cap_proxy,
    filter_dollar_vol,
)
from src.pead_small_backtest import (
    HORIZONS,
    _as_net_frame,
    beat_vs_miss_spread,
)


def _prices(days, close=10.0, volume=1_000_000):
    idx = pd.bdate_range("2026-01-01", periods=days)
    return pd.DataFrame({"close": close, "volume": volume}, index=idx)


def test_filter_cap_proxy_band_keeps_only_in_band():
    events = pd.DataFrame({
        "cap_proxy": [4e8, 5e8, 1.5e9, 2.5e9, np.nan],
    })
    out = filter_cap_proxy(events, min_cap=5e8, max_cap=2.5e9)
    assert list(out["cap_proxy"]) == [5e8, 1.5e9]  # 2.5e9 excluded (< max_cap)


def test_filter_cap_proxy_floor_only_unchanged():
    events = pd.DataFrame({"cap_proxy": [1e9, 3e9, np.nan]})
    out = filter_cap_proxy(events, min_cap=2.5e9)
    assert list(out["cap_proxy"]) == [3e9]


def test_attach_dollar_vol_uses_pre_event_window():
    prices = _prices(40, close=10.0, volume=500_000)  # $5M/day
    events = pd.DataFrame({
        "ticker": ["AAA"],
        "event_date": [prices.index[30].date().isoformat()],
    })
    out = attach_dollar_vol(events, {"AAA": prices})
    assert out["dollar_vol_20d"].iloc[0] == 5_000_000.0


def test_attach_dollar_vol_nan_when_history_short():
    prices = _prices(10)
    events = pd.DataFrame({
        "ticker": ["AAA"],
        "event_date": [prices.index[9].date().isoformat()],
    })
    out = attach_dollar_vol(events, {"AAA": prices})
    assert pd.isna(out["dollar_vol_20d"].iloc[0])


def test_filter_dollar_vol_floor():
    events = pd.DataFrame({"dollar_vol_20d": [1e6, 2e6, np.nan, 5e6]})
    out = filter_dollar_vol(events, min_dollar_vol=2e6)
    assert list(out["dollar_vol_20d"]) == [2e6, 5e6]


def test_band_cost_tiers():
    assert band_cost(6e8) == 0.0040
    assert band_cost(1.5e9) == 0.0025
    assert band_cost(5e9) == 0.0015
    assert pd.isna(band_cost(float("nan")))


def test_attach_net_returns_subtracts_band_cost():
    events = pd.DataFrame({
        "cap_proxy": [6e8, 1.5e9],
        "abn_ret_5d": [0.02, 0.02],
    })
    out = attach_net_returns(events, horizons=(5,))
    assert out["abn_ret_net_5d"].iloc[0] == 0.02 - 0.0040
    assert out["abn_ret_net_5d"].iloc[1] == 0.02 - 0.0025
    assert out["abn_ret_5d"].iloc[0] == 0.02  # gross column untouched


def test_as_net_frame_swaps_columns():
    row = {"cap_proxy": 6e8}
    for h in HORIZONS:
        row[f"abn_ret_{h}d"] = 0.03
        row[f"abn_ret_net_{h}d"] = 0.01
    events = pd.DataFrame([row])
    net = _as_net_frame(events)
    for h in HORIZONS:
        assert net[f"abn_ret_{h}d"].iloc[0] == 0.01


def test_beat_vs_miss_spread_detects_separation():
    rows = []
    rng = np.random.default_rng(0)
    for _ in range(30):
        r = {"category": "pead_beat_large"}
        for h in HORIZONS:
            r[f"abn_ret_{h}d"] = 0.03 + rng.normal(0, 0.005)
        rows.append(r)
    for _ in range(30):
        r = {"category": "pead_miss"}
        for h in HORIZONS:
            r[f"abn_ret_{h}d"] = -0.01 + rng.normal(0, 0.005)
        rows.append(r)
    out = beat_vs_miss_spread(pd.DataFrame(rows))
    assert (out["p_spread"] < 0.01).all()
    assert (out["spread"] > 0.03).all()


def test_beat_vs_miss_spread_small_n_returns_none():
    rows = []
    for cat in ("pead_beat_small", "pead_miss"):
        r = {"category": cat}
        for h in HORIZONS:
            r[f"abn_ret_{h}d"] = 0.01
        rows.append(r)
    out = beat_vs_miss_spread(pd.DataFrame(rows))
    assert out["p_spread"].isna().all()
