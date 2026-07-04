import pandas as pd

from src.backtest_recipe import as_net_frame
from src.insider_small_backtest import (
    GATE_HORIZONS,
    HORIZONS,
    LOOKBACK_DAYS,
    filter_lookback,
)


def test_gate_horizons_are_preregistered():
    # Registry Q10: h20/h40 declared (front-loaded ~1 month), h5 report-only.
    assert HORIZONS == (5, 20, 40)
    assert GATE_HORIZONS == (20, 40)


def test_filter_lookback_drops_old_events():
    events = pd.DataFrame({
        "filing_date": pd.to_datetime(["2020-01-02", "2026-01-02"]),
        "ticker": ["OLD", "NEW"],
    })
    out = filter_lookback(events, today=pd.Timestamp("2026-07-04"),
                          lookback_days=LOOKBACK_DAYS)
    assert list(out["ticker"]) == ["NEW"]


def test_as_net_frame_swaps_columns_shared_helper():
    row = {"cap_proxy": 6e8}
    for h in (5, 20, 40):
        row[f"abn_ret_{h}d"] = 0.03
        row[f"abn_ret_net_{h}d"] = 0.01
    net = as_net_frame(pd.DataFrame([row]), horizons=(5, 20, 40))
    for h in (5, 20, 40):
        assert net[f"abn_ret_{h}d"].iloc[0] == 0.01
