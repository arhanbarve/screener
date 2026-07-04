import pandas as pd

from src.sc13d_small_backtest import (
    GATE_HORIZONS,
    HORIZONS,
    control_cap_for,
    split_half,
)


def test_gate_horizons_are_preregistered():
    # Registry Q11: h5/h20/h40 all declared (activist drift is weeks-scale).
    assert HORIZONS == (5, 20, 40)
    assert GATE_HORIZONS == (5, 20, 40)


def test_control_cap_is_three_x_signal():
    assert control_cap_for(100) == 300


def test_split_half_handles_empty():
    assert split_half(pd.DataFrame(columns=["file_date"])).empty
