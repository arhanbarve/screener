"""Regime ladder signal tests. All series synthetic; index = business days."""
import numpy as np
import pandas as pd
import pytest

from src.regime import (
    trend_signal, breadth_signal, vol_signal, credit_signal,
)
from src.regime import (
    ladder_points, ladder_exposure, thrust_override, combined_exposure,
    THRUST_FLOOR,
)


def _bdays(n, start="2020-01-01"):
    return pd.bdate_range(start, periods=n)


def test_trend_signal_fires_below_sma200():
    idx = _bdays(260)
    # 210 flat days at 100 establish SMA200 ~100, then close drops to 90
    vals = np.concatenate([np.full(210, 100.0), np.full(50, 90.0)])
    sig = trend_signal(pd.Series(vals, index=idx))
    assert bool(sig.iloc[-1]) is True
    assert bool(sig.iloc[209]) is False        # at 100, not below SMA
    assert bool(sig.iloc[100]) is False        # SMA200 not formed yet -> False


def test_breadth_signal_thresholds():
    idx = _bdays(3)
    pct = pd.Series([0.55, 0.39, np.nan], index=idx)
    sig = breadth_signal(pct)
    assert list(sig) == [False, True, False]   # NaN -> False


def test_vol_signal_level_or_inversion():
    idx = _bdays(4)
    vix = pd.Series([15.0, 26.0, 18.0, 15.0], index=idx)
    vix3m = pd.Series([17.0, 28.0, 17.0, np.nan], index=idx)
    sig = vol_signal(vix, vix3m)
    # day0: level ok, 15<17 no inversion -> False
    # day1: 26>25 -> True
    # day2: 18>17 inverted -> True
    # day3: level ok, vix3m NaN -> inversion unknown -> False
    assert list(sig) == [False, True, True, False]


def test_credit_signal_ratio_below_sma100():
    idx = _bdays(160)
    hyg = pd.Series(np.concatenate([np.full(120, 80.0), np.full(40, 72.0)]), index=idx)
    ief = pd.Series(np.full(160, 100.0), index=idx)
    sig = credit_signal(hyg, ief)
    assert bool(sig.iloc[-1]) is True          # ratio dropped below its SMA100
    assert bool(sig.iloc[110]) is False        # flat ratio == SMA -> not below
    assert bool(sig.iloc[50]) is False         # SMA100 not formed -> False


def test_ladder_points_sums_aligned_signals():
    idx = _bdays(3)
    a = pd.Series([True, False, True], index=idx)
    b = pd.Series([True, False, False], index=idx)
    c = pd.Series([False, False, True], index=idx)
    d = pd.Series([False, False, True], index=idx)
    pts = ladder_points(a, b, c, d)
    assert list(pts) == [2, 0, 3]


def test_ladder_exposure_mapping():
    idx = _bdays(5)
    pts = pd.Series([0, 1, 2, 3, 4], index=idx)
    exp = ladder_exposure(pts)
    assert list(exp) == [1.00, 1.00, 0.66, 0.33, 0.00]


def test_thrust_fires_on_cross_and_expires():
    # breadth50: 12 days low (0.10), then jumps to 0.60 -> fires for THRUST_HOLD days
    idx = _bdays(60)
    vals = np.concatenate([np.full(12, 0.10), np.full(48, 0.60)])
    active = thrust_override(pd.Series(vals, index=idx))
    assert bool(active.iloc[11]) is False       # before cross
    assert bool(active.iloc[12]) is True        # cross day
    assert bool(active.iloc[12 + 19]) is True   # last day of hold window
    assert bool(active.iloc[12 + 20]) is False  # expired (no re-fire: no dip below LOW)


def test_thrust_does_not_fire_without_prior_low():
    idx = _bdays(30)
    vals = np.concatenate([np.full(15, 0.45), np.full(15, 0.60)])  # never below 0.20
    active = thrust_override(pd.Series(vals, index=idx))
    assert not active.any()


def test_combined_exposure_applies_floor_only_when_thrust_active():
    idx = _bdays(4)
    pts = pd.Series([4, 4, 2, 0], index=idx)
    thrust = pd.Series([True, False, True, False], index=idx)
    exp = combined_exposure(pts, thrust)
    assert list(exp) == [THRUST_FLOOR, 0.00, 0.66, 1.00]
