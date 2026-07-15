"""Regime ladder signal tests. All series synthetic; index = business days."""
import numpy as np
import pandas as pd
import pytest

from src.regime import (
    trend_signal, breadth_signal, vol_signal, credit_signal,
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
