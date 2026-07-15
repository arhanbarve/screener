"""Regime exposure ladder (spec: docs/superpowers/specs/2026-07-15-walkforward-harness-regime-ladder.md).

Four independent stress signals, each worth 1 point; points map to a target
equity exposure. Shared by the portfolio backtest and (after validation) the
live screener, so thresholds live here as module constants.

All signal functions are NaN-safe: any day where the underlying data is
missing or the lookback window hasn't formed yet contributes False (0 points),
never NaN. Early-history gaps (e.g. ^VIX3M before 2008) therefore weaken the
ladder rather than break it.
"""
import pandas as pd

SPY_SMA_WINDOW = 200
BREADTH_200D_THRESHOLD = 0.40
VIX_LEVEL_THRESHOLD = 25.0
CREDIT_SMA_WINDOW = 100

EXPOSURE_MAP = {0: 1.00, 1: 1.00, 2: 0.66, 3: 0.33, 4: 0.00}

THRUST_LOW = 0.20      # breadth-50d must have been below this...
THRUST_HIGH = 0.55     # ...and cross above this...
THRUST_WINDOW = 10     # ...within this many sessions
THRUST_HOLD = 20       # override lasts this many sessions
THRUST_FLOOR = 0.66    # forced minimum exposure while active


def trend_signal(spy_close: pd.Series) -> pd.Series:
    """SPY close < its SMA200. False until the SMA has formed."""
    sma = spy_close.rolling(SPY_SMA_WINDOW).mean()
    return ((spy_close < sma) & sma.notna()).fillna(False)


def breadth_signal(pct_above_200: pd.Series) -> pd.Series:
    """Fraction of universe above own SMA200 < threshold. NaN -> False."""
    return (pct_above_200 < BREADTH_200D_THRESHOLD).fillna(False)


def vol_signal(vix_close: pd.Series, vix3m_close: pd.Series) -> pd.Series:
    """VIX above absolute threshold, or term structure inverted (VIX > VIX3M)."""
    vix, v3 = vix_close.align(vix3m_close, join="left")
    level = (vix > VIX_LEVEL_THRESHOLD).fillna(False)
    inverted = (vix > v3).fillna(False)
    return level | inverted


def credit_signal(hyg_close: pd.Series, ief_close: pd.Series) -> pd.Series:
    """HYG/IEF ratio below its own SMA100 — credit risk-off leads equities."""
    hyg, ief = hyg_close.align(ief_close, join="inner")
    ratio = hyg / ief
    sma = ratio.rolling(CREDIT_SMA_WINDOW).mean()
    return ((ratio < sma) & sma.notna()).fillna(False)


def ladder_points(trend: pd.Series, breadth: pd.Series,
                  vol: pd.Series, credit: pd.Series) -> pd.Series:
    """Sum the four boolean signals into 0-4 daily points (int)."""
    df = pd.concat(
        {"t": trend, "b": breadth, "v": vol, "c": credit}, axis=1
    ).fillna(False)
    return df.sum(axis=1).astype(int)


def ladder_exposure(points: pd.Series) -> pd.Series:
    """Map daily points to target equity exposure per EXPOSURE_MAP."""
    return points.map(EXPOSURE_MAP).astype(float)


def thrust_override(pct_above_50: pd.Series) -> pd.Series:
    """Breadth-thrust re-entry: True while the override is active.

    Fires when pct_above_50 crosses above THRUST_HIGH having been below
    THRUST_LOW within the prior THRUST_WINDOW sessions; stays active for
    THRUST_HOLD sessions from the firing day (re-fires reset the clock).
    """
    vals = pct_above_50.to_numpy()
    active = pd.Series(False, index=pct_above_50.index)
    fire_until = -1
    for i in range(len(vals)):
        crossed = (
            i > 0
            and vals[i] > THRUST_HIGH
            and not (vals[i - 1] > THRUST_HIGH)
        )
        if crossed:
            lo = max(0, i - THRUST_WINDOW)
            window = vals[lo:i]
            if (window < THRUST_LOW).any():
                fire_until = i + THRUST_HOLD - 1
        if i <= fire_until:
            active.iloc[i] = True
    return active


def combined_exposure(points: pd.Series, thrust_active: pd.Series) -> pd.Series:
    """Ladder exposure with the thrust floor applied on active days."""
    exp = ladder_exposure(points)
    floored = exp.clip(lower=THRUST_FLOOR)
    return exp.where(~thrust_active.reindex(exp.index, fill_value=False), floored)
