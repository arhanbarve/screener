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
