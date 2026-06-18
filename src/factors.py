import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

DAYS_1M  = 21
DAYS_3M  = 63
DAYS_6M  = 126
DAYS_12M = 252

def mom_12_1(close: pd.Series) -> float:
    if len(close) < DAYS_12M:
        raise ValueError(f"Need at least 252 bars; got {len(close)}")
    return float(close.iloc[-DAYS_1M] / close.iloc[-DAYS_12M]) - 1.0

def mom_1m(close: pd.Series) -> float:
    if len(close) < DAYS_1M + 1:
        raise ValueError(f"Need at least {DAYS_1M+1} bars; got {len(close)}")
    return float(close.iloc[-1] / close.iloc[-DAYS_1M]) - 1.0

def compute_sue(actuals: list, estimates: list) -> float:
    """Standardized Unexpected Earnings. actuals/estimates oldest-first."""
    if len(actuals) < 4 or len(estimates) < 4:
        latest_a = actuals[-1]
        latest_e = estimates[-1]
        if abs(latest_e) < 1e-12:
            return 0.0
        return (latest_a - latest_e) / abs(latest_e)
    surprises = [a - e for a, e in zip(actuals, estimates)]
    std_s = float(np.std(surprises, ddof=1))
    if std_s < 1e-12:
        return 0.0
    return (surprises[-1]) / std_s

def rev_breadth_score(
    n_up: int,
    n_down: int,
    n_total: int,
    consensus_now: float = 0.0,
    consensus_90d_ago: float = 0.0,
) -> float:
    if n_total > 0:
        return (n_up - n_down) / n_total
    if abs(consensus_90d_ago) < 1e-12:
        return 0.0
    return (consensus_now - consensus_90d_ago) / abs(consensus_90d_ago)

def gp_assets_score(revenue: float, cogs: float, assets: float) -> float:
    if assets <= 0:
        return float("nan")
    return (revenue - cogs) / assets

def rs_vs_spy(stock_close: pd.Series, spy_close: pd.Series, window: int = DAYS_6M) -> float:
    if len(stock_close) < window + 1 or len(spy_close) < window + 1:
        return float("nan")
    stock_ret = float(stock_close.iloc[-1] / stock_close.iloc[-window]) - 1.0
    spy_ret   = float(spy_close.iloc[-1] / spy_close.iloc[-window]) - 1.0
    return stock_ret - spy_ret

def rs_slope(stock_close: pd.Series, spy_close: pd.Series, window: int = DAYS_3M) -> float:
    """Slope of stock/spy ratio over last `window` days."""
    if len(stock_close) < window or len(spy_close) < window:
        return float("nan")
    ratio = (stock_close / spy_close).iloc[-window:]
    x = np.arange(len(ratio), dtype=float)
    slope, *_ = scipy_stats.linregress(x, ratio.values)
    return float(slope)

def pct_from_52w_high(close: pd.Series) -> float:
    if len(close) < DAYS_12M:
        return float("nan")
    high_52w = close.iloc[-DAYS_12M:].max()
    return float(close.iloc[-1] / high_52w) - 1.0

def breakout_flag(close: pd.Series, volume: pd.Series) -> bool:
    if len(close) < DAYS_12M or len(volume) < 50:
        return False
    high_52w   = close.iloc[-DAYS_12M:].max()
    pct_off    = float(close.iloc[-1] / high_52w) - 1.0
    avg_vol_50 = float(volume.iloc[-50:].mean())
    return pct_off >= -0.05 and float(volume.iloc[-1]) > 1.5 * avg_vol_50

def avg_dollar_vol(close: pd.Series, volume: pd.Series, window: int = 20) -> float:
    n = min(window, len(close), len(volume))
    return float((close.iloc[-n:] * volume.iloc[-n:]).mean())

def squeeze_flag(short_float: float, days_to_cover: float, mom_1m_val: float) -> bool:
    return short_float > 0.15 and days_to_cover > 5.0 and mom_1m_val > 0.0
