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
    if not actuals or not estimates:
        return 0.0
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

def residual_momentum(close: pd.Series, market_close: pd.Series) -> float:
    """
    Risk-adjusted momentum (Blitz, Huij, Martens 2011).
    Regress monthly stock returns on monthly market returns,
    sum residuals over 12 months (skip last), scale by residual vol.
    Lower crash risk than raw 12-1 momentum.
    """
    monthly_stock = close.resample("ME").last().pct_change().dropna()
    monthly_mkt   = market_close.resample("ME").last().pct_change().dropna()
    df = pd.DataFrame({"r": monthly_stock, "m": monthly_mkt}).dropna()
    if len(df) < 13:
        return float("nan")
    df = df.iloc[-13:-1]  # 12 months, skip most recent
    m_var = float(df["m"].var())
    if m_var < 1e-12:
        return float("nan")
    beta = float(df["r"].cov(df["m"])) / m_var
    residuals = df["r"] - beta * df["m"]
    std = float(residuals.std())
    if std < 1e-12:
        return float("nan")
    return float(residuals.sum() / std)


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


def stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    smooth_k: int = 3,
    d_period: int = 3,
) -> tuple[float, float, bool]:
    """Returns (smooth_%K, %D, bull_cross). bull_cross = %K just crossed above %D."""
    if len(close) < k_period + smooth_k + d_period:
        return float("nan"), float("nan"), False
    lowest_low    = low.rolling(window=k_period).min()
    highest_high  = high.rolling(window=k_period).max()
    denom = (highest_high - lowest_low).replace(0, np.nan)
    raw_k         = 100.0 * (close - lowest_low) / denom
    sk_series     = raw_k.rolling(window=smooth_k).mean()
    d_series      = sk_series.rolling(window=d_period).mean()
    k_now, k_prev = float(sk_series.iloc[-1]), float(sk_series.iloc[-2])
    d_now, d_prev = float(d_series.iloc[-1]),  float(d_series.iloc[-2])
    bull_cross = (k_now > d_now) and (k_prev <= d_prev)
    return k_now, d_now, bull_cross


def bollinger_pct_b(
    close: pd.Series, window: int = 20, num_std: float = 2.0
) -> tuple[float, float]:
    """Returns (%B, bandwidth). %B: 0=lower band, 1=upper band. BW = (upper-lower)/mid."""
    if len(close) < window:
        return float("nan"), float("nan")
    sma   = close.rolling(window).mean()
    std   = close.rolling(window).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    band_range = (upper - lower).replace(0, np.nan)
    pct_b  = float(((close - lower) / band_range).iloc[-1])
    bw     = float(((upper - lower) / sma).iloc[-1])
    return pct_b, bw


def adx_14(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14
) -> float:
    """Wilder's ADX. < 20 = no trend, 20-25 = emerging, > 25 = trending."""
    if len(close) < window * 2:
        return float("nan")
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    up_move   = high - high.shift(1)
    down_move = low.shift(1) - low
    plus_dm  = np.where((up_move > down_move) & (up_move > 0),   up_move,   0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm_s  = pd.Series(plus_dm,  index=high.index)
    minus_dm_s = pd.Series(minus_dm, index=high.index)
    atr_s    = tr.ewm(com=window - 1, min_periods=window).mean()
    plus_di  = 100.0 * plus_dm_s.ewm(com=window - 1, min_periods=window).mean()  / atr_s
    minus_di = 100.0 * minus_dm_s.ewm(com=window - 1, min_periods=window).mean() / atr_s
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-12)
    adx_val = dx.ewm(com=window - 1, min_periods=window).mean()
    return float(adx_val.iloc[-1])


def mfi_14(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    window: int = 14,
) -> float:
    """Money Flow Index: RSI weighted by volume. Overbought >80, oversold <20."""
    if len(close) < window + 1:
        return float("nan")
    tp   = (high + low + close) / 3.0
    rmf  = tp * volume
    prev_tp = tp.shift(1)
    pos_mf = rmf.where(tp > prev_tp, 0.0)
    neg_mf = rmf.where(tp < prev_tp, 0.0)
    pos_sum = pos_mf.rolling(window).sum()
    neg_sum = neg_mf.rolling(window).sum()
    mfr = pos_sum / (neg_sum + 1e-12)
    return float((100.0 - 100.0 / (1.0 + mfr)).iloc[-1])


def rsi_14(close: pd.Series, window: int = 14) -> float:
    if len(close) < window + 1:
        return float("nan")
    delta = close.diff().iloc[1:]
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=window - 1, min_periods=window).mean().iloc[-1]
    avg_loss = loss.ewm(com=window - 1, min_periods=window).mean().iloc[-1]
    if avg_loss < 1e-12:
        return 100.0
    return float(100.0 - 100.0 / (1.0 + avg_gain / avg_loss))


def macd_state(close: pd.Series) -> str:
    if len(close) < 35:
        return "unknown"
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    above_now  = bool(macd_line.iloc[-1] > signal_line.iloc[-1])
    above_prev = bool(macd_line.iloc[-2] > signal_line.iloc[-2])
    if above_now and not above_prev:
        return "bullish_cross"
    if above_now:
        return "bullish"
    if not above_now and above_prev:
        return "bearish_cross"
    return "bearish"


def vol_surge_ratio(volume: pd.Series) -> float:
    if len(volume) < 20:
        return float("nan")
    avg5  = float(volume.iloc[-5:].mean())
    avg20 = float(volume.iloc[-20:].mean())
    if avg20 < 1:
        return float("nan")
    return avg5 / avg20


# ── Exit / trend-structure indicators ────────────────────────────────────────
# Added for the tiered exit engine. Kept as pure functions so they can be
# hand-verified on historical data independently of positions.py.

def sma(close: pd.Series, window: int) -> float:
    """Latest simple moving average, or NaN if insufficient history."""
    if len(close) < window:
        return float("nan")
    return float(close.iloc[-window:].mean())


def atr_series(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14
) -> pd.Series:
    """Wilder's Average True Range as a series (same smoothing as adx_14)."""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=window - 1, min_periods=window).mean()


def atr_14(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14
) -> float:
    """Latest ATR value."""
    if len(close) < window + 1:
        return float("nan")
    return float(atr_series(high, low, close, window).iloc[-1])


def chandelier_stop_long(
    high: pd.Series, low: pd.Series, close: pd.Series,
    lookback: int = 22, mult: float = 3.0, atr_window: int = 14,
) -> float:
    """
    Chandelier exit level for a long: highest-high(lookback) - mult*ATR.
    Snapshot approximation of a ratcheting trailing stop (recomputed each bar,
    not carried statefully). Exit when close falls below this level.
    """
    if len(close) < max(lookback, atr_window) + 1:
        return float("nan")
    atr = atr_series(high, low, close, atr_window)
    hh  = high.rolling(lookback).max()
    return float((hh - mult * atr).iloc[-1])


def obv_series(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume. Cumulative signed volume by daily price direction."""
    direction = np.sign(close.diff().fillna(0.0))
    return (direction * volume).cumsum()


def obv_slope(close: pd.Series, volume: pd.Series, window: int = 20) -> float:
    """
    Slope of OBV over `window` bars, normalized by average volume so it is
    comparable across tickers. Negative = net distribution over the window.
    """
    if len(close) < window or len(volume) < window:
        return float("nan")
    o = obv_series(close, volume).iloc[-window:]
    x = np.arange(len(o), dtype=float)
    slope, *_ = scipy_stats.linregress(x, o.values)
    avg_vol = float(volume.iloc[-window:].mean())
    if avg_vol < 1:
        return float("nan")
    return float(slope / avg_vol)


def directional_indicators(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14
) -> tuple[float, float, float]:
    """Latest (ADX, +DI, -DI). Shares Wilder math with adx_14."""
    if len(close) < window * 2:
        return float("nan"), float("nan"), float("nan")
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    up_move   = high - high.shift(1)
    down_move = low.shift(1) - low
    plus_dm  = np.where((up_move > down_move) & (up_move > 0),   up_move,   0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm_s  = pd.Series(plus_dm,  index=high.index)
    minus_dm_s = pd.Series(minus_dm, index=high.index)
    atr_s    = tr.ewm(com=window - 1, min_periods=window).mean()
    plus_di  = 100.0 * plus_dm_s.ewm(com=window - 1, min_periods=window).mean()  / atr_s
    minus_di = 100.0 * minus_dm_s.ewm(com=window - 1, min_periods=window).mean() / atr_s
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-12)
    adx_val = dx.ewm(com=window - 1, min_periods=window).mean()
    return float(adx_val.iloc[-1]), float(plus_di.iloc[-1]), float(minus_di.iloc[-1])


def parabolic_sar(
    high: pd.Series, low: pd.Series,
    af_step: float = 0.02, af_max: float = 0.20,
) -> tuple[float, bool, bool]:
    """
    Parabolic SAR. Returns (sar_now, is_uptrend_now, flipped_to_bear_this_bar).
    Standard Wilder iterative construction.
    """
    n = len(high)
    if n < 5:
        return float("nan"), False, False
    h = high.values
    l = low.values
    # Seed: assume uptrend, SAR = first low, EP = first high
    up = True
    sar = l[0]
    ep = h[0]
    af = af_step
    trend = [up]
    for i in range(1, n):
        prev_sar = sar
        sar = prev_sar + af * (ep - prev_sar)
        if up:
            # SAR cannot exceed the prior two lows
            sar = min(sar, l[i - 1], l[i - 2] if i >= 2 else l[i - 1])
            if l[i] < sar:               # flip to downtrend
                up = False
                sar = ep                 # SAR resets to prior EP
                ep = l[i]
                af = af_step
            else:
                if h[i] > ep:
                    ep = h[i]
                    af = min(af + af_step, af_max)
        else:
            sar = max(sar, h[i - 1], h[i - 2] if i >= 2 else h[i - 1])
            if h[i] > sar:               # flip to uptrend
                up = True
                sar = ep
                ep = h[i]
                af = af_step
            else:
                if l[i] < ep:
                    ep = l[i]
                    af = min(af + af_step, af_max)
        trend.append(up)
    flipped_to_bear = (trend[-2] is True) and (trend[-1] is False)
    return float(sar), bool(trend[-1]), bool(flipped_to_bear)


def trend_signal_score(row: pd.Series) -> int:
    """Trend persistence cluster (0-5): ADX + MACD + SMA50. Trend signals persist."""
    score = 0
    adx = row.get("adx")
    if pd.notna(adx):
        if float(adx) > 25.0:
            score += 2   # strong trend
        elif float(adx) > 20.0:
            score += 1   # emerging trend
    macd = row.get("macd")
    if pd.notna(macd):
        s = str(macd)
        if s == "bullish_cross":
            score += 2   # fresh cross outscores steady bullish
        elif s == "bullish":
            score += 1
    sma50 = row.get("above_sma50")
    if pd.notna(sma50) and bool(sma50):
        score += 1
    return score


def oscillator_signal_score(row: pd.Series) -> int:
    """Momentum oscillator cluster (0-4): RSI + Stoch + MFI + BB in healthy zones."""
    score = 0
    rsi = row.get("rsi_14")
    if pd.notna(rsi) and 40.0 <= float(rsi) <= 70.0:
        score += 1
    sk = row.get("stoch_k")
    cross = row.get("stoch_cross")
    if pd.notna(sk) and float(sk) > 20.0 and (bool(cross) or float(sk) > 50.0):
        score += 1
    mfi = row.get("mfi")
    if pd.notna(mfi) and 40.0 <= float(mfi) <= 80.0:   # wider upper zone than entry_grade
        score += 1
    bb = row.get("bb_pct_b")
    if pd.notna(bb) and 0.2 <= float(bb) <= 0.9:
        score += 1
    return score


def volume_signal_score(row: pd.Series) -> int:
    """Volume confirmation cluster (0-1)."""
    vol = row.get("vol_surge")
    if pd.notna(vol) and float(vol) > 1.1:
        return 1
    return 0


def tech_signal_score(row: pd.Series) -> int:
    """Count of 8 technical thresholds passing. Higher = more aligned for entry."""
    score = 0

    rsi = row.get("rsi_14")
    if pd.notna(rsi) and 40.0 <= float(rsi) <= 70.0:
        score += 1

    macd = row.get("macd")
    if pd.notna(macd) and str(macd) in ("bullish", "bullish_cross"):
        score += 1

    sk = row.get("stoch_k")
    cross = row.get("stoch_cross")
    if pd.notna(sk) and float(sk) > 20.0 and (bool(cross) or float(sk) > 50.0):
        score += 1

    adx = row.get("adx")
    if pd.notna(adx) and float(adx) > 20.0:
        score += 1

    vol = row.get("vol_surge")
    if pd.notna(vol) and float(vol) > 1.1:
        score += 1

    sma50 = row.get("above_sma50")
    if pd.notna(sma50) and bool(sma50):
        score += 1

    bb = row.get("bb_pct_b")
    if pd.notna(bb) and 0.2 <= float(bb) <= 0.9:
        score += 1

    mfi = row.get("mfi")
    if pd.notna(mfi) and float(mfi) > 40.0:
        score += 1

    return score


def entry_grade(
    rsi_val: float,
    macd: str,
    vol_ratio: float,
    above_sma20: bool,
    adx_val: float = float("nan"),
    stoch_k: float = float("nan"),
    stoch_d: float = float("nan"),
    stoch_cross: bool = False,
    bb_pct_b: float = float("nan"),
    mfi_val: float = float("nan"),
    obv_slope_val: float = float("nan"),
    sar_bullish: bool = False,
) -> str:
    """
    Short-term entry timing grade. Only uses signals NOT already captured
    in the composite technical scores (trend_score, momo_osc_score, volume_score).
    ADX, BB %B, vol_surge are in the composite — removed from here to avoid
    double-counting. OBV slope and Parabolic SAR are new families (not in the
    composite) added as accumulation / trend-flip confirmations. Hard vetoes
    remain unchanged.
    """
    def _ok(v: float) -> bool:
        return not np.isnan(v)

    # Hard vetoes — unconditional blockers
    if macd in ("bearish", "bearish_cross"):
        return "WAIT"
    if not above_sma20:
        return "WAIT"
    if rsi_val > 75:
        return "WAIT"
    if _ok(mfi_val) and mfi_val > 80:
        return "WAIT"

    score = 0

    # MACD direction (primary timing spine)
    if macd == "bullish_cross":
        score += 2
    elif macd == "bullish":
        score += 1

    # RSI in healthy momentum zone (40-65 consistent with composite)
    if 40.0 <= rsi_val <= 65.0:
        score += 1

    # MFI confirms money-flow alignment (40-65)
    if _ok(mfi_val) and 40.0 <= mfi_val <= 65.0:
        score += 1

    # Stochastic: momentum not overbought, extra for fresh cross
    if _ok(stoch_k) and _ok(stoch_d) and stoch_k > stoch_d and stoch_k < 70.0:
        score += 2 if stoch_cross else 1

    # OBV accumulation — volume flowing in (new family, not in composite)
    if _ok(obv_slope_val) and obv_slope_val > 0:
        score += 1

    # Parabolic SAR in an uptrend (new family, not in composite)
    if sar_bullish:
        score += 1

    # Max score = 8 (MACD_cross=2 + RSI=1 + MFI=1 + Stoch_cross=2 + OBV=1 + SAR=1)
    if score >= 6:
        return "STRONG"
    if score >= 3:
        return "OK"
    return "WAIT"
