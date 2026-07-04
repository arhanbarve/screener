import json
import math
import os
import tempfile
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import date, datetime, timedelta
from pathlib import Path

from src.factors import (
    rsi_14, macd_state, adx_14, mfi_14,
    sma, chandelier_stop_long, obv_slope, directional_indicators,
    parabolic_sar,
)

POSITIONS_FILE = Path("positions.json")


def load_positions() -> list[dict]:
    """Load positions from positions.json. Returns [] if file missing."""
    if not POSITIONS_FILE.exists():
        return []
    with open(POSITIONS_FILE, "r") as f:
        return json.load(f)


def save_positions(positions: list[dict]) -> None:
    """Atomically write positions to positions.json."""
    tmp = POSITIONS_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(positions, f, indent=2)
    os.replace(tmp, POSITIONS_FILE)


def fetch_price_on_date(ticker: str, entry_date: str) -> float | None:
    """Fetch closing price for ticker on or before entry_date. Returns None on failure."""
    try:
        dt = datetime.strptime(entry_date, "%Y-%m-%d")
        start = (dt - timedelta(days=7)).strftime("%Y-%m-%d")
        end = (dt + timedelta(days=2)).strftime("%Y-%m-%d")
        raw = yf.download(ticker, start=start, end=end, interval="1d",
                          auto_adjust=True, progress=False)
        if raw.empty:
            return None
        df = raw.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].lower() for c in df.columns]
        else:
            df.columns = [c.lower() for c in df.columns]
        df = df.dropna(how="all")
        if df.empty:
            return None
        target = pd.Timestamp(entry_date)
        available = df.index[df.index <= target]
        if available.empty:
            return float(df["close"].iloc[0])
        return float(df.loc[available[-1], "close"])
    except Exception:
        return None


def add_position(ticker: str, entry_date: str, entry_price: float | None = None) -> float:
    """Append a new position. Auto-fetches close price on entry_date if not provided.
    Returns the entry price used. Raises ValueError if ticker already open or price unavailable."""
    positions = load_positions()
    if any(p["ticker"] == ticker.upper() for p in positions):
        raise ValueError(f"{ticker.upper()} already in open positions")
    if entry_price is None:
        entry_price = fetch_price_on_date(ticker, entry_date)
        if entry_price is None:
            raise ValueError(f"Could not fetch price for {ticker.upper()} on {entry_date}")
    positions.append({
        "ticker": ticker.upper(),
        "entry_date": entry_date,
        "entry_price": float(entry_price),
    })
    save_positions(positions)
    return float(entry_price)


def remove_position(ticker: str) -> None:
    """Remove a position by ticker. No-op if not found."""
    positions = [p for p in load_positions() if p["ticker"] != ticker.upper()]
    save_positions(positions)


def _stoch_bear_cross(
    high: pd.Series, low: pd.Series, close: pd.Series,
    k_period: int = 14, smooth_k: int = 3, d_period: int = 3,
) -> tuple[float | None, float | None, bool]:
    """K_now, D_now, bear_cross. bear_cross = K_prev>80 AND K just crossed below D."""
    if len(close) < k_period + smooth_k + d_period:
        return None, None, False
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    denom = (highest_high - lowest_low).replace(0, np.nan)
    raw_k = 100.0 * (close - lowest_low) / denom
    sk = raw_k.rolling(smooth_k).mean()
    d_ser = sk.rolling(d_period).mean()
    k_now = float(sk.iloc[-1])
    k_prev = float(sk.iloc[-2])
    d_now = float(d_ser.iloc[-1])
    d_prev = float(d_ser.iloc[-2])
    bear_cross = (k_now < d_now) and (k_prev >= d_prev) and (k_prev > 80)
    return k_now, d_now, bear_cross


def fetch_ohlcv(ticker: str, days: int = 60) -> pd.DataFrame:
    """Fetch OHLCV via yfinance. Returns empty DataFrame on failure."""
    try:
        raw = yf.download(
            ticker,
            period=f"{days}d",
            interval="1d",
            auto_adjust=True,
            progress=False,
        )
        if raw.empty:
            return pd.DataFrame()
        df = raw.copy()
        # Flatten MultiIndex columns if present (single-ticker download)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].lower() for c in df.columns]
        else:
            df.columns = [c.lower() for c in df.columns]
        return df.dropna(how="all")
    except Exception:
        return pd.DataFrame()


def get_live_quote(ticker: str) -> tuple[float | None, float | None]:
    """Fetch latest price and previous close via yfinance fast_info. (None, None) on failure."""
    try:
        info = yf.Ticker(ticker).fast_info
        last = float(info.last_price)
        try:
            prev = float(info.previous_close)
        except Exception:
            prev = None
        return last, prev
    except Exception:
        return None, None




def days_to_next_earnings(ticker: str) -> int | None:
    """Calendar days until the next scheduled earnings date. None on failure.

    Reuses the yfinance earnings-dates endpoint (same source as
    src/pead_backtest.py). Best-effort: returns None if the call fails or no
    future date is listed.
    """
    try:
        df = yf.Ticker(ticker).get_earnings_dates(limit=8)
        if df is None or df.empty:
            return None
        today = pd.Timestamp.now(tz=df.index.tz) if df.index.tz else pd.Timestamp.now()
        future = [ts for ts in df.index if ts >= today]
        if not future:
            return None
        return int((min(future) - today).days)
    except Exception:
        return None


# Soft-signal point weights. Trend breaks and MACD carry more weight than a
# single oscillator flag. Max soft score = 12.
_SOFT_WEIGHTS = {
    "macd":        2,   # bearish_cross / bearish state
    "sma50_break": 2,   # close below the 50-day (intermediate trend break)
    "rsi":         1,   # overbought and rolling over
    "stoch":       1,   # bear cross from overbought
    "mfi":         1,   # money flow negative
    "obv_dist":    1,   # OBV distribution divergence
    "adx":         1,   # trend weakening, direction-confirmed
    "sma20_break": 1,   # close below the 20-day (short-term break)
    "bb_blowoff":  1,   # blow-off reversal back inside upper band
    "rs_decay":    1,   # underperforming SPY and falling
}


def compute_exit_signals(
    df: pd.DataFrame,
    spy_close: pd.Series | None = None,
    days_to_earn: int | None = None,
) -> dict:
    """
    Tiered exit engine over an OHLCV DataFrame.

    Two tiers, mirroring factors.entry_grade's veto + scored-points split:

    HARD (any one True -> hard_exit; overrides the soft score). Both hard
    conditions require close < SMA200, so a hard exit is impossible while the
    stock trades above its 200-day (verified on real data to suppress
    false-fires during uptrend pullbacks):
        chandelier   — close < chandelier stop (highest-high(22) - 3*ATR14)
                       AND close < SMA200  (trend-confirmed volatility stop)
        death_cross  — close < SMA200 AND SMA50 < SMA200 (downtrend regime)

    SOFT (weighted points, see _SOFT_WEIGHTS, max 12):
        macd(2)        — bearish or bearish_cross
        sma50_break(2) — close < SMA50
        rsi(1)         — RSI(14) > 70 AND declining vs 3 bars ago
        stoch(1)       — %K was >80 then crossed below %D
        mfi(1)         — MFI(14) < 50
        obv_dist(1)    — OBV 20d slope < 0 while price flat/up (distribution)
        adx(1)         — ADX down >5 vs 10 bars ago AND -DI > +DI
        sma20_break(1) — close < SMA20
        bb_blowoff(1)  — prior close above upper Bollinger, now back inside
        rs_decay(1)    — 21d return < SPY 21d return AND 21d return < 0

    grade: STRONG EXIT (hard, or soft>=7) / TRIM (soft 4-6) /
           WATCH (soft 2-3) / HOLD (<2).

    `days_to_earn` (if provided) is surfaced as a non-scoring risk flag — it is
    deliberately NOT folded into the sell score.
    """
    keys = list(_SOFT_WEIGHTS.keys())
    base = {
        # hard tier
        "hard_exit": False, "hard_reasons": [],
        "chandelier": None, "death_cross": None,
        # soft flags
        **{k: None for k in keys},
        # display values
        "rsi_val": None, "adx_val": None, "mfi_val": None,
        "stoch_k": None, "stoch_d": None, "macd_state": None,
        "obv_slope_val": None, "chand_val": None,
        "sma20_val": None, "sma50_val": None, "sma200_val": None,
        "close_val": None, "rs_1m_val": None,
        # scoring
        "soft_score": 0, "soft_max": sum(_SOFT_WEIGHTS.values()),
        "grade": "HOLD", "score": 0,
        "days_to_earnings": days_to_earn,
    }

    required_cols = {"close", "high", "low", "volume"}
    if df.empty or not required_cols.issubset(df.columns) or len(df) < 40:
        return base

    close, high, low, volume = df["close"], df["high"], df["low"], df["volume"]
    price = float(close.iloc[-1])
    base["close_val"] = price
    has_long = len(close) >= 210  # enough for SMA200 / chandelier

    # ── Moving-average structure ────────────────────────────────────────────
    s20 = sma(close, 20)
    s50 = sma(close, 50)
    s200 = sma(close, 200) if has_long else float("nan")
    base["sma20_val"] = None if math.isnan(s20) else s20
    base["sma50_val"] = None if math.isnan(s50) else s50
    base["sma200_val"] = None if math.isnan(s200) else s200
    if not math.isnan(s20):
        base["sma20_break"] = bool(price < s20)
    if not math.isnan(s50):
        base["sma50_break"] = bool(price < s50)

    # ── HARD tier (both gated on close < SMA200) ────────────────────────────
    if has_long and not math.isnan(s200):
        below_200 = price < s200
        try:
            chand = chandelier_stop_long(high, low, close)
            base["chand_val"] = None if math.isnan(chand) else chand
            base["chandelier"] = bool(below_200 and not math.isnan(chand) and price < chand)
        except Exception:
            base["chandelier"] = False
        base["death_cross"] = bool(below_200 and not math.isnan(s50) and s50 < s200)
        if base["chandelier"]:
            base["hard_reasons"].append("ATR trailing stop breached below 200-day")
        if base["death_cross"]:
            base["hard_reasons"].append("Death-cross downtrend (price<200d, 50d<200d)")
        base["hard_exit"] = bool(base["chandelier"] or base["death_cross"])

    # ── SOFT tier ───────────────────────────────────────────────────────────
    # RSI: >70 AND declining vs 3 bars ago
    try:
        rsi_now = rsi_14(close)
        base["rsi_val"] = None if math.isnan(rsi_now) else rsi_now
        if len(close) >= 17 and not math.isnan(rsi_now):
            rsi_3ago = rsi_14(close.iloc[:-3])
            if not math.isnan(rsi_3ago):
                base["rsi"] = bool(rsi_now > 70 and rsi_now < rsi_3ago)
    except Exception:
        pass

    # MACD bearish
    try:
        m_state = macd_state(close)
        base["macd_state"] = m_state
        base["macd"] = m_state in ("bearish", "bearish_cross")
    except Exception:
        pass

    # Stochastic bear cross from overbought
    try:
        k_now, d_now, bear_cross = _stoch_bear_cross(high, low, close)
        if k_now is not None:
            base["stoch_k"], base["stoch_d"] = k_now, d_now
            base["stoch"] = bear_cross
    except Exception:
        pass

    # ADX weakening AND direction-confirmed (-DI > +DI)
    try:
        adx_now, pdi, ndi = directional_indicators(high, low, close)
        base["adx_val"] = None if math.isnan(adx_now) else adx_now
        if len(close) >= 40 and not math.isnan(adx_now):
            adx_past, _, _ = directional_indicators(high.iloc[:-10], low.iloc[:-10], close.iloc[:-10])
            if not math.isnan(adx_past) and not math.isnan(ndi) and not math.isnan(pdi):
                base["adx"] = bool(adx_past > adx_now + 5 and ndi > pdi)
    except Exception:
        pass

    # MFI < 50
    try:
        mfi_now = mfi_14(high, low, close, volume)
        base["mfi_val"] = None if math.isnan(mfi_now) else mfi_now
        if not math.isnan(mfi_now):
            base["mfi"] = bool(mfi_now < 50)
    except Exception:
        pass

    # OBV distribution divergence: OBV falling while price flat/up
    try:
        slp = obv_slope(close, volume, window=20)
        base["obv_slope_val"] = None if math.isnan(slp) else slp
        if not math.isnan(slp) and len(close) >= 21:
            price_20ago = float(close.iloc[-21])
            base["obv_dist"] = bool(slp < 0 and price >= price_20ago)
    except Exception:
        pass

    # Bollinger blow-off: prior close above upper band, now back inside
    try:
        if len(close) >= 21:
            mid = close.rolling(20).mean()
            sd = close.rolling(20).std()
            upper = mid + 2.0 * sd
            base["bb_blowoff"] = bool(
                close.iloc[-2] > upper.iloc[-2] and close.iloc[-1] <= upper.iloc[-1]
            )
    except Exception:
        pass

    # RS decay: underperforming SPY over 21d AND falling
    try:
        if spy_close is not None and len(close) >= 22 and len(spy_close) >= 22:
            stk_ret = price / float(close.iloc[-22]) - 1.0
            spy_ret = float(spy_close.iloc[-1]) / float(spy_close.iloc[-22]) - 1.0
            base["rs_1m_val"] = stk_ret - spy_ret
            base["rs_decay"] = bool(stk_ret < spy_ret and stk_ret < 0)
    except Exception:
        pass

    # ── Aggregate ───────────────────────────────────────────────────────────
    # A fresh MACD bearish_cross gets full weight; a lingering bearish state is
    # lower-conviction (it flickers on 1-2 day dips mid-uptrend), so worth 1.
    soft = 0
    for k, w in _SOFT_WEIGHTS.items():
        if base[k] is True:
            if k == "macd" and base["macd_state"] == "bearish":
                soft += 1
            else:
                soft += w
    base["soft_score"] = soft
    base["score"] = soft  # legacy alias

    if base["hard_exit"] or soft >= 7:
        base["grade"] = "STRONG EXIT"
    elif soft >= 4:
        base["grade"] = "TRIM"
    elif soft >= 2:
        base["grade"] = "WATCH"
    else:
        base["grade"] = "HOLD"
    return base
