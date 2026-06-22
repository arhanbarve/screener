import json
import math
import os
import tempfile
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import date, datetime
from pathlib import Path

from src.factors import rsi_14, macd_state, adx_14, mfi_14

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


def add_position(ticker: str, entry_date: str, entry_price: float) -> None:
    """Append a new position. Raises ValueError if ticker already open."""
    positions = load_positions()
    if any(p["ticker"] == ticker.upper() for p in positions):
        raise ValueError(f"{ticker} already in open positions")
    positions.append({
        "ticker": ticker.upper(),
        "entry_date": entry_date,
        "entry_price": float(entry_price),
    })
    save_positions(positions)


def remove_position(ticker: str) -> None:
    """Remove a position by ticker. No-op if not found."""
    positions = [p for p in load_positions() if p["ticker"] != ticker.upper()]
    save_positions(positions)


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


def get_current_price(ticker: str) -> float | None:
    """Fetch latest price via yfinance fast_info. Returns None on failure."""
    try:
        info = yf.Ticker(ticker).fast_info
        return float(info.last_price)
    except Exception:
        return None


def compute_exit_signals(df: pd.DataFrame) -> dict:
    """
    Compute 5 exit signals from an OHLCV DataFrame.

    Returns:
        {
            "rsi": bool | None,      # True = exit signal triggered
            "macd": bool | None,
            "stoch": bool | None,
            "adx": bool | None,
            "mfi": bool | None,
            "score": int,            # count of True signals (0-5)
            "rsi_val": float | None,
            "adx_val": float | None,
            "mfi_val": float | None,
            "stoch_k": float | None,
            "stoch_d": float | None,
            "macd_state": str | None,
        }

    Signal definitions:
        RSI   — RSI(14) > 70 AND declining vs 3 bars ago
        MACD  — macd_state in ("bearish", "bearish_cross")
        Stoch — %K was >80 on prev bar AND %K just crossed below %D (bear cross)
        ADX   — ADX(14) now < ADX(10 bars ago) by >5 pts (trend weakening)
        MFI   — MFI(14) < 50 (money flow turned negative)
    """
    base = {
        "rsi": None, "macd": None, "stoch": None, "adx": None, "mfi": None,
        "score": 0,
        "rsi_val": None, "adx_val": None, "mfi_val": None,
        "stoch_k": None, "stoch_d": None, "macd_state": None,
    }

    required_cols = {"close", "high", "low", "volume"}
    if df.empty or not required_cols.issubset(df.columns) or len(df) < 30:
        return base

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # --- RSI: >70 AND declining vs 3 bars ago ---
    try:
        rsi_now = rsi_14(close)
        base["rsi_val"] = rsi_now
        if len(close) >= 17 and not math.isnan(rsi_now):
            rsi_3ago = rsi_14(close.iloc[:-3])
            if not math.isnan(rsi_3ago):
                base["rsi"] = bool(rsi_now > 70 and rsi_now < rsi_3ago)
    except Exception:
        pass

    # --- MACD: bearish or bearish_cross ---
    try:
        m_state = macd_state(close)
        base["macd_state"] = m_state
        base["macd"] = m_state in ("bearish", "bearish_cross")
    except Exception:
        pass

    # --- Stochastic: was overbought (K_prev > 80) AND bear cross (K crossed below D) ---
    try:
        k_period, smooth_k, d_period = 14, 3, 3
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
        base["stoch_k"] = k_now
        base["stoch_d"] = d_now
        bear_cross = (k_now < d_now) and (k_prev >= d_prev)
        base["stoch"] = bool(k_prev > 80 and bear_cross)
    except Exception:
        pass

    # --- ADX: current < (10-bars-ago value) by more than 5 pts ---
    try:
        adx_now = adx_14(high, low, close)
        base["adx_val"] = adx_now
        if len(close) >= 40 and not math.isnan(adx_now):
            adx_past = adx_14(high.iloc[:-10], low.iloc[:-10], close.iloc[:-10])
            if not math.isnan(adx_past):
                base["adx"] = bool(adx_past > adx_now + 5)
    except Exception:
        pass

    # --- MFI: < 50 ---
    try:
        mfi_now = mfi_14(high, low, close, volume)
        base["mfi_val"] = mfi_now
        if not math.isnan(mfi_now):
            base["mfi"] = bool(mfi_now < 50)
    except Exception:
        pass

    base["score"] = sum(
        1 for k in ("rsi", "macd", "stoch", "adx", "mfi")
        if base[k] is True
    )
    return base
