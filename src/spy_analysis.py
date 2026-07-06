"""SPY market regime analysis using 12+ technical, macro, and sentiment indicators."""
import math

import pandas as pd
from datetime import datetime

from src.factors import rsi_14, macd_state, adx_14, mfi_14, bollinger_pct_b
from src.positions import fetch_ohlcv as _fetch_ohlcv


def _fetch_last(ticker: str) -> float | None:
    try:
        df = _fetch_ohlcv(ticker, days=10)
        if df.empty or "close" not in df.columns:
            return None
        v = df["close"].iloc[-1]
        return float(v) if not math.isnan(float(v)) else None
    except Exception:
        return None


def compute_spy_regime() -> dict:
    """
    Compute SPY market regime using 12+ indicators.

    Returns:
        {
            regime: "BULL" | "CAUTION" | "BEAR",
            score: float 0-10,
            signals: list of {name, label, value, is_bull, weight},
            bull_count: int,
            total_count: int,
            as_of: str,
            error: str | None,
        }
    """
    out = {
        "regime": "UNKNOWN",
        "score": 5.0,
        "signals": [],
        "bull_count": 0,
        "total_count": 0,
        "as_of": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "error": None,
    }

    spy_df = _fetch_ohlcv("SPY", days=400)
    if spy_df.empty or len(spy_df) < 200:
        out["error"] = "Insufficient SPY data"
        return out

    close = spy_df["close"]
    high = spy_df["high"] if "high" in spy_df.columns else close
    low = spy_df["low"] if "low" in spy_df.columns else close
    volume = spy_df["volume"] if "volume" in spy_df.columns else pd.Series(dtype=float)
    price_now = float(close.iloc[-1])

    signals: list[dict] = []
    total_weight = 0.0
    bull_weight = 0.0

    def add(name: str, label: str, value_str: str, is_bull: bool, weight: float) -> None:
        nonlocal total_weight, bull_weight
        signals.append({
            "name": name,
            "label": label,
            "value": value_str,
            "is_bull": is_bull,
            "weight": weight,
        })
        total_weight += weight
        if is_bull:
            bull_weight += weight

    # --- Trend: SMA50 / SMA200 ---
    sma50 = float(close.rolling(50).mean().iloc[-1])
    sma200 = float(close.rolling(200).mean().iloc[-1])
    add("price_vs_sma50", "Price > SMA50",
        f"${price_now:.0f} vs ${sma50:.0f}", price_now > sma50, 1.0)
    add("price_vs_sma200", "Price > SMA200",
        f"${price_now:.0f} vs ${sma200:.0f}", price_now > sma200, 1.5)
    add("golden_cross", "Golden Cross (SMA50 > SMA200)",
        f"SMA50 ${sma50:.0f} / SMA200 ${sma200:.0f}", sma50 > sma200, 2.0)

    # --- RSI ---
    try:
        rsi_val = rsi_14(close)
        if not math.isnan(rsi_val):
            add("rsi", "RSI(14) 40–70 zone", f"{rsi_val:.1f}",
                40.0 <= rsi_val <= 70.0, 1.0)
    except Exception:
        pass

    # --- MACD ---
    try:
        m_state = macd_state(close)
        macd_bull = m_state in ("bullish", "bullish_cross")
        add("macd", "MACD bullish", m_state, macd_bull, 1.5)
    except Exception:
        pass

    # --- ADX ---
    try:
        adx_val = adx_14(high, low, close)
        if not math.isnan(adx_val):
            add("adx", "ADX > 20 (trend exists)", f"{adx_val:.1f}", adx_val > 20.0, 0.5)
    except Exception:
        pass

    # --- Bollinger %B ---
    try:
        bb_b, _ = bollinger_pct_b(close)
        if not math.isnan(bb_b):
            add("bollinger_pct_b", "Bollinger %B 0.2–0.9",
                f"{bb_b:.2f}", 0.2 <= bb_b <= 0.9, 0.5)
    except Exception:
        pass

    # --- MFI ---
    try:
        if not volume.empty:
            mfi_val = mfi_14(high, low, close, volume)
            if not math.isnan(mfi_val):
                add("mfi", "MFI > 50 (money inflow)", f"{mfi_val:.1f}", mfi_val >= 50.0, 1.0)
    except Exception:
        pass

    # --- VIX ---
    vix_val = _fetch_last("^VIX")
    if vix_val is not None:
        add("vix", "VIX < 20 (low fear)", f"{vix_val:.1f}", vix_val < 20.0, 2.0)

    # --- Yield curve (10Y minus 3M) ---
    try:
        tnx = _fetch_last("^TNX")   # 10-year Treasury yield (%)
        irx = _fetch_last("^IRX")   # 13-week T-bill rate (%)
        if tnx is not None and irx is not None:
            spread = tnx - irx
            add("yield_curve", "Yield curve (10Y−3M > 0)",
                f"{spread:+.2f}%", spread > 0.0, 1.5)
    except Exception:
        pass

    # --- Momentum ---
    for period_days, name, label, weight in [
        (21,  "mom_1m",  "1M Momentum > 0",  0.5),
        (63,  "mom_3m",  "3M Momentum > 0",  0.5),
        (126, "mom_6m",  "6M Momentum > 0",  1.0),
        (252, "mom_12m", "12M Momentum > 0", 1.0),
    ]:
        if len(close) > period_days:
            mom = float(close.iloc[-1] / close.iloc[-period_days]) - 1.0
            add(name, label, f"{mom:+.1%}", mom > 0.0, weight)

    # --- Finalize ---
    out["signals"] = signals
    out["bull_count"] = sum(1 for s in signals if s["is_bull"])
    out["total_count"] = len(signals)
    norm = (bull_weight / total_weight * 10.0) if total_weight > 0 else 5.0
    out["score"] = round(norm, 1)
    out["regime"] = "BULL" if norm >= 6.5 else ("BEAR" if norm < 3.5 else "CAUTION")
    return out


def compute_market_stress_overlay() -> dict:
    """
    Momentum-crash protection overlay. Scales top_n down (or to zero)
    when the market is in a stress regime that historically causes momentum
    reversals. Returns scale_factor in [0.0, 1.0].

    Stress signals:
    - SPY below SMA200 (confirmed downtrend)
    - VIX > 28 (elevated fear)
    - SPY 20-day realized vol > 2× 126-day baseline (volatility spike)

    0 signals → scale_factor=1.0 (normal)
    1 signal  → scale_factor=0.5 (run top-10 only)
    2+ signals → scale_factor=0.0 (skip screen, market in crash mode)
    """
    df = _fetch_ohlcv("SPY", days=420)
    if df.empty or "close" not in df.columns:
        return {"scale_factor": 1.0, "regime": "UNKNOWN", "reason": "SPY data unavailable",
                "stress_signals": []}

    close = df["close"].dropna()
    stress: list[str] = []

    # SMA200 check
    if len(close) >= 252:
        sma200 = float(close.iloc[-252:].mean())
        price  = float(close.iloc[-1])
        if price < sma200:
            stress.append(f"SPY below SMA200 ({price:.0f} < {sma200:.0f})")

    # Realized vol spike check
    if len(close) >= 126:
        ret = close.pct_change().dropna()
        vol20  = float(ret.iloc[-20:].std()  * (252 ** 0.5)) if len(ret) >= 20  else float("nan")
        vol126 = float(ret.iloc[-126:].std() * (252 ** 0.5)) if len(ret) >= 126 else float("nan")
        if not (math.isnan(vol20) or math.isnan(vol126)) and vol126 > 0 and vol20 > 2.0 * vol126:
            stress.append(f"vol spike (20d={vol20:.0%} > 2×126d={vol126:.0%})")
    else:
        vol20 = vol126 = float("nan")

    # VIX check
    vix = _fetch_last("^VIX")
    if vix is not None and vix > 28:
        stress.append(f"VIX={vix:.1f}")

    n = len(stress)
    if n == 0:
        scale_factor, regime = 1.0, "NORMAL"
    elif n == 1:
        scale_factor, regime = 0.5, "CAUTION"
    else:
        scale_factor, regime = 0.0, "STRESS"

    return {
        "scale_factor": scale_factor,
        "regime":       regime,
        "reason":       "; ".join(stress) if stress else "all clear",
        "stress_signals": stress,
        "vix":          vix,
    }
