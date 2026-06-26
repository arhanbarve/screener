import time
import logging
import pandas as pd
import yfinance as yf
from typing import Optional
from src.factors import (
    mom_12_1, mom_1m, rs_vs_spy, rs_slope, residual_momentum,
    pct_from_52w_high, breakout_flag, avg_dollar_vol,
    rsi_14, macd_state, vol_surge_ratio, entry_grade,
    stochastic, bollinger_pct_b, adx_14, mfi_14,
)
from src.cache import get_prices, put_prices, get_market_cap, put_market_cap

logger = logging.getLogger(__name__)

BATCH_SIZE = 200
HISTORY_DAYS = 420


def _fetch_batch_yfinance(tickers: list[str]) -> dict[str, pd.DataFrame]:
    joined = " ".join(tickers)
    try:
        raw = yf.download(
            joined,
            period="420d",
            interval="1d",
            auto_adjust=True,
            progress=False,
            group_by="ticker",
            threads=True,
        )
    except Exception as e:
        logger.warning(f"yfinance batch failed: {e}")
        return {}

    if raw.empty:
        return {}

    result = {}
    # yfinance 1.x always returns MultiIndex columns (Price, Ticker)
    if isinstance(raw.columns, pd.MultiIndex):
        # level 0 = price field, level 1 = ticker (or vice versa)
        # detect orientation: if tickers appear in level 1, use (field, ticker)
        l0 = set(raw.columns.get_level_values(0))
        l1 = set(raw.columns.get_level_values(1))
        ticker_set = {t.upper() for t in tickers}
        if ticker_set & l1:
            # (Price, Ticker) format
            for t in tickers:
                tu = t.upper()
                cols = [(p, tu) for p in l0 if (p, tu) in raw.columns]
                if not cols:
                    continue
                df = raw[[c for c in raw.columns if c[1] == tu]].copy()
                df.columns = [c[0].lower() for c in df.columns]
                df = df.dropna(how="all")
                if len(df) > 0:
                    result[t] = df
        else:
            # (Ticker, Price) format
            for t in tickers:
                tu = t.upper()
                if tu not in l0:
                    continue
                df = raw[tu].copy()
                df.columns = [c.lower() for c in df.columns]
                df = df.dropna(how="all")
                if len(df) > 0:
                    result[t] = df
    else:
        # Flat columns — single ticker fallback
        t = tickers[0]
        df = raw.copy()
        df.columns = [c.lower() for c in df.columns]
        df = df.dropna(how="all")
        if len(df) > 0:
            result[t] = df
    return result


def _get_market_cap(ticker: str, db_path: str, ttl_hours: int) -> Optional[float]:
    try:
        cached = get_market_cap(db_path, ticker, ttl_hours)
    except Exception:
        cached = None
    if cached is not None:
        return cached
    try:
        info = yf.Ticker(ticker).fast_info
        val = float(getattr(info, "market_cap", None) or 0) or None
        if val:
            put_market_cap(db_path, ticker, val)
        return val
    except Exception:
        return None


def compute_price_factors(
    ticker: str,
    df: pd.DataFrame,
    spy_df: pd.DataFrame,
    market_cap: Optional[float],
) -> Optional[dict]:
    if len(df) < 252:
        return None
    close  = df["close"]
    volume = df["volume"]
    spy_close = spy_df["close"]

    try:
        high   = df["high"]
        low    = df["low"]

        price_now   = float(close.iloc[-1])
        sma20       = float(close.iloc[-20:].mean())
        sma50       = float(close.iloc[-50:].mean())
        rsi_val     = rsi_14(close)
        macd        = macd_state(close)
        vol_ratio   = vol_surge_ratio(volume)
        above_sma20 = price_now >= sma20

        stoch_k, stoch_d, stoch_cross = stochastic(high, low, close)
        bb_pct_b, bb_width            = bollinger_pct_b(close)
        adx_val                       = adx_14(high, low, close)
        mfi_val                       = mfi_14(high, low, close, volume)

        return {
            "ticker": ticker,
            "market_cap": market_cap,
            "mom_12_1": mom_12_1(close),
            "mom_1m": mom_1m(close),
            "residual_mom": residual_momentum(close, spy_close),
            "rs_6m": rs_vs_spy(close, spy_close, window=126),
            "rs_3m": rs_vs_spy(close, spy_close, window=63),
            "rs_slope": rs_slope(close, spy_close),
            "pct_from_high": pct_from_52w_high(close),
            "breakout": breakout_flag(close, volume),
            "avg_dollar_vol_20d": avg_dollar_vol(close, volume, window=20),
            "price": price_now,
            "sma_20": sma20,
            "sma_50": sma50,
            "sma_200": float(close.iloc[-252:].mean()),
            "above_sma20": above_sma20,
            "above_sma50": price_now >= sma50,
            "rsi_14": rsi_val,
            "macd": macd,
            "vol_surge": vol_ratio,
            "stoch_k": stoch_k,
            "stoch_d": stoch_d,
            "stoch_cross": stoch_cross,
            "bb_pct_b": bb_pct_b,
            "bb_width": bb_width,
            "adx": adx_val,
            "mfi": mfi_val,
            "entry": entry_grade(
                rsi_val, macd, vol_ratio, above_sma20,
                adx_val, stoch_k, stoch_d, stoch_cross, bb_pct_b, mfi_val,
            ),
            "close_series": close,
        }
    except Exception as e:
        logger.warning(f"[prices] factor error for {ticker}: {e}")
        return None


def apply_liquidity_gate(factors_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    gate = cfg["liquidity_gate"]
    min_mcap = gate["min_market_cap"]
    min_vol  = gate["min_avg_dollar_vol_20d"]
    before = len(factors_df)
    result = factors_df[
        (factors_df["market_cap"] >= min_mcap) &
        (factors_df["avg_dollar_vol_20d"] >= min_vol)
    ].reset_index(drop=True)
    logger.info(f"[liquidity_gate] {before} → {len(result)} survivors")
    print(f"[liquidity_gate] {before} in → {len(result)} survivors")
    return result


def fetch_all_prices(
    universe_df: pd.DataFrame,
    cfg: dict,
    db_path: str,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    ttl = cfg["cache"]["price_ttl_hours"]
    tickers = universe_df["ticker"].tolist()

    spy_cached = get_prices(db_path, "SPY", ttl_hours=ttl)
    if spy_cached is not None and len(spy_cached) >= 252:
        spy_df = spy_cached
    else:
        spy_data = _fetch_batch_yfinance(["SPY"])
        spy_df   = spy_data.get("SPY", pd.DataFrame())
        if not spy_df.empty:
            put_prices(db_path, "SPY", spy_df)
        elif spy_cached is not None and not spy_cached.empty:
            # Rate-limited: fall back to stale cache rather than hard-fail
            logger.warning("[prices] SPY live fetch failed — using stale cache")
            spy_df = spy_cached
    if spy_df is None or spy_df.empty:
        raise RuntimeError("Failed to fetch SPY — cannot compute relative strength")

    price_store: dict[str, pd.DataFrame] = {"SPY": spy_df}
    factor_rows = []

    batches = [tickers[i:i+BATCH_SIZE] for i in range(0, len(tickers), BATCH_SIZE)]
    for batch_idx, batch in enumerate(batches):
        logger.info(f"[prices] batch {batch_idx+1}/{len(batches)} ({len(batch)} tickers)")

        to_fetch = []
        for t in batch:
            cached = get_prices(db_path, t, ttl_hours=ttl)
            if cached is not None and len(cached) >= 252:
                price_store[t] = cached
            else:
                to_fetch.append(t)

        if to_fetch:
            fetched = _fetch_batch_yfinance(to_fetch)
            for t, df in fetched.items():
                put_prices(db_path, t, df)
                price_store[t] = df

        for t in batch:
            df = price_store.get(t)
            if df is None or len(df) < 252:
                continue
            mcap = _get_market_cap(t, db_path, ttl)
            row = compute_price_factors(t, df, spy_df, mcap)
            if row is not None:
                row["name"] = universe_df.loc[universe_df["ticker"] == t, "name"].values[0] if len(universe_df.loc[universe_df["ticker"] == t]) > 0 else ""
                row["cik"]  = universe_df.loc[universe_df["ticker"] == t, "cik"].values[0]  if len(universe_df.loc[universe_df["ticker"] == t]) > 0 else ""
                factor_rows.append(row)

        if batch_idx < len(batches) - 1:
            time.sleep(1.0)

    factors_df = pd.DataFrame([{k: v for k, v in r.items() if k != "close_series"} for r in factor_rows])
    factors_df = apply_liquidity_gate(factors_df, cfg)
    print(f"[prices] {len(tickers)} universe → {len(factors_df)} passed liquidity gate")
    return price_store, factors_df
