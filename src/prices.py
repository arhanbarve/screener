import time
import logging
import pandas as pd
import numpy as np
import yfinance as yf
from typing import Optional
from src.factors import (
    mom_12_1, mom_1m, rs_vs_spy, rs_slope,
    pct_from_52w_high, breakout_flag, avg_dollar_vol,
)
from src.cache import get_prices, put_prices

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

    result = {}
    if len(tickers) == 1:
        t = tickers[0]
        raw.columns = [c.lower() for c in raw.columns]
        result[t] = raw.dropna(how="all")
    else:
        for t in tickers:
            if t not in raw.columns.get_level_values(0):
                continue
            df = raw[t].copy()
            df.columns = [c.lower() for c in df.columns]
            df = df.dropna(how="all")
            if len(df) > 0:
                result[t] = df
    return result


def _get_market_cap(ticker: str) -> Optional[float]:
    try:
        info = yf.Ticker(ticker).fast_info
        return float(getattr(info, "market_cap", None) or 0) or None
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
        return {
            "ticker": ticker,
            "market_cap": market_cap,
            "mom_12_1": mom_12_1(close),
            "mom_1m": mom_1m(close),
            "rs_6m": rs_vs_spy(close, spy_close, window=126),
            "rs_3m": rs_vs_spy(close, spy_close, window=63),
            "rs_slope": rs_slope(close, spy_close),
            "pct_from_high": pct_from_52w_high(close),
            "breakout": breakout_flag(close, volume),
            "avg_dollar_vol_20d": avg_dollar_vol(close, volume, window=20),
            "price": float(close.iloc[-1]),
            "sma_200": float(close.iloc[-252:].mean()),
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

    spy_data = _fetch_batch_yfinance(["SPY"])
    spy_df   = spy_data.get("SPY", pd.DataFrame())
    if spy_df.empty:
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
            mcap = _get_market_cap(t)
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
