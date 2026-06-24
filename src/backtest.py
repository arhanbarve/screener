"""
Point-in-time backtest for price-based factors.

Uses the 420-day OHLCV history in cache.db (no look-ahead).
Fundamental factors (sue, rev_breadth, gp_assets) require the
fundamentals_history table populated by daily archiving — available
only after 90+ days of production runs.

Usage:
    python3 -m src.backtest
    python3 -m src.backtest --horizon 42 --start 2025-06-01
"""
import argparse
import logging
import sqlite3
from datetime import date, timedelta

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.config import load_config
from src.factors import (
    mom_12_1, mom_1m, rs_vs_spy, rs_slope, residual_momentum,
    pct_from_52w_high, rsi_14, macd_state, vol_surge_ratio,
    stochastic, bollinger_pct_b, adx_14, mfi_14,
    trend_signal_score, oscillator_signal_score, volume_signal_score,
)

logger = logging.getLogger(__name__)

PRICE_FACTORS = [
    "mom_12_1", "residual_mom", "rs_6m", "rs_3m", "rs_accel",
    "rs_slope", "pct_from_high", "mom_1m",
    "trend_score", "momo_osc_score", "volume_score",
]


def _load_prices_from_db(db_path: str) -> dict[str, pd.DataFrame]:
    """Load all tickers with ≥252 days from cache.db prices table."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql(
        "SELECT ticker, date, close, high, low, volume FROM prices ORDER BY ticker, date",
        conn,
        parse_dates=["date"],
    )
    conn.close()
    result = {}
    for ticker, g in df.groupby("ticker"):
        g = g.set_index("date").sort_index()
        if len(g) >= 252:
            result[ticker] = g
    logger.info(f"[backtest] loaded {len(result)} tickers with ≥252 days")
    return result


def _compute_factors_at(
    ticker: str,
    prices: pd.DataFrame,
    spy_prices: pd.DataFrame,
    as_of: pd.Timestamp,
) -> dict | None:
    """Compute all price-based factors for a ticker as-of a given date (no look-ahead)."""
    p = prices.loc[:as_of]
    s = spy_prices.loc[:as_of]
    if len(p) < 252 or len(s) < 252:
        return None

    close  = p["close"]
    volume = p["volume"]
    high   = p["high"]
    low    = p["low"]
    spy_close = s["close"]

    try:
        sk, sd, scross = stochastic(high, low, close)
        bb_b, _        = bollinger_pct_b(close)
        adx_val        = adx_14(high, low, close)
        mfi_val        = mfi_14(high, low, close, volume)
        rsi_val        = rsi_14(close)
        macd           = macd_state(close)
        vol_ratio      = vol_surge_ratio(volume)
        above50        = float(close.iloc[-1]) >= float(close.iloc[-50:].mean())

        row = pd.Series({
            "rsi_14": rsi_val, "macd": macd, "vol_surge": vol_ratio,
            "stoch_k": sk, "stoch_d": sd, "stoch_cross": scross,
            "bb_pct_b": bb_b, "adx": adx_val, "mfi": mfi_val, "above_sma50": above50,
        })

        return {
            "ticker":        ticker,
            "mom_12_1":      mom_12_1(close),
            "mom_1m":        mom_1m(close),
            "residual_mom":  residual_momentum(close, spy_close),
            "rs_6m":         rs_vs_spy(close, spy_close, window=126),
            "rs_3m":         rs_vs_spy(close, spy_close, window=63),
            "rs_accel":      rs_vs_spy(close, spy_close, window=63) * 2.0 - rs_vs_spy(close, spy_close, window=126),
            "rs_slope":      rs_slope(close, spy_close),
            "pct_from_high": pct_from_52w_high(close),
            "trend_score":   trend_signal_score(row),
            "momo_osc_score": oscillator_signal_score(row),
            "volume_score":  volume_signal_score(row),
        }
    except Exception:
        return None


def _forward_return(prices: pd.DataFrame, as_of: pd.Timestamp, horizon: int) -> float | None:
    future = prices.loc[as_of:]
    trading_days = future.index[future.index > as_of]
    if len(trading_days) < horizon:
        return None
    price_t  = float(prices.loc[as_of, "close"])
    price_th = float(prices.loc[trading_days[horizon - 1], "close"])
    return (price_th - price_t) / price_t


def run_factor_ic_analysis(
    db_path: str,
    horizons: list[int] | None = None,
    start_date: str = "2025-01-01",
    end_date: str | None = None,
    min_market_cap: float = 300e6,
) -> pd.DataFrame:
    """
    Compute per-factor Information Coefficients (IC) over the cached price history.

    IC = Spearman rank correlation between factor value at time t
         and forward return at t + horizon.

    Returns DataFrame with columns: factor, horizon, ic_mean, ic_std, ir, hit_rate, n_periods
    """
    if horizons is None:
        horizons = [21, 42, 63]
    if end_date is None:
        end_date = (date.today() - timedelta(days=max(horizons))).isoformat()

    prices = _load_prices_from_db(db_path)
    if "SPY" not in prices:
        raise RuntimeError("SPY not in cache — required for RS factors")

    spy_prices = prices["SPY"]

    # Build rebalance dates: every 21 trading days within window
    all_dates = spy_prices.loc[start_date:end_date].index
    rebal_dates = all_dates[::21]
    logger.info(f"[backtest] {len(rebal_dates)} rebalance dates, {len(prices)} tickers")

    records = []
    for rebal_date in rebal_dates:
        factor_rows = []
        for ticker, p in prices.items():
            if ticker == "SPY":
                continue
            row = _compute_factors_at(ticker, p, spy_prices, rebal_date)
            if row is None:
                continue
            # Liquidity proxy: require ≥10 non-zero volume days
            p_slice = p.loc[:rebal_date].iloc[-20:]
            if float(p_slice["volume"].mean()) < 1e4:
                continue
            row["date"] = rebal_date
            factor_rows.append(row)

        if len(factor_rows) < 30:
            continue

        cross = pd.DataFrame(factor_rows).set_index("ticker")

        for horizon in horizons:
            fwd_rets = {}
            for ticker, p in prices.items():
                if ticker not in cross.index:
                    continue
                r = _forward_return(p, rebal_date, horizon)
                if r is not None:
                    fwd_rets[ticker] = r
            if len(fwd_rets) < 20:
                continue

            fwd_series = pd.Series(fwd_rets)
            common = cross.index.intersection(fwd_series.index)
            if len(common) < 20:
                continue

            for factor in PRICE_FACTORS:
                if factor not in cross.columns:
                    continue
                f_vals = cross.loc[common, factor].dropna()
                f_rets = fwd_series.loc[f_vals.index]
                if len(f_vals) < 15:
                    continue
                ic, _ = spearmanr(f_vals.values, f_rets.values)
                records.append({
                    "date": rebal_date,
                    "factor": factor,
                    "horizon": horizon,
                    "ic": ic,
                    "n_stocks": len(f_vals),
                })

    if not records:
        logger.warning("[backtest] no IC records computed — check date range and cache")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    summary = (
        df.groupby(["factor", "horizon"])["ic"]
        .agg(ic_mean="mean", ic_std="std", n_periods="count")
        .reset_index()
    )
    summary["ir"]       = summary["ic_mean"] / summary["ic_std"].replace(0, float("nan"))
    summary["hit_rate"] = (
        df[df["ic"] > 0].groupby(["factor", "horizon"])["ic"].count()
        / df.groupby(["factor", "horizon"])["ic"].count()
    ).reset_index(drop=True)
    # Re-merge hit_rate properly
    hit = (df.groupby(["factor", "horizon"])
             .apply(lambda g: (g["ic"] > 0).mean())
             .reset_index(name="hit_rate"))
    summary = summary.merge(hit, on=["factor", "horizon"])
    return summary.sort_values(["horizon", "ir"], ascending=[True, False])


def test_decile_monotonicity(
    db_path: str,
    weights: dict,
    horizon: int = 42,
    start_date: str = "2025-01-01",
) -> pd.DataFrame:
    """
    Sort universe into 10 deciles by composite score and measure average
    forward returns. Monotonic increase = signal is real. Flat = fragile.
    """
    from src.config import load_config
    prices = _load_prices_from_db(db_path)
    spy_prices = prices.get("SPY")
    if spy_prices is None:
        raise RuntimeError("SPY required")

    end_date = (date.today() - timedelta(days=horizon)).isoformat()
    all_dates = spy_prices.loc[start_date:end_date].index
    rebal_dates = all_dates[::21]

    decile_rets = {i: [] for i in range(1, 11)}

    for rebal_date in rebal_dates:
        rows = []
        for ticker, p in prices.items():
            if ticker == "SPY":
                continue
            row = _compute_factors_at(ticker, p, spy_prices, rebal_date)
            if row is None:
                continue
            fwd = _forward_return(p, rebal_date, horizon)
            if fwd is None:
                continue
            row["fwd_return"] = fwd
            rows.append(row)

        if len(rows) < 50:
            continue

        df = pd.DataFrame(rows)
        composite = pd.Series(0.0, index=df.index)
        for factor, w in weights.items():
            if factor in df.columns:
                s = df[factor].fillna(df[factor].mean())
                z = (s - s.mean()) / (s.std() + 1e-12)
                composite += w * z
        df["composite"] = composite
        df["decile"] = pd.qcut(composite, 10, labels=False, duplicates="drop") + 1
        for d, g in df.groupby("decile"):
            decile_rets[d].append(g["fwd_return"].mean())

    result = pd.DataFrame({
        "decile":    list(range(1, 11)),
        "avg_return": [np.mean(decile_rets[d]) if decile_rets[d] else float("nan")
                       for d in range(1, 11)],
        "n_periods": [len(decile_rets[d]) for d in range(1, 11)],
    })
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Run factor IC backtest")
    parser.add_argument("--db",      default="data/cache.db")
    parser.add_argument("--horizon", type=int, default=42)
    parser.add_argument("--start",   default="2025-01-01")
    args = parser.parse_args()

    cfg = load_config()
    print(f"\nRunning IC analysis — horizon={args.horizon}d, start={args.start}")
    ic_df = run_factor_ic_analysis(args.db, horizons=[21, args.horizon, 63], start_date=args.start)
    if ic_df.empty:
        print("No results. Run screener first to populate cache.")
    else:
        print("\n=== Factor IC Summary ===")
        print(ic_df.to_string(index=False, float_format="{:.3f}".format))

    print(f"\nRunning decile monotonicity test — horizon={args.horizon}d")
    weights = cfg["factors"]["weights"]
    price_weights = {k: v for k, v in weights.items()
                     if k in ["mom_12_1", "residual_mom", "rs_6m", "rs_accel", "rs_slope",
                               "pct_from_high", "trend_score", "momo_osc_score", "volume_score"]}
    mono = test_decile_monotonicity(args.db, price_weights, horizon=args.horizon, start_date=args.start)
    print("\n=== Decile Monotonicity ===")
    print(mono.to_string(index=False, float_format="{:.3f}".format))
