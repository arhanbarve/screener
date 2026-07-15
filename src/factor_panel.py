"""Stage A of the portfolio backtest: point-in-time price-block factor panel.

SURVIVORSHIP BIAS WARNING (carried from the retired backtest/backtest.py):
yfinance contains only currently-listed tickers. Delisted names
(bankruptcies, mergers) are absent, which inflates measured returns.
Treat all results as an upper bound / directional sanity check, NOT a
reliable estimate of live performance. The spec compensates by demanding a
wide pass margin and judging the regime ladder on relative drawdown
reduction vs our own unhedged portfolio.

LOOK-AHEAD NOTE: every factor at date t uses only bars <= t. Market cap at t
is approximated as (cached market cap today) x (close_t / close_today) —
i.e. constant share count. Breadth series use all candidate tickers with
enough data at t, not gate survivors.

Only the price momentum block is computed (mom_12_1, residual_mom, rs_6m,
rs_accel, rs_slope, pct_from_high); fundamental factors cannot be
reconstructed point-in-time from free sources. Weights renormalized to 1.
"""
import argparse
import logging

import numpy as np
import pandas as pd

from src.cache import get_market_cap_stale
from src.event_backtest import get_history, get_history_bulk
from src.factors import residual_momentum_from_monthly

logger = logging.getLogger(__name__)

# config.yaml price-block weights (sum 0.50), renormalized to sum 1.0
PRICE_BLOCK_WEIGHTS = {
    "mom_12_1": 0.24,
    "residual_mom": 0.28,
    "rs_6m": 0.20,
    "rs_accel": 0.12,
    "rs_slope": 0.08,
    "pct_from_high": 0.08,
}

MIN_MCAP_TODAY = 150e6        # candidate floor (half the live $300M gate)
GATE_MIN_MCAP = 300e6         # point-in-time liquidity gate
GATE_MIN_DOLLAR_VOL = 5e6
GATE_MAX_BELOW_HIGH = 0.35    # config.yaml confirmation.max_pct_below_52w_high
WARMUP_DAYS = 400             # calendar days of history before sim start

PANEL_PATH = "output/factor_panel.parquet"
BREADTH_PATH = "output/breadth.parquet"


def rebalance_dates(index: pd.DatetimeIndex) -> list[pd.Timestamp]:
    """Last trading day of each ISO week present in `index`."""
    s = pd.Series(index, index=index)
    iso = index.isocalendar()
    return list(s.groupby([iso.year, iso.week]).max().sort_values())


def _rolling_slope(y: pd.Series, window: int) -> pd.Series:
    """Least-squares slope of y against x=0..window-1, rolling. Matches
    scipy.stats.linregress slope used by factors.rs_slope."""
    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    x_var_sum = ((x - x_mean) ** 2).sum()

    def _slope(vals: np.ndarray) -> float:
        return float(((x - x_mean) * (vals - vals.mean())).sum() / x_var_sum)

    return y.rolling(window).apply(_slope, raw=True)


def _monthly_asof(monthly: pd.Series, daily: pd.Series, t: pd.Timestamp,
                   monthly_periods: pd.PeriodIndex) -> pd.Series:
    """Month-end-resampled `monthly` series truncated to periods <= t's
    month, with the in-progress (current) month's bin overwritten by
    `daily.loc[t]` so it reflects only data through t. Reproduces
    factors.residual_momentum's behavior of resampling close.loc[:t] fresh
    at each t, without leaking future bars into t's own still-in-progress
    month (which a single upfront resample("ME") of the full series would
    do, since "ME" labels a bin at the true calendar month-end regardless
    of how much data is actually available)."""
    t_period = t.to_period("M")
    m = monthly[monthly_periods <= t_period].copy()
    if len(m) and m.index[-1].to_period("M") == t_period:
        m.iloc[-1] = daily.loc[t]
    return m


def ticker_factor_frame(prices: pd.DataFrame, spy_close: pd.Series,
                        dates: list[pd.Timestamp]) -> pd.DataFrame:
    """All price-block factors + gate ingredients for one ticker, evaluated
    at `dates`. `prices` needs 'close' and 'volume' columns, daily index."""
    close = prices["close"]
    volume = prices["volume"]
    spy = spy_close.reindex(close.index).ffill()

    # NOTE: shift amounts are (window - 1), not window: the scalar functions
    # in factors.py compare close.iloc[-1] to close.iloc[-window], which
    # spans (window - 1) bars, not `window` bars as a naive .shift(window)
    # would. This keeps the panel numerically identical to the scalar
    # functions on sliced history (see test_factor_panel.py).
    mom_12_1 = close.shift(20) / close.shift(251) - 1.0
    stock_6m = close / close.shift(125) - 1.0
    spy_6m = spy / spy.shift(125) - 1.0
    rs_6m = stock_6m - spy_6m
    stock_3m = close / close.shift(62) - 1.0
    spy_3m = spy / spy.shift(62) - 1.0
    rs_3m = stock_3m - spy_3m
    rs_accel = 2 * rs_3m - rs_6m
    rs_slope = _rolling_slope(close / spy, 63)
    high_52w = close.rolling(252).max()
    pct_from_high = close / high_52w - 1.0
    dollar_vol = (close * volume).rolling(20).mean()
    sma200 = close.rolling(200).mean()

    # residual momentum: resample once, evaluate per date on month-ends <= t.
    # Resampling the full series ahead of time would leak future bars into
    # the bin for t's own (still in-progress) month, since "ME" labels that
    # bin at the true calendar month-end regardless of how much data is
    # actually available. Reproduce factors.residual_momentum's behavior
    # (which resamples close.loc[:t]) by keeping only bins whose period is
    # <= t's period, and overwriting the in-progress bin's value with
    # close.loc[t] so it reflects only data through t.
    monthly_close = close.resample("ME").last()
    monthly_spy = spy.resample("ME").last()
    monthly_periods = monthly_close.index.to_period("M")

    rows = []
    for t in dates:
        if t not in close.index or len(close.loc[:t]) < 252:
            rows.append({c: np.nan for c in
                         ["mom_12_1", "residual_mom", "rs_6m", "rs_accel",
                          "rs_slope", "pct_from_high", "dollar_vol_20d",
                          "close", "above_sma200"]})
            continue
        res_mom = residual_momentum_from_monthly(
            _monthly_asof(monthly_close, close, t, monthly_periods),
            _monthly_asof(monthly_spy, spy, t, monthly_periods))
        rows.append({
            "mom_12_1": float(mom_12_1.loc[t]),
            "residual_mom": res_mom,
            "rs_6m": float(rs_6m.loc[t]),
            "rs_accel": float(rs_accel.loc[t]),
            "rs_slope": float(rs_slope.loc[t]),
            "pct_from_high": float(pct_from_high.loc[t]),
            "dollar_vol_20d": float(dollar_vol.loc[t]),
            "close": float(close.loc[t]),
            "above_sma200": bool(close.loc[t] >= sma200.loc[t]),
        })
    frame = pd.DataFrame(rows, index=pd.DatetimeIndex(dates))
    # Mixing NaN placeholder rows with computed bool rows makes pandas
    # coerce this column to a plain float/object dtype where NaN is
    # indistinguishable from a truthy value under naive `if row[...]:`
    # checks (bool(float('nan')) is True). Cast to the nullable boolean
    # dtype so True/False/<NA> stay a genuine tri-state column.
    frame["above_sma200"] = frame["above_sma200"].astype("boolean")
    return frame


FACTOR_COLS = list(PRICE_BLOCK_WEIGHTS)


def apply_gates(panel: pd.DataFrame) -> pd.DataFrame:
    """Point-in-time liquidity + confirmation gates -> `passes_gates` column.
    Rows with any missing factor fail (price-block factors are all-or-nothing
    from the same OHLCV history, so partial coverage means short history)."""
    p = panel.copy()
    has_factors = p[FACTOR_COLS].notna().all(axis=1)
    p["passes_gates"] = (
        has_factors
        & (p["mcap"] >= GATE_MIN_MCAP)
        & (p["dollar_vol_20d"] >= GATE_MIN_DOLLAR_VOL)
        & p["above_sma200"].fillna(False)
        & (p["pct_from_high"] >= -GATE_MAX_BELOW_HIGH)
    )
    return p


def _winsorize(s: pd.Series, pct: float = 0.01) -> pd.Series:
    if s.notna().sum() < 3:
        return s
    lo, hi = s.quantile(pct), s.quantile(1 - pct)
    return s.clip(lo, hi)


def composite_scores(panel: pd.DataFrame) -> pd.DataFrame:
    """Winsorize + z-score each factor cross-sectionally per date among gate
    survivors; composite = weighted sum. Non-survivors get composite NaN."""
    p = panel.copy()
    p["composite"] = np.nan

    for date, idx in p.groupby("date").groups.items():
        rows = p.loc[idx]
        surv = rows.index[rows["passes_gates"]]
        if len(surv) < 3:
            continue
        comp = pd.Series(0.0, index=surv)
        for col, w in PRICE_BLOCK_WEIGHTS.items():
            vals = _winsorize(p.loc[surv, col])
            std = vals.std()
            if std < 1e-12 or np.isnan(std):
                continue
            comp += w * (vals - vals.mean()) / std
        p.loc[surv, "composite"] = comp
    return p


def breadth_series(prices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Daily fraction of tickers above their own SMA200 / SMA50.
    Denominator = tickers whose SMA window has formed by that day."""
    above200, above50 = [], []
    for t, df in prices.items():
        c = df["close"]
        above200.append((c >= c.rolling(200).mean()).where(c.rolling(200).mean().notna()))
        above50.append((c >= c.rolling(50).mean()).where(c.rolling(50).mean().notna()))
    a200 = pd.concat(above200, axis=1)
    a50 = pd.concat(above50, axis=1)
    return pd.DataFrame({
        "pct_above_200": a200.mean(axis=1),
        "pct_above_50": a50.mean(axis=1),
    })
