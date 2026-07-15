"""Stage B of the portfolio backtest: band-portfolio simulator + report.

Consumes the parquet artifacts produced by src/factor_panel.py. See the
survivorship-bias warning in that module's docstring — it applies to every
number this simulator prints, and is repeated in the report output.
"""
import argparse
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRANSACTION_COST_BPS = 20.0   # per side: ~10 commission-equivalent + ~10 spread/slippage


def cagr(equity: pd.Series) -> float:
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    if years <= 0 or equity.iloc[0] <= 0:
        return float("nan")
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1)


def max_drawdown(equity: pd.Series) -> tuple[float, int]:
    """Returns (max drawdown as negative fraction, longest drawdown length in
    trading days measured peak-to-recovery; unrecovered runs count to end)."""
    peak = equity.cummax()
    dd = equity / peak - 1.0
    max_dd = float(dd.min()) if len(dd) else float("nan")

    below = equity < peak
    longest = current = 0
    for b in below:
        current = current + 1 if b else 0
        longest = max(longest, current)
    return max_dd, longest


def sharpe(equity: pd.Series) -> float:
    rets = equity.pct_change().dropna()
    std = rets.std()
    if std < 1e-12 or np.isnan(std):
        return float("nan")
    return float(rets.mean() / std * np.sqrt(252))


def per_year(equity: pd.Series) -> pd.DataFrame:
    """Calendar-year return and max drawdown."""
    rows = {}
    for year, eq in equity.groupby(equity.index.year):
        dd, _ = max_drawdown(eq)
        rows[year] = {"return": float(eq.iloc[-1] / eq.iloc[0] - 1), "max_dd": dd}
    return pd.DataFrame(rows).T
