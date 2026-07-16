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


def simulate(
    panel: pd.DataFrame,
    closes: pd.DataFrame,
    exposure: pd.Series | None = None,
    score_col: str = "composite",
    entry_band: int = 20,
    exit_band: int = 35,
    max_positions: int = 15,
    cost_bps: float = TRANSACTION_COST_BPS,
    initial_capital: float = 100_000.0,
) -> dict:
    """Band-portfolio simulation.

    - Daily mark-to-market on `closes` (columns = tickers).
    - On each panel date (weekly): rank gate-survivors by `score_col` desc;
      sell holdings ranked beyond `exit_band` (or absent from survivors);
      buy best-ranked names within `entry_band` until `max_positions` held.
      New positions sized equal-weight against target invested capital;
      existing positions drift (no weight rebalancing).
    - Exposure (daily target fraction, default 1.0): on days the target
      changes, all positions scale pro-rata toward target invested value.
      Known limitation: the scale factor is computed over the whole book but
      only applied to positions untouched by that day's step-2 rebalance
      trades, so when a large fraction of the book turns over on the same
      day the target also changes, the achieved exposure can miss the
      target (proportional to turnover fraction) until the next exposure
      change fires a fresh rescale.
    - Costs: `cost_bps` per side on traded notional. Cash earns 0%.
    """
    exposure = (pd.Series(1.0, index=closes.index) if exposure is None
                else exposure.reindex(closes.index).ffill().fillna(1.0))
    cost_rate = cost_bps / 1e4

    rebal_by_date = {d: g for d, g in panel.groupby("date")}
    positions: dict[str, float] = {}     # ticker -> current dollar value
    cash = initial_capital
    trades, daily_rows = [], []
    equity_curve = {}
    total_costs = 0.0
    turnover_notional = 0.0
    prev_exposure = None

    rets = closes.pct_change().fillna(0.0)

    for day in closes.index:
        # 1) mark to market
        for t in list(positions):
            positions[t] *= 1.0 + float(rets.loc[day].get(t, 0.0))
        equity = cash + sum(positions.values())

        def _trade(ticker: str, notional: float, side: str):
            nonlocal cash, total_costs, turnover_notional
            fee = abs(notional) * cost_rate
            if side == "buy":
                positions[ticker] = positions.get(ticker, 0.0) + notional
                cash -= notional + fee
            else:
                positions[ticker] = positions.get(ticker, 0.0) - notional
                if positions[ticker] < 1e-9:
                    positions.pop(ticker, None)
                cash += notional - fee
            total_costs += fee
            turnover_notional += abs(notional)
            trades.append({"date": day, "ticker": ticker, "side": side,
                           "notional": abs(notional)})

        target_exposure = float(exposure.loc[day])

        # 2) weekly band rebalance at this close
        rebalanced_today = day in rebal_by_date
        traded_in_step2: set[str] = set()
        if rebalanced_today:
            g = rebal_by_date[day]
            surv = g[g["passes_gates"]].dropna(subset=[score_col])
            ranked = surv.sort_values(score_col, ascending=False)["ticker"].tolist()
            rank_of = {t: i + 1 for i, t in enumerate(ranked)}

            for t in list(positions):
                r = rank_of.get(t)
                if r is None or r > exit_band:
                    _trade(t, positions[t], "sell")
                    traded_in_step2.add(t)

            equity = cash + sum(positions.values())
            target_invested = equity * target_exposure
            slot = target_invested / max_positions if max_positions else 0.0
            for t in ranked:
                if len(positions) >= max_positions:
                    break
                if t in positions or rank_of[t] > entry_band:
                    continue
                # fees come out of cash too: shrink the final slot to what
                # cash affords rather than skipping it over a fee-sized
                # shortfall, but don't open dust positions
                notional = min(slot, cash / (1.0 + cost_rate))
                if notional <= 1e-9 or notional < slot * 0.5:
                    break
                _trade(t, notional, "buy")
                traded_in_step2.add(t)

        # 3) daily exposure adjustment (pro-rata) when target changed.
        # Runs every day, including rebalance days -- but skips any ticker
        # step 2 already traded today (opened/closed), since re-scaling
        # those would just churn (buy-then-sell the same ticker step 2
        # just opened) without changing the end-of-day allocation. Other,
        # untouched (drifting) positions still need pro-rata rescaling
        # toward the new target -- otherwise an exposure change that lands
        # on a rebalance day with little band turnover never gets applied
        # to them.
        invested = sum(positions.values())
        if prev_exposure is not None and target_exposure != prev_exposure and invested > 0:
            equity = cash + invested
            target_invested = equity * target_exposure
            scale = target_invested / invested
            if scale > 1.0:
                # Buying: all deltas below are positive (pro-rata scale-up
                # of long-only positions), so cap their sum to what cash
                # affords -- same principle as step 2's cash-affordability
                # cap -- to keep cash from going negative. Only sum deltas
                # for positions step 3 actually touches (untouched ones);
                # step-2-traded tickers are excluded from the loop below.
                total_buy = sum(v for t, v in positions.items()
                                 if t not in traded_in_step2) * (scale - 1.0)
                if total_buy > 1e-9:
                    affordable = cash / (1.0 + cost_rate)
                    if total_buy > affordable:
                        shrink = max(0.0, affordable / total_buy)
                        scale = 1.0 + (scale - 1.0) * shrink
            for t in list(positions):
                if t in traded_in_step2:
                    continue
                delta = positions[t] * (scale - 1.0)
                if delta > 0:
                    _trade(t, delta, "buy")
                elif delta < 0:
                    _trade(t, -delta, "sell")
        prev_exposure = target_exposure

        invested = sum(positions.values())
        equity = cash + invested
        equity_curve[day] = equity
        daily_rows.append({"date": day, "equity": equity, "invested": invested,
                           "n_positions": len(positions),
                           "exposure_target": target_exposure})

    equity_s = pd.Series(equity_curve).sort_index()
    daily = pd.DataFrame(daily_rows).set_index("date")
    avg_exposure = float((daily["invested"] / daily["equity"]).mean())
    return {
        "equity": equity_s,
        "daily": daily,
        "trades": pd.DataFrame(trades, columns=["date", "ticker", "side", "notional"]),
        "costs": total_costs,
        "turnover_notional": turnover_notional,
        "avg_exposure": avg_exposure,
    }
