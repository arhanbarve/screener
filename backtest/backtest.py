"""
Monthly-rebalanced long-only backtest of the composite factor vs SPY.

WARNING: This backtest uses yfinance, which contains only currently-listed tickers.
Any results therefore suffer SURVIVORSHIP BIAS — delisted names (bankruptcies,
mergers, delistings) are absent from the universe, which inflates measured returns.
Treat results as directional sanity checks only, NOT as reliable estimates of
live performance. For a defensible backtest, use a point-in-time universe with
delisted securities (CRSP, Compustat) which are not available for free.

LOOK-AHEAD BIAS WARNING: Factors computed on date t must only use data that was
publicly available on or before t. Earnings estimates and revisions have reporting
lags that are approximated here but not precisely modeled. Treat all results
with appropriate skepticism.
"""

import logging
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date
from src.factors import mom_12_1, rs_vs_spy

logger = logging.getLogger(__name__)

TRANSACTION_COST_BPS = 10  # per side, conservative minimum


def _download_history(tickers: list, start: str, end: str) -> dict:
    joined = " ".join(tickers)
    try:
        raw = yf.download(joined, start=start, end=end, auto_adjust=True, progress=False, group_by="ticker")
    except Exception as e:
        logger.error(f"yfinance download failed: {e}")
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
            result[t] = df.dropna(how="all")
    return result


def _monthly_rebalance_dates(start: date, end: date) -> list:
    dates = []
    current = date(start.year, start.month, 1)
    while current <= end:
        dates.append(current)
        if current.month == 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, current.month + 1, 1)
    return dates


def run_backtest(
    tickers: list,
    start: str = "2018-01-01",
    end: str | None = None,
    top_decile_n: int = 50,
    out_path: str = "output/backtest_results.md",
):
    """
    Build monthly-rebalanced long-only portfolios from the top decile of
    12-1 momentum + 6m RS composite, compare to SPY total return.
    """
    if end is None:
        end = date.today().isoformat()

    print("[backtest] Downloading price history — this may take several minutes...")
    all_tickers = tickers + ["SPY"]
    prices = _download_history(all_tickers, start, end)
    spy_df = prices.get("SPY", pd.DataFrame())
    if spy_df.empty:
        raise RuntimeError("Could not fetch SPY")

    rebalance_dates = _monthly_rebalance_dates(
        date.fromisoformat(start), date.fromisoformat(end)
    )

    portfolio_returns = []
    spy_returns = []

    for i, rb_date in enumerate(rebalance_dates[:-1]):
        next_date = rebalance_dates[i + 1]
        rb_str   = rb_date.isoformat()
        next_str = next_date.isoformat()

        scored = []
        for ticker in tickers:
            df = prices.get(ticker, pd.DataFrame())
            if df.empty or len(df) < 252:
                continue
            hist = df[df.index.date <= rb_date]
            if len(hist) < 252:
                continue
            spy_hist = spy_df[spy_df.index.date <= rb_date]
            if len(spy_hist) < 252:
                continue
            try:
                score = (
                    0.5 * mom_12_1(hist["close"]) +
                    0.5 * rs_vs_spy(hist["close"], spy_hist["close"], window=126)
                )
                scored.append((ticker, score))
            except Exception:
                continue

        if not scored:
            continue

        scored.sort(key=lambda x: x[1], reverse=True)
        top = [t for t, _ in scored[:top_decile_n]]

        monthly_rets = []
        for ticker in top:
            df = prices.get(ticker, pd.DataFrame())
            period = df[(df.index.date >= rb_date) & (df.index.date < next_date)]
            if len(period) < 2:
                continue
            ret = float(period["close"].iloc[-1] / period["close"].iloc[0]) - 1.0
            ret -= 2 * TRANSACTION_COST_BPS / 10000
            monthly_rets.append(ret)

        if monthly_rets:
            portfolio_returns.append({"date": rb_str, "return": np.mean(monthly_rets)})

        spy_period = spy_df[(spy_df.index.date >= rb_date) & (spy_df.index.date < next_date)]
        if len(spy_period) >= 2:
            spy_ret = float(spy_period["close"].iloc[-1] / spy_period["close"].iloc[0]) - 1.0
            spy_returns.append({"date": rb_str, "return": spy_ret})

    if not portfolio_returns:
        print("[backtest] Insufficient data to produce results.")
        return

    port_df = pd.DataFrame(portfolio_returns)
    spy_df2 = pd.DataFrame(spy_returns)

    port_cum = (1 + port_df["return"]).cumprod().iloc[-1] - 1
    spy_cum  = (1 + spy_df2["return"]).cumprod().iloc[-1] - 1 if len(spy_df2) else float("nan")

    port_ann = (1 + port_cum) ** (12 / len(port_df)) - 1
    spy_ann  = (1 + spy_cum)  ** (12 / len(spy_df2)) - 1 if len(spy_df2) else float("nan")

    sharpe_port = port_df["return"].mean() / (port_df["return"].std() + 1e-12) * (12 ** 0.5)

    summary = f"""# Backtest Results

> **SURVIVORSHIP BIAS WARNING:** This backtest uses yfinance, which only contains
> currently-listed tickers. Delisted names (bankruptcies, acquisitions, failures)
> are absent. Results OVERSTATE actual returns. Do NOT use for capital allocation.
>
> **LOOK-AHEAD BIAS:** Earnings estimate data is approximated with reporting lags.
> Treat results as directional sanity checks only.
>
> **Transaction costs:** {TRANSACTION_COST_BPS}bps/side (20bps round-trip) applied.

## Summary ({start} to {end})

| Metric | Portfolio | SPY |
|--------|-----------|-----|
| Cumulative return | {port_cum:.1%} | {spy_cum:.1%} |
| Annualized return | {port_ann:.1%} | {spy_ann:.1%} |
| Monthly Sharpe (ann.) | {sharpe_port:.2f} | — |
| Months tracked | {len(port_df)} | {len(spy_df2)} |

*Model: top-{top_decile_n} by 12-1 momentum + 6m RS, equal-weight, monthly rebalance.*
"""
    f = open(out_path, "w")
    f.write(summary)
    f.close()
    print(summary)
    print(f"[backtest] Results written to {out_path}")


if __name__ == "__main__":
    universe = pd.read_parquet("data/universe.parquet")
    tickers = universe["ticker"].tolist()[:500]
    run_backtest(tickers, start="2018-01-01")
