"""Stage B of the portfolio backtest: band-portfolio simulator + report.

Consumes the parquet artifacts produced by src/factor_panel.py. See the
survivorship-bias warning in that module's docstring — it applies to every
number this simulator prints, and is repeated in the report output.
"""
import argparse
import logging
import os

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


CRASH_WINDOWS = {"2020": ("2020-01-01", "2020-12-31"),
                 "2022": ("2022-01-01", "2022-12-31")}
DD_REDUCTION_REQUIRED = 1 / 3      # spec pass criterion 1
CAGR_GIVEUP_MAX = 0.02             # spec pass criterion 2


def _window_dd(run: dict, start: str, end: str) -> float:
    eq = run["equity"].loc[start:end]
    if len(eq) < 2:
        return float("nan")
    dd, _ = max_drawdown(eq)
    return dd


def verdict(unhedged: dict, laddered: dict, naive: dict) -> dict:
    """Spec pass criteria, all a priori. NaN window (period doesn't cover a
    crash year) -> that check passes vacuously (counts toward `overall` per
    original intent -- a window absent from the data shouldn't fail the run)
    but is flagged via the `dd_2020_covered`/`dd_2022_covered` keys so callers
    and the report can distinguish a vacuous pass from a real one."""
    dd_u, _ = max_drawdown(unhedged["equity"])
    dd_l, _ = max_drawdown(laddered["equity"])
    reduced = (abs(dd_l) <= abs(dd_u) * (1 - DD_REDUCTION_REQUIRED))

    window_ok, window_covered = {}, {}
    for name, (s, e) in CRASH_WINDOWS.items():
        wu, wl = _window_dd(unhedged, s, e), _window_dd(laddered, s, e)
        if np.isnan(wu) or np.isnan(wl):
            window_ok[name] = True   # window outside sim period
            window_covered[name] = False
        else:
            window_ok[name] = abs(wl) <= abs(wu) * (1 - DD_REDUCTION_REQUIRED)
            window_covered[name] = True

    giveup = cagr(unhedged["equity"]) - cagr(laddered["equity"])
    cagr_ok = giveup <= CAGR_GIVEUP_MAX
    sharpe_ok = sharpe(laddered["equity"]) >= sharpe(naive["equity"])

    overall = (reduced and window_ok["2020"] and window_ok["2022"]
               and cagr_ok and sharpe_ok)
    return {
        "dd_unhedged": dd_u, "dd_laddered": dd_l,
        "dd_reduced_third": reduced,
        "dd_2020_ok": window_ok["2020"], "dd_2022_ok": window_ok["2022"],
        "dd_2020_covered": window_covered["2020"], "dd_2022_covered": window_covered["2022"],
        "cagr_giveup": giveup, "cagr_giveup_ok": cagr_ok,
        "sharpe_vs_naive_ok": sharpe_ok,
        "overall": overall,
    }


def write_report(path: str, runs: dict[str, dict], v: dict, meta: dict) -> None:
    lines = [
        "# Portfolio Backtest Report",
        "",
        f"Period: {meta.get('start')} → {meta.get('end')} · "
        f"costs {meta.get('cost_bps')} bps/side · cash earns 0%",
        "",
        "> **SURVIVORSHIP BIAS WARNING:** universe contains only currently-",
        "> listed tickers (yfinance). Delisted losers are absent; every CAGR",
        "> here is an upper bound. Ladder value is judged on *relative*",
        "> drawdown reduction, which is far less bias-sensitive.",
        "",
        "## Summary",
        "",
        "| Strategy | CAGR | Vol | Sharpe | MaxDD | LongestDD (d) | AvgExp | ExpAdj CAGR | Costs $ |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, run in runs.items():
        eq = run["equity"]
        dd, dd_days = max_drawdown(eq)
        c = cagr(eq)
        ae = run.get("avg_exposure", 1.0)
        vol = eq.pct_change().std() * np.sqrt(252)
        exp_adj = c / ae if ae > 0 else float("nan")
        lines.append(
            f"| {name} | {c:.2%} | {vol:.2%} | {sharpe(eq):.2f} | {dd:.2%} "
            f"| {dd_days} | {ae:.2f} | {exp_adj:.2%} | {run.get('costs', 0):,.0f} |")

    lines += ["", "## Per-year", ""]
    for name, run in runs.items():
        table = per_year(run["equity"])
        lines += [f"### {name}", "", "| Year | Return | MaxDD |", "|---|---|---|"]
        for year, row in table.iterrows():
            lines.append(f"| {year} | {row['return']:.2%} | {row['max_dd']:.2%} |")
        lines.append("")

    lines += [
        "## Verdict (spec pass criteria, a priori)",
        "",
        f"- Max drawdown cut ≥ 1/3 overall: **{v['dd_reduced_third']}** "
        f"(unhedged {v['dd_unhedged']:.2%} → laddered {v['dd_laddered']:.2%})",
        f"- 2020 window cut ≥ 1/3: **{v['dd_2020_ok']}**"
        f"{' (window not in sim period)' if not v['dd_2020_covered'] else ''}",
        f"- 2022 window cut ≥ 1/3: **{v['dd_2022_ok']}**"
        f"{' (window not in sim period)' if not v['dd_2022_covered'] else ''}",
        f"- CAGR give-up ≤ 2pts: **{v['cagr_giveup_ok']}** ({v['cagr_giveup']:.2%})",
        f"- Ladder Sharpe ≥ naive momentum Sharpe: **{v['sharpe_vs_naive_ok']}**",
        "",
        f"**OVERALL: {'PASS' if v['overall'] else 'FAIL'}**",
        "",
        meta.get("note", ""),   # raw markdown block (e.g. sensitivity table)
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"[report] wrote {path}")


from src import regime
from src.event_backtest import get_history

SENSITIVITY_MAPS = [
    {0: 1.00, 1: 1.00, 2: 0.66, 3: 0.33, 4: 0.00},   # spec (chosen)
    {0: 1.00, 1: 0.75, 2: 0.50, 3: 0.25, 4: 0.00},   # linear
    {0: 1.00, 1: 1.00, 2: 0.50, 3: 0.00, 4: 0.00},   # aggressive
    {0: 1.00, 1: 0.66, 2: 0.33, 3: 0.00, 4: 0.00},   # early
]


def _fetch_instrument(name: str, db_path: str, start: str, end: str) -> pd.DataFrame:
    df = get_history(name, db_path, start=start, end=end)
    if df is None or df.empty:
        raise RuntimeError(f"Could not fetch {name}")
    return df


def compute_exposure(breadth: pd.DataFrame, db_path: str, start: str, end: str,
                     exposure_map: dict | None = None) -> pd.Series:
    """Daily combined ladder+thrust exposure over breadth's index."""
    spy = _fetch_instrument("SPY", db_path, start, end)["close"]
    vix = _fetch_instrument("^VIX", db_path, start, end)["close"]
    vix3m = _fetch_instrument("^VIX3M", db_path, start, end)["close"]
    hyg = _fetch_instrument("HYG", db_path, start, end)["close"]
    ief = _fetch_instrument("IEF", db_path, start, end)["close"]

    idx = breadth.index
    pts = regime.ladder_points(
        regime.trend_signal(spy).reindex(idx, fill_value=False),
        regime.breadth_signal(breadth["pct_above_200"]),
        regime.vol_signal(vix, vix3m).reindex(idx, fill_value=False),
        regime.credit_signal(hyg, ief).reindex(idx, fill_value=False),
    )
    thrust = regime.thrust_override(breadth["pct_above_50"])
    if exposure_map is None:
        return regime.combined_exposure(pts, thrust)
    exp = pts.map(exposure_map).astype(float)
    floored = exp.clip(lower=regime.THRUST_FLOOR)
    return exp.where(~thrust.reindex(exp.index, fill_value=False), floored)


def run_backtest(panel_path: str = "output/factor_panel.parquet",
                 breadth_path: str = "output/breadth.parquet",
                 db_path: str = "data/cache.db",
                 report_path: str | None = None,
                 sensitivity: bool = False) -> dict:
    panel = pd.read_parquet(panel_path)
    panel["date"] = pd.to_datetime(panel["date"])
    breadth = pd.read_parquet(breadth_path)
    breadth.index = pd.to_datetime(breadth.index)

    start = str(breadth.index.min().date())
    end = str(breadth.index.max().date())
    # warm-up for SMA200/SMA100 in regime signals
    fetch_start = str((breadth.index.min() - pd.Timedelta(days=400)).date())

    closes = (panel.pivot_table(index="date", columns="ticker", values="close")
              .reindex(breadth.index).ffill())

    spy = _fetch_instrument("SPY", db_path, fetch_start, end)
    spy_close = spy["close"].reindex(breadth.index).ffill()
    spy_run = {"equity": 100_000.0 * spy_close / spy_close.iloc[0],
               "avg_exposure": 1.0, "costs": 0.0, "turnover_notional": 0.0,
               "daily": pd.DataFrame({"equity": spy_close, "invested": spy_close,
                                      "n_positions": 1, "exposure_target": 1.0})}

    exposure = compute_exposure(breadth, db_path, fetch_start, end)

    runs = {
        "composite": simulate(panel, closes),
        "composite+ladder": simulate(panel, closes, exposure=exposure),
        "naive_momentum": simulate(panel, closes, score_col="mom_12_1"),
        "SPY": spy_run,
    }
    v = verdict(runs["composite"], runs["composite+ladder"], runs["naive_momentum"])

    note = ""
    if sensitivity:
        note_lines = ["", "## Sensitivity (alternative exposure maps — robustness, not tuning)", "",
                      "| Map | CAGR | MaxDD | Sharpe |", "|---|---|---|---|"]
        for m in SENSITIVITY_MAPS:
            exp_m = compute_exposure(breadth, db_path, fetch_start, end, exposure_map=m)
            r = simulate(panel, closes, exposure=exp_m)
            dd, _ = max_drawdown(r["equity"])
            note_lines.append(f"| {m} | {cagr(r['equity']):.2%} | {dd:.2%} "
                              f"| {sharpe(r['equity']):.2f} |")
        note = "\n".join(note_lines)

    caveat = ("Note: composite/composite+ladder/naive_momentum use weekly-rebalance "
              "close prices (forward-filled to daily); intra-week moves and the SPY "
              "baseline (genuine daily data) are not directly comparable on "
              "Vol/Sharpe/MaxDD.")
    note = f"{caveat}\n\n{note}" if note else caveat

    if report_path is None:
        report_path = f"output/portfolio_backtest_{pd.Timestamp.today().date()}.md"
    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
    write_report(report_path, runs, v,
                 meta={"start": start, "end": end,
                       "cost_bps": TRANSACTION_COST_BPS, "note": note})
    for name, run in runs.items():
        run["equity"].to_csv(report_path.replace(".md", f"_{name.replace('+','_')}_equity.csv"))
    return {"runs": runs, "verdict": v, "report_path": report_path}


# --- index-overlay mode (SPY x ladder) -----------------------------------
# Pre-registered criteria: docs/CONCLUSION-2026-07-16-composite-stock-picker.md
# (fixed before the 2024+ holdout run). The composite stock-picker is retired;
# this evaluates the regime ladder as drawdown insurance on the index itself.

INDEX_OVERLAY_COST_BPS = 5.0       # per side on exposure-change notional (SPY spread ~1bp)
INDEX_DD_REDUCTION_REQUIRED = 1 / 3
INDEX_CAGR_GIVEUP_MAX = 0.04


def overlay_equity(close: pd.Series, exposure: pd.Series,
                   cost_bps: float = INDEX_OVERLAY_COST_BPS,
                   initial_capital: float = 100_000.0) -> pd.Series:
    """Single-instrument exposure overlay: daily returns scaled by the prior
    close's exposure target (signal at close t trades at that close, earns
    from t+1), minus cost_bps on each day's |exposure change|."""
    rets = close.pct_change().fillna(0.0)
    exp = exposure.reindex(close.index).ffill().fillna(1.0)
    eff = exp.shift(1).fillna(exp.iloc[0])
    tc = exp.diff().abs().fillna(0.0) * cost_bps / 1e4
    return initial_capital * (1.0 + rets * eff - tc).cumprod()


def verdict_index_overlay(index_eq: pd.Series, overlay_eq: pd.Series) -> dict:
    """Pre-registered pass criteria for the ladder-on-index overlay:
    MaxDD cut >= 1/3, CAGR give-up <= 4pts, Sharpe >= index Sharpe."""
    dd_i, _ = max_drawdown(index_eq)
    dd_o, _ = max_drawdown(overlay_eq)
    dd_cut = abs(dd_o) <= abs(dd_i) * (1 - INDEX_DD_REDUCTION_REQUIRED)
    giveup = cagr(index_eq) - cagr(overlay_eq)
    giveup_ok = giveup <= INDEX_CAGR_GIVEUP_MAX
    sharpe_ok = sharpe(overlay_eq) >= sharpe(index_eq)
    return {
        "dd_index": dd_i, "dd_overlay": dd_o, "dd_cut_third": dd_cut,
        "cagr_giveup": giveup, "cagr_giveup_ok": giveup_ok,
        "sharpe_ok": sharpe_ok,
        "overall": bool(dd_cut and giveup_ok and sharpe_ok),
    }


def run_index_overlay(breadth_path: str = "output/breadth.parquet",
                      db_path: str = "data/cache.db",
                      report_path: str | None = None) -> dict:
    """SPY buy-and-hold vs SPY x (regime ladder + asymmetric dwell)."""
    breadth = pd.read_parquet(breadth_path)
    breadth.index = pd.to_datetime(breadth.index)
    start = str(breadth.index.min().date())
    end = str(breadth.index.max().date())
    fetch_start = str((breadth.index.min() - pd.Timedelta(days=400)).date())

    spy = _fetch_instrument("SPY", db_path, fetch_start, end)["close"]
    spy = spy.reindex(breadth.index).ffill()

    raw_exp = compute_exposure(breadth, db_path, fetch_start, end)
    exp = regime.asymmetric_dwell(raw_exp)

    runs = {
        "SPY": overlay_equity(spy, pd.Series(1.0, index=spy.index), cost_bps=0.0),
        "SPY x ladder+dwell": overlay_equity(spy, exp),
    }
    v = verdict_index_overlay(runs["SPY"], runs["SPY x ladder+dwell"])

    lines = [
        "# Index Overlay Backtest Report (SPY x regime ladder)",
        "",
        f"Period: {start} → {end} · overlay costs {INDEX_OVERLAY_COST_BPS} bps/side "
        f"on exposure changes · cash earns 0% · dwell confirm {regime.DWELL_CONFIRM}d",
        "",
        "Criteria pre-registered in docs/CONCLUSION-2026-07-16-composite-stock-picker.md.",
        "Caveat: breadth + thrust signals derive from a survivorship-biased universe;",
        "trend/vol/credit signals are clean.",
        "",
        "| Strategy | CAGR | Vol | Sharpe | MaxDD | LongestDD (d) | AvgExp |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, eq in runs.items():
        dd, dd_days = max_drawdown(eq)
        vol = eq.pct_change().std() * np.sqrt(252)
        ae = 1.0 if name == "SPY" else float(exp.reindex(eq.index).ffill().mean())
        lines.append(f"| {name} | {cagr(eq):.2%} | {vol:.2%} | {sharpe(eq):.2f} "
                     f"| {dd:.2%} | {dd_days} | {ae:.2f} |")

    lines += ["", "## Per-year", ""]
    for name, eq in runs.items():
        table = per_year(eq)
        lines += [f"### {name}", "", "| Year | Return | MaxDD |", "|---|---|---|"]
        for year, row in table.iterrows():
            lines.append(f"| {year} | {row['return']:.2%} | {row['max_dd']:.2%} |")
        lines.append("")

    lines += [
        "## Verdict (pre-registered criteria)",
        "",
        f"- MaxDD cut ≥ 1/3: **{v['dd_cut_third']}** "
        f"(SPY {v['dd_index']:.2%} → overlay {v['dd_overlay']:.2%})",
        f"- CAGR give-up ≤ 4pts: **{v['cagr_giveup_ok']}** ({v['cagr_giveup']:.2%})",
        f"- Overlay Sharpe ≥ SPY Sharpe: **{v['sharpe_ok']}**",
        "",
        f"**OVERALL: {'PASS' if v['overall'] else 'FAIL'}**",
    ]

    if report_path is None:
        report_path = f"output/index_overlay_{pd.Timestamp.today().date()}.md"
    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"[report] wrote {report_path}")
    for name, eq in runs.items():
        safe = name.replace(" ", "_").replace("+", "_")
        eq.to_csv(report_path.replace(".md", f"_{safe}_equity.csv"))
    return {"runs": runs, "verdict": v, "report_path": report_path}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description="Run the portfolio backtest + regime ladder")
    ap.add_argument("--panel", default="output/factor_panel.parquet")
    ap.add_argument("--breadth", default="output/breadth.parquet")
    ap.add_argument("--db", default="data/cache.db")
    ap.add_argument("--report", default=None)
    ap.add_argument("--sensitivity", action="store_true")
    ap.add_argument("--index-overlay", action="store_true",
                    help="run SPY x ladder overlay instead of the composite harness")
    args = ap.parse_args()
    if args.index_overlay:
        out = run_index_overlay(breadth_path=args.breadth, db_path=args.db,
                                report_path=args.report)
    else:
        out = run_backtest(panel_path=args.panel, breadth_path=args.breadth,
                           db_path=args.db, report_path=args.report,
                           sensitivity=args.sensitivity)
    print(f"OVERALL: {'PASS' if out['verdict']['overall'] else 'FAIL'}"
          f" -> {out['report_path']}")
