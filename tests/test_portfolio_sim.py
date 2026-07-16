"""Portfolio simulator tests — synthetic panels, no network."""
import numpy as np
import pandas as pd
import pytest

from src.portfolio_sim import cagr, max_drawdown, sharpe, per_year, simulate


def test_cagr_known_curve():
    idx = pd.bdate_range("2020-01-01", periods=504)  # ~2 years
    equity = pd.Series(np.linspace(100.0, 144.0, 504), index=idx)
    years = (idx[-1] - idx[0]).days / 365.25
    assert cagr(equity) == pytest.approx((144 / 100) ** (1 / years) - 1, rel=1e-9)


def test_max_drawdown_known_curve():
    idx = pd.bdate_range("2020-01-01", periods=5)
    equity = pd.Series([100, 120, 90, 110, 130], index=idx, dtype=float)
    dd, dd_days = max_drawdown(equity)
    assert dd == pytest.approx(-0.25)  # 120 -> 90
    # convention: dd_days = consecutive trading days spent below the running
    # peak. 90 and 110 are below the 120 peak; 130 recovers -> 2 days.
    assert dd_days == 2


def test_sharpe_zero_vol_is_nan():
    idx = pd.bdate_range("2020-01-01", periods=10)
    flat = pd.Series(100.0, index=idx)
    assert np.isnan(sharpe(flat))


def test_per_year_table():
    idx = pd.bdate_range("2020-06-01", "2021-06-01")
    equity = pd.Series(np.linspace(100, 120, len(idx)), index=idx)
    table = per_year(equity)
    assert set(table.index) == {2020, 2021}
    assert (table["return"] > 0).all()


def _mini_market():
    """5 tickers, 15 business days, 3 weekly rebalances (Fridays).
    Prices constant except winners drift; panel makes ranks deterministic."""
    idx = pd.bdate_range("2024-01-01", periods=15)  # Mon 1/1 .. Fri 1/19
    fridays = [pd.Timestamp("2024-01-05"), pd.Timestamp("2024-01-12"),
               pd.Timestamp("2024-01-19")]
    tickers = ["T1", "T2", "T3", "T4", "T5"]
    closes = pd.DataFrame(
        {t: 100.0 * (1.01 ** np.arange(15)) if t in ("T1", "T2")
         else np.full(15, 100.0) for t in tickers},
        index=idx,
    )

    rows = []
    for d in fridays:
        for rank, t in enumerate(tickers, start=1):
            rows.append({
                "date": d, "ticker": t, "passes_gates": True,
                "composite": float(len(tickers) - rank),  # T1 best ... T5 worst
                "close": float(closes.loc[d, t]),
            })
    return pd.DataFrame(rows), closes, fridays


def test_simulate_enters_top_ranked_equal_weight():
    panel, closes, fridays = _mini_market()
    res = simulate(panel, closes, max_positions=2, entry_band=2, exit_band=3,
                   cost_bps=0.0)
    first_trades = res["trades"][res["trades"]["date"] == fridays[0]]
    assert set(first_trades["ticker"]) == {"T1", "T2"}
    assert (first_trades["side"] == "buy").all()
    # equal weight: each buy ~50% of equity
    assert first_trades["notional"].iloc[0] == pytest.approx(50_000, rel=1e-6)


def test_simulate_band_exit_only_below_exit_band():
    panel, closes, fridays = _mini_market()
    # At second rebalance, demote T2 to rank 3 (within exit band of 3 -> hold),
    # then at third rebalance to rank 4 (beyond band -> sell).
    p = panel.copy()
    p.loc[(p["date"] == fridays[1]) & (p["ticker"] == "T2"), "composite"] = 1.5  # rank 3
    p.loc[(p["date"] == fridays[2]) & (p["ticker"] == "T2"), "composite"] = 0.5  # rank 4
    res = simulate(p, closes, max_positions=2, entry_band=2, exit_band=3, cost_bps=0.0)
    t2_sells = res["trades"][(res["trades"]["ticker"] == "T2") &
                             (res["trades"]["side"] == "sell")]
    assert list(t2_sells["date"]) == [fridays[2]]


def test_simulate_costs_reduce_equity():
    panel, closes, _ = _mini_market()
    free = simulate(panel, closes, max_positions=2, entry_band=2, exit_band=3,
                    cost_bps=0.0)
    costly = simulate(panel, closes, max_positions=2, entry_band=2, exit_band=3,
                      cost_bps=20.0)
    assert costly["equity"].iloc[-1] < free["equity"].iloc[-1]
    assert costly["costs"] > 0


def test_simulate_exposure_scales_invested_fraction():
    panel, closes, fridays = _mini_market()
    exposure = pd.Series(1.0, index=closes.index)
    exposure.loc[exposure.index >= fridays[1]] = 0.5   # de-risk halfway through
    res = simulate(panel, closes, max_positions=2, entry_band=2, exit_band=3,
                   cost_bps=0.0, exposure=exposure)
    # day after de-risking: invested value ~= 50% of equity
    day_after = closes.index[closes.index.get_loc(fridays[1]) + 1]
    snap = res["daily"].loc[day_after]
    assert snap["invested"] / snap["equity"] == pytest.approx(0.5, abs=0.02)
    # and average exposure < 1
    assert res["avg_exposure"] < 0.85


def test_simulate_full_exposure_zero_goes_all_cash():
    panel, closes, fridays = _mini_market()
    exposure = pd.Series(0.0, index=closes.index)
    res = simulate(panel, closes, max_positions=2, entry_band=2, exit_band=3,
                   cost_bps=0.0, exposure=exposure)
    assert (res["daily"]["invested"] < 1e-9).all()
    assert res["equity"].iloc[-1] == pytest.approx(100_000.0)
