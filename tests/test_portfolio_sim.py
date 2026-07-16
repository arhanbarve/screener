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
    # costs are fully deterministic in this synthetic fixture (fee = notional
    # * cost_rate on every trade, so total costs = turnover * cost_rate) --
    # pin the exact magnitude so a fee-doubling (or halving) bug can't slip
    # through a direction-only check.
    cost_rate = 20.0 / 1e4
    assert costly["costs"] == pytest.approx(costly["turnover_notional"] * cost_rate)


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


def test_simulate_exposure_change_on_rebalance_day_still_rescales_untouched_positions():
    """Exposure change lands exactly on a rebalance day where band membership
    doesn't turn over (T1/T2 stay top-ranked every week in _mini_market()).
    Step 2 only sizes newly-traded positions -- it never touches the
    already-held, untouched T1/T2 -- so step 3 is the only mechanism that can
    rescale them toward the new exposure target. Regression test for the bug
    where step 3 used to no-op entirely on rebalance days, permanently
    dropping the de-risk signal for untouched positions (invested/equity
    stayed pinned at the old exposure for the rest of the run)."""
    panel, closes, fridays = _mini_market()
    exposure = pd.Series(1.0, index=closes.index)
    exposure.loc[exposure.index >= fridays[1]] = 0.5   # de-risk exactly on rebal day
    res = simulate(panel, closes, max_positions=2, entry_band=2, exit_band=3,
                   cost_bps=0.0, exposure=exposure)
    # no band turnover happened on fridays[1] (T1/T2 remain rank 1/2 all
    # along), so step 2 traded nothing that day -- the trades below are all
    # step 3's pro-rata rescale, confirming this exercises the
    # untouched-position rescale path, not step 2's sizing.
    same_day_trades = res["trades"][res["trades"]["date"] == fridays[1]]
    assert set(same_day_trades["ticker"]) == {"T1", "T2"}
    assert (same_day_trades["side"] == "sell").all()
    # the de-risk should still take effect the same day, via step 3.
    snap = res["daily"].loc[fridays[1]]
    assert snap["invested"] / snap["equity"] == pytest.approx(0.5, abs=0.02)
    # and it must stick -- not get silently dropped and drift back toward
    # full exposure over the following days/rebalances.
    tail = res["daily"].loc[fridays[1]:]
    assert (tail["invested"] / tail["equity"] < 0.7).all()


def test_simulate_rebalance_and_exposure_change_same_day_no_double_trade():
    """Original bug (pre-7bb4054): a same-day rebalance (step 2) + exposure
    change (step 3) caused wasteful buy-then-immediate-sell churn on the same
    ticker. Force real band turnover (T2 exits, T3 enters) on the same day
    exposure changes, and confirm each step-2-traded ticker is traded at most
    once that day -- step 3 must skip tickers step 2 already sized, even
    though (per the newer fix) it still rescales the untouched one (T1)."""
    idx = pd.bdate_range("2024-01-01", periods=15)
    fridays = [pd.Timestamp("2024-01-05"), pd.Timestamp("2024-01-12"),
               pd.Timestamp("2024-01-19")]
    tickers = ["T1", "T2", "T3", "T4", "T5"]
    closes = pd.DataFrame({t: np.full(15, 100.0) for t in tickers}, index=idx)

    rows = []
    for d in fridays:
        for rank, t in enumerate(tickers, start=1):
            rows.append({"date": d, "ticker": t, "passes_gates": True,
                         "composite": float(len(tickers) - rank), "close": 100.0})
    panel = pd.DataFrame(rows)
    # at the second rebalance: demote T2 out of the exit band (sold) and
    # promote T3 into the entry band (bought); T1 stays top-ranked (untouched).
    panel.loc[(panel["date"] == fridays[1]) & (panel["ticker"] == "T2"), "composite"] = -5
    panel.loc[(panel["date"] == fridays[1]) & (panel["ticker"] == "T3"), "composite"] = 10

    exposure = pd.Series(1.0, index=closes.index)
    exposure.loc[exposure.index >= fridays[1]] = 0.5   # exposure change same day as turnover

    res = simulate(panel, closes, max_positions=2, entry_band=2, exit_band=3,
                   cost_bps=20.0, exposure=exposure)
    same_day = res["trades"][res["trades"]["date"] == fridays[1]]
    assert set(same_day["ticker"]) == {"T1", "T2", "T3"}
    # no ticker traded more than once (no buy-then-sell churn on step 2's picks)
    assert (same_day.groupby("ticker").size() == 1).all()
    trade_by_ticker = same_day.set_index("ticker")
    assert trade_by_ticker.loc["T2", "side"] == "sell"   # step 2: exited the band
    assert trade_by_ticker.loc["T3", "side"] == "buy"    # step 2: entered the band
    assert trade_by_ticker.loc["T1", "side"] == "sell"   # step 3: untouched, rescaled down


def test_simulate_full_exposure_zero_goes_all_cash():
    panel, closes, fridays = _mini_market()
    exposure = pd.Series(0.0, index=closes.index)
    res = simulate(panel, closes, max_positions=2, entry_band=2, exit_band=3,
                   cost_bps=0.0, exposure=exposure)
    assert (res["daily"]["invested"] < 1e-9).all()
    assert res["equity"].iloc[-1] == pytest.approx(100_000.0)


def _full_turnover_market(seed):
    """8 tickers, 2 weekly rebalances. Week 2 flips the ranking entirely
    (all 3 held positions fall outside exit_band=4; 3 new ones fill in),
    so every held position is traded by step 2 that day -- step 3's
    untouched-position set is empty. Same day, exposure jumps 0.9 -> 1.15,
    which cash-caps step 2's last buy (its notional lands a hair below the
    full slot from float rounding, tipping cash to ~-1e-12). Regression
    scenario for the ZeroDivisionError in step 3's cash-affordability cap:
    total_buy over the (empty) untouched set is exactly 0.0, and 0.0 was
    compared/divided against that tiny negative `affordable`."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-01", periods=15)
    fri0, fri1 = idx[4], idx[9]
    tickers = [f"T{i}" for i in range(8)]
    rets = rng.normal(0.0, 0.03, size=(15, 8))
    prices = 100.0 * np.cumprod(1.0 + rets, axis=0)
    closes = pd.DataFrame(prices, index=idx, columns=tickers)

    rows = []
    for rank, t in enumerate(["T0", "T1", "T2", "T3", "T4", "T5", "T6", "T7"], start=1):
        rows.append({"date": fri0, "ticker": t, "passes_gates": True,
                     "composite": float(8 - rank), "close": float(closes.loc[fri0, t])})
    for rank, t in enumerate(["T5", "T6", "T7", "T4", "T3", "T2", "T1", "T0"], start=1):
        rows.append({"date": fri1, "ticker": t, "passes_gates": True,
                     "composite": float(8 - rank), "close": float(closes.loc[fri1, t])})
    panel = pd.DataFrame(rows)

    exposure = pd.Series(0.9, index=idx)
    exposure.loc[exposure.index >= fri1] = 1.15
    return panel, closes, exposure, fri1


def test_simulate_full_turnover_cash_capped_no_zero_division():
    """Regression test: step 3's cash-affordability cap used to divide
    `affordable / total_buy` unconditionally whenever scale > 1.0. On a day
    where step 2 trades every held position (full turnover) and its last
    buy is cash-capped, float rounding can leave `cash` a hair below zero
    while `total_buy` (summed only over positions step 2 didn't touch) is
    exactly 0.0 -- `0.0 > affordable` was True, so `affordable / total_buy`
    raised ZeroDivisionError. Seed below is a known-good deterministic
    trigger for that exact condition."""
    panel, closes, exposure, fri1 = _full_turnover_market(seed=861908)
    res = simulate(panel, closes, entry_band=3, exit_band=4, max_positions=3,
                   cost_bps=51.34574847743024, exposure=exposure)  # must not raise
    cash = res["daily"]["equity"] - res["daily"]["invested"]
    assert (cash > -1e-6).all()
    # step 3 shouldn't fire spurious trades on top of step 2's full turnover
    # (its untouched-position set is empty, so it has nothing left to do)
    same_day = res["trades"][res["trades"]["date"] == fri1]
    assert len(same_day) == 6  # 3 step-2 sells + 3 step-2 buys, nothing else


from src.portfolio_sim import verdict, write_report


def _fake_run(final=150_000.0, dd_curve=None, avg_exposure=1.0, n=756,
              start="2019-01-01"):
    idx = pd.bdate_range(start, periods=n)
    if dd_curve is None:
        equity = pd.Series(np.linspace(100_000, final, n), index=idx)
    else:
        equity = pd.Series(dd_curve, index=idx[:len(dd_curve)])
    return {"equity": equity, "avg_exposure": avg_exposure, "costs": 0.0,
            "turnover_notional": 0.0,
            "daily": pd.DataFrame({"equity": equity, "invested": equity,
                                   "n_positions": 10, "exposure_target": 1.0})}


def _curve_with_dd(n, dd_frac, seed=0):
    """Linear up, one crash of dd_frac in the middle, recovery."""
    third = n // 3
    up1 = np.linspace(100_000, 130_000, third)
    crash = np.linspace(130_000, 130_000 * (1 - dd_frac), third)
    up2 = np.linspace(130_000 * (1 - dd_frac), 160_000, n - 2 * third)
    return np.concatenate([up1, crash, up2])


def _curve_with_window_crash(idx, window_start, window_end, dd_frac,
                             start_val=100_000.0, end_val=160_000.0):
    """Equity curve over `idx`: linear rise to a local peak exactly at
    `window_start`, a monotonic dd_frac crash down to `window_end`, then a
    linear recovery to end_val. The crash is located via idx.searchsorted so
    it lands precisely inside the given calendar window regardless of
    business-day alignment -- unlike `_curve_with_dd`'s fixed 1/3-of-n
    crash position, which (for the default start="2019-01-01") does NOT
    actually land inside the literal 2020 or 2022 calendar windows, so it
    can't genuinely exercise the window drawdown comparison. Since cagr()
    and max_drawdown() only look at endpoint/extremum values (not the
    interior path), this also gives exact control over both the window's
    max_drawdown (== dd_frac) and, via end_val, the overall CAGR."""
    n = len(idx)
    i0 = idx.searchsorted(pd.Timestamp(window_start))
    i1 = idx.searchsorted(pd.Timestamp(window_end), side="right") - 1
    assert 0 < i0 < i1 < n - 1, "window crash must sit strictly inside idx"
    peak_val = start_val + (end_val - start_val) * 0.5
    trough_val = peak_val * (1 - dd_frac)
    up1 = np.linspace(start_val, peak_val, i0)
    crash = np.linspace(peak_val, trough_val, i1 - i0 + 1)
    up2 = np.linspace(trough_val, end_val, n - i1 - 1)
    return np.concatenate([up1, crash, up2])


def test_verdict_pass_when_all_criteria_met():
    # ~7.7 years covering 2019-2026 (idx ends 2026-09-22), includes 2020 and
    # 2022. cagr()/max_drawdown() depend only on endpoint/extremum values, so
    # end_val below was solved analytically (see scratch calc in the PR) to
    # produce a known, nonzero cagr_giveup of exactly 0.01 -- well within
    # CAGR_GIVEUP_MAX (0.02) but genuinely nonzero, so this test can't pass
    # via the degenerate giveup == 0.0 case a same-final-value fixture would
    # produce.
    idx = pd.bdate_range("2019-01-01", periods=2016)
    n = len(idx)
    u_curve = _curve_with_window_crash(idx, "2020-01-01", "2020-12-31", 0.30,
                                       end_val=200_000.0)
    l_curve = _curve_with_window_crash(idx, "2020-01-01", "2020-12-31", 0.15,
                                       end_val=186_305.4309686459)
    naive_curve = _curve_with_window_crash(idx, "2020-01-01", "2020-12-31", 0.35,
                                           end_val=150_000.0)
    unhedged = _fake_run(dd_curve=u_curve, n=n)
    laddered = _fake_run(dd_curve=l_curve, avg_exposure=0.8, n=n)
    naive = _fake_run(dd_curve=naive_curve, n=n)
    v = verdict(unhedged, laddered, naive)

    # The 2020 window is genuinely covered and its dd comparison is real
    # (0.15 <= 0.30 * 2/3 == 0.20), not the degenerate 0.0 <= 0.0 case the
    # old fixture produced.
    assert v["dd_2020_covered"] is True
    assert v["dd_2022_covered"] is True
    assert v["dd_2020_ok"] is True
    assert v["dd_reduced_third"] is True

    # Giveup is a real, known nonzero number and passes because it's under
    # the 2pt threshold -- not because it's exactly zero.
    assert v["cagr_giveup"] == pytest.approx(0.01, abs=1e-6)
    assert v["cagr_giveup_ok"] is True

    assert isinstance(v["sharpe_vs_naive_ok"], bool)
    assert v["overall"] == (v["dd_reduced_third"] and v["dd_2020_ok"]
                            and v["dd_2022_ok"] and v["cagr_giveup_ok"]
                            and v["sharpe_vs_naive_ok"])


def test_verdict_cagr_giveup_fails_when_exceeds_max():
    """Same endpoint-only-dependence trick as above, but solved for a giveup
    of exactly 0.05 -- comfortably over CAGR_GIVEUP_MAX (0.02) -- to confirm
    cagr_giveup_ok actually flips to False rather than always reading True."""
    idx = pd.bdate_range("2019-01-01", periods=2016)
    n = len(idx)
    unhedged = _fake_run(dd_curve=np.linspace(100_000, 200_000.0, n), n=n)
    laddered = _fake_run(dd_curve=np.linspace(100_000, 139_347.17832624083, n), n=n)
    naive = _fake_run(dd_curve=np.linspace(100_000, 150_000.0, n), n=n)
    v = verdict(unhedged, laddered, naive)
    assert v["cagr_giveup"] == pytest.approx(0.05, abs=1e-6)
    assert v["cagr_giveup_ok"] is False
    assert v["overall"] is False


def test_verdict_vacuous_window_not_covered():
    """A short-horizon run that never reaches 2020 or 2022 at all: both
    window checks fall back to the NaN/vacuous branch. Confirms dd_2020_ok
    (and dd_2022_ok) still read True -- vacuous pass, unchanged pre-existing
    behavior toward `overall` -- but the new coverage flags now correctly
    report that no real comparison happened, closing the gap where a
    holdout run outside both crash years would otherwise render an
    indistinguishable-from-genuine PASS."""
    unhedged = _fake_run(final=110_000.0, n=100, start="2023-01-01")
    laddered = _fake_run(final=108_000.0, n=100, start="2023-01-01")
    naive = _fake_run(final=105_000.0, n=100, start="2023-01-01")
    v = verdict(unhedged, laddered, naive)
    assert v["dd_2020_ok"] is True
    assert v["dd_2020_covered"] is False
    assert v["dd_2022_ok"] is True
    assert v["dd_2022_covered"] is False


def test_write_report_renders(tmp_path):
    # composite+ladder uses a curve with a genuine (nonzero) drawdown, not a
    # monotonically-increasing one -- a monotonic curve's MaxDD is exactly
    # 0.0, and 0.0 formats identically whether or not the sign is flipped
    # (`{0.0:.2%}` == `{-0.0:.2%}` == "0.00%"), which would make the value
    # assertion below pass even against a sabotaged sign.
    runs = {"composite": _fake_run(),
            "composite+ladder": _fake_run(dd_curve=_curve_with_dd(756, 0.18)),
            "naive_momentum": _fake_run(final=130_000.0), "SPY": _fake_run(final=120_000.0)}
    v = verdict(runs["composite"], runs["composite+ladder"], runs["naive_momentum"])
    path = tmp_path / "report.md"
    write_report(str(path), runs, v, meta={"start": "2019-01-01", "end": "2026-01-01",
                                           "cost_bps": 20.0, "note": "test"})
    text = path.read_text()
    assert "SURVIVORSHIP" in text.upper()
    assert "composite+ladder" in text
    assert "CAGR" in text
    assert "Verdict" in text

    # Value-level check, not just label presence: compute the actual CAGR and
    # MaxDD for the "composite+ladder" run directly and confirm the exact
    # formatted string appears in its summary-table row. This is deliberately
    # non-tautological -- sabotaging write_report's formatting (e.g. flipping
    # the MaxDD sign to `{-dd:.2%}`, or swapping which run's CAGR is printed)
    # was manually confirmed to break this assertion while the old
    # substring-only checks above kept passing.
    ladder_run = runs["composite+ladder"]
    expected_cagr = cagr(ladder_run["equity"])
    expected_dd, _ = max_drawdown(ladder_run["equity"])
    lines = text.splitlines()
    row = next(l for l in lines if l.startswith("| composite+ladder |"))
    assert f"{expected_cagr:.2%}" in row
    assert f"{expected_dd:.2%}" in row


def test_run_backtest_end_to_end_synthetic(tmp_path, monkeypatch):
    """Wiring smoke test: panel/breadth/instruments synthetic, full CLI path."""
    import src.portfolio_sim as ps

    n = 300
    idx = pd.bdate_range("2023-01-02", periods=n)
    rng = np.random.default_rng(11)
    tickers = [f"T{i}" for i in range(30)]
    closes = pd.DataFrame(
        {t: 100 * np.cumprod(1 + rng.normal(0.0004, 0.015, n)) for t in tickers},
        index=idx)

    fridays = [d for d in idx if d.dayofweek == 4]
    rows = []
    for d in fridays:
        for i, t in enumerate(tickers):
            rows.append({"date": d, "ticker": t, "passes_gates": True,
                         "composite": float(rng.normal()), "mom_12_1": float(rng.normal()),
                         "close": float(closes.loc[d, t])})
    panel = pd.DataFrame(rows)
    breadth = pd.DataFrame({"pct_above_200": np.full(n, 0.6),
                            "pct_above_50": np.full(n, 0.6)}, index=idx)

    def synth_instr(base, vol):
        r = np.random.default_rng(hash(base) % 2**31)
        return pd.DataFrame({"close": base * np.cumprod(1 + r.normal(0.0003, vol, n)),
                             "volume": np.full(n, 1e6)}, index=idx)

    instruments = {"SPY": synth_instr(400, 0.01), "^VIX": synth_instr(18, 0.05),
                   "^VIX3M": synth_instr(20, 0.04), "HYG": synth_instr(75, 0.005),
                   "IEF": synth_instr(95, 0.004)}
    monkeypatch.setattr(ps, "_fetch_instrument",
                        lambda name, db_path, start, end: instruments[name])

    panel_path = tmp_path / "panel.parquet"
    breadth_path = tmp_path / "breadth.parquet"
    panel.to_parquet(panel_path, index=False)
    breadth.to_parquet(breadth_path)
    report_path = tmp_path / "report.md"

    ps.run_backtest(panel_path=str(panel_path), breadth_path=str(breadth_path),
                    db_path="unused.db", report_path=str(report_path),
                    sensitivity=True)

    text = report_path.read_text()
    assert "composite+ladder" in text
    assert "naive_momentum" in text
    assert "Sensitivity" in text
    assert "OVERALL" in text
