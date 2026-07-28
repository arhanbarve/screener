import numpy as np
import pandas as pd
import pytest

from src.exit_plan import init_plan, evaluate_day
from src.exit_plan import weekly_health, apply_weekly_tightener


def trend_df(n, start_price, end_price, **kw):
    return make_df(list(np.linspace(start_price, end_price, n)), **kw)


def make_df(closes, start="2026-01-02", spread=0.01, volume=1_000_000):
    """OHLCV frame from a list of closes. High/low = close ±spread fraction."""
    closes = pd.Series(closes, dtype=float)
    idx = pd.bdate_range(start=start, periods=len(closes))
    return pd.DataFrame({
        "open":   closes.values,
        "high":   closes.values * (1 + spread),
        "low":    closes.values * (1 - spread),
        "close":  closes.values,
        "volume": [volume] * len(closes),
    }, index=idx)


def flat_df(n=80, price=100.0, **kw):
    return make_df([price] * n, **kw)


def entry_pos(df, entry_price=None):
    """Build a position dict with a plan initialized at the last bar of df."""
    entry_date = df.index[-1].strftime("%Y-%m-%d")
    price = float(df["close"].iloc[-1]) if entry_price is None else entry_price
    pos = {"ticker": "TEST", "entry_date": entry_date, "entry_price": price}
    pos["plan"] = init_plan(df, price)
    return pos


class TestInitPlan:
    def test_initial_stop_is_entry_minus_2_atr(self):
        df = flat_df(80)
        plan = init_plan(df, 100.0)
        # flat closes, high/low = ±1% → TR = 2, Wilder ATR ≈ 2 → stop ≈ 96
        assert plan["initial_stop"] == pytest.approx(96.0, abs=0.5)
        assert plan["risk_R"] == pytest.approx(100.0 - plan["initial_stop"])
        assert plan["stop_floor"] == plan["initial_stop"]
        assert plan["trail_mult"] == 3.0
        assert plan["trims_fired"] == []
        assert plan["verdict"] == "HOLD"

    def test_init_plan_short_history_uses_pct_fallback(self):
        df = flat_df(5)  # too short for ATR14
        plan = init_plan(df, 100.0)
        assert plan["initial_stop"] == pytest.approx(92.0)  # 8% fallback


class TestHold:
    def test_flat_market_stays_hold(self):
        df = flat_df(80)
        pos = entry_pos(df)
        newdf = make_df([100.0] * 81)
        pos, events = evaluate_day(pos, newdf)
        assert pos["plan"]["verdict"] == "HOLD"
        assert events == []


class TestSellReasons:
    def test_floor_breach_sells_with_max_loss_floor_reason(self):
        df = flat_df(80)
        pos = entry_pos(df)  # entry_price = 100
        # de-risk trim fires and moves stop_floor to breakeven (100)
        day81 = make_df([100.0] * 80 + [115.0])
        pos, _ = evaluate_day(pos, day81)
        assert pos["plan"]["stop_floor"] == pytest.approx(100.0)
        # close back below entry breaches the floor (and the higher trailing
        # stop_level too) — the floor reason must win the reason string
        day82 = make_df([100.0] * 80 + [115.0, 99.0])
        pos, events = evaluate_day(pos, day82)
        assert pos["plan"]["verdict"] == "SELL"
        assert "max-loss floor" in events[0]["reason"]


class TestReplayIdempotent:
    def test_replay_same_bar_twice_does_not_double_count_streak(self):
        df = flat_df(80)
        pos = entry_pos(df)
        newdf = make_df([100.0] * 79 + [50.0])  # single close well below 50-day SMA
        pos, _ = evaluate_day(pos, newdf, replay=True)
        streak_after_first = pos["plan"]["below_50d_streak"]
        pos, _ = evaluate_day(pos, newdf, replay=True)
        assert pos["plan"]["below_50d_streak"] == streak_after_first


class TestSell:
    def test_gap_down_through_stop_sells_and_is_terminal(self):
        df = flat_df(80)
        pos = entry_pos(df)
        crash = make_df([100.0] * 80 + [80.0])
        pos, events = evaluate_day(pos, crash)
        assert pos["plan"]["verdict"] == "SELL"
        assert len(events) == 1 and events[0]["type"] == "SELL"
        # recovery next day does NOT un-sell in live mode
        recover = make_df([100.0] * 80 + [80.0, 105.0])
        pos, events = evaluate_day(pos, recover)
        assert pos["plan"]["verdict"] == "SELL"
        assert events == []

    def test_replay_mode_clears_sell_on_recovery(self):
        df = flat_df(80)
        pos = entry_pos(df)
        crash = make_df([100.0] * 80 + [80.0])
        pos, _ = evaluate_day(pos, crash, replay=True)
        assert pos["plan"]["verdict"] == "SELL"
        recover = make_df([100.0] * 80 + [80.0, 105.0])
        pos, _ = evaluate_day(pos, recover, replay=True)
        assert pos["plan"]["verdict"] == "HOLD"

    # High-vol fixture (±5% bars → ATR≈10): the trailing stop sits far below
    # the 50d SMA, so the trend-break rule is the binding SELL trigger.
    # Entry at end of a 60-bar plateau @120: initial stop 100, s50 = 120.
    _HV = [100.0] * 80 + list(np.linspace(100, 120, 20)) + [120.0] * 60

    def test_50d_break_needs_3_consecutive_closes(self):
        pos = entry_pos(make_df(self._HV, spread=0.05))
        for i in range(1, 4):
            df = make_df(self._HV + [112.0] * i, spread=0.05)
            pos, events = evaluate_day(pos, df)
        assert pos["plan"]["verdict"] == "SELL"
        assert "50-day" in events[0]["reason"]

    def test_50d_streak_resets_on_close_above(self):
        pos = entry_pos(make_df(self._HV, spread=0.05))
        seq = [112.0, 112.0, 121.0, 112.0, 112.0]  # 121 > s50: never 3 in a row
        for i in range(1, len(seq) + 1):
            df = make_df(self._HV + seq[:i], spread=0.05)
            pos, events = evaluate_day(pos, df)
        assert pos["plan"]["verdict"] == "HOLD"


class TestTrim:
    # Ramp to +21% over 25 sessions rather than the plan's original
    # single-day 100 -> 121 jump. Two separate things go wrong with a
    # single-day jump (or a perfectly monotonic multi-day replacement):
    #   1. It trips the blowoff burst rule too (>3xATR within 5 sessions —
    #      3xATR is far smaller than a 20%+ move for this fixture's vol),
    #      firing a second TRIM alongside derisk.
    #   2. A monotonic ramp has zero down weeks in its weekly resample, so
    #      rsi_14 hits its own zero-avg-loss branch and reports a degenerate
    #      RSI of 100 (formula edge case, not a real overbought read) —
    #      firing blowoff again, this time via the "overheated" path.
    # This ramp rises to 114 by day 10, pulls back to 106 by day 15 (a
    # genuine down week — real weekly RSI ~78, under the 80 threshold), then
    # climbs to 121 by day 24 and flattens near the end so the last 5-session
    # window stays under 3xATR. Net effect: only the derisk rung fires.
    _DERISK_RAMP = list(np.interp(
        np.arange(25),
        [0, 5, 10, 15, 20, 24],
        [101.0, 108.0, 114.0, 106.0, 120.0, 121.0],
    ))

    def test_derisk_fires_once_at_20pct_and_moves_stop_to_breakeven(self):
        df = flat_df(80)
        pos = entry_pos(df)
        ramp = self._DERISK_RAMP
        up = make_df([100.0] * 80 + ramp)
        pos, events = evaluate_day(pos, up)
        assert pos["plan"]["verdict"] == "TRIM"
        assert [e["type"] for e in events] == ["TRIM"]
        assert "derisk" in pos["plan"]["trims_fired"]
        assert pos["plan"]["stop_floor"] == pytest.approx(100.0)
        # next day, still up: no second derisk event
        up2 = make_df([100.0] * 80 + ramp + [122.0])
        pos, events = evaluate_day(pos, up2)
        assert events == []
        assert pos["plan"]["verdict"] == "HOLD"

    def test_derisk_fires_at_2R_before_20pct(self):
        df = flat_df(80)          # ATR≈2 → R≈4 → 2R = +8 = +8% (< 20%)
        pos = entry_pos(df)
        up = make_df([100.0] * 80 + [109.0])
        pos, events = evaluate_day(pos, up)
        assert "derisk" in pos["plan"]["trims_fired"]

    def test_blowoff_burst_fires_once(self):
        df = flat_df(80)
        pos = entry_pos(df)
        # +15 move in 3 days on ATR≈2 → burst (>3×ATR within 5 sessions)
        up = make_df([100.0] * 80 + [105.0, 110.0, 115.0])
        pos, events = evaluate_day(pos, up)
        assert "blowoff" in pos["plan"]["trims_fired"]
        types = [e["type"] for e in events]
        assert types.count("TRIM") >= 1


class TestInvariants:
    def test_whipsaw_market_never_flips_verdict(self):
        """Alternating ±3% days around entry: verdict must stay HOLD throughout.

        Absolute alternation (97 / 103, not compounding) so closes straddle the
        50d SMA — the below-50d streak resets every other day, and neither the
        96 stop nor the +8 (2R) derisk rung is ever touched."""
        df = flat_df(80)
        pos = entry_pos(df)
        closes = [100.0] * 80
        for i in range(20):
            closes.append(97.0 if i % 2 == 0 else 103.0)
            pos, events = evaluate_day(pos, make_df(closes))
            assert pos["plan"]["verdict"] == "HOLD", f"flipped on day {i}"
            assert events == []

    def test_stop_level_never_decreases(self):
        df = flat_df(80)
        pos = entry_pos(df)
        closes = [100.0] * 80
        rng = np.random.default_rng(7)
        prev_stop = pos["plan"]["stop_level"]
        price = 100.0
        for _ in range(60):
            price = max(50.0, price * float(rng.normal(1.0, 0.02)))
            closes.append(round(price, 2))
            # replay=True so a SELL triggered by this random walk can clear
            # and re-enter HOLD; stop_level must still never decrease across
            # that transition (the ratchet math itself is replay-agnostic).
            pos, _ = evaluate_day(pos, make_df(closes), replay=True)
            assert pos["plan"]["stop_level"] >= prev_stop - 1e-9
            prev_stop = pos["plan"]["stop_level"]

    def test_same_day_reeval_is_noop(self):
        df = flat_df(80)
        pos = entry_pos(df)
        up = make_df([100.0] * 80 + [121.0])
        pos, events1 = evaluate_day(pos, up)
        pos, events2 = evaluate_day(pos, up)   # re-fire same date
        assert events1 and events2 == []


class TestWeeklyTightener:
    def test_healthy_uptrend_scores_low_and_keeps_mult(self):
        df = trend_df(300, 50, 150)          # strong uptrend
        spy = trend_df(300, 100, 110)["close"]  # weaker than stock
        health = weekly_health(df, spy)
        assert health["bearish"] < 2
        plan = init_plan(df, 150.0)
        plan2 = apply_weekly_tightener(plan, health)
        assert plan2["trail_mult"] == 3.0

    def test_deteriorating_tightens_one_step(self):
        # up then rolling over: bearish weekly MACD + lagging SPY
        closes = list(np.linspace(50, 150, 200)) + list(np.linspace(150, 110, 100))
        df = make_df(closes)
        spy = trend_df(300, 100, 130)["close"]
        health = weekly_health(df, spy)
        assert health["bearish"] >= 2
        plan = init_plan(df, 110.0)
        plan = apply_weekly_tightener(plan, health)
        assert plan["trail_mult"] == 2.5
        plan = apply_weekly_tightener(plan, health)
        assert plan["trail_mult"] == 2.0
        plan = apply_weekly_tightener(plan, health)
        assert plan["trail_mult"] == 2.0   # floor, never below

    def test_recovery_never_loosens(self):
        df = trend_df(300, 50, 150)
        spy = trend_df(300, 100, 110)["close"]
        plan = init_plan(df, 150.0)
        plan["trail_mult"] = 2.0
        plan = apply_weekly_tightener(plan, weekly_health(df, spy))
        assert plan["trail_mult"] == 2.0   # healthy again, but no loosening
