from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from src.exit_plan import init_plan, evaluate_day
from src.exit_plan import weekly_health, apply_weekly_tightener
from src.exit_plan import bootstrap_position, run_daily_eval
from src.exit_plan import health_label


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

    def test_short_history_is_distinguishable_from_healthy(self):
        # ~8 weeks of history: too little for any check to run. bearish must
        # read 0 (same as a fully-evaluated healthy result), but "weeks"
        # must reveal it was never actually checked so callers (and the UI
        # badge) don't mistake "unknown" for "4/4 healthy".
        df = trend_df(40, 50, 60)
        spy = trend_df(40, 100, 105)["close"]
        health = weekly_health(df, spy)
        assert health["bearish"] == 0
        assert health["weeks"] < 15

    def test_check_exception_is_recorded_not_silently_healthy(self, monkeypatch):
        # Rollover fixture where macd/rs/obv/adx would all normally fire.
        # Force obv_slope to throw and confirm the failure is recorded in
        # "errors" (not silently treated as "not bearish"), while the other
        # three checks still run independently.
        closes = list(np.linspace(50, 150, 200)) + list(np.linspace(150, 110, 100))
        df = make_df(closes)
        spy = trend_df(300, 100, 130)["close"]

        def boom(*args, **kwargs):
            raise ValueError("bad volume data")

        monkeypatch.setattr("src.exit_plan.obv_slope", boom)
        health = weekly_health(df, spy)
        assert "obv" in health["errors"]
        assert "obv" not in health["parts"]
        assert health["bearish"] == 3   # macd, rs, adx still fired
        assert set(health["parts"]) == {"macd", "rs", "adx"}


class TestHealthLabel:
    """health_label() is the single source of truth shared by the digest
    email (src/exit_alerts.py) and the positions page (app_shared.py) — both
    callers must show the same score text."""

    def test_none_health_is_unknown(self):
        assert health_label(None) == "?/4"

    def test_short_history_is_unknown(self):
        health = {"bearish": 0, "parts": [], "errors": [], "weeks": 8, "asof": "2026-01-02"}
        assert health_label(health) == "?/4"

    def test_errors_keep_the_real_bearish_count_with_asterisk(self):
        # A check that threw must not hide the count the other checks did
        # produce — this is the exact bug that made the page show a bare
        # "?/4*" while the email showed the real number.
        health = {"bearish": 3, "parts": ["macd", "rs", "adx"], "errors": ["obv"],
                   "weeks": 20, "asof": "2026-01-02"}
        assert health_label(health) == "3/4*"

    def test_clean_count_has_no_asterisk(self):
        health = {"bearish": 1, "parts": ["macd"], "errors": [], "weeks": 20, "asof": "2026-01-02"}
        assert health_label(health) == "1/4"


class TestBootstrap:
    def test_backfill_marks_earned_trims_and_ratchets(self):
        # entry at bar 60 @100, runs to 130 (derisk earned), settles at 120
        closes = [100.0] * 60 + list(np.linspace(100, 130, 30)) + list(np.linspace(130, 120, 10))
        df = make_df(closes)
        entry_date = df.index[59].strftime("%Y-%m-%d")
        pos = {"ticker": "TEST", "entry_date": entry_date, "entry_price": 100.0}
        pos = bootstrap_position(pos, df)
        plan = pos["plan"]
        assert "derisk" in plan["trims_fired"]
        assert plan["stop_floor"] == pytest.approx(100.0)   # breakeven after derisk
        assert plan["peak_close"] == pytest.approx(130.0, abs=0.5)
        assert plan["verdict"] in ("HOLD", "TRIM", "SELL")

    def test_backfill_survivor_of_old_dip_is_hold(self):
        # dip below initial stop mid-history, then recovery to new highs
        closes = ([100.0] * 60 + list(np.linspace(100, 90, 10))
                  + list(np.linspace(90, 140, 40)))
        df = make_df(closes)
        entry_date = df.index[59].strftime("%Y-%m-%d")
        pos = {"ticker": "TEST", "entry_date": entry_date, "entry_price": 100.0}
        pos = bootstrap_position(pos, df)
        assert pos["plan"]["verdict"] != "SELL"   # replay clears historical breach

    def test_backfill_ends_sell_when_trend_break_survives_a_cleared_price_sell(self):
        # Day 1 after entry: crash through the max-loss floor -> fresh SELL.
        # Day 2: partial recovery, still under the (unmoved) stop -> stays SELL.
        # Day 3: recovers just above the stop_level (clears the price-based
        # SELL under replay) while ALSO closing below the 50-day SMA for the
        # 3rd straight day. The clear must not skip the trend-break check for
        # that same bar -- this is the SPY case from the dry-run: a price
        # recovery cleared the SELL while the trend-break condition, which is
        # genuinely true on that final bar, went unexamined.
        closes = [100.0] * 80 + [80.0, 95.0, 99.0]
        df = make_df(closes)
        entry_date = df.index[79].strftime("%Y-%m-%d")
        pos = {"ticker": "TEST", "entry_date": entry_date, "entry_price": 100.0}
        pos = bootstrap_position(pos, df)
        plan = pos["plan"]
        assert plan["below_50d_streak"] >= 3
        assert plan["last_close"] > plan["stop_level"]   # price-based SELL did clear
        assert plan["verdict"] == "SELL"   # but the trend-break condition still holds


class TestBootstrapCatchup:
    """DEFECT 1: a bootstrap that ends SELL (or a fresh TRIM on the final
    replayed bar) must actually reach the user via run_daily_eval, not just
    sit silently in the plan. These drive run_daily_eval end-to-end with
    monkeypatched src.positions calls and a fake src.exit_alerts module,
    following the pattern in TestPendingNotify."""

    def test_bootstrap_ending_sell_emits_one_catchup_event_and_reaches_action_email(self, monkeypatch):
        import sys
        import types

        # Entry at 100 over 80 flat bars, then a single bar that crashes
        # through the max-loss floor -> bootstrap's only replay day fires a
        # fresh SELL and it never clears, so the plan ends SELL.
        crash_df = make_df([100.0] * 80 + [70.0])
        entry_date = crash_df.index[79].strftime("%Y-%m-%d")
        positions = [{"ticker": "AAA", "entry_date": entry_date, "entry_price": 100.0}]  # no "plan" -> bootstrap runs

        def fake_fetch_ohlcv(ticker, days=60):
            if ticker == "AAA":
                return crash_df
            if ticker == "SPY":
                return trend_df(300, 100, 110)
            return pd.DataFrame()

        monkeypatch.setattr("src.positions.load_positions", lambda: positions)
        monkeypatch.setattr("src.positions.save_positions", lambda p: None)
        monkeypatch.setattr("src.positions.fetch_ohlcv", fake_fetch_ohlcv)
        monkeypatch.setattr("src.positions.days_to_next_earnings", lambda t: None)

        captured = {"events": None}

        def fake_send_action_alert(events, positions):
            captured["events"] = list(events)

        def fake_send_daily_digest(positions, skipped, stale=None, errored=None):
            pass

        fake_module = types.ModuleType("src.exit_alerts")
        fake_module.send_action_alert = fake_send_action_alert
        fake_module.send_daily_digest = fake_send_daily_digest
        monkeypatch.setitem(sys.modules, "src.exit_alerts", fake_module)

        today = crash_df.index[-1].date()
        result = run_daily_eval(send_emails=True, today=today)

        assert positions[0]["plan"]["verdict"] == "SELL"
        assert len(result["events"]) == 1
        assert result["events"][0]["ticker"] == "AAA"
        assert result["events"][0]["type"] == "SELL"
        assert "history" in result["events"][0]["reason"]
        assert "70.00" in result["events"][0]["reason"]   # concrete last close

        # reached the action email
        assert captured["events"], "the catch-up SELL must reach send_action_alert"
        assert captured["events"][0]["type"] == "SELL"

        # and pending_notify cleared after the successful send (same
        # persist/clear contract as any other event)
        assert positions[0]["plan"]["pending_notify"] == []

    def test_bootstrap_ending_hold_emits_no_catchup_event(self, monkeypatch):
        import sys
        import types

        # A clean winner: modest, steady gain with no stop breach, no trim
        # rung, no trend break. Short history (< 15 weeks) so the weekly-RSI
        # blowoff check can't fire spuriously on a degenerate read.
        winner_df = make_df([100.0] * 20 + [101, 101.5, 102, 102.5, 103, 103,
                                             103.5, 104, 104.5, 105])
        entry_date = winner_df.index[19].strftime("%Y-%m-%d")
        positions = [{"ticker": "BBB", "entry_date": entry_date, "entry_price": 100.0}]

        def fake_fetch_ohlcv(ticker, days=60):
            if ticker == "BBB":
                return winner_df
            if ticker == "SPY":
                return trend_df(300, 100, 110)
            return pd.DataFrame()

        monkeypatch.setattr("src.positions.load_positions", lambda: positions)
        monkeypatch.setattr("src.positions.save_positions", lambda p: None)
        monkeypatch.setattr("src.positions.fetch_ohlcv", fake_fetch_ohlcv)
        monkeypatch.setattr("src.positions.days_to_next_earnings", lambda t: None)

        captured = {"events": None}

        def fake_send_action_alert(events, positions):
            captured["events"] = list(events)

        def fake_send_daily_digest(positions, skipped, stale=None, errored=None):
            pass

        fake_module = types.ModuleType("src.exit_alerts")
        fake_module.send_action_alert = fake_send_action_alert
        fake_module.send_daily_digest = fake_send_daily_digest
        monkeypatch.setitem(sys.modules, "src.exit_alerts", fake_module)

        today = winner_df.index[-1].date()
        result = run_daily_eval(send_emails=True, today=today)

        assert positions[0]["plan"]["verdict"] == "HOLD"
        assert result["events"] == []
        assert positions[0]["plan"]["pending_notify"] == []
        assert captured["events"] is None, "no events -> send_action_alert must not even be called"

    def test_bootstrap_ending_trim_on_final_bar_emits_a_catchup_event(self):
        # A derisk rung that fires on the LAST replayed bar specifically is
        # freshly actionable (unlike one that fired months earlier and is
        # merely a trims_fired state record) -- see report for the decision
        # to also catch this up.
        ramp = list(np.linspace(100.0, 109.0, 5))   # ATR~2 -> 2R=+8 hits on the way up
        df = make_df([100.0] * 80 + ramp)
        entry_date = df.index[79].strftime("%Y-%m-%d")
        pos = {"ticker": "AAA", "entry_date": entry_date, "entry_price": 100.0}
        pos = bootstrap_position(pos, df)
        assert pos["plan"]["verdict"] == "TRIM"
        assert "derisk" in pos["plan"]["trims_fired"]

        from src.exit_plan import _bootstrap_catchup_events
        final_events = pos.pop("_bootstrap_final_events")
        events = _bootstrap_catchup_events(pos, final_events)
        assert len(events) == 1
        assert events[0]["type"] == "TRIM"
        assert "history" in events[0]["reason"]
        assert "derisk" in events[0]["reason"] or "de-risk" in events[0]["reason"]


class TestRunDailyEval:
    def test_bootstrap_skip_save_and_evaluated_count(self, monkeypatch):
        import copy

        aaa_df = flat_df(80, price=100.0)
        aaa_entry_date = aaa_df.index[59].strftime("%Y-%m-%d")

        bbb_existing_plan = init_plan(flat_df(80, price=50.0), 50.0)
        bbb_existing_plan["verdict"] = "TRIM"
        bbb_existing_plan["trims_fired"] = ["derisk"]
        bbb_plan_before = copy.deepcopy(bbb_existing_plan)

        positions = [
            {"ticker": "AAA", "entry_date": aaa_entry_date, "entry_price": 100.0},
            {"ticker": "BBB", "entry_date": "2025-01-02", "entry_price": 50.0,
             "plan": bbb_existing_plan},
        ]

        spy_df = trend_df(300, 100, 110)

        def fake_fetch_ohlcv(ticker, days=60):
            if ticker == "AAA":
                return aaa_df
            if ticker == "SPY":
                return spy_df
            return pd.DataFrame()   # BBB: simulated data outage

        saved = {}

        def fake_save_positions(pos_list):
            saved["positions"] = pos_list

        monkeypatch.setattr("src.positions.load_positions", lambda: positions)
        monkeypatch.setattr("src.positions.save_positions", fake_save_positions)
        monkeypatch.setattr("src.positions.fetch_ohlcv", fake_fetch_ohlcv)
        monkeypatch.setattr("src.positions.days_to_next_earnings", lambda ticker: 10)

        # Pin "today" to AAA's last bar so the staleness check (a real
        # wall-clock comparison) doesn't reject this synthetic 2026-01-02
        # fixture as stale relative to whatever day the suite actually runs.
        result = run_daily_eval(send_emails=False, today=aaa_df.index[-1].date())

        # AAA had no plan key -> bootstrapped and now carries a plan
        aaa = next(p for p in positions if p["ticker"] == "AAA")
        assert "plan" in aaa and aaa["plan"] is not None

        # BBB's fetch returned empty -> skipped, never evaluated, plan untouched
        assert "BBB" in result["skipped"]
        bbb = next(p for p in positions if p["ticker"] == "BBB")
        assert bbb["plan"] == bbb_plan_before

        # save_positions called with the mutated list
        assert saved["positions"] is positions

        # evaluated count excludes the skipped ticker
        assert result["evaluated"] == 1


class TestErrorIsolation:
    def test_one_ticker_error_does_not_abort_the_run(self, monkeypatch):
        aaa_df = flat_df(80, price=100.0)
        aaa_entry_date = aaa_df.index[59].strftime("%Y-%m-%d")
        positions = [
            {"ticker": "AAA", "entry_date": aaa_entry_date, "entry_price": 100.0},
            {"ticker": "BAD", "entry_date": aaa_entry_date, "entry_price": 100.0},
        ]

        def fake_fetch_ohlcv(ticker, days=60):
            if ticker == "AAA":
                return aaa_df
            if ticker == "SPY":
                return trend_df(300, 100, 110)
            if ticker == "BAD":
                raise RuntimeError("yfinance blew up")
            return pd.DataFrame()

        saved = {}
        monkeypatch.setattr("src.positions.load_positions", lambda: positions)
        monkeypatch.setattr("src.positions.save_positions", lambda p: saved.setdefault("positions", p))
        monkeypatch.setattr("src.positions.fetch_ohlcv", fake_fetch_ohlcv)
        monkeypatch.setattr("src.positions.days_to_next_earnings", lambda t: None)

        result = run_daily_eval(send_emails=False, today=aaa_df.index[-1].date())

        # AAA still evaluated and bootstrapped despite BAD's fetch throwing
        aaa = next(p for p in positions if p["ticker"] == "AAA")
        assert "plan" in aaa and aaa["plan"] is not None

        # BAD recorded as errored, not silently dropped, not counted as evaluated
        assert result["errored"] == [{"ticker": "BAD", "error": repr(RuntimeError("yfinance blew up"))}]
        assert result["evaluated"] == 1

        # the run still persisted (positions that did evaluate must still save)
        assert saved["positions"] is positions


class TestPendingNotify:
    def test_pending_notify_persists_on_failed_send_and_clears_on_success(self, monkeypatch):
        import sys
        import types

        entry_df = flat_df(80, price=100.0)
        live_df = make_df([100.0] * 80 + [121.0])   # 21% jump -> fires a TRIM event
        entry_date = entry_df.index[-1].strftime("%Y-%m-%d")
        plan = init_plan(entry_df, 100.0)
        positions = [{"ticker": "AAA", "entry_date": entry_date, "entry_price": 100.0, "plan": plan}]

        def fake_fetch_ohlcv(ticker, days=60):
            if ticker == "AAA":
                return live_df
            if ticker == "SPY":
                return trend_df(300, 100, 110)
            return pd.DataFrame()

        monkeypatch.setattr("src.positions.load_positions", lambda: positions)
        monkeypatch.setattr("src.positions.save_positions", lambda p: None)
        monkeypatch.setattr("src.positions.fetch_ohlcv", fake_fetch_ohlcv)
        monkeypatch.setattr("src.positions.days_to_next_earnings", lambda t: None)

        fail_flag = {"fail": True}
        captured = {"events": None}

        def fake_send_action_alert(events, positions):
            if fail_flag["fail"]:
                raise RuntimeError("smtp down")
            captured["events"] = list(events)

        def fake_send_daily_digest(positions, skipped, stale=None, errored=None):
            pass

        fake_module = types.ModuleType("src.exit_alerts")
        fake_module.send_action_alert = fake_send_action_alert
        fake_module.send_daily_digest = fake_send_daily_digest
        monkeypatch.setitem(sys.modules, "src.exit_alerts", fake_module)

        today = live_df.index[-1].date()

        run_daily_eval(send_emails=True, today=today)
        assert positions[0]["plan"]["pending_notify"], "leftover events must survive a failed send"

        fail_flag["fail"] = False
        run_daily_eval(send_emails=True, today=today)   # retry: same leftover, this time it sends
        assert positions[0]["plan"]["pending_notify"] == []
        assert captured["events"], "leftover event must have been retried on the successful send"


class TestWeeklyTightenerGuard:
    _ROLLOVER = list(np.linspace(50, 150, 200)) + list(np.linspace(150, 110, 100))

    def test_duplicate_run_on_same_bar_steps_trail_mult_once(self, monkeypatch):
        df = make_df(self._ROLLOVER)
        entry_price = float(df["close"].iloc[-1])
        entry_date = df.index[-1].strftime("%Y-%m-%d")
        positions = [{"ticker": "AAA", "entry_date": entry_date, "entry_price": entry_price}]

        def fake_fetch_ohlcv(ticker, days=60):
            if ticker == "AAA":
                return df
            if ticker == "SPY":
                return trend_df(300, 100, 130)
            return pd.DataFrame()

        monkeypatch.setattr("src.positions.load_positions", lambda: positions)
        monkeypatch.setattr("src.positions.save_positions", lambda p: None)
        monkeypatch.setattr("src.positions.fetch_ohlcv", fake_fetch_ohlcv)
        monkeypatch.setattr("src.positions.days_to_next_earnings", lambda t: None)

        last_bar_date = df.index[-1].date()
        friday = last_bar_date + timedelta(days=(4 - last_bar_date.weekday()) % 7)

        run_daily_eval(send_emails=False, today=friday)
        assert positions[0]["plan"]["trail_mult"] == 2.5

        # duplicate cron fire / manual retry / Saturday catch-up: same bar,
        # same week -> must not step a second time
        run_daily_eval(send_emails=False, today=friday)
        assert positions[0]["plan"]["trail_mult"] == 2.5

    def test_thursday_final_bar_from_a_holiday_friday_still_tightens(self, monkeypatch):
        start = pd.Timestamp("2026-01-01")
        while start.weekday() != 3:   # walk forward to a Thursday
            start += pd.Timedelta(days=1)
        closes = list(np.linspace(50, 150, 201)) + list(np.linspace(150, 110, 100))  # 301 bars
        df = make_df(closes, start=start.strftime("%Y-%m-%d"))
        assert df.index[-1].weekday() == 3   # sanity: final bar really is a Thursday

        entry_price = float(df["close"].iloc[-1])
        entry_date = df.index[-1].strftime("%Y-%m-%d")
        positions = [{"ticker": "AAA", "entry_date": entry_date, "entry_price": entry_price}]

        def fake_fetch_ohlcv(ticker, days=60):
            if ticker == "AAA":
                return df
            if ticker == "SPY":
                return trend_df(300, 100, 130)
            return pd.DataFrame()

        monkeypatch.setattr("src.positions.load_positions", lambda: positions)
        monkeypatch.setattr("src.positions.save_positions", lambda p: None)
        monkeypatch.setattr("src.positions.fetch_ohlcv", fake_fetch_ohlcv)
        monkeypatch.setattr("src.positions.days_to_next_earnings", lambda t: None)

        # Friday (the next calendar day) is a market holiday: no newer bar
        # ever shows up, so wall-clock "today" reaches Friday while the data's
        # last close stays on Thursday.
        friday = df.index[-1].date() + timedelta(days=1)
        run_daily_eval(send_emails=False, today=friday)
        assert positions[0]["plan"]["trail_mult"] == 2.5


class TestStaleness:
    def test_stale_bar_lands_in_stale_and_leaves_plan_untouched(self, monkeypatch):
        import copy

        df = flat_df(80, price=100.0)
        entry_date = df.index[59].strftime("%Y-%m-%d")
        existing_plan = init_plan(flat_df(80, price=100.0), 100.0)
        existing_plan["verdict"] = "TRIM"
        plan_before = copy.deepcopy(existing_plan)
        positions = [{"ticker": "STALE", "entry_date": entry_date, "entry_price": 100.0,
                      "plan": existing_plan}]

        def fake_fetch_ohlcv(ticker, days=60):
            return df if ticker != "SPY" else pd.DataFrame()

        monkeypatch.setattr("src.positions.load_positions", lambda: positions)
        monkeypatch.setattr("src.positions.save_positions", lambda p: None)
        monkeypatch.setattr("src.positions.fetch_ohlcv", fake_fetch_ohlcv)
        monkeypatch.setattr("src.positions.days_to_next_earnings", lambda t: None)

        far_future = df.index[-1].date() + timedelta(days=10)   # 10 days stale (> 4)
        result = run_daily_eval(send_emails=False, today=far_future)

        assert result["stale"] == [{"ticker": "STALE", "bar_date": df.index[-1].strftime("%Y-%m-%d")}]
        assert result["evaluated"] == 0
        assert positions[0]["plan"] == plan_before   # untouched, exactly like "skipped"

    def test_evaluate_day_does_not_regress_state_on_an_older_bar(self):
        import copy

        df = flat_df(80)
        pos = entry_pos(df)
        day81 = make_df([100.0] * 81)
        pos, _ = evaluate_day(pos, day81)   # last_eval is now day81's date
        plan_before = copy.deepcopy(pos["plan"])

        # a stale re-fetch hands back a frame whose last bar predates last_eval
        pos, events = evaluate_day(pos, df)
        assert events == []
        assert pos["plan"] == plan_before   # nothing regressed backward


class TestDigestWiring:
    def test_stale_and_errored_tickers_reach_digest_with_not_evaluated_marker(self, monkeypatch):
        import sys
        import types

        from src.exit_alerts import build_digest_email as real_build_digest_email

        good_df = flat_df(90, price=100.0)
        stale_df = good_df.iloc[:-10]   # last bar well over 4 calendar days behind "today"

        positions = [
            {"ticker": "GOOD", "entry_date": "2025-01-02", "entry_price": 100.0,
             "plan": init_plan(flat_df(80, price=100.0), 100.0)},
            {"ticker": "STALE", "entry_date": "2025-01-02", "entry_price": 100.0,
             "plan": init_plan(flat_df(80, price=100.0), 100.0)},
            {"ticker": "BAD", "entry_date": "2025-01-02", "entry_price": 100.0,
             "plan": init_plan(flat_df(80, price=100.0), 100.0)},
        ]

        def fake_fetch_ohlcv(ticker, days=600):
            if ticker == "GOOD":
                return good_df
            if ticker == "STALE":
                return stale_df
            if ticker == "BAD":
                raise RuntimeError("yfinance timed out")
            return pd.DataFrame()   # SPY: weekly tightener degrades gracefully with no benchmark

        monkeypatch.setattr("src.positions.load_positions", lambda: positions)
        monkeypatch.setattr("src.positions.save_positions", lambda p: None)
        monkeypatch.setattr("src.positions.fetch_ohlcv", fake_fetch_ohlcv)
        monkeypatch.setattr("src.positions.days_to_next_earnings", lambda t: None)

        captured = {}

        def fake_send_action_alert(events, positions):
            pass

        def fake_send_daily_digest(positions, skipped, stale=None, errored=None):
            # Use the real builder so this test exercises production rendering,
            # not a second hand-rolled assertion of the call args.
            captured["subject"], captured["html"] = real_build_digest_email(
                positions, skipped, stale or [], errored or [])

        fake_module = types.ModuleType("src.exit_alerts")
        fake_module.send_action_alert = fake_send_action_alert
        fake_module.send_daily_digest = fake_send_daily_digest
        monkeypatch.setitem(sys.modules, "src.exit_alerts", fake_module)

        run_daily_eval(send_emails=True, today=good_df.index[-1].date())

        assert "html" in captured, "send_daily_digest must be called with this run's stale/errored lists"
        html = captured["html"]
        assert "STALE" in html and "NOT EVALUATED TODAY" in html
        assert stale_df.index[-1].strftime("%Y-%m-%d") in html
        assert "BAD" in html and "yfinance timed out" in html
