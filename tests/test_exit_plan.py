import numpy as np
import pandas as pd
import pytest

from src.exit_plan import init_plan, evaluate_day


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
