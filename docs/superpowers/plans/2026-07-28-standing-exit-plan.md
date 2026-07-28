# Standing Exit Plan Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the daily-flipping 10-signal exit grade with per-position standing exit plans: one-way ratcheting stops, once-only trim rungs, weekly-only tightener, final SELL/TRIM/HOLD verdicts delivered by email.

**Architecture:** New pure-function engine in `src/exit_plan.py` persists plan state inside each position dict in `positions.json` (additive `"plan"` key). A new step in `run_screener.sh` (separate process from the screener, so screener crashes can't block it) evaluates all positions on closing data at 4:30pm ET, saves state, and sends emails via a new `src/exit_alerts.py` (reusing `notify.send_email`). The Streamlit positions page reads the stored plan state instead of recomputing indicators — verdicts are identical locally and on the cloud deploy (positions.json is already git-committed daily by run_screener.sh).

**Tech Stack:** Python 3.14 (`/Library/Frameworks/Python.framework/Versions/3.14/bin/python3`), pandas, yfinance (via existing `positions.fetch_ohlcv`), existing `src/factors.py` indicators, Gmail SMTP via `notify.send_email`, pytest.

**Spec:** `docs/superpowers/specs/2026-07-28-standing-exit-plan-design.md`

---

## Plan-state schema (reference for all tasks)

Each entry in `positions.json` gains a `"plan"` key (additive — old readers ignore it):

```json
{
  "ticker": "ANET",
  "entry_date": "2026-06-18",
  "entry_price": 100.25,
  "plan": {
    "initial_stop": 90.00,          // entry − 2×ATR14(entry date); fixed
    "risk_R": 11.47,                // entry − initial_stop; fixed
    "stop_floor": 90.00,            // = initial_stop, ratchets to entry after derisk trim; never lowers
    "peak_close": 110.00,            // highest close since entry; only rises
    "trail_mult": 3.0,              // weekly tightener may step 3.0→2.5→2.0; never rises
    "stop_level": 102.0,            // max(stop_floor, peak−mult×ATR, yesterday's); one-way up
    "trims_fired": ["derisk"],      // append-only; subset of {"derisk","blowoff"}
    "below_50d_streak": 0,          // consecutive closes < SMA50; resets on close above
    "verdict": "HOLD",              // HOLD | TRIM | SELL; SELL terminal in live mode
    "health": {"bearish": 1, "parts": ["macd"], "asof": "2026-07-24"},
    "days_to_earnings": 12,         // refreshed at eval; null if unknown
    "last_eval": "2026-07-28",
    "last_close": 181.3
  }
}
```

Events emitted by evaluation (drive emails): `{"ticker","type","reason","instruction"}` where `type ∈ {"SELL","TRIM"}`.

---

### Task 1: Engine core — `init_plan` + `evaluate_day` (pure functions, TDD)

**Files:**
- Create: `src/exit_plan.py`
- Create: `tests/test_exit_plan.py`

- [ ] **Step 1: Write failing tests for `init_plan` and basic HOLD evaluation**

Create `tests/test_exit_plan.py`:

```python
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
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `cd /Users/arhanbarve/Code/screener && /Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m pytest tests/test_exit_plan.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.exit_plan'`

- [ ] **Step 3: Implement `init_plan` + `evaluate_day` skeleton (HOLD path)**

Create `src/exit_plan.py`:

```python
"""Standing exit plan engine.

Per-position exit plans with one-way ratcheting stops, once-only trim rungs,
and final SELL/TRIM/HOLD verdicts. All triggers anchor to the position's own
history (entry, R, peak) or need multi-day confirmation, so verdicts cannot
flip on ordinary market-wide moves.

Spec: docs/superpowers/specs/2026-07-28-standing-exit-plan-design.md
"""
import math

import pandas as pd

from src.factors import atr_14, macd_state, obv_slope, rsi_14, sma, directional_indicators

INITIAL_STOP_ATR_MULT = 2.0
INITIAL_STOP_PCT_FALLBACK = 0.08   # when history too short for ATR14
TRAIL_MULT_START = 3.0
TRAIL_MULT_STEPS = [3.0, 2.5, 2.0]  # weekly tightener walks down, floor 2.0
DERISK_R_MULT = 2.0
DERISK_GAIN_PCT = 0.20
TRIM_FRACTION = "1/3"
BLOWOFF_SMA50_STRETCH = 1.25
BLOWOFF_WEEKLY_RSI = 80.0
BLOWOFF_ATR_BURST = 3.0
BLOWOFF_BURST_DAYS = 5
TREND_BREAK_DAYS = 3


def init_plan(df: pd.DataFrame, entry_price: float) -> dict:
    """Create a fresh plan. `df` must end on the entry date (bars after entry
    excluded) so ATR reflects conditions at entry."""
    atr = atr_14(df["high"], df["low"], df["close"]) if len(df) >= 15 else float("nan")
    if math.isnan(atr):
        initial_stop = entry_price * (1 - INITIAL_STOP_PCT_FALLBACK)
    else:
        initial_stop = entry_price - INITIAL_STOP_ATR_MULT * atr
    return {
        "initial_stop": round(initial_stop, 4),
        "risk_R": round(entry_price - initial_stop, 4),
        "stop_floor": round(initial_stop, 4),
        "peak_close": round(float(entry_price), 4),
        "trail_mult": TRAIL_MULT_START,
        "stop_level": round(initial_stop, 4),
        "trims_fired": [],
        "below_50d_streak": 0,
        "verdict": "HOLD",
        "health": None,
        "days_to_earnings": None,
        "last_eval": None,
        "last_close": None,
    }


def evaluate_day(pos: dict, df: pd.DataFrame, replay: bool = False) -> tuple[dict, list[dict]]:
    """Evaluate one position on the latest close in `df`.

    Mutates and returns `pos` (with updated pos["plan"]) plus a list of action
    events. Idempotent per date: re-running on the same last bar produces no
    duplicate events (trims are once-only, SELL is sticky).

    `replay=True` (bootstrap): SELL is not terminal — it clears when the close
    regains stop_level, because the user demonstrably held through it.
    """
    plan = pos["plan"]
    events: list[dict] = []
    if df.empty or len(df) < 2:
        return pos, events
    close = df["close"]
    price = float(close.iloc[-1])
    today = df.index[-1].strftime("%Y-%m-%d")
    entry = float(pos["entry_price"])

    # already evaluated this bar in live mode → no-op (idempotent re-fire)
    if not replay and plan.get("last_eval") == today:
        return pos, events

    # ── ratchets (always run, even while verdict is SELL) ──────────────────
    plan["peak_close"] = max(plan["peak_close"], price)
    atr = atr_14(df["high"], df["low"], df["close"])
    if not math.isnan(atr):
        trail = plan["peak_close"] - plan["trail_mult"] * atr
        plan["stop_level"] = round(max(plan["stop_level"], plan["stop_floor"], trail), 4)

    s50 = sma(close, 50)
    if not math.isnan(s50) and price < s50:
        plan["below_50d_streak"] += 1
    else:
        plan["below_50d_streak"] = 0

    # ── terminal SELL handling ─────────────────────────────────────────────
    if plan["verdict"] == "SELL":
        if replay and price > plan["stop_level"]:
            plan["verdict"] = "HOLD"   # held through it; plan resumes
        plan["last_eval"], plan["last_close"] = today, round(price, 4)
        return pos, events

    # ── SELL triggers ──────────────────────────────────────────────────────
    sell_reason = None
    if price < plan["stop_level"]:
        sell_reason = f"close {price:.2f} below stop level {plan['stop_level']:.2f}"
    elif price < plan["stop_floor"]:
        sell_reason = f"close {price:.2f} below max-loss floor {plan['stop_floor']:.2f}"
    elif plan["below_50d_streak"] >= TREND_BREAK_DAYS:
        sell_reason = f"{plan['below_50d_streak']} consecutive closes below 50-day SMA {s50:.2f}"
    if sell_reason:
        plan["verdict"] = "SELL"
        events.append({"ticker": pos["ticker"], "type": "SELL", "reason": sell_reason,
                       "instruction": "Sell entire remaining position at next open."})
        plan["last_eval"], plan["last_close"] = today, round(price, 4)
        return pos, events

    # ── TRIM rungs (each fires once, append-only) ──────────────────────────
    if "derisk" not in plan["trims_fired"]:
        gain = price - entry
        if gain >= DERISK_R_MULT * plan["risk_R"] or gain >= DERISK_GAIN_PCT * entry:
            plan["trims_fired"].append("derisk")
            plan["stop_floor"] = round(max(plan["stop_floor"], entry), 4)
            plan["stop_level"] = round(max(plan["stop_level"], plan["stop_floor"]), 4)
            events.append({
                "ticker": pos["ticker"], "type": "TRIM",
                "reason": f"gain {gain / entry:+.1%} hit de-risk target "
                          f"(+{DERISK_R_MULT:g}R={DERISK_R_MULT * plan['risk_R']:.2f} "
                          f"or +{DERISK_GAIN_PCT:.0%}); stop moved to breakeven {entry:.2f}",
                "instruction": f"Sell {TRIM_FRACTION} of position at next open.",
            })

    if "blowoff" not in plan["trims_fired"]:
        stretched = (not math.isnan(s50)) and price > BLOWOFF_SMA50_STRETCH * s50
        wk_rsi = _weekly_rsi(close)
        overheated = wk_rsi is not None and wk_rsi > BLOWOFF_WEEKLY_RSI
        burst = False
        if not math.isnan(atr) and len(close) > BLOWOFF_BURST_DAYS:
            burst = (price - float(close.iloc[-1 - BLOWOFF_BURST_DAYS])) > BLOWOFF_ATR_BURST * atr
        if stretched or overheated or burst:
            why = ("close >25% above 50-day SMA" if stretched
                   else f"weekly RSI {wk_rsi:.0f} > 80" if overheated
                   else f"gained >{BLOWOFF_ATR_BURST:g}×ATR in {BLOWOFF_BURST_DAYS} sessions")
            plan["trims_fired"].append("blowoff")
            events.append({"ticker": pos["ticker"], "type": "TRIM",
                           "reason": f"blowoff extension: {why}",
                           "instruction": f"Sell {TRIM_FRACTION} of position at next open."})

    plan["verdict"] = "TRIM" if events else "HOLD"
    plan["last_eval"], plan["last_close"] = today, round(price, 4)
    return pos, events


def _weekly_rsi(close: pd.Series) -> float | None:
    """RSI(14) on weekly (Friday) closes. None if under 15 weeks of data."""
    wk = close.resample("W-FRI").last().dropna()
    if len(wk) < 15:
        return None
    val = rsi_14(wk)
    return None if math.isnan(val) else float(val)
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m pytest tests/test_exit_plan.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/exit_plan.py tests/test_exit_plan.py
git commit -m "feat(exit-plan): plan init + daily evaluation engine core"
```

---

### Task 2: SELL, TRIM, and invariance tests (TDD hardening)

**Files:**
- Modify: `tests/test_exit_plan.py` (append)
- Modify: `src/exit_plan.py` (only if a test exposes a bug)

- [ ] **Step 1: Append failing/behavioral tests**

Append to `tests/test_exit_plan.py`:

```python
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
    def test_derisk_fires_once_at_20pct_and_moves_stop_to_breakeven(self):
        df = flat_df(80)
        pos = entry_pos(df)
        up = make_df([100.0] * 80 + [121.0])
        pos, events = evaluate_day(pos, up)
        assert pos["plan"]["verdict"] == "TRIM"
        assert [e["type"] for e in events] == ["TRIM"]
        assert "derisk" in pos["plan"]["trims_fired"]
        assert pos["plan"]["stop_floor"] == pytest.approx(100.0)
        # next day, still up: no second derisk event
        up2 = make_df([100.0] * 80 + [121.0, 122.0])
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
```

- [ ] **Step 2: Run tests**

Run: `/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m pytest tests/test_exit_plan.py -v`
Expected: all PASS. If any fail, fix `src/exit_plan.py` (the engine has a bug — tests encode the spec).

Note: the whipsaw test alternates absolute 97/103 closes (not compounding percentages) deliberately — closes must straddle the 50d SMA so the below-50d streak resets, and stay above both the 96 stop and the +8 derisk rung. Don't "fix" it to compounding moves; a drifting sequence slides under the SMA and legitimately trips the 3-day trend break.

- [ ] **Step 3: Commit**

```bash
git add tests/test_exit_plan.py src/exit_plan.py
git commit -m "test(exit-plan): SELL/TRIM triggers, whipsaw + ratchet invariants"
```

---

### Task 3: Weekly tightener (health check)

**Files:**
- Modify: `src/exit_plan.py`
- Modify: `tests/test_exit_plan.py` (append)

- [ ] **Step 1: Write failing tests**

Append to `tests/test_exit_plan.py`:

```python
from src.exit_plan import weekly_health, apply_weekly_tightener


def trend_df(n, start_price, end_price, **kw):
    return make_df(list(np.linspace(start_price, end_price, n)), **kw)


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
```

- [ ] **Step 2: Run, verify fail**

Run: `/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m pytest tests/test_exit_plan.py -k Tightener -v`
Expected: FAIL — `ImportError: cannot import name 'weekly_health'`

- [ ] **Step 3: Implement**

Add to `src/exit_plan.py`:

```python
def weekly_health(df: pd.DataFrame, spy_close: pd.Series | None) -> dict:
    """Weekly-bar health check. Returns {"bearish": n, "parts": [...], "asof": date}.

    Four checks on W-FRI resampled bars: MACD state, 13-week RS vs SPY,
    OBV slope, ADX direction (-DI dominant and ADX falling). Each bearish
    check adds 1. Used ONLY to tighten trail_mult — never to sell.
    """
    parts: list[str] = []
    wk = df.resample("W-FRI").agg(
        {"high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()
    asof = df.index[-1].strftime("%Y-%m-%d") if len(df) else None
    if len(wk) < 15:
        return {"bearish": 0, "parts": [], "asof": asof}

    try:
        if macd_state(wk["close"]) in ("bearish", "bearish_cross"):
            parts.append("macd")
    except Exception:
        pass
    try:
        if spy_close is not None:
            spy_wk = spy_close.resample("W-FRI").last().dropna()
            if len(spy_wk) >= 14 and len(wk) >= 14:
                stk = float(wk["close"].iloc[-1] / wk["close"].iloc[-14] - 1)
                spy = float(spy_wk.iloc[-1] / spy_wk.iloc[-14] - 1)
                if stk < spy:
                    parts.append("rs")
    except Exception:
        pass
    try:
        slp = obv_slope(wk["close"], wk["volume"], window=10)
        if not math.isnan(slp) and slp < 0:
            parts.append("obv")
    except Exception:
        pass
    try:
        adx_now, pdi, ndi = directional_indicators(wk["high"], wk["low"], wk["close"])
        if len(wk) >= 20:
            adx_past, _, _ = directional_indicators(
                wk["high"].iloc[:-4], wk["low"].iloc[:-4], wk["close"].iloc[:-4])
            if (not math.isnan(adx_now) and not math.isnan(adx_past)
                    and adx_past > adx_now and ndi > pdi):
                parts.append("adx")
    except Exception:
        pass
    return {"bearish": len(parts), "parts": parts, "asof": asof}


def apply_weekly_tightener(plan: dict, health: dict) -> dict:
    """If ≥2 weekly checks bearish, step trail_mult down one notch (floor 2.0).
    Never loosens. Stores health on the plan either way."""
    plan["health"] = health
    if health["bearish"] >= 2:
        steps = [m for m in TRAIL_MULT_STEPS if m < plan["trail_mult"]]
        if steps:
            plan["trail_mult"] = steps[0]
    return plan
```

- [ ] **Step 4: Run full test file**

Run: `/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m pytest tests/test_exit_plan.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/exit_plan.py tests/test_exit_plan.py
git commit -m "feat(exit-plan): weekly health check tightens trail multiplier one-way"
```

---

### Task 4: Daily runner + bootstrap migration

**Files:**
- Modify: `src/exit_plan.py` (add `run_daily_eval`, `bootstrap_position`, `__main__`)
- Modify: `tests/test_exit_plan.py` (append)

- [ ] **Step 1: Write failing tests**

Append to `tests/test_exit_plan.py`:

```python
from src.exit_plan import bootstrap_position


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
```

- [ ] **Step 2: Run, verify fail**

Run: `/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m pytest tests/test_exit_plan.py -k Bootstrap -v`
Expected: FAIL — `ImportError: cannot import name 'bootstrap_position'`

- [ ] **Step 3: Implement bootstrap + daily runner + CLI**

Add to `src/exit_plan.py`:

```python
def bootstrap_position(pos: dict, df: pd.DataFrame) -> dict:
    """One-time backfill: init plan at entry, replay history day-by-day.
    Events are discarded (trims get marked, no emails). Replay mode clears
    historical SELLs the user held through."""
    entry_ts = pd.Timestamp(pos["entry_date"])
    upto_entry = df[df.index <= entry_ts]
    if upto_entry.empty:
        upto_entry = df.iloc[:1]
    pos["plan"] = init_plan(upto_entry, float(pos["entry_price"]))
    after = df.index[df.index > entry_ts]
    for ts in after:
        pos, _ = evaluate_day(pos, df.loc[:ts], replay=True)
    return pos


def run_daily_eval(send_emails: bool = True) -> dict:
    """Evaluate every position on today's close. Saves positions.json,
    applies weekly tightener on Fridays, sends action + digest emails.
    Returns {"events": [...], "evaluated": n, "skipped": [...]}."""
    from src.positions import load_positions, save_positions, fetch_ohlcv, days_to_next_earnings

    positions = load_positions()
    if not positions:
        return {"events": [], "evaluated": 0, "skipped": []}

    spy_df = fetch_ohlcv("SPY", days=600)
    spy_close = spy_df["close"] if not spy_df.empty else None

    all_events: list[dict] = []
    skipped: list[str] = []
    for pos in positions:
        df = fetch_ohlcv(pos["ticker"], days=600)
        if df.empty or len(df) < 2:
            skipped.append(pos["ticker"])   # keep yesterday's state, never fabricate
            continue
        if "plan" not in pos or pos["plan"] is None:
            pos = bootstrap_position(pos, df)
            print(f"[exit-plan] bootstrapped {pos['ticker']}: "
                  f"verdict={pos['plan']['verdict']} trims={pos['plan']['trims_fired']}")
        pos, events = evaluate_day(pos, df)
        if df.index[-1].weekday() == 4:   # Friday close → weekly tightener
            pos["plan"] = apply_weekly_tightener(pos["plan"], weekly_health(df, spy_close))
        pos["plan"]["days_to_earnings"] = days_to_next_earnings(pos["ticker"])
        all_events.extend(events)

    save_positions(positions)

    result = {"events": all_events, "evaluated": len(positions) - len(skipped),
              "skipped": skipped}
    if send_emails:
        from src.exit_alerts import send_action_alert, send_daily_digest
        if all_events:
            send_action_alert(all_events, positions)
        send_daily_digest(positions, skipped)
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Standing exit plan daily evaluation")
    parser.add_argument("--no-email", action="store_true", help="evaluate and save only")
    args = parser.parse_args()
    res = run_daily_eval(send_emails=not args.no_email)
    print(f"[exit-plan] evaluated={res['evaluated']} events={len(res['events'])} "
          f"skipped={res['skipped']}")
    for e in res["events"]:
        print(f"[exit-plan] ACTION {e['type']} {e['ticker']}: {e['reason']}")
```

(`send_action_alert` / `send_daily_digest` arrive in Task 5 — the import is lazy and only hit with `send_emails=True`, so Task 4 tests pass without them.)

- [ ] **Step 4: Run full test file**

Run: `/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m pytest tests/test_exit_plan.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/exit_plan.py tests/test_exit_plan.py
git commit -m "feat(exit-plan): daily runner with lazy bootstrap and Friday tightener"
```

---

### Task 5: Emails — action alert + daily digest

**Files:**
- Create: `src/exit_alerts.py`
- Create: `tests/test_exit_alerts.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_exit_alerts.py`:

```python
from src.exit_alerts import build_action_email, build_digest_email

POSITIONS = [
    {"ticker": "ANET", "entry_date": "2026-06-18", "entry_price": 100.25,
     "plan": {"initial_stop": 90.00, "risk_R": 11.47, "stop_floor": 100.25,
              "peak_close": 110.00, "trail_mult": 3.0, "stop_level": 102.0,
              "trims_fired": ["derisk"], "below_50d_streak": 0, "verdict": "SELL",
              "health": {"bearish": 2, "parts": ["macd", "rs"], "asof": "2026-07-24"},
              "days_to_earnings": 12, "last_eval": "2026-07-28", "last_close": 171.0}},
    {"ticker": "CRDO", "entry_date": "2026-06-18", "entry_price": 200.00,
     "plan": {"initial_stop": 250.0, "risk_R": 20.00, "stop_floor": 250.0,
              "peak_close": 300.0, "trail_mult": 3.0, "stop_level": 280.0,
              "trims_fired": [], "below_50d_streak": 0, "verdict": "HOLD",
              "health": {"bearish": 0, "parts": [], "asof": "2026-07-24"},
              "days_to_earnings": None, "last_eval": "2026-07-28", "last_close": 295.0}},
]

EVENTS = [{"ticker": "ANET", "type": "SELL",
           "reason": "close 171.00 below stop level 102.00",
           "instruction": "Sell entire remaining position at next open."}]


def test_action_email_subject_names_action_and_ticker():
    subject, html = build_action_email(EVENTS, POSITIONS)
    assert "ACTION" in subject and "SELL" in subject and "ANET" in subject
    assert "102.00" in html
    assert "Sell entire remaining position" in html


def test_digest_lists_every_position_with_verdict():
    subject, html = build_digest_email(POSITIONS, skipped=[])
    for t in ("ANET", "CRDO"):
        assert t in html
    assert "SELL" in html and "HOLD" in html


def test_digest_flags_skipped_tickers():
    subject, html = build_digest_email(POSITIONS, skipped=["RSI"])
    assert "RSI" in html and ("stale" in html.lower() or "skipped" in html.lower())
```

- [ ] **Step 2: Run, verify fail**

Run: `/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m pytest tests/test_exit_alerts.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.exit_alerts'`

- [ ] **Step 3: Implement**

Create `src/exit_alerts.py`:

```python
"""Emails for the standing exit plan engine: action alerts + daily digest.

Builders are pure (subject, html) functions — testable without SMTP.
Sending reuses notify.send_email (Gmail SMTP, env-var credentials).
"""
import html as _html
from datetime import date

from src.notify import send_email

_VERDICT_COLOR = {"SELL": "#ef4444", "TRIM": "#f59e0b", "HOLD": "#22c55e"}

_CSS = """
body{background:#0b0d17;color:#e5e7eb;font-family:-apple-system,Segoe UI,sans-serif;margin:0;padding:24px}
.card{max-width:640px;margin:0 auto;background:#11142a;border:1px solid #232748;border-radius:12px;padding:24px}
h1{font-size:18px;margin:0 0 16px}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{text-align:left;padding:8px 10px;border-bottom:1px solid #232748}
th{color:#9ca3af;font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:.06em}
.mono{font-family:ui-monospace,Menlo,monospace}
.badge{display:inline-block;padding:2px 10px;border-radius:999px;font-weight:700;font-size:12px}
.reason{color:#9ca3af;font-size:12px;margin:4px 0 12px}
.instruction{background:#1a1e3a;border-left:3px solid #ef4444;padding:10px 14px;margin:8px 0 20px;font-weight:600}
.foot{color:#6b7280;font-size:11px;margin-top:16px;text-align:center}
"""


def _badge(verdict: str) -> str:
    c = _VERDICT_COLOR.get(verdict, "#9ca3af")
    return (f'<span class="badge" style="color:{c};background:{c}22;'
            f'border:1px solid {c}55">{verdict}</span>')


def _wrap(title: str, body: str) -> str:
    return (f'<!doctype html><html><head><meta charset="utf-8">'
            f'<style>{_CSS}</style></head><body><div class="card">'
            f'<h1>{title}</h1>{body}'
            f'<div class="foot">Standing exit plan · evaluated on closing prices · '
            f'{date.today().isoformat()} · Not investment advice</div>'
            f'</div></body></html>')


def _plan_row(pos: dict) -> str:
    p = pos.get("plan") or {}
    close = p.get("last_close")
    stop = p.get("stop_level")
    dist = (f"{(close - stop) / close:+.1%}" if close and stop else "—")
    trims = ",".join(p.get("trims_fired") or []) or "—"
    health = p.get("health") or {}
    hstr = f"{health.get('bearish', '—')}/4"
    dte = p.get("days_to_earnings")
    earn = f"{int(dte)}d" if isinstance(dte, (int, float)) else "—"
    return (f'<tr><td class="mono"><b>{_html.escape(pos["ticker"])}</b></td>'
            f'<td>{_badge(p.get("verdict", "?"))}</td>'
            f'<td class="mono">{close if close is not None else "—"}</td>'
            f'<td class="mono">{stop if stop is not None else "—"} ({dist})</td>'
            f'<td class="mono">{trims}</td>'
            f'<td class="mono">{hstr}</td>'
            f'<td class="mono">{earn}</td></tr>')


def build_action_email(events: list[dict], positions: list[dict]) -> tuple[str, str]:
    parts = sorted({f"{e['type']} {e['ticker']}" for e in events})
    subject = f"🚨 ACTION: {' · '.join(parts)}"
    by_ticker = {p["ticker"]: p for p in positions}
    body = ""
    for e in events:
        p = (by_ticker.get(e["ticker"], {}).get("plan")) or {}
        body += (f'<div style="margin-bottom:8px">{_badge(e["type"])} '
                 f'<span class="mono" style="font-size:16px;font-weight:700">'
                 f'{_html.escape(e["ticker"])}</span></div>'
                 f'<div class="reason">{_html.escape(e["reason"])}</div>'
                 f'<div class="instruction">{_html.escape(e["instruction"])}</div>')
        if p:
            body += (f'<div class="reason mono">entry {p.get("stop_floor")}fl · '
                     f'stop {p.get("stop_level")} · peak {p.get("peak_close")} · '
                     f'trims [{",".join(p.get("trims_fired") or [])}]</div>')
    return subject, _wrap("Exit plan action required", body)


def build_digest_email(positions: list[dict], skipped: list[str]) -> tuple[str, str]:
    sells = sum(1 for p in positions if (p.get("plan") or {}).get("verdict") == "SELL")
    trims = sum(1 for p in positions if (p.get("plan") or {}).get("verdict") == "TRIM")
    if sells or trims:
        subject = f"📋 Positions digest — {sells} SELL · {trims} TRIM"
    else:
        subject = "📋 Positions digest — all HOLD"
    rows = "".join(_plan_row(p) for p in positions)
    body = (f'<table><tr><th>Ticker</th><th>Verdict</th><th>Close</th>'
            f'<th>Stop (dist)</th><th>Trims</th><th>Health</th><th>Earn</th></tr>'
            f'{rows}</table>')
    if skipped:
        body += (f'<div class="reason">⚠ Skipped (data fetch failed, state stale): '
                 f'{_html.escape(", ".join(skipped))}</div>')
    return subject, _wrap("Daily positions digest", body)


def send_action_alert(events: list[dict], positions: list[dict]) -> dict:
    subject, html = build_action_email(events, positions)
    return send_email(subject, html)


def send_daily_digest(positions: list[dict], skipped: list[str]) -> dict:
    subject, html = build_digest_email(positions, skipped)
    return send_email(subject, html)
```

- [ ] **Step 4: Run tests**

Run: `/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m pytest tests/test_exit_alerts.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/exit_alerts.py tests/test_exit_alerts.py
git commit -m "feat(exit-plan): action alert + daily digest emails"
```

---

### Task 6: Wire into run_screener.sh

**Files:**
- Modify: `run_screener.sh` (after the `src.run_status` block, before the git-commit block)

- [ ] **Step 1: Add eval step**

Insert into `run_screener.sh` after the run_status block (currently ends `|| echo "=== run_status failed (non-fatal) ===" >> "$LOG_FILE"`) and before `# Commit and push today's output`:

```bash
# Standing exit plan: evaluate positions on today's close, email verdicts.
# Separate process from the screener so a screener crash can't block it.
/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m src.exit_plan \
    >> "$LOG_FILE" 2>&1 || echo "=== exit-plan eval failed (non-fatal) ===" >> "$LOG_FILE"
```

(The existing `COMMIT_PATHS` already includes `positions.json`, so plan state gets committed and pushed for the cloud dashboard — no change needed there.)

- [ ] **Step 2: Verify script syntax**

Run: `bash -n run_screener.sh && echo OK`
Expected: `OK`

- [ ] **Step 3: Dry-run the eval end-to-end (real data, no email)**

Run: `/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m src.exit_plan --no-email`
Expected: `[exit-plan] evaluated=5 events=... skipped=[]`, bootstrap lines for all 5 positions (ANET, CRDO, RSI, ENVA, SPY), `positions.json` now contains `"plan"` objects. Inspect: `python3 -c "import json;print(json.dumps(json.load(open('positions.json'))[0]['plan'],indent=1))"`

- [ ] **Step 4: Preview emails without sending**

Run: `/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -c "
from src.positions import load_positions
from src.exit_alerts import build_digest_email
subject, html = build_digest_email(load_positions(), [])
open('logs/exit_digest_preview.html','w').write(html)
print(subject)"`
Expected: subject printed; open `logs/exit_digest_preview.html` in a browser and confirm the table renders.

- [ ] **Step 5: Commit**

```bash
git add run_screener.sh positions.json
git commit -m "feat(exit-plan): daily eval step in screener cron; bootstrap live positions"
```

---

### Task 7: Positions page UI — verdict from stored plan

**Files:**
- Modify: `app_shared.py` (`_render_position_card` ~line 2037, `_render_positions` ~line 2186, `_batch_position_data` ~line 1263)

The page stops computing exit signals — it reads `pos["plan"]` (written daily by the eval, present in the cloud deploy via the git-committed positions.json). `_batch_position_data` now only fetches live quotes.

- [ ] **Step 1: Slim `_batch_position_data`**

In `app_shared.py`, replace the body of `_fetch_one` inside `_batch_position_data` so it no longer calls `fetch_ohlcv`/`compute_exit_signals`/`days_to_next_earnings`:

```python
    def _fetch_one(ticker: str) -> tuple[str, tuple[dict, float | None, float | None]]:
        try:
            price, prev_close = get_live_quote(ticker)
            if price is None:
                print(f"[positions] live quote fetch returned no price for {ticker} — falling back to Fidelity snapshot", flush=True)
            return ticker, ({}, price, prev_close)
        except Exception as e:
            print(f"[positions] live quote fetch failed for {ticker}: {e!r}", flush=True)
            return ticker, ({}, None, None)
```

Also remove the now-unused `spy_df`/`spy_close` lines at the top of `_batch_position_data`, and delete the `_cached_position_data` function (only caller is `_render_position_card`'s fallback — updated next step). Drop `compute_exit_signals` and `days_to_next_earnings` from the `from src.positions import (...)` block at ~line 1167 if no other references remain (grep first).

- [ ] **Step 2: Rewrite `_render_position_card` verdict/badges sections**

In `_render_position_card`, replace the signal-derived locals (`grade`, `soft_score`, `soft_max`, `hard_exit`, `hard_reasons` at ~lines 2040–2044) with plan-derived state:

```python
    plan       = pos.get("plan") or {}
    verdict    = plan.get("verdict", "—")
    stop_level = plan.get("stop_level")
    trims      = plan.get("trims_fired") or []
    health     = plan.get("health") or {}
    dte        = plan.get("days_to_earnings")
    last_eval  = plan.get("last_eval") or "never"
```

Replace the `top_c` mapping (~2098) with:

```python
    top_c = {"SELL": "var(--bear)", "TRIM": "var(--wait)", "HOLD": "var(--bull)"}.get(verdict, "var(--muted)")
```

Replace the `hard_badges`/`soft_badges` block (~2105–2120) with a compact plan row (stop distance uses the live price when available, else the eval close):

```python
    ref_price = live_price if live_price is not None else plan.get("last_close")
    if stop_level is not None and ref_price:
        dist = (ref_price - stop_level) / ref_price
        stop_chip = f'STOP ${stop_level:,.2f} ({dist:+.1%})'
    else:
        stop_chip = "STOP —"
    trims_chip = "TRIMS " + ("+".join(trims) if trims else "none")
    health_n = health.get("bearish")
    health_chip = f'HEALTH {health_n}/4' if health_n is not None else "HEALTH —"
    plan_badges = (
        _signal_badge(stop_chip, verdict == "SELL")
        + " " + _signal_badge(trims_chip, bool(trims))
        + " " + _signal_badge(health_chip, (health_n or 0) >= 2)
    )
```

Keep the existing `earn_chip` block but source `dte` from the plan (variable already set above). Set `badges = plan_badges + (" " + earn_chip if earn_chip else "")`.

Replace the `exit_html` block (~2137–2155): SELL shows the red banner with the instruction, TRIM the amber one:

```python
    exit_html = ""
    if verdict == "SELL":
        exit_html = (
            f'<div style="margin-top:12px;padding:10px 14px;background:var(--bear-dim);'
            f'border:1px solid rgba(239,68,68,0.3);border-radius:var(--radius-sm);'
            f'animation:shockwave 0.55s ease-out 0.25s 1 both">'
            f'<span style="font-family:var(--mono);font-size:0.7rem;font-weight:700;'
            f'color:var(--bear);letter-spacing:0.08em">SELL — exit remaining position at next open</span>'
            f'</div>'
        )
    elif verdict == "TRIM":
        exit_html = (
            f'<div style="margin-top:12px;padding:10px 14px;background:rgba(245,158,11,0.08);'
            f'border:1px solid rgba(245,158,11,0.3);border-radius:var(--radius-sm)">'
            f'<span style="font-family:var(--mono);font-size:0.7rem;font-weight:700;'
            f'color:var(--wait);letter-spacing:0.08em">TRIM — sell 1/3 at next open ({_html.escape("+".join(trims))})</span>'
            f'</div>'
        )
```

Replace the grade line (~2177) with:

```python
        f'<div style="font-family:var(--mono);font-size:0.55rem;color:{top_c};letter-spacing:0.09em;margin-bottom:8px">VERDICT {_html.escape(verdict)} · evaluated {_html.escape(str(last_eval))}</div>'
```

Update the signature: `cached` still supplies `(signals, live_price, prev_close)`; `signals` is now always `{}` and unused — change unpack to `_, live_price, prev_close = cached ...` and drop the `_cached_position_data` fallback call (pass `cached` always from `_render_positions`; if `None`, use `({}, None, None)`).

- [ ] **Step 3: Update `_render_positions` ranking and summary**

Replace the `enriched` ranking block (~2230–2239) with plan-based ordering:

```python
    _order = {"SELL": 2, "TRIM": 1, "HOLD": 0}
    enriched = []
    for p in positions:
        v = (p.get("plan") or {}).get("verdict", "HOLD")
        enriched.append({**p, "_score": _order.get(v, 0), "_grade": v})
    enriched.sort(key=lambda x: x["_score"], reverse=True)

    exits = sum(1 for e in enriched if e["_grade"] == "SELL")
    exit_cls = "bear" if exits > 0 else ""
```

Check the summary-strip HTML below (~2254+) for references to old grade names ("STRONG EXIT") and update labels to SELL/TRIM/HOLD counts.

- [ ] **Step 4: Verify with Playwright/browser**

Run the app locally: `streamlit run app.py` (or existing project run command). Open the Positions page. Confirm: each card shows VERDICT badge, stop chip with distance, trims chip, health chip; SELL cards sort first; no Python errors in terminal.
Per user global CLAUDE.md this is a UI change — verify rendering in the browser before calling done.

- [ ] **Step 5: Run the FULL test suite**

Run: `/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m pytest tests/ -v`
Expected: all PASS (positions tests still green — storage functions untouched).

- [ ] **Step 6: Commit**

```bash
git add app_shared.py
git commit -m "feat(positions-ui): cards read standing exit plan verdicts, drop live signal grid"
```

---

### Task 8: Retire the old grade engine + docs

**Files:**
- Modify: `src/positions.py` (remove `compute_exit_signals`, `_stoch_bear_cross`, `_SOFT_WEIGHTS` and now-unused imports — ONLY if grep shows no remaining callers)
- Modify: `tests/test_positions.py` (remove tests of removed functions)
- Modify: `app_shared.py` (~line 1686: update the in-app "Exit tiers" doc text to describe the standing-plan system)

- [ ] **Step 1: Confirm no remaining callers**

Run: `grep -rn "compute_exit_signals\|_stoch_bear_cross\|_SOFT_WEIGHTS" --include="*.py" . | grep -v test_ | grep -v "src/positions.py"`
Expected: no output. If anything appears, fix that caller first — do not delete while referenced.

- [ ] **Step 2: Delete `compute_exit_signals`, `_stoch_bear_cross`, `_SOFT_WEIGHTS` from `src/positions.py`**

Also trim the `from src.factors import (...)` block to only what's still used in the file (check each name with grep). Keep `fetch_ohlcv`, `get_live_quote`, `days_to_next_earnings`, storage functions — the eval runner uses them.

- [ ] **Step 3: Remove dead tests**

In `tests/test_positions.py`, delete tests that import/exercise `compute_exit_signals` (e.g. `test_exit_signals_empty_df`, `test_exit_signals_too_short`, and any others found by: `grep -n "exit_signals\|soft_score\|grade" tests/test_positions.py`). Keep storage/add/remove tests.

- [ ] **Step 4: Update in-app doc text**

At `app_shared.py:1686` the docs describe "Exit tiers (mirrors the entry grade's veto + scored-points design)". Rewrite that block to describe: standing exit plans, ratcheting stops (initial 2×ATR / trailing peak−mult×ATR / 3-day 50d break), once-only trims (de-risk +2R or +20% with breakeven ratchet; blowoff extension), Friday-only tightener, SELL/TRIM/HOLD verdict evaluated at each close by the 4:30pm pipeline.

- [ ] **Step 5: Full suite + commit**

Run: `/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m pytest tests/ -v`
Expected: all PASS

```bash
git add src/positions.py tests/test_positions.py app_shared.py
git commit -m "refactor(positions): retire snapshot grade engine in favor of standing plans"
```

---

## Deferred / explicitly out of scope

- No automated order placement (system instructs; user executes).
- No intraday monitoring; closing prices only.
- Trader (Alpaca) daily email untouched — positions digest is its own email.
- The cloud Streamlit deploy needs no code beyond Task 7: it reads the same committed positions.json.

## Rollback

Each task is one commit. Plan state is additive to positions.json — reverting code leaves harmless extra keys. To fully reset state: `python3 -c "import json;ps=json.load(open('positions.json'));[p.pop('plan',None) for p in ps];json.dump(ps,open('positions.json','w'),indent=2)"`.
