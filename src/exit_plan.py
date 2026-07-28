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

    # already evaluated this bar → no-op (idempotent re-fire, live or replay)
    if plan.get("last_eval") == today:
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

    # ── SELL triggers ────────────────────────────────────────────────────
    sell_reason = None
    if price < plan["stop_floor"]:
        sell_reason = f"close {price:.2f} below max-loss floor {plan['stop_floor']:.2f}"
    elif price < plan["stop_level"]:
        sell_reason = f"close {price:.2f} below trailing stop {plan['stop_level']:.2f}"
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
                           "instruction": f"Sell {TRIM_FRACTION} of position at next open.",})

    plan["verdict"] = "TRIM" if events else "HOLD"
    plan["last_eval"], plan["last_close"] = today, round(price, 4)
    return pos, events


def weekly_health(df: pd.DataFrame, spy_close: pd.Series | None) -> dict:
    """Weekly-bar health check. Returns
    {"bearish": n, "parts": [...], "errors": [...], "weeks": n, "asof": date}.

    Four checks on W-FRI resampled bars: MACD state, 13-week RS vs SPY,
    OBV slope, ADX direction (-DI dominant and ADX falling). Each bearish
    check adds 1. Used ONLY to tighten trail_mult — never to sell.

    "weeks" records how many weekly bars were actually evaluated, so callers
    can distinguish "checked, all clear" (weeks >= 15, bearish 0) from
    "never checked" (weeks < 15, bearish 0 by construction — short history,
    e.g. a recent IPO). "errors" names any check that threw and was skipped,
    so a swallowed exception reads as "unknown" rather than silently
    counting toward a healthy score.
    """
    parts: list[str] = []
    errors: list[str] = []
    wk = df.resample("W-FRI").agg(
        {"high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()
    asof = df.index[-1].strftime("%Y-%m-%d") if len(df) else None
    if len(wk) < 15:
        return {"bearish": 0, "parts": [], "errors": [], "weeks": len(wk), "asof": asof}

    try:
        if macd_state(wk["close"]) in ("bearish", "bearish_cross"):
            parts.append("macd")
    except Exception:
        errors.append("macd")
    try:
        if spy_close is not None:
            spy_wk = spy_close.resample("W-FRI").last().dropna()
            if len(spy_wk) >= 14 and len(wk) >= 14:
                stk = float(wk["close"].iloc[-1] / wk["close"].iloc[-14] - 1)
                spy = float(spy_wk.iloc[-1] / spy_wk.iloc[-14] - 1)
                if stk < spy:
                    parts.append("rs")
    except Exception:
        errors.append("rs")
    try:
        slp = obv_slope(wk["close"], wk["volume"], window=10)
        if not math.isnan(slp) and slp < 0:
            parts.append("obv")
    except Exception:
        errors.append("obv")
    try:
        adx_now, pdi, ndi = directional_indicators(wk["high"], wk["low"], wk["close"])
        if len(wk) >= 20:
            adx_past, _, _ = directional_indicators(
                wk["high"].iloc[:-4], wk["low"].iloc[:-4], wk["close"].iloc[:-4])
            if (not math.isnan(adx_now) and not math.isnan(adx_past)
                    and adx_past > adx_now and ndi > pdi):
                parts.append("adx")
    except Exception:
        errors.append("adx")
    return {"bearish": len(parts), "parts": parts, "errors": errors, "weeks": len(wk), "asof": asof}


def apply_weekly_tightener(plan: dict, health: dict) -> dict:
    """If ≥2 weekly checks bearish, step trail_mult down one notch (floor 2.0).
    Never loosens. Stores health on the plan either way."""
    plan["health"] = health
    if health.get("bearish", 0) >= 2:
        steps = [m for m in TRAIL_MULT_STEPS if m < plan["trail_mult"]]
        if steps:
            plan["trail_mult"] = steps[0]
    return plan


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


def _weekly_rsi(close: pd.Series) -> float | None:
    """RSI(14) on weekly (Friday) closes. None if under 15 weeks of data or
    the series has no variance (a perfectly constant series carries no
    signal, so any RSI value it produces — degenerate 100 included — isn't
    a meaningful overbought read)."""
    wk = close.resample("W-FRI").last().dropna()
    if len(wk) < 15 or wk.std() < 1e-9:
        return None
    val = rsi_14(wk)
    return None if math.isnan(val) else float(val)


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
