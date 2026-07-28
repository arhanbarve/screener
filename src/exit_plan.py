"""Standing exit plan engine.

Per-position exit plans with one-way ratcheting stops, once-only trim rungs,
and final SELL/TRIM/HOLD verdicts. All triggers anchor to the position's own
history (entry, R, peak) or need multi-day confirmation, so verdicts cannot
flip on ordinary market-wide moves.

Spec: docs/superpowers/specs/2026-07-28-standing-exit-plan-design.md
"""
import math
from datetime import date

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
        "verdict_reason": None,
        "health": None,
        "days_to_earnings": None,
        "last_eval": None,
        "last_close": None,
        "pending_notify": [],
        "tightened_asof": None,
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

    # already evaluated this bar (idempotent re-fire) OR this bar is OLDER
    # than the last one we evaluated (stale/out-of-order data) → no-op either
    # way. String dates compare lexicographically same as chronologically.
    last_eval = plan.get("last_eval")
    if last_eval is not None and last_eval >= today:
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
        cleared = replay and price > plan["stop_level"]
        if cleared:
            # Held through it; plan resumes — but do NOT early-return here.
            # Clearing one (price-based) SELL condition says nothing about
            # whether a *different* SELL condition (trend-break streak, or a
            # fresh stop breach) also holds on this same bar. Fall through to
            # the normal SELL trigger checks below so this bar only ends
            # HOLD when no SELL condition is true for it.
            plan["verdict"] = "HOLD"
            plan["verdict_reason"] = None
        else:
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
        plan["verdict_reason"] = sell_reason
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
    plan["verdict_reason"] = "; ".join(e["reason"] for e in events) if events else None
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


_MIN_HEALTH_WEEKS = 15  # weekly_health() treats fewer weeks as "never checked"


def health_label(health: dict | None) -> str:
    """Weekly-health score as plain text, shared by the digest email
    (src/exit_alerts.py) and the positions page (app_shared.py) so the two
    surfaces can never disagree about what a position's health chip says.

    '?/4' — no health recorded, or too little history for the weekly check
    to have run yet (weeks < _MIN_HEALTH_WEEKS): unknown, not a clean 0/4.
    'N/4' — scored cleanly, N of 4 checks bearish.
    'N/4*' — scored, but one or more checks threw (health["errors"] is
    non-empty): the asterisk flags an unknown-quality read WITHOUT hiding
    the real bearish count that the other checks did produce.
    """
    if not health:
        return "?/4"
    weeks = health.get("weeks") or 0
    if weeks < _MIN_HEALTH_WEEKS:
        return "?/4"
    bearish = health.get("bearish", 0)
    errors = health.get("errors") or []
    return f"{bearish}/4*" if errors else f"{bearish}/4"


def bootstrap_position(pos: dict, df: pd.DataFrame) -> dict:
    """One-time backfill: init plan at entry, replay history day-by-day.
    Historical events are discarded (trims get marked, no emails) — the user
    must never be emailed a trim/sell that fired months ago. Replay mode
    clears historical SELLs the user held through.

    The final replayed bar's events are kept, transiently, on
    `pos["_bootstrap_final_events"]` (not part of the persisted plan) so the
    caller (run_daily_eval) can tell whether today's standing verdict is
    freshly actionable — as opposed to a stale replay artifact — without
    resurrecting any other historical event.
    """
    entry_ts = pd.Timestamp(pos["entry_date"])
    upto_entry = df[df.index <= entry_ts]
    if upto_entry.empty:
        upto_entry = df.iloc[:1]
    pos["plan"] = init_plan(upto_entry, float(pos["entry_price"]))
    after = df.index[df.index > entry_ts]
    final_events: list[dict] = []
    for ts in after:
        pos, final_events = evaluate_day(pos, df.loc[:ts], replay=True)
    pos["_bootstrap_final_events"] = final_events
    return pos


def _bootstrap_catchup_events(pos: dict, final_replay_events: list[dict]) -> list[dict]:
    """Turn a freshly-bootstrapped plan's standing verdict into the action
    event(s) needed to actually notify the user, without resurrecting any
    *historical* event bootstrap correctly discarded.

    SELL: reconstructed from the plan's own persisted numbers (last_close vs
    stop_floor/stop_level/below_50d_streak). These describe the position's
    CURRENT state — they don't depend on which bar first crossed them — so
    they're honestly reconstructable even when the SELL verdict has been
    sitting terminal since a bar many days before the final one (the common
    case: bootstrap's replay never cleared it).

    TRIM: NOT reconstructed from persisted numbers — trims_fired is a state
    record with no per-rung date, so guessing "why" from current numbers
    could misattribute an old, already-settled trim to today. Instead this
    uses `final_replay_events`, the actual events evaluate_day produced on
    the *last* replayed bar specifically. That's reliable because (unlike
    SELL) TRIM/HOLD is recomputed from scratch every single day — a plan
    only ends bootstrap as TRIM if a rung fired on that literal final bar,
    so final_replay_events, when non-empty, is exactly that rung's own event.
    """
    plan = pos["plan"]
    verdict = plan["verdict"]
    events: list[dict] = []

    if verdict == "SELL":
        last_close = plan["last_close"]
        stop_floor = plan["stop_floor"]
        stop_level = plan["stop_level"]
        streak = plan["below_50d_streak"]
        if last_close < stop_floor:
            breach = f"below the max-loss floor {stop_floor:.2f}"
        elif last_close < stop_level:
            breach = f"below the trailing stop {stop_level:.2f}"
        else:
            breach = f"the {streak}th straight close below the 50-day SMA"
        events.append({
            "ticker": pos["ticker"], "type": "SELL",
            "reason": (f"standing exit plan just established from history: "
                       f"last close {last_close:.2f} is already {breach}"),
            "instruction": "Sell entire remaining position at next open.",
        })
    elif verdict == "TRIM":
        for fe in final_replay_events:
            events.append({
                "ticker": pos["ticker"], "type": fe["type"],
                "reason": f"standing exit plan just established from history: {fe['reason']}",
                "instruction": fe["instruction"],
            })

    return events


def run_daily_eval(send_emails: bool = True, today: date | None = None) -> dict:
    """Evaluate every position on today's close. Saves positions.json,
    applies the weekly tightener once per ISO week, sends action + digest
    emails. Returns {"events": [...], "evaluated": n, "skipped": [...],
    "stale": [...], "errored": [...]}.

    `today` overrides the wall-clock date used for the staleness and
    week-closed checks below; defaults to the real current date. Tests pass
    it explicitly so synthetic historical fixtures aren't flagged stale.

    Per-ticker failures (bad fetch, a throwing indicator, etc.) are caught
    and recorded in "errored" rather than aborting the whole run — one bad
    ticker must never cost every other position its daily evaluation and
    email. A bar more than 4 calendar days behind `today` is distrusted and
    the ticker is parked in "stale" with its plan left untouched, same as
    "skipped", rather than evaluated as if it were current.
    """
    from src.positions import load_positions, save_positions, fetch_ohlcv, days_to_next_earnings

    positions = load_positions()
    if not positions:
        return {"events": [], "evaluated": 0, "skipped": [], "stale": [], "errored": []}

    today = today or date.today()
    spy_df = fetch_ohlcv("SPY", days=600)
    spy_close = spy_df["close"] if not spy_df.empty else None

    all_events: list[dict] = []
    skipped: list[str] = []
    stale: list[dict] = []
    errored: list[dict] = []
    for pos in positions:
        try:
            df = fetch_ohlcv(pos["ticker"], days=600)
            if df.empty or len(df) < 2:
                skipped.append(pos["ticker"])   # keep yesterday's state, never fabricate
                continue

            last_bar_date = df.index[-1].date()
            if (today - last_bar_date).days > 4:
                stale.append({"ticker": pos["ticker"], "bar_date": last_bar_date.strftime("%Y-%m-%d")})
                continue   # bar too old to trust — leave the plan untouched

            if "plan" not in pos or pos["plan"] is None:
                pos = bootstrap_position(pos, df)
                final_replay_events = pos.pop("_bootstrap_final_events", [])
                print(f"[exit-plan] bootstrapped {pos['ticker']}: "
                      f"verdict={pos['plan']['verdict']} trims={pos['plan']['trims_fired']}")
                catchup_events = _bootstrap_catchup_events(pos, final_replay_events)
                if catchup_events:
                    pos["plan"].setdefault("pending_notify", []).extend(catchup_events)
                    all_events.extend(catchup_events)
                    # Keep the page's verdict_reason in lockstep with the
                    # catch-up wording the user is emailed — otherwise the
                    # replay-derived reason (from whichever historical bar
                    # first tripped it) could read differently than what the
                    # email says just went out.
                    pos["plan"]["verdict_reason"] = "; ".join(e["reason"] for e in catchup_events)

            pos, events = evaluate_day(pos, df)

            # Weekly tightener: fire once per ISO week, on whichever weekday
            # turns out to be that week's final trading bar. A bar is
            # "closed" for its week either because it's a Friday, or because
            # real time has already reached Friday-or-later (same week or a
            # later one) without a newer bar showing up — i.e. the rest of
            # the week was a holiday. tightened_asof (keyed by ISO year+week,
            # not weekday) blocks a duplicate/retry run from double-stepping.
            iso_last = last_bar_date.isocalendar()[:2]
            iso_today = today.isocalendar()[:2]
            week_closed = iso_today != iso_last or today.weekday() >= 4
            if week_closed:
                week_key = f"{iso_last[0]}-W{iso_last[1]:02d}"
                if pos["plan"].get("tightened_asof") != week_key:
                    pos["plan"] = apply_weekly_tightener(pos["plan"], weekly_health(df, spy_close))
                    pos["plan"]["tightened_asof"] = week_key

            pos["plan"]["days_to_earnings"] = days_to_next_earnings(pos["ticker"])
            if events:
                pos["plan"].setdefault("pending_notify", []).extend(events)
            all_events.extend(events)
        except Exception as e:
            errored.append({"ticker": pos["ticker"], "error": repr(e)})
            print(f"[exit-plan] ERROR evaluating {pos['ticker']}: {e!r}")
            continue

    save_positions(positions)   # ratchets/peaks are real regardless of email outcome

    result = {
        "events": all_events,
        "evaluated": len(positions) - len(skipped) - len(stale) - len(errored),
        "skipped": skipped,
        "stale": stale,
        "errored": errored,
    }

    if send_emails:
        # Union of this run's fresh events (already folded into pending_notify
        # above) and any leftovers a previous run failed to send — a leftover
        # entry means the previous send failed, so it must go out now too.
        notify_positions = [p for p in positions if (p.get("plan") or {}).get("pending_notify")]
        combined_events: list[dict] = []
        seen = set()
        for p in notify_positions:
            for e in p["plan"]["pending_notify"]:
                key = (e.get("ticker"), e.get("type"), e.get("reason"))
                if key not in seen:
                    seen.add(key)
                    combined_events.append(e)
        try:
            from src.exit_alerts import send_action_alert, send_daily_digest
            if combined_events:
                send_action_alert(combined_events, positions)
            send_daily_digest(positions, skipped, stale, errored)
        except Exception as e:
            # State is already saved above; leave pending_notify intact so
            # the next run retries these instructions instead of losing them.
            print(f"[exit-plan] EMAIL SEND FAILED: {e!r}")
        else:
            if notify_positions:
                for p in notify_positions:
                    p["plan"]["pending_notify"] = []
                save_positions(positions)
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
