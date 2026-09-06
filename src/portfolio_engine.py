"""Deterministic portfolio engine for the Alpaca paper book.

Turns tonight's inputs into a list of order *legs* with reasons:

    screen   output/screen_latest.json (health, rows, ranking_tail)
    book     Alpaca account + positions + open orders
    plans    per-position exit plans (data/alpaca/plans.json, src.paper_stops)
    regime   src.spy_analysis.compute_market_stress_overlay()
    policy   config.yaml `trader` block (risk rules)
    state    trading/engine_state.json (peak equity, breaker, rank history)

Everything here is pure: no network, no clock reads, no files. src.trader_cli
gathers the inputs, calls build_plan, and executes the legs; the LLM reviewer
may SKIP or DOWNSIZE legs marked `overridable`, never the others.

Design rules (decided 2026-09-04, "aggressive" posture):

  * Two sleeves. Alpha = screener picks, sized by regime; core = SPY holds
    whatever equity the alpha sleeve does not, above a cash buffer. The book
    is never parked in cash by accident — the measured drag on this account
    was -4pp of absence from the tape, not bad entries.
  * Risk-budgeted sizing: a new position risks `risk_per_trade` of equity at
    its max-loss floor (entry - 2xATR14, the same floor the resting stop
    uses). A tight stop means a bigger position, capped by `name_cap`.
  * Exits come from the standing exit plan (SELL / TRIM verdicts), from
    falling out of the ranking (`exit_band` for `rotate_out_runs` screens),
    and from an earnings print inside `earnings_trim_days` on an oversized
    position. None of these can be skipped by the reviewer except the
    rotate-out, which may be deferred one session.
  * Drawdown breaker: equity below peak x (1 - breaker_drawdown) halves the
    alpha target until equity recovers to peak x (1 - breaker_recover).
  * Nothing new is bought from a screen whose health.ok is False or that is
    older than the session being decided for. Exits still happen.

Leg ordering is the execution order: sells first so their (haircut) proceeds
can fund the buys at the same open.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from datetime import date

DEFAULT_POLICY: dict = {
    "risk_per_trade": 0.0125,          # of equity, at the max-loss floor
    "name_cap": 0.12,                  # max weight of one alpha name
    "sector_cap": 0.35,                # max weight of one sector (alpha sleeve)
    "max_positions": 8,                # alpha positions (excludes core, dust)
    "alpha_target": {"NORMAL": 0.70, "CAUTION": 0.40, "STRESS": 0.15, "UNKNOWN": 0.40},
    "core_symbol": "SPY",
    "cash_buffer": 0.05,               # kept uninvested
    "core_band": 0.03,                 # of equity; rebalance core only outside it
    "breaker_drawdown": 0.15,
    "breaker_recover": 0.07,
    "earnings_no_entry_days": 5,
    "earnings_trim_days": 2,
    "earnings_max_weight": 0.06,
    "entry_band": 20,
    "exit_band": 35,
    "rotate_out_runs": 3,
    "min_entry_notional": 500.0,
    "min_stop_distance_pct": 0.02,     # a stop closer than this is noise
    "max_stop_distance_pct": 0.15,     # a stop further than this caps the risk unit
    "fallback_stop_pct": 0.08,         # when atr_14 is missing (matches exit_plan)
    "add_trigger_r": 1.0,
    "add_fraction": 0.5,               # of the original risk budget
    "dust_notional": 50.0,
    "min_price": 5.0,
    "entry_grades": ["OK", "STRONG"],
    "avoid_signals": ["avoid"],
    "fund_floor": 0.30,
    "sell_haircut": 0.97,              # fraction of sell proceeds assumed available
    "trim_fraction": 1 / 3,
    "rank_history_keep": 6,
}

# Execution order by kind. Sells first so their proceeds fund the buys.
KIND_ORDER = ["exit", "trim", "earnings_trim", "rotate_out", "dust", "core_sell",
              "entry", "add", "core_buy"]
RISK_REDUCING_KINDS = {"exit", "trim", "earnings_trim", "rotate_out", "core_sell"}
NON_OVERRIDABLE_KINDS = {"exit", "trim", "earnings_trim"}


@dataclass
class Leg:
    id: str
    kind: str
    symbol: str
    side: str
    qty: float | None
    notional: float | None
    reason: str
    risk_reducing: bool
    overridable: bool
    requires_rth: bool = False
    cancel_stop_first: bool = False
    source: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        if d["qty"] is not None:
            d["qty"] = round(d["qty"], 6)
        if d["notional"] is not None:
            d["notional"] = round(d["notional"], 2)
        return d


class EngineError(ValueError):
    """The inputs cannot produce a safe plan (e.g. no equity figure)."""


# ── helpers ───────────────────────────────────────────────────────────────────

def _f(v, default: float = 0.0) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(x) else x


def policy_from_config(cfg: dict) -> dict:
    """DEFAULT_POLICY overlaid with config.yaml's `trader` block."""
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in DEFAULT_POLICY.items()}
    for k, v in ((cfg or {}).get("trader") or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k].update(v)
        else:
            out[k] = v
    return out


def screen_usable(screen: dict | None, target_date: str, latest_session: str) -> tuple[bool, str]:
    """A screen may feed new entries only if it passed its health gate and is
    dated the latest completed session (the one whose close the evening
    session reads). `latest_session` is the most recent trading day ≤ today."""
    if not screen:
        return False, "no screen"
    health = screen.get("health") or {}
    if health.get("ok") is not True:
        return False, "screen health not ok: " + "; ".join(health.get("reasons") or ["unknown"])
    d = str(screen.get("date") or "")
    if d < latest_session:
        return False, f"screen dated {d} is older than the last session {latest_session}"
    if not screen.get("rows"):
        return False, "screen has no rows (stress regime or empty)"
    return True, "ok"


def rank_lookup(screen: dict | None) -> dict[str, int]:
    """ticker -> rank from ranking_tail (top ~100) falling back to rows."""
    out: dict[str, int] = {}
    for r in (screen or {}).get("ranking_tail") or []:
        t = str(r.get("ticker") or "").upper()
        if t and r.get("rank"):
            out[t] = int(r["rank"])
    for r in (screen or {}).get("rows") or []:
        t = str(r.get("ticker") or "").upper()
        if t and r.get("rank") and t not in out:
            out[t] = int(r["rank"])
    return out


def update_breaker(state: dict, equity: float, policy: dict) -> dict:
    """Peak-equity tracking with hysteresis. Returns the breaker block."""
    peak = max(_f(state.get("peak_equity"), 0.0), equity)
    dd = 0.0 if peak <= 0 else 1.0 - equity / peak
    active = bool(state.get("breaker_active", False))
    if active and dd <= policy["breaker_recover"]:
        active = False
    elif not active and dd >= policy["breaker_drawdown"]:
        active = True
    return {"active": active, "drawdown": round(dd, 6), "peak_equity": round(peak, 2)}


def update_rank_history(state: dict, held: list[str], ranks: dict[str, int],
                        screen_date: str | None, keep: int) -> dict:
    """Append today's rank (or None) for every held alpha name; one entry per
    screen date so a rerun does not double-count."""
    hist = dict(state.get("rank_history") or {})
    if not screen_date:
        return hist
    for sym in held:
        entries = [e for e in (hist.get(sym) or []) if e.get("date") != screen_date]
        entries.append({"date": screen_date, "rank": ranks.get(sym)})
        hist[sym] = entries[-keep:]
    # Forget names no longer held.
    return {s: v for s, v in hist.items() if s in held}


def consecutive_out_of_band(entries: list[dict], exit_band: int) -> int:
    """How many most-recent screens in a row the name ranked worse than
    exit_band (None = not in the top ~100 at all)."""
    n = 0
    for e in reversed(entries or []):
        r = e.get("rank")
        if r is None or int(r) > exit_band:
            n += 1
        else:
            break
    return n


def stop_distance(price: float, atr: float | None, policy: dict) -> tuple[float, float, str]:
    """(floor, distance, note) for a new entry: entry − 2×ATR14 clamped to
    [min, max] stop distance; percentage fallback without ATR."""
    if atr is None or not (atr == atr) or atr <= 0:
        dist = price * policy["fallback_stop_pct"]
        note = "fallback 8% stop (no ATR)"
    else:
        dist = 2.0 * atr
        note = "2×ATR14"
    max_d = price * policy["max_stop_distance_pct"]
    if dist > max_d:
        dist, note = max_d, note + f", capped at {policy['max_stop_distance_pct']:.0%}"
    return price - dist, dist, note


# ── the plan ──────────────────────────────────────────────────────────────────

def build_plan(
    screen: dict | None,
    book: dict,
    plans: dict,
    regime: dict | None,
    policy: dict,
    state: dict,
    today: date,
    target_date: str,
    latest_session: str,
    earnings: dict[str, int | None] | None = None,
    assets: dict[str, dict] | None = None,
) -> tuple[dict, dict]:
    """Return (plan, new_state). Pure.

    book: {"account": {...}, "positions": [...], "open_orders": [...]}
    plans: {ticker: {"entry_price", "risk_R", "stop_floor", "stop_level",
                     "verdict", "verdict_reason", "trims_fired", "last_close"}}
    earnings: {ticker: days_to_earnings or None} for held names
    assets: {alpaca_symbol: {"fractionable": bool}} or None
    """
    account = book.get("account") or {}
    equity = _f(account.get("equity"))
    cash = _f(account.get("cash"))
    if equity <= 0:
        raise EngineError("account equity missing or non-positive")
    core_sym = policy["core_symbol"].upper()
    earnings = earnings or {}
    legs: list[Leg] = []
    rejected: list[dict] = []
    warnings: list[str] = []
    counter = iter(range(1, 10_000))

    def new_leg(**kw) -> Leg:
        leg = Leg(id=f"L{next(counter)}", **kw)
        legs.append(leg)
        return leg

    # ── positions ────────────────────────────────────────────────────────────
    positions = {}
    for p in book.get("positions") or []:
        sym = str(p.get("symbol") or "").upper()
        if not sym:
            continue
        positions[sym] = {
            "symbol": sym,
            "qty": _f(p.get("qty")),
            "mv": _f(p.get("market_value")),
            "price": _f(p.get("current_price")),
            "avg_cost": _f(p.get("avg_entry_price")),
            "qty_available": _f(p.get("qty_available"), _f(p.get("qty"))),
        }
    core_pos = positions.get(core_sym)
    dust = {s: p for s, p in positions.items() if s != core_sym and p["mv"] < policy["dust_notional"]}
    alpha = {s: p for s, p in positions.items() if s != core_sym and s not in dust}

    # Open non-stop orders block re-entry in the same symbol tonight.
    pending_syms = {str(o.get("symbol") or "").upper() for o in (book.get("open_orders") or [])
                    if o.get("type") != "stop"}
    stop_syms = {str(o.get("symbol") or "").upper() for o in (book.get("open_orders") or [])
                 if o.get("type") == "stop"}

    # ── regime, breaker, targets ─────────────────────────────────────────────
    regime_name = str((regime or {}).get("regime") or "UNKNOWN").upper()
    if regime_name not in policy["alpha_target"]:
        regime_name = "UNKNOWN"
    breaker = update_breaker(state, equity, policy)
    alpha_target = float(policy["alpha_target"][regime_name])
    if breaker["active"]:
        alpha_target *= 0.5
        warnings.append(f"drawdown breaker active ({breaker['drawdown']:.1%} from peak): alpha target halved")

    # ── screen ───────────────────────────────────────────────────────────────
    usable, why = screen_usable(screen, target_date, latest_session)
    if not usable:
        warnings.append(f"no new entries: {why}")
    ranks = rank_lookup(screen)
    rows = {str(r.get("ticker") or "").upper(): r for r in (screen or {}).get("rows") or []}
    screen_date = (screen or {}).get("date") if screen else None
    rank_history = update_rank_history(state, sorted(alpha), ranks, screen_date if usable else None,
                                       int(policy["rank_history_keep"]))

    # ── exits from the standing exit plan ────────────────────────────────────
    exiting: set[str] = set()
    sell_proceeds = 0.0
    trims_executed = dict(state.get("trims_executed") or {})
    for sym, pos in sorted(alpha.items()):
        plan = plans.get(sym) or {}
        verdict = str(plan.get("verdict") or "HOLD").upper()
        if verdict == "SELL":
            new_leg(kind="exit", symbol=sym, side="sell", qty=pos["qty"], notional=None,
                    reason=f"exit plan SELL: {plan.get('verdict_reason') or 'stop or trend break'}",
                    risk_reducing=True, overridable=False, cancel_stop_first=sym in stop_syms,
                    source={"verdict": verdict, "stop_level": plan.get("stop_level"),
                            "stop_floor": plan.get("stop_floor"), "last_close": plan.get("last_close")})
            exiting.add(sym)
            sell_proceeds += pos["mv"]
            continue
        fired = [t for t in (plan.get("trims_fired") or []) if t not in (trims_executed.get(sym) or [])]
        if verdict == "TRIM" and fired:
            qty = pos["qty"] * float(policy["trim_fraction"])
            new_leg(kind="trim", symbol=sym, side="sell", qty=qty, notional=None,
                    reason=f"exit plan TRIM ({', '.join(fired)}): {plan.get('verdict_reason') or ''}".strip(),
                    risk_reducing=True, overridable=False, cancel_stop_first=sym in stop_syms,
                    source={"rungs": fired})
            sell_proceeds += qty * pos["price"]
            trims_executed[sym] = sorted(set(trims_executed.get(sym) or []) | set(fired))

    # ── earnings trims ───────────────────────────────────────────────────────
    for sym, pos in sorted(alpha.items()):
        if sym in exiting:
            continue
        dte = earnings.get(sym)
        if dte is None or dte > policy["earnings_trim_days"]:
            continue
        weight = pos["mv"] / equity
        if weight > policy["earnings_max_weight"]:
            target_mv = policy["earnings_max_weight"] * equity
            qty = pos["qty"] * (1 - target_mv / pos["mv"])
            new_leg(kind="earnings_trim", symbol=sym, side="sell", qty=qty, notional=None,
                    reason=(f"earnings in {dte}d at {weight:.1%} of equity; trim to "
                            f"{policy['earnings_max_weight']:.0%}"),
                    risk_reducing=True, overridable=False, cancel_stop_first=sym in stop_syms,
                    source={"days_to_earnings": dte, "weight": round(weight, 4)})
            sell_proceeds += qty * pos["price"]

    # ── rotate out of names the screener has dropped ─────────────────────────
    if usable:
        for sym, pos in sorted(alpha.items()):
            if sym in exiting:
                continue
            n_out = consecutive_out_of_band(rank_history.get(sym) or [], int(policy["exit_band"]))
            if n_out >= int(policy["rotate_out_runs"]):
                new_leg(kind="rotate_out", symbol=sym, side="sell", qty=pos["qty"], notional=None,
                        reason=f"ranked below {policy['exit_band']} (or unranked) for {n_out} consecutive screens",
                        risk_reducing=True, overridable=True, cancel_stop_first=sym in stop_syms,
                        source={"rank_history": rank_history.get(sym)})
                exiting.add(sym)
                sell_proceeds += pos["mv"]

    # ── dust ─────────────────────────────────────────────────────────────────
    for sym, pos in sorted(dust.items()):
        if pos["qty"] <= 0:
            continue
        new_leg(kind="dust", symbol=sym, side="sell", qty=pos["qty"], notional=None,
                reason=f"dust position (${pos['mv']:.2f}) — market sell in regular hours",
                risk_reducing=False, overridable=True, requires_rth=True,
                cancel_stop_first=sym in stop_syms)

    # ── exposure after sells ─────────────────────────────────────────────────
    alpha_mv_after = sum(p["mv"] for s, p in alpha.items() if s not in exiting)
    for leg in legs:
        if leg.kind in ("trim", "earnings_trim"):
            alpha_mv_after -= leg.qty * alpha[leg.symbol]["price"]
    n_alpha_after = len([s for s in alpha if s not in exiting])
    sector_mv: dict[str, float] = {}
    for s, p in alpha.items():
        if s in exiting:
            continue
        sec = str((rows.get(s) or {}).get("sector") or "Unknown")
        sector_mv[sec] = sector_mv.get(sec, 0.0) + p["mv"]

    cash_after = cash + sell_proceeds * float(policy["sell_haircut"])
    alpha_budget = alpha_target * equity - alpha_mv_after
    risk_dollars = equity * float(policy["risk_per_trade"])
    buys_total = 0.0

    def room_for(sym: str, notional: float, sector: str) -> tuple[float, str | None]:
        """Cap a buy by name cap, sector cap, alpha budget and cash."""
        held_mv = alpha.get(sym, {}).get("mv", 0.0) if sym not in exiting else 0.0
        caps = [
            (policy["name_cap"] * equity - held_mv, "name cap"),
            (policy["sector_cap"] * equity - sector_mv.get(sector, 0.0), f"sector cap ({sector})"),
            (alpha_budget - buys_total, "alpha budget"),
            (cash_after - buys_total, "cash"),
        ]
        binding = None
        for cap, label in caps:
            if cap < notional:
                notional, binding = cap, label
        return max(notional, 0.0), binding

    # ── entries ──────────────────────────────────────────────────────────────
    if usable:
        candidates = sorted(
            [r for r in rows.values() if int(r.get("rank") or 999) <= int(policy["entry_band"])],
            key=lambda r: int(r.get("rank") or 999))
        for r in candidates:
            sym = str(r.get("ticker") or "").upper()
            asym = str(r.get("alpaca_symbol") or sym).upper()
            price = _f(r.get("price"))
            reason_skip = None
            if sym in alpha or sym in exiting or sym == core_sym:
                continue                                   # adds handled below
            if str(r.get("entry") or "") not in policy["entry_grades"]:
                reason_skip = f"entry grade {r.get('entry')}"
            elif str(r.get("entry_signal") or "") in policy["avoid_signals"]:
                reason_skip = "news signal avoid"
            elif price < policy["min_price"]:
                reason_skip = f"price {price:.2f} below ${policy['min_price']:g}"
            elif r.get("fund_score") is not None and _f(r.get("fund_score"), float("nan")) == _f(r.get("fund_score"), float("nan")) \
                    and _f(r.get("fund_score")) < policy["fund_floor"]:
                reason_skip = f"fund_score {_f(r.get('fund_score')):.2f} below floor"
            elif assets is not None and asym not in assets:
                reason_skip = "not tradable on Alpaca"
            elif asym in pending_syms:
                reason_skip = "an order is already resting for this symbol"
            elif r.get("days_to_earnings") is not None and _f(r.get("days_to_earnings"), 999) <= policy["earnings_no_entry_days"]:
                reason_skip = f"earnings in {int(_f(r.get('days_to_earnings')))}d"
            elif n_alpha_after >= int(policy["max_positions"]):
                reason_skip = f"max positions ({policy['max_positions']}) reached"
            if reason_skip:
                rejected.append({"symbol": sym, "rank": r.get("rank"), "reason": reason_skip})
                continue

            floor, dist, note = stop_distance(price, r.get("atr_14"), policy)
            if dist / price < policy["min_stop_distance_pct"]:
                rejected.append({"symbol": sym, "rank": r.get("rank"), "reason": f"stop too tight ({dist / price:.1%})"})
                continue
            qty = risk_dollars / dist
            notional = qty * price
            sector = str(r.get("sector") or "Unknown")
            notional, binding = room_for(sym, notional, sector)
            fractionable = True if assets is None else bool((assets.get(asym) or {}).get("fractionable", False))
            if not fractionable:
                qty_whole = math.floor(notional / price)
                notional = qty_whole * price
            if notional < policy["min_entry_notional"]:
                rejected.append({"symbol": sym, "rank": r.get("rank"),
                                 "reason": f"sized below ${policy['min_entry_notional']:g}" + (f" ({binding})" if binding else "")})
                continue
            qty = notional / price
            new_leg(kind="entry", symbol=asym, side="buy", qty=None, notional=notional,
                    reason=(f"rank {r.get('rank')}, entry {r.get('entry')}, composite {_f(r.get('composite_final', r.get('composite'))):.3f}, "
                            f"fund {_f(r.get('fund_score'), float('nan')):.2f}; risk ${risk_dollars:,.0f} at floor "
                            f"{floor:.2f} ({note})" + (f"; capped by {binding}" if binding else "")),
                    risk_reducing=False, overridable=True,
                    source={"ticker": sym, "rank": r.get("rank"), "price": price, "floor": round(floor, 4),
                            "atr_14": r.get("atr_14"), "sector": sector, "qty_est": round(qty, 4),
                            "days_to_earnings": r.get("days_to_earnings"), "conviction": r.get("conviction")})
            buys_total += notional
            n_alpha_after += 1
            sector_mv[sector] = sector_mv.get(sector, 0.0) + notional

        # ── adds to winners ──────────────────────────────────────────────────
        for sym, pos in sorted(alpha.items()):
            if sym in exiting:
                continue
            plan = plans.get(sym) or {}
            r = rows.get(sym)
            if not r or int(r.get("rank") or 999) > int(policy["entry_band"]):
                continue
            if str(r.get("entry") or "") not in policy["entry_grades"]:
                continue
            risk_R = _f(plan.get("risk_R"))
            entry = _f(plan.get("entry_price"))
            last = _f(plan.get("last_close"), pos["price"])
            if risk_R <= 0 or entry <= 0:
                continue
            gain_R = (last - entry) / risk_R
            if gain_R < policy["add_trigger_r"]:
                continue
            if _f(r.get("days_to_earnings"), 999) <= policy["earnings_no_entry_days"]:
                continue
            stop = _f(plan.get("stop_level"))
            dist = last - stop if stop > 0 else 0.0
            if dist <= 0 or dist / last < policy["min_stop_distance_pct"]:
                continue
            notional = policy["add_fraction"] * risk_dollars / dist * last
            sector = str(r.get("sector") or "Unknown")
            notional, binding = room_for(sym, notional, sector)
            if notional < policy["min_entry_notional"]:
                continue
            asym = str(r.get("alpaca_symbol") or sym).upper()
            new_leg(kind="add", symbol=asym, side="buy", qty=None, notional=notional,
                    reason=(f"add: +{gain_R:.1f}R with rank {r.get('rank')} and entry {r.get('entry')}; "
                            f"half risk at trailing stop {stop:.2f}" + (f"; capped by {binding}" if binding else "")),
                    risk_reducing=False, overridable=True, cancel_stop_first=asym in stop_syms,
                    source={"gain_R": round(gain_R, 3), "stop_level": stop, "rank": r.get("rank")})
            buys_total += notional
            sector_mv[sector] = sector_mv.get(sector, 0.0) + notional

    # ── core sleeve ──────────────────────────────────────────────────────────
    core_mv = core_pos["mv"] if core_pos else 0.0
    core_target = max(0.0, equity * (1.0 - policy["cash_buffer"]) - (alpha_mv_after + buys_total))
    diff = core_target - core_mv
    band = max(policy["core_band"] * equity, policy["min_entry_notional"])
    cash_left = cash_after - buys_total
    haircut = float(policy["sell_haircut"])
    if cash_left < 0 and core_mv > 0:
        # Sells may not fill or cash was already negative: restore cash ≥ 0 first,
        # sized on haircut proceeds so the projection actually reaches zero.
        qty = min(core_pos["qty"], (-cash_left) / (core_pos["price"] * haircut)) if core_pos["price"] > 0 else 0.0
        if qty > 0:
            new_leg(kind="core_sell", symbol=core_sym, side="sell", qty=qty, notional=None,
                    reason=f"projected cash {cash_left:,.0f} < 0: sell core to restore",
                    risk_reducing=True, overridable=False, cancel_stop_first=core_sym in stop_syms)
    elif diff > band:
        notional = min(diff, max(cash_left, 0.0))
        if notional >= policy["min_entry_notional"]:
            new_leg(kind="core_buy", symbol=core_sym, side="buy", qty=None, notional=notional,
                    reason=(f"core sleeve {core_mv / equity:.1%} → target {core_target / equity:.1%} "
                            f"(alpha {(alpha_mv_after + buys_total) / equity:.1%} of {alpha_target:.0%} target)"),
                    risk_reducing=False, overridable=True, cancel_stop_first=core_sym in stop_syms)
    elif diff < -band and core_pos:
        qty = min(core_pos["qty"], (-diff) / core_pos["price"]) if core_pos["price"] > 0 else 0.0
        if qty * core_pos["price"] >= policy["min_entry_notional"]:
            new_leg(kind="core_sell", symbol=core_sym, side="sell", qty=qty, notional=None,
                    reason=f"core sleeve {core_mv / equity:.1%} above target {core_target / equity:.1%}",
                    risk_reducing=breaker["active"], overridable=not breaker["active"],
                    cancel_stop_first=core_sym in stop_syms)

    legs.sort(key=lambda l: KIND_ORDER.index(l.kind))
    core_price = core_pos["price"] if core_pos else 0.0
    core_buy_total = sum(l.notional or 0.0 for l in legs if l.kind == "core_buy")
    core_sell_qty = sum(l.qty or 0.0 for l in legs if l.kind == "core_sell")
    projected_cash = cash_after - buys_total - core_buy_total + core_sell_qty * core_price * haircut
    plan = {
        "as_of": today.isoformat(),
        "target_date": target_date,
        "screen_date": screen_date,
        "screen_usable": usable,
        "screen_note": why,
        "regime": {"regime": regime_name, "detail": (regime or {}).get("reason", "")},
        "equity": round(equity, 2),
        "cash": round(cash, 2),
        "targets": {"alpha_pct": round(alpha_target, 4),
                    "core_pct": round(max(0.0, 1.0 - alpha_target - policy["cash_buffer"]), 4),
                    "cash_pct": policy["cash_buffer"]},
        "breaker": breaker,
        "exposure_after": {"alpha_mv": round(alpha_mv_after + buys_total, 2),
                           "core_mv": round(core_mv + core_buy_total - core_sell_qty * core_price, 2),
                           "cash": round(projected_cash, 2)},
        "legs": [l.to_dict() for l in legs],
        "rejected": rejected,
        "warnings": warnings,
        "policy": {k: v for k, v in policy.items() if not isinstance(v, dict)} | {"alpha_target": policy["alpha_target"]},
    }
    new_state = {
        "peak_equity": breaker["peak_equity"],
        "breaker_active": breaker["active"],
        "rank_history": rank_history,
        "trims_executed": trims_executed,
        "last_plan_date": target_date,
    }
    validate_plan(plan)
    return plan, new_state


def validate_plan(plan: dict) -> None:
    """Invariants every plan must satisfy. Raises EngineError."""
    legs = plan.get("legs") or []
    if not plan.get("screen_usable") and any(l["kind"] in ("entry", "add") for l in legs):
        raise EngineError("entries planned from an unusable screen")
    if plan["exposure_after"]["cash"] < -1.0:
        raise EngineError(f"plan leaves projected cash negative: {plan['exposure_after']['cash']}")
    for l in legs:
        if l["side"] == "buy" and (l.get("notional") or 0) <= 0:
            raise EngineError(f"buy leg {l['id']} has no notional")
        if l["side"] == "sell" and (l.get("qty") or 0) <= 0:
            raise EngineError(f"sell leg {l['id']} has no qty")
        if l["kind"] in NON_OVERRIDABLE_KINDS and l["overridable"]:
            raise EngineError(f"{l['kind']} leg {l['id']} must not be overridable")
    ids = [l["id"] for l in legs]
    if len(ids) != len(set(ids)):
        raise EngineError("duplicate leg ids")


def summarize(plan: dict) -> str:
    """Human-readable plan for the journal and the log."""
    lines = [f"PLAN for {plan['target_date']} (as of {plan['as_of']}) — regime {plan['regime']['regime']}, "
             f"equity ${plan['equity']:,.0f}, alpha target {plan['targets']['alpha_pct']:.0%}"]
    if plan["breaker"]["active"]:
        lines.append(f"  BREAKER ACTIVE: {plan['breaker']['drawdown']:.1%} below peak ${plan['breaker']['peak_equity']:,.0f}")
    if not plan["screen_usable"]:
        lines.append(f"  screen unusable: {plan['screen_note']}")
    for w in plan.get("warnings") or []:
        lines.append(f"  ! {w}")
    if not plan["legs"]:
        lines.append("  no legs — hold")
    for l in plan["legs"]:
        size = f"${l['notional']:,.0f}" if l.get("notional") else f"{l['qty']:g} sh"
        flags = " ".join(f for f, on in (("RISK", l["risk_reducing"]), ("fixed", not l["overridable"]),
                                          ("RTH-only", l["requires_rth"]), ("cancel-stop", l["cancel_stop_first"])) if on)
        lines.append(f"  {l['id']:>4} {l['kind']:<13} {l['side']:<4} {l['symbol']:<6} {size:>10}  [{flags}]  {l['reason']}")
    if plan["rejected"]:
        lines.append("  rejected: " + "; ".join(f"{r['symbol']} ({r['reason']})" for r in plan["rejected"][:12]))
    e = plan["exposure_after"]
    lines.append(f"  after: alpha ${e['alpha_mv']:,.0f}, core ${e['core_mv']:,.0f}, cash ${e['cash']:,.0f}")
    return "\n".join(lines)
