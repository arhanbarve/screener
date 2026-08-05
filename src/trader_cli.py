"""CLI used by the daily trading agent. Every command prints JSON to stdout.

Commands:
  status              account + clock + positions + open orders
  gate                {"run": bool, "reason": str} — should today's session run now?
  buy SYM (--notional D | --qty N)
  sell SYM (--notional D | --qty N)
  close SYM           liquidate entire position
  orders [--status open|closed|all]
  activity-today      {"count": int, "safe_to_retry": bool} — did today touch the book?
  quote SYM           last printed trade
  cancel ORDER_ID     cancel one open order
  cancel-all          cancel every open order
  cancel-stops [SYM]  cancel resting protective stops (one symbol, or all)
  sync-stops          rest protective stops at each position's max-loss floor

Order types. buy/sell default to --auto, which picks the order that fits the
current session: market inside regular hours, otherwise a marketable limit that
queues with a price cap. A bare market order outside regular hours is refused —
on 2026-08-04 five of them sat unpriced for 14 hours ahead of the open. Override
with --type/--limit/--stop when you mean something specific.
"""
import argparse
import json
import sys
from datetime import date, datetime
from zoneinfo import ZoneInfo

from src import broker

ET = ZoneInfo("America/New_York")

# Two decision windows, because the machine is usually not awake at 09:00.
#
# EVENING is the primary one. It runs after the close, decides for the *next*
# open, and leaves marketable limit orders resting so they execute at 09:30
# without anybody present. It is also better informed than the morning session
# ever was: run_screener.sh fires at 16:30 ET, so an evening session reads the
# same day's screener plus the same day's closing prices, where a 09:00 session
# reads yesterday's screener and has no fresh close at all.
#
# MORNING is kept as a fallback for a day the evening session was missed, or for
# reacting to a pre-market event when the machine happens to be on.
GATE_MORNING = ((8, 30), (15, 45))   # intraday; 15:45 stops market orders near the close
GATE_EVENING = ((16, 15), (23, 59))  # after the close, deciding for the next open

# Kept as aliases so anything importing the old names still works.
GATE_EARLIEST = GATE_MORNING[0]
GATE_LATEST = GATE_MORNING[1]


def cmd_status() -> dict:
    return {
        "account": broker.get_account(),
        "clock": broker.get_clock(),
        "positions": broker.get_positions(),
        "open_orders": broker.get_orders("open"),
    }


def cmd_gate(now: datetime | None = None) -> dict:
    """Should a session run now, and which trading day is it deciding for?

    Returns target_date — the session whose open these orders are aimed at. The
    daily stamp is keyed to that, not to the calendar date, so an evening session
    that decides for tomorrow stops tomorrow morning from deciding again and
    doubling the position.
    """
    now = now or datetime.now(ET)
    clock = broker.get_clock()
    today = now.date().isoformat()
    next_open_date = str(clock.get("next_open") or "")[:10]
    hm = (now.hour, now.minute)

    opens_today = bool(clock.get("is_open")) or next_open_date == today

    # Morning: the market trades today and we are inside the intraday window.
    if opens_today and GATE_MORNING[0] <= hm <= GATE_MORNING[1]:
        return {"run": True, "window": "morning", "target_date": today,
                "reason": "trading day, within 08:30-15:45 ET window"}

    # Evening: today's session is over; decide for the next open. Weekdays only —
    # a weekend evening adds no information the Friday close did not already have.
    if (next_open_date and next_open_date > today and now.weekday() < 5
            and GATE_EVENING[0] <= hm <= GATE_EVENING[1]):
        return {"run": True, "window": "evening", "target_date": next_open_date,
                "reason": (f"after the close, deciding for the {next_open_date} open "
                           f"— orders rest overnight and execute at 09:30")}

    # Refusals, specific enough to diagnose from a log line.
    if opens_today and hm < GATE_MORNING[0]:
        return {"run": False, "target_date": today,
                "reason": "before the 08:30 ET morning window"}
    if opens_today and hm > GATE_MORNING[1]:
        return {"run": False, "target_date": today,
                "reason": ("between the 15:45 ET intraday cutoff and the 16:15 ET "
                           "evening window")}
    if next_open_date and next_open_date > today and now.weekday() >= 5:
        return {"run": False, "target_date": next_open_date,
                "reason": f"weekend — next open {next_open_date}"}
    if next_open_date and next_open_date > today:
        return {"run": False, "target_date": next_open_date,
                "reason": f"outside both windows (next open {next_open_date})"}
    return {"run": False, "target_date": next_open_date or today,
            "reason": f"not a trading day (next open {clock.get('next_open')})"}


def orders_today(orders: list[dict], today: str | None = None) -> list[dict]:
    """Orders submitted today, in any state — filled, pending, cancelled, rejected.

    Deliberately not just fills. A cancelled or rejected order still means the
    session reached the point of acting on the book, and re-running from scratch
    after that risks duplicating intent.
    """
    today = today or date.today().isoformat()
    out = []
    for o in orders:
        stamp = o.get("submitted_at") or o.get("created_at") or o.get("filled_at") or ""
        if stamp[:10] == today:
            out.append(o)
    return out


def cmd_activity_today(today: str | None = None) -> dict:
    """Whether a crashed session may safely be retried.

    run_trader.sh used to stamp the day *before* the session so a crash could
    never retry, on the reasoning that partial orders plus a retry is worse than
    a missed day. That is right, but it turned every transient disconnect into a
    permanently skipped trading day — 4 of the account's first 7 sessions.

    Retrying is safe only when the crashed session never touched the book. This
    is the check that makes the difference decidable instead of assumed.
    """
    orders = orders_today(broker.get_orders("all"), today)
    return {
        "count": len(orders),
        "safe_to_retry": len(orders) == 0,
        "symbols": sorted({(o.get("symbol") or "").upper() for o in orders}),
    }


def cmd_quote(symbol: str) -> dict:
    return broker.get_latest_trade(symbol)


def cmd_order(
    symbol: str,
    side: str,
    notional: float | None = None,
    qty: float | None = None,
    order_type: str = "auto",
    limit_price: float | None = None,
    stop_price: float | None = None,
    buffer_bps: float | None = None,
    allow_extended: bool = False,
) -> dict:
    """Place an order, choosing the type from the session unless told otherwise.

    "auto" is the default because the failure this guards against is not a bad
    limit price, it is an unpriced market order resting overnight.
    """
    from src import orders as orders_mod

    session = orders_mod.market_session(broker.get_clock())

    if order_type == "auto":
        plan = orders_mod.plan_order(
            symbol, side, session,
            last=broker.get_latest_trade(symbol)["price"],
            notional=notional, qty=qty, buffer_bps=buffer_bps,
            allow_extended=allow_extended,
        )
    else:
        if order_type == "market" and session != "open":
            raise broker.BrokerError(
                f"refusing a market order while the market is {session}: it would "
                f"rest unpriced until the next auction. Use --auto (marketable "
                f"limit), or --type limit --limit PRICE if you have a level in mind."
            )
        plan = {"symbol": symbol.upper(), "side": side, "order_type": order_type,
                "notional": notional, "qty": qty, "limit_price": limit_price,
                "stop_price": stop_price,
                "_why": f"{order_type}: explicitly requested"}
        if allow_extended:
            plan["extended_hours"] = True

    why = plan.pop("_why", "")
    plan = {k: v for k, v in plan.items() if v is not None}
    result = broker.submit_order(**plan)
    return {"session": session, "reason": why, "submitted": plan, "order": result}


def cmd_cancel_stops(apply: bool = True, symbol: str | None = None) -> dict:
    """Cancel resting protective stops, optionally just one symbol's.

    A stop holds the shares a sell needs, so it has to go before selling that
    position. Pass a symbol and only that one is touched — cancelling all of them
    up front would leave every position unprotected for the length of the session,
    and if the session then crashed (four of the first nine did) they would stay
    unprotected until the next run or the watchdog noticed.

    Blanket cancel-all is deliberately not used here: it would also take out a
    pending entry somebody meant to leave working.
    """
    from src import orders as orders_mod

    open_orders = broker.get_orders("open")
    if symbol:
        want = symbol.upper()
        open_orders_scoped = [o for o in open_orders
                              if (o.get("symbol") or "").upper() == want]
    else:
        open_orders_scoped = open_orders
    ids = orders_mod.protective_stop_ids(open_orders_scoped)
    others = [o["id"] for o in open_orders if o.get("id") and o["id"] not in ids]
    out = {"symbol": symbol.upper() if symbol else None,
           "stops": ids, "left_alone": others, "cancelled": []}
    if apply:
        out["cancelled"] = [broker.cancel_order(i) for i in ids]
    return out


def cmd_sync_stops(apply: bool = False) -> dict:
    """Rest a protective stop at each position's max-loss floor.

    Idempotent: running twice places nothing the second time. Without that the
    retry path would stack duplicate stops until they oversold the position.
    """
    from src import orders as orders_mod
    from src import paper_stops

    positions = broker.get_positions()
    tickers = [(p.get("symbol") or "").upper() for p in positions]
    if not tickers:
        return {"place": [], "cancel": [], "keep": [], "skip": [], "applied": False,
                "note": "no positions"}

    earliest = min((p.get("held_since") or date.today().isoformat())
                   for p in _with_held_since(positions))
    history = paper_stops.fetch_history(tickers, earliest)
    plans = paper_stops.build_plans(_with_held_since(positions), history)
    paper_stops.save_plans(plans)
    floors = paper_stops.floors_from_plans(plans)

    # Drop any floor that would fire on placement — that is a plan error, and
    # dumping a position on a bad plan is worse than resting no stop.
    rejected = []
    for p in positions:
        sym = (p.get("symbol") or "").upper()
        if sym not in floors:
            continue
        problem = paper_stops.sanity_check_floor(
            floors[sym], float(p.get("current_price") or 0))
        if problem:
            rejected.append({"ticker": sym, "floor": floors.pop(sym), "reason": problem})

    result = orders_mod.reconcile_stops(positions, floors, broker.get_orders("open"))
    result["rejected"] = rejected
    result["floors"] = floors
    result["applied"] = False

    if apply:
        cancelled, placed = [], []
        for o in result["cancel"]:
            cancelled.append(broker.cancel_order(o["id"]))
        for spec in result["place"]:
            spec = {k: v for k, v in spec.items() if not k.startswith("_")}
            placed.append(broker.submit_order(**spec))
        result.update(applied=True, cancelled=cancelled, placed=placed)
    return result


def _with_held_since(positions: list[dict]) -> list[dict]:
    """Attach FIFO held_since from the paper snapshot when available.

    After a partial sale the shares still held began later than the first fill,
    and the plan's ATR must be taken at the entry that actually applies.
    """
    from src import paper

    snap = paper.load_snapshot() or {}
    by_ticker = {p["ticker"]: p for p in (snap.get("positions") or [])}
    out = []
    for p in positions:
        sym = (p.get("symbol") or "").upper()
        snapped = by_ticker.get(sym, {})
        out.append({
            "ticker": sym,
            "avg_cost": p.get("avg_entry_price"),
            "held_since": snapped.get("held_since"),
        })
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="trader_cli", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    sub.add_parser("gate")
    for name in ("buy", "sell"):
        sp = sub.add_parser(name)
        sp.add_argument("symbol")
        g = sp.add_mutually_exclusive_group(required=True)
        g.add_argument("--notional", type=float)
        g.add_argument("--qty", type=float)
        sp.add_argument("--type", dest="order_type", default="auto",
                        choices=["auto", "market", "limit", "stop", "stop_limit"],
                        help="auto (default) picks by session; market is refused "
                             "outside regular hours")
        sp.add_argument("--limit", dest="limit_price", type=float)
        sp.add_argument("--stop", dest="stop_price", type=float)
        sp.add_argument("--buffer-bps", dest="buffer_bps", type=float,
                        help="max slippage for an auto limit (default 50bp buy / 200bp sell)")
        sp.add_argument("--extended", dest="allow_extended", action="store_true",
                        help="allow filling in pre/post-market (thin books)")
    sp = sub.add_parser("close")
    sp.add_argument("symbol")
    sp = sub.add_parser("orders")
    sp.add_argument("--status", default="open", choices=["open", "closed", "all"])
    sub.add_parser("activity-today")
    sp = sub.add_parser("quote")
    sp.add_argument("symbol")
    sp = sub.add_parser("cancel")
    sp.add_argument("order_id")
    sub.add_parser("cancel-all")
    sp = sub.add_parser("cancel-stops")
    sp.add_argument("symbol", nargs="?", help="only this symbol's stop (recommended)")
    sp.add_argument("--report-only", action="store_true")
    sp = sub.add_parser("sync-stops")
    sp.add_argument("--apply", action="store_true",
                    help="actually place/cancel stops (default reports only)")
    args = p.parse_args(argv)

    try:
        if args.cmd == "status":
            out = cmd_status()
        elif args.cmd == "gate":
            out = cmd_gate()
        elif args.cmd in ("buy", "sell"):
            out = cmd_order(args.symbol.upper(), args.cmd,
                            notional=args.notional, qty=args.qty,
                            order_type=args.order_type,
                            limit_price=args.limit_price,
                            stop_price=args.stop_price,
                            buffer_bps=args.buffer_bps,
                            allow_extended=args.allow_extended)
        elif args.cmd == "close":
            out = broker.close_position(args.symbol.upper())
        elif args.cmd == "activity-today":
            out = cmd_activity_today()
        elif args.cmd == "quote":
            out = cmd_quote(args.symbol.upper())
        elif args.cmd == "cancel":
            out = broker.cancel_order(args.order_id)
        elif args.cmd == "cancel-all":
            out = broker.cancel_all_orders()
        elif args.cmd == "cancel-stops":
            out = cmd_cancel_stops(apply=not args.report_only, symbol=args.symbol)
        elif args.cmd == "sync-stops":
            out = cmd_sync_stops(apply=args.apply)
        else:  # orders
            out = broker.get_orders(args.status)
    except broker.BrokerError as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        return 1
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
