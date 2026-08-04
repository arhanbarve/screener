"""CLI used by the daily trading agent. Every command prints JSON to stdout.

Commands:
  status              account + clock + positions + open orders
  gate                {"run": bool, "reason": str} — should today's session run now?
  buy SYM (--notional D | --qty N)
  sell SYM (--notional D | --qty N)
  close SYM           liquidate entire position
  orders [--status open|closed|all]
  activity-today      {"count": int, "safe_to_retry": bool} — did today touch the book?
"""
import argparse
import json
import sys
from datetime import date, datetime
from zoneinfo import ZoneInfo

from src import broker

ET = ZoneInfo("America/New_York")
GATE_EARLIEST = (8, 30)   # don't trade on a midnight reboot with stale context
GATE_LATEST = (15, 45)    # market orders too close to 16:00 close are pointless


def cmd_status() -> dict:
    return {
        "account": broker.get_account(),
        "clock": broker.get_clock(),
        "positions": broker.get_positions(),
        "open_orders": broker.get_orders("open"),
    }


def cmd_gate(now: datetime | None = None) -> dict:
    now = now or datetime.now(ET)
    clock = broker.get_clock()
    today = now.date().isoformat()
    is_trading_day = clock["is_open"] or clock["next_open"][:10] == today
    if not is_trading_day:
        return {"run": False, "reason": f"not a trading day (next open {clock['next_open']})"}
    hm = (now.hour, now.minute)
    if hm < GATE_EARLIEST:
        return {"run": False, "reason": "before 08:30 ET window"}
    if hm > GATE_LATEST:
        return {"run": False, "reason": "after 15:45 ET cutoff"}
    return {"run": True, "reason": "trading day, within 08:30-15:45 ET window"}


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
    sp = sub.add_parser("close")
    sp.add_argument("symbol")
    sp = sub.add_parser("orders")
    sp.add_argument("--status", default="open", choices=["open", "closed", "all"])
    sub.add_parser("activity-today")
    args = p.parse_args(argv)

    try:
        if args.cmd == "status":
            out = cmd_status()
        elif args.cmd == "gate":
            out = cmd_gate()
        elif args.cmd in ("buy", "sell"):
            out = broker.submit_order(args.symbol.upper(), args.cmd,
                                      notional=args.notional, qty=args.qty)
        elif args.cmd == "close":
            out = broker.close_position(args.symbol.upper())
        elif args.cmd == "activity-today":
            out = cmd_activity_today()
        else:  # orders
            out = broker.get_orders(args.status)
    except broker.BrokerError as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        return 1
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
