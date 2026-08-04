"""Order-type intelligence: choose the right order for the current session.

Everything here is pure — it takes a clock, a last price and a size, and returns
the arguments to hand src.broker.submit_order. No network, no clock reads, so the
decisions are testable without a broker.

Why this module exists. Until 2026-08-04 every order was a market DAY order, and
on that evening five market orders were queued 14 hours ahead of the open. A
market order inside regular hours is fine — the 08-03 fills were all good — but a
market order sitting overnight fills at whatever the auction prints. This account
has already seen a 9.6% gap between a pre-market indication and the actual fill
(TSEM, 08-03), so unpriced overnight orders are the real exposure.

Two asymmetries shape the defaults:

1. **Limits must be marketable, not clever.** The measured drag on this account
   is absence from the tape (-4.04pp while holding 40% cash), not bad entries. An
   unfilled limit costs more than a few basis points of slippage, so buffers sit
   wide enough to fill through normal open volatility.
2. **Extended hours is usually a trap.** Alpaca accepts limit orders in
   pre/post-market, but on 08-04 at 19:15 ET, ORKA, VSXY and HUT had last printed
   at 15:59 — no post-market liquidity at all. Filling into that is worse than
   waiting for the auction, so extended_hours is opt-in per order, never default.
"""
from datetime import datetime, time
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

# Regular trading hours, and the extended windows Alpaca will accept limit orders in.
RTH_OPEN = time(9, 30)
RTH_CLOSE = time(16, 0)
PRE_OPEN = time(4, 0)
POST_CLOSE = time(20, 0)

# Default marketable buffers, in basis points. 50bp = 0.5%.
ENTRY_BUFFER_BPS = 50    # pay up to 0.5% above last to get filled
EXIT_BUFFER_BPS = 200    # accept up to 2% below last to get out

# Alpaca sub-penny rule: limit prices take 4 decimals below $1, 2 decimals at or
# above $1. Sending 3 decimals on a $50 stock is rejected.
SUBDOLLAR_TICK = 1.0


class OrderPlanError(ValueError):
    """The requested order cannot be planned for this session."""


# ── session ───────────────────────────────────────────────────────────────────

def market_session(clock: dict, now: datetime | None = None) -> str:
    """Classify the current session: "open", "extended" or "closed".

    Alpaca's clock is authoritative for whether regular hours are live (it knows
    holidays and half-days). The extended windows are derived from ET wall time,
    but only on a day the exchange actually trades — otherwise a Sunday evening
    would look like post-market.
    """
    if clock.get("is_open"):
        return "open"

    now = now or datetime.now(ET)
    if now.tzinfo is None:
        now = now.replace(tzinfo=ET)
    now = now.astimezone(ET)

    # A trading day is one where the exchange opens today. next_open falling on
    # today means the session is still ahead of us (pre-market); a next_open on a
    # later date with the clock closed means we are after today's close.
    next_open = str(clock.get("next_open") or "")[:10]
    today = now.date().isoformat()
    is_trading_day = next_open == today or (
        next_open > today and now.weekday() < 5 and now.time() >= RTH_CLOSE
    )
    if not is_trading_day:
        return "closed"
    if PRE_OPEN <= now.time() < RTH_OPEN or RTH_CLOSE <= now.time() < POST_CLOSE:
        return "extended"
    return "closed"


# ── sizing and pricing ────────────────────────────────────────────────────────

def qty_for_notional(notional: float, price: float) -> float:
    """Shares for a dollar amount. Alpaca accepts 9 decimal places.

    Needed because notional sizing is only reliable for market orders; every
    limit or stop order has to be expressed in shares.
    """
    if price <= 0:
        raise OrderPlanError(f"cannot size ${notional:,.2f} at price {price}")
    if notional <= 0:
        raise OrderPlanError(f"notional must be positive, got {notional}")
    return round(notional / price, 9)


def round_tick(price: float) -> float:
    """Round to a price Alpaca will accept: 4dp under $1, 2dp at or above."""
    if price <= 0:
        raise OrderPlanError(f"price must be positive, got {price}")
    return round(price, 4) if price < SUBDOLLAR_TICK else round(price, 2)


def limit_price_for(side: str, last: float, buffer_bps: float) -> float:
    """A marketable limit: above last to buy, below last to sell.

    The buffer is the most slippage accepted, not a target — the order still
    fills at the best available price inside it.
    """
    if side not in ("buy", "sell"):
        raise OrderPlanError(f"side must be buy or sell, got {side!r}")
    if last <= 0:
        raise OrderPlanError(f"last price must be positive, got {last}")
    if buffer_bps < 0:
        raise OrderPlanError("buffer_bps must not be negative")
    factor = 1 + buffer_bps / 10_000 if side == "buy" else 1 - buffer_bps / 10_000
    return round_tick(last * factor)


# ── order planning ────────────────────────────────────────────────────────────

def plan_order(
    symbol: str,
    side: str,
    session: str,
    last: float,
    notional: float | None = None,
    qty: float | None = None,
    buffer_bps: float | None = None,
    allow_extended: bool = False,
) -> dict:
    """Choose the order that fits the session, and explain the choice.

    Returns kwargs for broker.submit_order plus "_why", a human-readable reason
    that gets logged and journalled — an order type nobody can explain is an
    order type nobody should send.
    """
    if (notional is None) == (qty is None):
        raise OrderPlanError("provide exactly one of notional or qty")
    if buffer_bps is None:
        buffer_bps = ENTRY_BUFFER_BPS if side == "buy" else EXIT_BUFFER_BPS

    base = {"symbol": symbol.upper(), "side": side}

    if session == "open":
        # Inside regular hours a market order fills against a live book in
        # seconds. Notional sizing is exact here, so prefer it when given.
        if notional is not None:
            return {**base, "notional": round(float(notional), 2),
                    "order_type": "market",
                    "_why": "market: regular hours, notional sizing exact"}
        return {**base, "qty": qty, "order_type": "market",
                "_why": "market: regular hours"}

    if session == "closed":
        # Nothing can fill. A market order would sit unpriced until the auction —
        # the exact exposure this module exists to remove. A limit DAY order
        # queues with a price cap and becomes eligible at the next open.
        limit = limit_price_for(side, last, buffer_bps)
        size = qty if qty is not None else qty_for_notional(notional, limit)
        return {**base, "qty": size, "order_type": "limit", "limit_price": limit,
                "_why": (f"limit {limit:g}: market closed, queues for next open "
                         f"with a {buffer_bps:g}bp cap instead of filling at the "
                         f"auction unpriced")}

    # session == "extended": Alpaca accepts limit orders only, and thin books
    # make filling into them a choice rather than a default.
    limit = limit_price_for(side, last, buffer_bps)
    size = qty if qty is not None else qty_for_notional(notional, limit)
    if allow_extended:
        return {**base, "qty": size, "order_type": "limit", "limit_price": limit,
                "extended_hours": True,
                "_why": (f"limit {limit:g} extended-hours: explicitly opted in; "
                         f"pre/post-market books are thin")}
    return {**base, "qty": size, "order_type": "limit", "limit_price": limit,
            "_why": (f"limit {limit:g}: extended hours, but not opted in — "
                     f"queues for regular hours rather than filling into a thin "
                     f"book")}


def plan_stop(symbol: str, qty: float, stop_price: float) -> dict:
    """A resting protective stop for a long position.

    Only ever the max-loss floor. The trailing stop and the 50-day trend break
    stay close-based in src/exit_plan.py on purpose: a resting intraday stop
    fires on a wick, which is the flip-flopping the standing-verdict rewrite
    removed. The floor is far enough from noise to be worth resting, and it is
    the level whose whole job is surviving a session that never runs.
    """
    if qty <= 0:
        raise OrderPlanError(f"stop needs a positive qty, got {qty}")
    return {"symbol": symbol.upper(), "side": "sell", "qty": round(qty, 9),
            "order_type": "stop", "stop_price": round_tick(stop_price),
            "_why": f"protective stop at max-loss floor {round_tick(stop_price):g}"}


# ── stop reconciliation ───────────────────────────────────────────────────────

def _is_protective_stop(order: dict) -> bool:
    return (order.get("type") == "stop" and order.get("side") == "sell"
            and order.get("stop_price") is not None)


def reconcile_stops(
    positions: list[dict],
    floors: dict[str, float],
    open_orders: list[dict],
    tolerance: float = 0.01,
) -> dict:
    """Work out which resting stops to place, replace or cancel.

    Idempotent by design: run it twice and the second run is a no-op. Without
    that, a session that runs several times a day (the retry path can) would
    stack duplicate stops until they oversell the position.

    Sizing uses Alpaca's qty_available (shares not already committed to another
    order), not the raw position size. Otherwise a position with a pending sell
    would get a stop for its full quantity on top of that sell — HUT on
    2026-08-04 had qty 36.02 and qty_available 0 because a market close was
    already resting against every share. Shares held by protective stops we are
    about to replace are added back, so replacing a stop stays possible.

    Returns {"place": [...], "cancel": [...], "keep": [...], "skip": [...]}.
    """
    existing: dict[str, list[dict]] = {}
    for o in open_orders:
        if _is_protective_stop(o):
            existing.setdefault((o.get("symbol") or "").upper(), []).append(o)

    held: dict[str, float] = {}
    free: dict[str, float] = {}
    for p in positions:
        sym = (p.get("symbol") or "").upper()
        qty = float(p.get("qty") or 0)
        held[sym] = qty
        # qty_available may be absent on older payloads; fall back to full size.
        avail = p.get("qty_available")
        avail = qty if avail is None else float(avail)
        reserved_by_our_stops = sum(
            float(o.get("qty") or 0) for o in existing.get(sym, []))
        free[sym] = round(min(qty, avail + reserved_by_our_stops), 9)

    place, cancel, keep, skip = [], [], [], []

    for sym in sorted(held):
        floor = floors.get(sym)
        current = existing.get(sym, [])
        if floor is None:
            # No plan for this symbol — resting a stop would be guessing.
            skip.append({"ticker": sym, "reason": "no exit plan / max-loss floor"})
            cancel.extend(current)
            continue

        target_qty = free[sym]
        if target_qty <= 0:
            skip.append({
                "ticker": sym,
                "reason": (f"no free shares: all {held[sym]:g} are committed to "
                           f"another open order, so a stop would oversell"),
            })
            cancel.extend(current)
            continue

        matching = [
            o for o in current
            if abs(float(o["stop_price"]) - round_tick(floor)) < tolerance
            and abs(float(o.get("qty") or 0) - target_qty) < 1e-6
        ]
        if matching:
            keep.extend(matching)
            cancel.extend([o for o in current if o not in matching])
            continue

        cancel.extend(current)
        place.append(plan_stop(sym, target_qty, floor))

    # Stops on symbols no longer held must go — they would sell shares we do not
    # have and be rejected, or worse, sell a position re-entered later.
    for sym, orders in sorted(existing.items()):
        if sym not in held:
            cancel.extend(orders)

    return {"place": place, "cancel": cancel, "keep": keep, "skip": skip}
