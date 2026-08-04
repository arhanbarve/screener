"""Thin Alpaca paper-trading REST wrapper.

All calls hit the PAPER endpoint only — this module can never touch a
live brokerage account. Credentials come from env vars ALPACA_API_KEY /
ALPACA_SECRET_KEY. Set TRADER_DRY_RUN=1 to have order/close calls return
what they *would* send without sending it.
"""
import os

import requests

BASE_URL = "https://paper-api.alpaca.markets"
# Market data lives on a separate host from trading. Used only for last-trade
# prices, which limit and stop prices are derived from.
DATA_URL = "https://data.alpaca.markets"
TIMEOUT = 30

# Order types this wrapper will send. Alpaca also offers trailing_stop, which is
# deliberately unsupported here: src/exit_plan.py owns the trailing stop, walks
# it down weekly (TRAIL_MULT_STEPS 3.0 -> 2.5 -> 2.0) and evaluates it on closes.
# A broker-side trailing stop would drift from that and give one stop two
# sources of truth, which is worse than having no resting trail at all.
ORDER_TYPES = ("market", "limit", "stop", "stop_limit")


class BrokerError(Exception):
    pass


def _dry_run() -> bool:
    return os.environ.get("TRADER_DRY_RUN") == "1"


def _headers() -> dict:
    key = os.environ.get("ALPACA_API_KEY")
    secret = os.environ.get("ALPACA_SECRET_KEY")
    if not key or not secret:
        raise BrokerError("Missing ALPACA_API_KEY / ALPACA_SECRET_KEY env vars")
    return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}


def _get(path: str, params: dict | None = None):
    resp = requests.get(f"{BASE_URL}{path}", headers=_headers(), params=params, timeout=TIMEOUT)
    if resp.status_code != 200:
        raise BrokerError(f"GET {path} -> {resp.status_code}: {resp.text}")
    return resp.json()


def _post(path: str, payload: dict):
    resp = requests.post(f"{BASE_URL}{path}", headers=_headers(), json=payload, timeout=TIMEOUT)
    if resp.status_code not in (200, 201):
        raise BrokerError(f"POST {path} -> {resp.status_code}: {resp.text}")
    return resp.json()


def _delete(path: str):
    resp = requests.delete(f"{BASE_URL}{path}", headers=_headers(), timeout=TIMEOUT)
    if resp.status_code not in (200, 204, 207):
        raise BrokerError(f"DELETE {path} -> {resp.status_code}: {resp.text}")
    return resp.json() if resp.text else {}


def get_account() -> dict:
    return _get("/v2/account")


def get_clock() -> dict:
    return _get("/v2/clock")


def get_positions() -> list:
    return _get("/v2/positions")


def get_orders(status: str = "open") -> list:
    return _get("/v2/orders", params={"status": status, "limit": 100})


def submit_order(
    symbol: str,
    side: str,
    notional: float | None = None,
    qty: float | None = None,
    order_type: str = "market",
    limit_price: float | None = None,
    stop_price: float | None = None,
    time_in_force: str = "day",
    extended_hours: bool = False,
) -> dict:
    """Submit an order. Exactly one of notional (dollars) or qty (shares).

    Validation here mirrors the constraints Alpaca actually enforces, so a bad
    combination fails locally with a clear message instead of as an opaque 422:

    - limit needs limit_price, stop needs stop_price, stop_limit needs both
    - market must not carry either price
    - extended_hours accepts limit orders only (market/stop/stop_limit are
      rejected outright during pre/post-market)
    - notional sizing is only reliable for market orders; for the other types
      pass qty (src/orders.py converts dollars to shares)
    - fractional quantities require time_in_force="day"
    """
    if (notional is None) == (qty is None):
        raise BrokerError("Provide exactly one of notional or qty")
    if order_type not in ORDER_TYPES:
        raise BrokerError(f"order_type must be one of {ORDER_TYPES}, got {order_type!r}")
    if side not in ("buy", "sell"):
        raise BrokerError(f"side must be buy or sell, got {side!r}")

    needs_limit = order_type in ("limit", "stop_limit")
    needs_stop = order_type in ("stop", "stop_limit")
    if needs_limit and limit_price is None:
        raise BrokerError(f"{order_type} order requires limit_price")
    if needs_stop and stop_price is None:
        raise BrokerError(f"{order_type} order requires stop_price")
    if order_type == "market" and (limit_price is not None or stop_price is not None):
        raise BrokerError("market order must not carry limit_price or stop_price")
    if limit_price is not None and float(limit_price) <= 0:
        raise BrokerError("limit_price must be positive")
    if stop_price is not None and float(stop_price) <= 0:
        raise BrokerError("stop_price must be positive")
    if extended_hours and order_type != "limit":
        raise BrokerError(
            "extended_hours supports limit orders only — Alpaca rejects market, "
            "stop and stop_limit outside regular hours"
        )
    if notional is not None and order_type != "market":
        raise BrokerError(
            f"notional sizing is only supported for market orders; convert to qty "
            f"for {order_type} (see src.orders.qty_for_notional)"
        )
    if qty is not None and float(qty) != int(float(qty)) and time_in_force != "day":
        raise BrokerError("fractional qty requires time_in_force='day'")

    payload: dict = {
        "symbol": symbol.upper(),
        "side": side,
        "type": order_type,
        "time_in_force": time_in_force,
    }
    if notional is not None:
        payload["notional"] = round(float(notional), 2)
    else:
        payload["qty"] = f"{float(qty):.9f}".rstrip("0").rstrip(".")
    if limit_price is not None:
        payload["limit_price"] = str(limit_price)
    if stop_price is not None:
        payload["stop_price"] = str(stop_price)
    if extended_hours:
        payload["extended_hours"] = True

    if _dry_run():
        return {"dry_run": True, "would_submit": payload}
    return _post("/v2/orders", payload)


def close_position(symbol: str) -> dict:
    """Liquidate the entire position in symbol at market."""
    if _dry_run():
        return {"dry_run": True, "would_close": symbol.upper()}
    return _delete(f"/v2/positions/{symbol.upper()}")


def cancel_order(order_id: str) -> dict:
    """Cancel a single open order. Cancelling removes an intent; it never trades."""
    if _dry_run():
        return {"dry_run": True, "would_cancel": order_id}
    return _delete(f"/v2/orders/{order_id}")


def cancel_all_orders() -> dict:
    """Cancel every open order. Returns per-order status (Alpaca answers 207)."""
    if _dry_run():
        return {"dry_run": True, "would_cancel": "all open orders"}
    return _delete("/v2/orders")


def get_latest_trade(symbol: str) -> dict:
    """Last printed trade for symbol, from the market-data host.

    Limit and stop prices are derived from this. Note the timestamp: outside
    regular hours a thinly traded name may last have printed at the close, which
    is exactly when a limit derived from it is least meaningful.
    """
    resp = requests.get(
        f"{DATA_URL}/v2/stocks/{symbol.upper()}/trades/latest",
        headers=_headers(),
        timeout=TIMEOUT,
    )
    if resp.status_code != 200:
        raise BrokerError(f"latest trade {symbol} -> {resp.status_code}: {resp.text}")
    trade = resp.json().get("trade") or {}
    if not trade.get("p"):
        raise BrokerError(f"no last price available for {symbol}")
    return {"symbol": symbol.upper(), "price": float(trade["p"]),
            "size": trade.get("s"), "at": trade.get("t")}
