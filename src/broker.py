"""Thin Alpaca paper-trading REST wrapper.

All calls hit the PAPER endpoint only — this module can never touch a
live brokerage account. Credentials come from env vars ALPACA_API_KEY /
ALPACA_SECRET_KEY. Set TRADER_DRY_RUN=1 to have order/close calls return
what they *would* send without sending it.
"""
import os

import requests

BASE_URL = "https://paper-api.alpaca.markets"
TIMEOUT = 30


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


def submit_order(symbol: str, side: str, notional: float | None = None, qty: float | None = None) -> dict:
    """Market DAY order. Exactly one of notional (dollars) or qty (shares)."""
    if (notional is None) == (qty is None):
        raise BrokerError("Provide exactly one of notional or qty")
    payload = {
        "symbol": symbol.upper(),
        "side": side,
        "type": "market",
        "time_in_force": "day",
    }
    if notional is not None:
        payload["notional"] = round(float(notional), 2)
    else:
        payload["qty"] = f"{qty:g}"
    if _dry_run():
        return {"dry_run": True, "would_submit": payload}
    return _post("/v2/orders", payload)


def close_position(symbol: str) -> dict:
    """Liquidate the entire position in symbol at market."""
    if _dry_run():
        return {"dry_run": True, "would_close": symbol.upper()}
    return _delete(f"/v2/positions/{symbol.upper()}")
