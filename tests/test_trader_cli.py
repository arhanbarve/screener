import json
from datetime import datetime
from unittest.mock import patch
from zoneinfo import ZoneInfo

from src import trader_cli

ET = ZoneInfo("America/New_York")


def _clock(is_open, next_open):
    return {"is_open": is_open, "next_open": next_open, "next_close": "2026-07-23T16:00:00-04:00"}


def test_gate_weekend_skips():
    now = datetime(2026, 7, 25, 10, 0, tzinfo=ET)  # Saturday
    with patch("src.trader_cli.broker.get_clock",
               return_value=_clock(False, "2026-07-27T09:30:00-04:00")):
        out = trader_cli.cmd_gate(now=now)
    assert out["run"] is False
    assert "not a trading day" in out["reason"]


def test_gate_trading_morning_runs():
    now = datetime(2026, 7, 23, 9, 0, tzinfo=ET)  # Thursday pre-open
    with patch("src.trader_cli.broker.get_clock",
               return_value=_clock(False, "2026-07-23T09:30:00-04:00")):
        out = trader_cli.cmd_gate(now=now)
    assert out["run"] is True


def test_gate_market_open_midday_runs():
    now = datetime(2026, 7, 23, 13, 0, tzinfo=ET)
    with patch("src.trader_cli.broker.get_clock",
               return_value=_clock(True, "2026-07-24T09:30:00-04:00")):
        out = trader_cli.cmd_gate(now=now)
    assert out["run"] is True


def test_gate_before_window_skips():
    now = datetime(2026, 7, 23, 7, 0, tzinfo=ET)
    with patch("src.trader_cli.broker.get_clock",
               return_value=_clock(False, "2026-07-23T09:30:00-04:00")):
        out = trader_cli.cmd_gate(now=now)
    assert out["run"] is False
    assert "before" in out["reason"]


def test_gate_after_cutoff_skips():
    now = datetime(2026, 7, 23, 17, 30, tzinfo=ET)  # post-close
    with patch("src.trader_cli.broker.get_clock",
               return_value=_clock(False, "2026-07-24T09:30:00-04:00")):
        out = trader_cli.cmd_gate(now=now)
    assert out["run"] is False


def test_buy_prints_order_json(capsys):
    with patch("src.trader_cli.broker.submit_order", return_value={"id": "ord1"}) as s:
        rc = trader_cli.main(["buy", "STX", "--notional", "5000"])
    assert rc == 0
    s.assert_called_once_with("STX", "buy", notional=5000.0, qty=None)
    assert json.loads(capsys.readouterr().out) == {"id": "ord1"}


def test_sell_qty(capsys):
    with patch("src.trader_cli.broker.submit_order", return_value={"id": "ord2"}) as s:
        rc = trader_cli.main(["sell", "SPY", "--qty", "3"])
    assert rc == 0
    s.assert_called_once_with("SPY", "sell", notional=None, qty=3.0)


def test_close_position(capsys):
    with patch("src.trader_cli.broker.close_position", return_value={"ok": 1}) as c:
        rc = trader_cli.main(["close", "STX"])
    assert rc == 0
    c.assert_called_once_with("STX")


def test_status_aggregates(capsys):
    with patch("src.trader_cli.broker.get_account", return_value={"equity": "1"}), \
         patch("src.trader_cli.broker.get_clock", return_value={"is_open": True}), \
         patch("src.trader_cli.broker.get_positions", return_value=[]), \
         patch("src.trader_cli.broker.get_orders", return_value=[]):
        rc = trader_cli.main(["status"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert set(out) == {"account", "clock", "positions", "open_orders"}


def test_broker_error_returns_nonzero(capsys):
    from src.broker import BrokerError
    with patch("src.trader_cli.broker.get_account", side_effect=BrokerError("boom")):
        rc = trader_cli.main(["status"])
    assert rc == 1
    assert "boom" in capsys.readouterr().err


# ── activity-today: is a crashed session safe to retry? ───────────────────────
#
# run_trader.sh used to stamp the day before the session so a crash could never
# retry. That made a transient disconnect cost the whole trading day — 4 of the
# account's first 7 sessions. Retry is safe only when the crashed session never
# touched the book, and these tests pin that decision.

def _order(sym, day, status="filled", field="submitted_at"):
    return {"symbol": sym, "status": status, field: f"{day}T13:45:00Z"}


def test_orders_today_filters_by_date():
    orders = [_order("AAPL", "2026-08-04"), _order("MSFT", "2026-08-03")]
    got = trader_cli.orders_today(orders, today="2026-08-04")
    assert [o["symbol"] for o in got] == ["AAPL"]


def test_orders_today_counts_cancelled_and_rejected():
    """Not just fills — a cancelled order still means the session acted."""
    orders = [
        _order("AAPL", "2026-08-04", status="canceled"),
        _order("MSFT", "2026-08-04", status="rejected"),
    ]
    assert len(trader_cli.orders_today(orders, today="2026-08-04")) == 2


def test_orders_today_falls_back_to_created_and_filled_at():
    orders = [
        {"symbol": "AAPL", "created_at": "2026-08-04T13:00:00Z"},
        {"symbol": "MSFT", "filled_at": "2026-08-04T14:00:00Z"},
    ]
    assert len(trader_cli.orders_today(orders, today="2026-08-04")) == 2


def test_orders_today_ignores_undated():
    assert trader_cli.orders_today([{"symbol": "AAPL"}], today="2026-08-04") == []


def test_activity_today_safe_when_book_untouched():
    with patch("src.trader_cli.broker.get_orders", return_value=[_order("MSFT", "2026-08-03")]):
        out = trader_cli.cmd_activity_today(today="2026-08-04")
    assert out["count"] == 0
    assert out["safe_to_retry"] is True
    assert out["symbols"] == []


def test_activity_today_unsafe_after_any_order():
    orders = [_order("AAPL", "2026-08-04"), _order("HUT", "2026-08-04", status="canceled")]
    with patch("src.trader_cli.broker.get_orders", return_value=orders):
        out = trader_cli.cmd_activity_today(today="2026-08-04")
    assert out["count"] == 2
    assert out["safe_to_retry"] is False
    assert out["symbols"] == ["AAPL", "HUT"]


def test_activity_today_queries_all_statuses():
    """Querying only open orders would miss the fills that make retry unsafe."""
    with patch("src.trader_cli.broker.get_orders", return_value=[]) as m:
        trader_cli.cmd_activity_today(today="2026-08-04")
    m.assert_called_once_with("all")


def test_activity_today_cli_prints_json(capsys):
    with patch("src.trader_cli.broker.get_orders", return_value=[]):
        rc = trader_cli.main(["activity-today"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["safe_to_retry"] is True
