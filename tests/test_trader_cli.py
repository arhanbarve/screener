import json
from datetime import datetime
import pytest
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


# buy/sell now route through cmd_order, which picks the order type from the
# session. Inside regular hours that is still a market order; outside, it is a
# marketable limit — a bare market order resting overnight is the failure these
# tests exist to prevent.

def _open_clock():
    return {"is_open": True, "next_open": "2026-08-05T09:30:00-04:00",
            "next_close": "2026-08-04T16:00:00-04:00"}


def _closed_clock():
    return {"is_open": False, "next_open": "2026-08-05T09:30:00-04:00",
            "next_close": "2026-08-05T16:00:00-04:00"}


def test_buy_in_regular_hours_is_a_market_order(capsys):
    with patch("src.trader_cli.broker.get_clock", return_value=_open_clock()), \
         patch("src.trader_cli.broker.get_latest_trade", return_value={"price": 850.0}), \
         patch("src.trader_cli.broker.submit_order", return_value={"id": "ord1"}) as s:
        rc = trader_cli.main(["buy", "STX", "--notional", "5000"])
    assert rc == 0
    s.assert_called_once_with(symbol="STX", side="buy", notional=5000.0,
                              order_type="market")
    out = json.loads(capsys.readouterr().out)
    assert out["session"] == "open"
    assert out["order"] == {"id": "ord1"}
    assert "market" in out["reason"]


def test_sell_qty_in_regular_hours(capsys):
    with patch("src.trader_cli.broker.get_clock", return_value=_open_clock()), \
         patch("src.trader_cli.broker.get_latest_trade", return_value={"price": 770.0}), \
         patch("src.trader_cli.broker.submit_order", return_value={"id": "ord2"}) as s:
        rc = trader_cli.main(["sell", "SPY", "--qty", "3"])
    assert rc == 0
    s.assert_called_once_with(symbol="SPY", side="sell", qty=3.0,
                              order_type="market")


def test_buy_outside_hours_becomes_a_marketable_limit(capsys):
    """The 2026-08-04 regression: five market orders queued 14h before the open."""
    with patch("src.trader_cli.broker.get_clock", return_value=_closed_clock()), \
         patch("src.trader_cli.broker.get_latest_trade", return_value={"price": 100.0}), \
         patch("src.trader_cli.broker.submit_order", return_value={"id": "ord3"}) as s:
        rc = trader_cli.main(["buy", "VSXY", "--notional", "4000"])
    assert rc == 0
    kwargs = s.call_args.kwargs
    assert kwargs["order_type"] == "limit"
    assert kwargs["limit_price"] == 100.5          # 50bp entry buffer
    assert kwargs["qty"] == pytest.approx(4000 / 100.5, abs=1e-6)
    assert "notional" not in kwargs                # limits must be share-sized
    assert kwargs.get("extended_hours") is None    # thin books are opt-in


def test_explicit_market_order_refused_outside_hours(capsys):
    with patch("src.trader_cli.broker.get_clock", return_value=_closed_clock()), \
         patch("src.trader_cli.broker.submit_order") as s:
        rc = trader_cli.main(["buy", "VSXY", "--notional", "4000", "--type", "market"])
    assert rc == 1
    s.assert_not_called()
    assert "refusing a market order" in capsys.readouterr().err


def test_explicit_limit_is_passed_through(capsys):
    with patch("src.trader_cli.broker.get_clock", return_value=_closed_clock()), \
         patch("src.trader_cli.broker.submit_order", return_value={"id": "ord4"}) as s:
        rc = trader_cli.main(["buy", "VSXY", "--notional", "4000",
                              "--type", "limit", "--limit", "88.00"])
    assert rc == 0
    assert s.call_args.kwargs["limit_price"] == 88.0


def test_buffer_bps_override(capsys):
    with patch("src.trader_cli.broker.get_clock", return_value=_closed_clock()), \
         patch("src.trader_cli.broker.get_latest_trade", return_value={"price": 100.0}), \
         patch("src.trader_cli.broker.submit_order", return_value={"id": "ord5"}) as s:
        rc = trader_cli.main(["buy", "VSXY", "--notional", "4000",
                              "--buffer-bps", "10"])
    assert rc == 0
    assert s.call_args.kwargs["limit_price"] == 100.1


def test_cancel_and_cancel_all(capsys):
    with patch("src.trader_cli.broker.cancel_order", return_value={"ok": True}) as c:
        assert trader_cli.main(["cancel", "abc-123"]) == 0
    c.assert_called_once_with("abc-123")
    with patch("src.trader_cli.broker.cancel_all_orders", return_value={"ok": True}) as c:
        assert trader_cli.main(["cancel-all"]) == 0
    c.assert_called_once_with()


def test_quote_command(capsys):
    with patch("src.trader_cli.broker.get_latest_trade",
               return_value={"symbol": "SPY", "price": 771.7}):
        assert trader_cli.main(["quote", "SPY"]) == 0
    assert json.loads(capsys.readouterr().out)["price"] == 771.7


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
