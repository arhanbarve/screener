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
