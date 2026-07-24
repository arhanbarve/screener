import json
from unittest.mock import patch, MagicMock

import pytest

from src import broker


def _resp(status=200, body=None):
    m = MagicMock()
    m.status_code = status
    m.json.return_value = body if body is not None else {}
    m.text = json.dumps(body) if body is not None else ""
    return m


@pytest.fixture(autouse=True)
def alpaca_env(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "test-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test-secret")
    monkeypatch.delenv("TRADER_DRY_RUN", raising=False)


def test_headers_missing_keys_raises(monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    with pytest.raises(broker.BrokerError, match="Missing ALPACA"):
        broker._headers()


def test_get_account_hits_paper_endpoint():
    with patch("src.broker.requests.get", return_value=_resp(body={"equity": "100000"})) as g:
        out = broker.get_account()
    assert out == {"equity": "100000"}
    url = g.call_args[0][0]
    assert url == "https://paper-api.alpaca.markets/v2/account"
    headers = g.call_args[1]["headers"]
    assert headers["APCA-API-KEY-ID"] == "test-key"
    assert headers["APCA-API-SECRET-KEY"] == "test-secret"


def test_get_non_200_raises():
    with patch("src.broker.requests.get", return_value=_resp(status=403, body={"message": "forbidden"})):
        with pytest.raises(broker.BrokerError, match="403"):
            broker.get_clock()


def test_submit_order_notional_payload():
    with patch("src.broker.requests.post", return_value=_resp(body={"id": "abc"})) as p:
        out = broker.submit_order("stx", "buy", notional=5000.555)
    assert out == {"id": "abc"}
    payload = p.call_args[1]["json"]
    assert payload == {
        "symbol": "STX",
        "side": "buy",
        "type": "market",
        "time_in_force": "day",
        "notional": 5000.56,
    }


def test_submit_order_qty_payload():
    with patch("src.broker.requests.post", return_value=_resp(body={"id": "abc"})) as p:
        broker.submit_order("SPY", "sell", qty=3)
    assert p.call_args[1]["json"]["qty"] == "3"
    assert "notional" not in p.call_args[1]["json"]


def test_submit_order_requires_exactly_one_size():
    with pytest.raises(broker.BrokerError):
        broker.submit_order("SPY", "buy")
    with pytest.raises(broker.BrokerError):
        broker.submit_order("SPY", "buy", notional=100, qty=1)


def test_submit_order_dry_run(monkeypatch):
    monkeypatch.setenv("TRADER_DRY_RUN", "1")
    with patch("src.broker.requests.post") as p:
        out = broker.submit_order("STX", "buy", notional=1000)
    p.assert_not_called()
    assert out["dry_run"] is True
    assert out["would_submit"]["symbol"] == "STX"


def test_close_position_dry_run(monkeypatch):
    monkeypatch.setenv("TRADER_DRY_RUN", "1")
    with patch("src.broker.requests.delete") as d:
        out = broker.close_position("stx")
    d.assert_not_called()
    assert out == {"dry_run": True, "would_close": "STX"}


def test_close_position_deletes():
    with patch("src.broker.requests.delete", return_value=_resp(body={"id": "ord1"})) as d:
        out = broker.close_position("STX")
    assert d.call_args[0][0] == "https://paper-api.alpaca.markets/v2/positions/STX"
    assert out == {"id": "ord1"}


def test_get_orders_passes_status():
    with patch("src.broker.requests.get", return_value=_resp(body=[])) as g:
        broker.get_orders("closed")
    assert g.call_args[1]["params"] == {"status": "closed", "limit": 100}
