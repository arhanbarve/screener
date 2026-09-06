"""trader_plan: plan file lifecycle, reviews, crash-safe execution, journal draft."""
import json
from datetime import date, datetime
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest

from src import trader_plan as tp
from src.broker import BrokerError

ET = ZoneInfo("America/New_York")
TARGET = "2026-09-08"


@pytest.fixture
def paths(tmp_path, monkeypatch):
    monkeypatch.setattr(tp, "PLANS_DIR", tmp_path / "plans")
    monkeypatch.setattr(tp, "STATE_FILE", tmp_path / "engine_state.json")
    monkeypatch.setattr(tp, "JOURNAL_DIR", tmp_path / "journal")
    monkeypatch.setattr(tp, "SCREEN_FILE", tmp_path / "screen_latest.json")
    return tmp_path


def _screen():
    return {"date": "2026-09-04", "health": {"ok": True, "reasons": []},
            "rows": [{"ticker": "AAA", "alpaca_symbol": "AAA", "rank": 1, "price": 100.0, "atr_14": 2.0,
                      "entry": "OK", "entry_signal": "", "fund_score": 0.7, "sector": "Tech", "composite": 1.0}],
            "ranking_tail": [{"rank": 1, "ticker": "AAA", "composite": 1.0}]}


def _inputs(positions=(), plans=None, cash=100_000.0, open_orders=()):
    from src.portfolio_engine import DEFAULT_POLICY
    return {"screen": _screen(),
            "book": {"account": {"equity": "100000", "cash": str(cash), "last_equity": "99000"},
                     "positions": list(positions), "open_orders": list(open_orders)},
            "plans": plans or {}, "regime": {"regime": "NORMAL", "reason": ""}, "earnings": {},
            "assets": None, "state": {}, "policy": dict(DEFAULT_POLICY),
            "latest_session": "2026-09-04", "target_date": TARGET, "today": date(2026, 9, 4)}


def test_latest_completed_session():
    days = ["2026-09-01", "2026-09-02", "2026-09-03", "2026-09-04"]
    evening = datetime(2026, 9, 4, 17, 15, tzinfo=ET)
    assert tp.latest_completed_session({"is_open": False}, evening, days) == "2026-09-04"
    morning = datetime(2026, 9, 4, 9, 0, tzinfo=ET)
    assert tp.latest_completed_session({"is_open": False}, morning, days) == "2026-09-03"
    midday = datetime(2026, 9, 4, 13, 0, tzinfo=ET)
    assert tp.latest_completed_session({"is_open": True}, midday, days) == "2026-09-03"
    saturday = datetime(2026, 9, 5, 12, 0, tzinfo=ET)
    assert tp.latest_completed_session({"is_open": False}, saturday, days) == "2026-09-04"
    assert tp.latest_completed_session({"is_open": False}, saturday, []) == "2026-09-05"


def test_cmd_plan_writes_plan_and_state_without_trims(paths):
    inputs = _inputs(positions=[{"symbol": "WIN", "qty": "90", "market_value": "11700", "current_price": "130",
                                 "avg_entry_price": "100", "qty_available": "90"}],
                     plans={"WIN": {"entry_price": 100.0, "stop_floor": 90.0, "stop_level": 110.0, "verdict": "TRIM",
                                    "verdict_reason": "derisk", "trims_fired": ["derisk"], "last_close": 130.0, "risk_R": 10.0}},
                     cash=50_000)
    plan = tp.cmd_plan(TARGET, inputs=inputs)
    saved = json.loads(tp.plan_path(TARGET).read_text())
    assert saved["target_date"] == TARGET and saved["reviews"] == {} and saved["executed"] == {}
    assert any(l["kind"] == "trim" for l in saved["legs"])
    state = json.loads(tp.STATE_FILE.read_text())
    assert state["peak_equity"] == 100_000 and "trims_executed" not in state   # committed only on execution
    assert "summary" in plan and "PLAN for" in plan["summary"]


def test_cmd_plan_warns_about_held_names_without_exit_plan(paths):
    inputs = _inputs(positions=[{"symbol": "ORPHAN", "qty": "10", "market_value": "1000", "current_price": "100",
                                 "avg_entry_price": "90", "qty_available": "10"}], cash=90_000)
    plan = tp.cmd_plan(TARGET, inputs=inputs)
    assert any("ORPHAN" in w and "sync-stops" in w for w in plan["warnings"])


def test_cmd_plan_refuses_to_overwrite_an_executed_plan(paths):
    tp.cmd_plan(TARGET, inputs=_inputs())
    p = json.loads(tp.plan_path(TARGET).read_text())
    p["executed"] = {"L1": {"status": "submitted"}}
    tp.plan_path(TARGET).write_text(json.dumps(p))
    with pytest.raises(tp.PlanError):
        tp.cmd_plan(TARGET, inputs=_inputs())


def test_review_rules(paths):
    tp.cmd_plan(TARGET, inputs=_inputs(positions=[{"symbol": "DYING", "qty": "100", "market_value": "4000",
                                                    "current_price": "40", "avg_entry_price": "50", "qty_available": "100"}],
                                        plans={"DYING": {"entry_price": 50.0, "stop_floor": 45.0, "stop_level": 45.0,
                                                         "verdict": "SELL", "verdict_reason": "floor", "trims_fired": [],
                                                         "last_close": 40.0, "risk_R": 5.0}}, cash=50_000))
    plan = json.loads(tp.plan_path(TARGET).read_text())
    exit_leg = next(l for l in plan["legs"] if l["kind"] == "exit")
    entry_leg = next(l for l in plan["legs"] if l["kind"] == "entry")
    with pytest.raises(tp.PlanError):
        tp.cmd_review(TARGET, exit_leg["id"], "SKIP", "I think it will bounce back tomorrow")
    tp.cmd_review(TARGET, exit_leg["id"], "APPROVE", "stop breached, no argument")
    with pytest.raises(tp.PlanError):
        tp.cmd_review(TARGET, entry_leg["id"], "SKIP", "short")
    with pytest.raises(tp.PlanError):
        tp.cmd_review(TARGET, entry_leg["id"], "DOWNSIZE", "too big for a first tranche")            # no notional
    with pytest.raises(tp.PlanError):
        tp.cmd_review(TARGET, entry_leg["id"], "DOWNSIZE", "too big for a first tranche", notional=99_999)
    out = tp.cmd_review(TARGET, entry_leg["id"], "DOWNSIZE", "too big for a first tranche", notional=3_000)
    assert out["review"]["decision"] == "DOWNSIZE"
    with pytest.raises(tp.PlanError):
        tp.cmd_review(TARGET, entry_leg["id"], "DEFER", "defer is for risk legs only")
    with pytest.raises(tp.PlanError):
        tp.cmd_review(TARGET, "L99", "APPROVE", "no such leg exists here")


def _submit_ok(**spec):
    return {"id": f"ord-{spec['symbol']}", "status": "accepted"}


def test_execute_honours_reviews_order_and_is_crash_safe(paths):
    tp.cmd_plan(TARGET, inputs=_inputs(
        positions=[{"symbol": "DYING", "qty": "100", "market_value": "4000", "current_price": "40",
                    "avg_entry_price": "50", "qty_available": "100"},
                   {"symbol": "EVC", "qty": "0.318", "market_value": "2.6", "current_price": "8.2",
                    "avg_entry_price": "11", "qty_available": "0.318"}],
        plans={"DYING": {"entry_price": 50.0, "stop_floor": 45.0, "stop_level": 45.0, "verdict": "SELL",
                         "verdict_reason": "floor", "trims_fired": [], "last_close": 40.0, "risk_R": 5.0}},
        cash=50_000, open_orders=[{"symbol": "DYING", "type": "stop", "side": "sell", "id": "s1"}]))
    plan = json.loads(tp.plan_path(TARGET).read_text())
    entry_leg = next(l for l in plan["legs"] if l["kind"] == "entry")
    tp.cmd_review(TARGET, entry_leg["id"], "DOWNSIZE", "first tranche only, add on confirmation", notional=2_000)
    core_leg = next(l for l in plan["legs"] if l["kind"] == "core_buy")
    tp.cmd_review(TARGET, core_leg["id"], "SKIP", "keep cash tonight ahead of CPI print")

    closed = {"is_open": False, "next_open": "2026-09-08T09:30:00-04:00", "next_close": "2026-09-08T16:00:00-04:00"}
    with patch("src.trader_cli.broker.get_clock", return_value=closed), \
         patch("src.trader_cli.broker.get_latest_trade", side_effect=lambda s: {"price": 40.0 if s == "DYING" else 100.0}), \
         patch("src.trader_cli.broker.get_orders", return_value=[{"id": "s1", "symbol": "DYING", "side": "sell",
                                                                  "type": "stop", "stop_price": "45"}]), \
         patch("src.trader_cli.broker.cancel_order", return_value={"ok": True}) as cancel, \
         patch("src.trader_cli.broker.submit_order", side_effect=_submit_ok) as submit:
        out = tp.cmd_execute(TARGET, session="closed")

    statuses = {r["leg"]: r["status"] for r in out["results"]}
    exit_id = next(l["id"] for l in plan["legs"] if l["kind"] == "exit")
    dust_id = next(l["id"] for l in plan["legs"] if l["kind"] == "dust")
    assert statuses[exit_id] == "submitted" and statuses[entry_leg["id"]] == "submitted"
    assert statuses[core_leg["id"]] == "skipped"
    assert statuses[dust_id] == "deferred"                 # fractional market sell needs RTH
    cancel.assert_called_once_with("s1")                   # stop cleared before the exit sell
    sent = [c.kwargs for c in submit.call_args_list]
    assert sent[0]["symbol"] == "DYING" and sent[0]["side"] == "sell"          # sells first
    assert sent[1]["symbol"] == "AAA" and sent[1]["qty"] == pytest.approx(2_000 / 100.5, abs=1e-6)   # downsized
    saved = json.loads(tp.plan_path(TARGET).read_text())
    assert set(saved["executed"]) == {exit_id, entry_leg["id"]}
    assert saved["executed"][exit_id]["order"]["id"] == "ord-DYING"

    # Re-running places nothing new: idempotent.
    with patch("src.trader_cli.broker.get_clock", return_value=closed), \
         patch("src.trader_cli.broker.submit_order") as submit2:
        out2 = tp.cmd_execute(TARGET, session="closed")
    submit2.assert_not_called()
    assert {r["leg"]: r["status"] for r in out2["results"]}[exit_id] == "already_executed"


def test_execute_risk_only_and_failure_recorded(paths):
    tp.cmd_plan(TARGET, inputs=_inputs(
        positions=[{"symbol": "DYING", "qty": "100", "market_value": "4000", "current_price": "40",
                    "avg_entry_price": "50", "qty_available": "100"}],
        plans={"DYING": {"entry_price": 50.0, "stop_floor": 45.0, "stop_level": 45.0, "verdict": "SELL",
                         "verdict_reason": "floor", "trims_fired": [], "last_close": 40.0, "risk_R": 5.0}},
        cash=50_000))
    closed = {"is_open": False, "next_open": "2026-09-08T09:30:00-04:00", "next_close": "2026-09-08T16:00:00-04:00"}
    with patch("src.trader_cli.broker.get_clock", return_value=closed), \
         patch("src.trader_cli.broker.get_latest_trade", return_value={"price": 40.0}), \
         patch("src.trader_cli.broker.get_orders", return_value=[]), \
         patch("src.trader_cli.broker.submit_order", side_effect=BrokerError("403 wash trade")):
        out = tp.cmd_execute(TARGET, risk_only=True, session="closed")
    kinds = {r["leg"]: r["status"] for r in out["results"]}
    plan = json.loads(tp.plan_path(TARGET).read_text())
    exit_id = next(l["id"] for l in plan["legs"] if l["kind"] == "exit")
    entry_id = next(l["id"] for l in plan["legs"] if l["kind"] == "entry")
    assert kinds[exit_id] == "failed" and kinds[entry_id] == "skipped"
    assert out["failed"] == 1 and "wash trade" in plan["failures"][exit_id]["error"]
    assert exit_id not in plan["executed"]                 # a failed leg is retryable


def test_execute_trim_commits_trims_executed_state(paths):
    tp.cmd_plan(TARGET, inputs=_inputs(
        positions=[{"symbol": "WIN", "qty": "90", "market_value": "11700", "current_price": "130",
                    "avg_entry_price": "100", "qty_available": "90"}],
        plans={"WIN": {"entry_price": 100.0, "stop_floor": 90.0, "stop_level": 110.0, "verdict": "TRIM",
                       "verdict_reason": "derisk", "trims_fired": ["derisk"], "last_close": 130.0, "risk_R": 10.0}},
        cash=50_000))
    assert "trims_executed" not in json.loads(tp.STATE_FILE.read_text())
    closed = {"is_open": False, "next_open": "2026-09-08T09:30:00-04:00", "next_close": "2026-09-08T16:00:00-04:00"}
    with patch("src.trader_cli.broker.get_clock", return_value=closed), \
         patch("src.trader_cli.broker.get_latest_trade", return_value={"price": 130.0}), \
         patch("src.trader_cli.broker.get_orders", return_value=[]), \
         patch("src.trader_cli.broker.submit_order", side_effect=_submit_ok):
        tp.cmd_execute(TARGET, risk_only=True, session="closed")
    assert json.loads(tp.STATE_FILE.read_text())["trims_executed"] == {"WIN": ["derisk"]}


def test_execute_dry_run_sends_nothing(paths):
    tp.cmd_plan(TARGET, inputs=_inputs())
    with patch("src.trader_cli.broker.submit_order") as submit:
        out = tp.cmd_execute(TARGET, dry_run=True, session="closed")
    submit.assert_not_called()
    assert all(r["status"] == "dry_run" for r in out["results"]) and out["results"][0]["would_send"]["symbol"] == "AAA"


def test_journal_draft_written_once(paths):
    tp.cmd_plan(TARGET, inputs=_inputs())
    with patch("src.trader_plan.broker.get_account", return_value={"equity": "100000", "cash": "100000", "last_equity": "99000"}), \
         patch("src.trader_plan.broker.get_positions", return_value=[]), \
         patch("src.trader_plan.broker.get_latest_trade", return_value={"price": 770.0}):
        out = tp.cmd_journal_draft(TARGET)
        again = tp.cmd_journal_draft(TARGET)
    text = Path(out["path"]).read_text()
    assert out["written"] and not again["written"]
    assert text.startswith(f"# Trading Journal — {TARGET}")
    assert "**Status:** PLANNED" in text and "## Engine plan" in text and "| L1 |" in text
    assert "SPY vs start" in text
