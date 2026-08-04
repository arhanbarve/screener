"""Watchdog health checks.

Every failure this system has had was silent, so these tests are mostly about a
check actually going red — a watchdog that stays green through a real problem is
worse than none, because it manufactures confidence.
"""
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from src.watchdog import (
    check_cadence,
    check_git_clean,
    check_protective_stops,
    check_reconciliation,
    check_snapshot_freshness,
    check_stale_orders,
    check_tests,
    check_unpriced_orders,
    render,
    summarize,
)

ET = ZoneInfo("America/New_York")
TODAY = "2026-08-05"


# ── snapshot freshness ────────────────────────────────────────────────────────

def test_fresh_snapshot_ok():
    assert check_snapshot_freshness({"synced_at": "2026-08-05T09:40"}, TODAY)["status"] == "ok"


def test_yesterday_snapshot_tolerated():
    assert check_snapshot_freshness({"synced_at": "2026-08-04T17:40"}, TODAY)["status"] == "ok"


def test_week_old_snapshot_fails():
    r = check_snapshot_freshness({"synced_at": "2026-07-30T16:56"}, TODAY)
    assert r["status"] == "fail" and r["age_days"] == 6


def test_missing_snapshot_fails():
    assert check_snapshot_freshness(None, TODAY)["status"] == "fail"


def test_snapshot_without_timestamp_fails():
    assert check_snapshot_freshness({}, TODAY)["status"] == "fail"


def test_unparseable_timestamp_warns():
    assert check_snapshot_freshness({"synced_at": "nonsense"}, TODAY)["status"] == "warn"


# ── reconciliation ────────────────────────────────────────────────────────────

def test_reconciliation_ok():
    r = check_reconciliation({"reconciliation": {"ok": True, "drift": 0.08}})
    assert r["status"] == "ok"


def test_reconciliation_drift_fails():
    """The invariant that proved FIFO right; if it drifts the page lies."""
    r = check_reconciliation({"reconciliation": {"ok": False, "drift": 97.38}})
    assert r["status"] == "fail" and r["drift"] == 97.38


def test_reconciliation_absent_warns():
    assert check_reconciliation({})["status"] == "warn"


# ── cadence ───────────────────────────────────────────────────────────────────

def _day(d, status, label=None):
    return {"date": d, "status": status, "label": label or status.upper()}


def test_cadence_all_clean():
    cad = {"days": [_day("2026-08-04", "ok"), _day("2026-08-05", "ok")],
           "summary": {"broken": 0, "market_days": 2, "completed": 2}}
    assert check_cadence(cad, TODAY)["status"] == "ok"


def test_cadence_latest_crashed_fails():
    cad = {"days": [_day("2026-08-04", "ok"), _day("2026-08-05", "crashed")],
           "summary": {"broken": 1, "market_days": 2, "completed": 1}}
    r = check_cadence(cad, TODAY)
    assert r["status"] == "fail" and r["latest_status"] == "crashed"


def test_cadence_today_still_running_is_ok():
    """Mid-session must not read as a failure."""
    cad = {"days": [_day("2026-08-05", "running")],
           "summary": {"broken": 1, "market_days": 1, "completed": 0}}
    assert check_cadence(cad, TODAY)["status"] == "ok"


def test_cadence_history_broken_but_latest_ok_warns():
    cad = {"days": [_day("2026-08-04", "crashed"), _day("2026-08-05", "ok")],
           "summary": {"broken": 1, "market_days": 2, "completed": 1}}
    assert check_cadence(cad, TODAY)["status"] == "warn"


def test_cadence_missing_warns():
    assert check_cadence(None, TODAY)["status"] == "warn"


# ── the 2026-08-04 failure, as a standing check ───────────────────────────────

def test_resting_market_order_outside_hours_fails():
    orders = [{"symbol": "SPY", "side": "buy", "type": "market", "id": "a"},
              {"symbol": "HUT", "side": "sell", "type": "market", "id": "b"}]
    r = check_unpriced_orders(orders, "closed")
    assert r["status"] == "fail"
    assert "unknown price" in r["detail"]
    assert {o["ticker"] for o in r["orders"]} == {"SPY", "HUT"}


def test_resting_limit_orders_outside_hours_are_fine():
    orders = [{"symbol": "SPY", "side": "buy", "type": "limit", "id": "a"}]
    assert check_unpriced_orders(orders, "closed")["status"] == "ok"


def test_market_orders_fine_during_regular_hours():
    orders = [{"symbol": "SPY", "side": "buy", "type": "market", "id": "a"}]
    assert check_unpriced_orders(orders, "open")["status"] == "ok"


def test_stop_orders_are_not_flagged_as_unpriced():
    orders = [{"symbol": "MYE", "side": "sell", "type": "stop", "id": "a"}]
    assert check_unpriced_orders(orders, "closed")["status"] == "ok"


# ── stale orders ──────────────────────────────────────────────────────────────

def _now():
    return datetime(2026, 8, 6, 12, 0, tzinfo=ET)


def test_old_order_warns():
    orders = [{"symbol": "SPY", "id": "a",
               "submitted_at": "2026-08-04T23:03:48Z"}]
    r = check_stale_orders(orders, _now())
    assert r["status"] == "warn" and r["orders"][0]["ticker"] == "SPY"


def test_recent_order_ok():
    orders = [{"symbol": "SPY", "id": "a",
               "submitted_at": (_now() - timedelta(hours=2)).isoformat()}]
    assert check_stale_orders(orders, _now())["status"] == "ok"


def test_no_orders_ok():
    assert check_stale_orders([], _now())["status"] == "ok"


def test_order_without_timestamp_ignored():
    assert check_stale_orders([{"symbol": "SPY"}], _now())["status"] == "ok"


# ── protective stops ──────────────────────────────────────────────────────────

def test_missing_stops_warn():
    positions = [{"symbol": "MYE", "qty": "100"}, {"symbol": "GTX", "qty": "50"}]
    r = check_protective_stops(positions, [])
    assert r["status"] == "warn" and r["missing"] == ["GTX", "MYE"]


def test_stops_present_ok():
    positions = [{"symbol": "MYE", "qty": "100"}]
    orders = [{"symbol": "MYE", "side": "sell", "type": "stop", "stop_price": "31.45"}]
    assert check_protective_stops(positions, orders)["status"] == "ok"


def test_excused_position_not_flagged():
    """A position whose shares are committed elsewhere is legitimately stopless."""
    positions = [{"symbol": "HUT", "qty": "36"}]
    r = check_protective_stops(positions, [], skipped=[{"ticker": "HUT"}])
    assert r["status"] == "ok"


def test_no_positions_ok():
    assert check_protective_stops([], [])["status"] == "ok"


# ── tests + git ───────────────────────────────────────────────────────────────

def test_failing_suite_fails():
    assert check_tests(1, "2 failed, 551 passed")["status"] == "fail"


def test_passing_suite_ok():
    assert check_tests(0, "553 passed")["status"] == "ok"


def test_unrun_suite_warns():
    assert check_tests(None)["status"] == "warn"


def test_git_clean_ok():
    assert check_git_clean("", 0)["status"] == "ok"


def test_git_dirty_warns():
    assert check_git_clean(" M src/foo.py\n", 0)["status"] == "warn"


def test_git_unpushed_warns():
    r = check_git_clean("", 2)
    assert r["status"] == "warn" and "not pushed" in r["detail"]


# ── aggregation ───────────────────────────────────────────────────────────────

def test_summary_worst_status_wins():
    checks = [{"check": "a", "status": "ok", "detail": ""},
              {"check": "b", "status": "warn", "detail": ""},
              {"check": "c", "status": "fail", "detail": ""}]
    s = summarize(checks)
    assert s["overall"] == "fail" and len(s["problems"]) == 2


def test_summary_all_ok():
    assert summarize([{"check": "a", "status": "ok", "detail": ""}])["overall"] == "ok"


def test_render_puts_failures_first():
    report = {"generated_at": "2026-08-05T10:00:00-04:00",
              "checks": [{"check": "snapshot", "status": "ok", "detail": "fresh"},
                         {"check": "cadence", "status": "fail", "detail": "crashed"}],
              "summary": {"overall": "fail"}}
    out = render(report)
    assert out.splitlines()[1].strip().startswith("FAIL")
    assert "cadence" in out.splitlines()[1]


# ── a failing check must always say why ───────────────────────────────────────
# The first live run reported "tests: FAIL" with an empty detail, because the
# watchdog passed --timeout (needs pytest-timeout, not installed), pytest exited
# nonzero on the unrecognised flag, and the reason went to stderr while only
# stdout was read. A watchdog that fails without a reason is barely a watchdog.

def test_failing_tests_check_always_carries_a_reason():
    r = check_tests(2, "error: unrecognized arguments: --timeout=300")
    assert r["status"] == "fail"
    assert "unrecognized arguments" in r["detail"]


def test_failing_tests_check_with_no_output_still_flags():
    r = check_tests(1, "")
    assert r["status"] == "fail"
    assert r["detail"].strip()  # never an empty explanation


def test_watchdog_pytest_args_are_all_supported():
    """Guards the exact regression: --timeout needs a plugin this env lacks.

    Checks the argv literals rather than the whole source, so the comment
    explaining why the flag is absent does not trip the test — which it did on
    the first attempt at this guard.
    """
    import re
    import inspect
    from src import watchdog

    src = inspect.getsource(watchdog.run)
    m = re.search(r'\["python3",\s*"-m",\s*"pytest"(.*?)\]', src, re.S)
    assert m, "could not locate the pytest argv in watchdog.run"
    passed = re.findall(r'"(--?[\w-]+[^"]*)"', m.group(1))
    assert passed == ["-q"], f"unexpected pytest flags: {passed}"
