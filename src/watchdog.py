"""Health watchdog for the trading system.

Every failure this session diagnosed was silent. The Fidelity parser wrote zeros
for five days and reported success. The trader crashed on four of nine days and
nothing surfaced it. Five market orders sat unpriced overnight. In each case the
data needed to notice was already on disk — nothing was looking at it.

This module looks at it. Each check is a pure function over data someone else
gathered, so the whole set is testable without a broker or a filesystem. run()
does the gathering and returns a report; scripts/watchdog.sh wraps it for launchd
and notifies on anything that is not OK.

Severities: "ok" (nothing to do), "warn" (worth knowing), "fail" (acts wrong now).
"""
import json
import subprocess
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

SCREENER_DIR = Path(__file__).parent.parent
ET = ZoneInfo("America/New_York")

# A snapshot older than this on a trading day means the page is showing stale P&L.
SNAPSHOT_STALE_DAYS = 1
# An open order older than this has missed at least one session it should have
# filled in.
ORDER_STALE_HOURS = 30


def _r(name: str, status: str, detail: str, **extra) -> dict:
    return {"check": name, "status": status, "detail": detail, **extra}


# ── individual checks (pure) ──────────────────────────────────────────────────

def check_snapshot_freshness(snapshot: dict | None, today: str) -> dict:
    """The Paper page renders from the snapshot; a stale one shows stale money."""
    if not snapshot:
        return _r("snapshot", "fail", "no data/alpaca/portfolio.json — Paper page is empty")
    synced = str(snapshot.get("synced_at") or "")
    if not synced:
        return _r("snapshot", "fail", "snapshot has no synced_at")
    try:
        age = (date.fromisoformat(today) - datetime.fromisoformat(synced).date()).days
    except ValueError:
        return _r("snapshot", "warn", f"unparseable synced_at {synced!r}")
    if age > SNAPSHOT_STALE_DAYS:
        return _r("snapshot", "fail", f"snapshot is {age}d stale (synced {synced})", age_days=age)
    return _r("snapshot", "ok", f"synced {synced}", age_days=age)


def check_reconciliation(snapshot: dict | None) -> dict:
    """realized + unrealized must equal equity - deposits.

    This is the invariant that proved FIFO was the right lot-matching method. If
    it drifts, the P&L on the page is wrong in a way no other check would see.
    """
    if not snapshot:
        return _r("reconciliation", "warn", "no snapshot to reconcile")
    rec = snapshot.get("reconciliation") or {}
    if not rec:
        return _r("reconciliation", "warn", "snapshot carries no reconciliation block")
    if rec.get("ok"):
        return _r("reconciliation", "ok", f"drift ${rec.get('drift', 0):,.2f}")
    return _r("reconciliation", "fail",
              f"P&L does not reconcile against equity — drift ${rec.get('drift', 0):,.2f}",
              drift=rec.get("drift"))


def check_cadence(cadence: dict | None, today: str) -> dict:
    """Did the trader run and finish on the most recent market days?"""
    if not cadence or not cadence.get("days"):
        return _r("cadence", "warn", "no run history in snapshot")
    days = cadence["days"]
    summary = cadence.get("summary") or {}
    broken = summary.get("broken", 0)
    total = summary.get("market_days", len(days))
    latest = max(days, key=lambda d: d["date"])

    # Today is only judged once the window has had a fair chance.
    if latest["date"] == today and latest["status"] in ("running", "no_log"):
        return _r("cadence", "ok", f"today still in progress ({latest['status']})")
    if latest["status"] in ("crashed", "gate_failed", "no_log"):
        return _r("cadence", "fail",
                  f"most recent session {latest['date']} is {latest['label']}; "
                  f"{broken}/{total} market days broken",
                  latest=latest["date"], latest_status=latest["status"])
    if broken:
        return _r("cadence", "warn",
                  f"latest session ok but {broken}/{total} market days broken")
    return _r("cadence", "ok", f"{summary.get('completed', 0)}/{total} completed")


def check_stale_orders(open_orders: list[dict], now: datetime) -> dict:
    """An order open far longer than a session has quietly failed to fill."""
    stale = []
    for o in open_orders or []:
        submitted = str(o.get("submitted_at") or "")
        if not submitted:
            continue
        try:
            ts = datetime.fromisoformat(submitted.replace("Z", "+00:00"))
        except ValueError:
            continue
        hours = (now - ts).total_seconds() / 3600
        if hours > ORDER_STALE_HOURS:
            stale.append({"ticker": o.get("symbol"), "id": o.get("id"),
                          "hours": round(hours, 1)})
    if stale:
        return _r("stale_orders", "warn",
                  f"{len(stale)} order(s) open >{ORDER_STALE_HOURS}h: "
                  + ", ".join(f"{s['ticker']} {s['hours']}h" for s in stale),
                  orders=stale)
    return _r("stale_orders", "ok", f"{len(open_orders or [])} open, none stale")


def check_unpriced_orders(open_orders: list[dict], session: str) -> dict:
    """No market order may rest while the market cannot fill it.

    This is the 2026-08-04 failure as a standing check: five market orders queued
    14 hours ahead of the open, filling at whatever the auction printed.
    """
    if session == "open":
        return _r("unpriced_orders", "ok", "regular hours — market orders fill now")
    bad = [o for o in (open_orders or []) if o.get("type") == "market"]
    if bad:
        return _r("unpriced_orders", "fail",
                  f"{len(bad)} market order(s) resting while market is {session}: "
                  + ", ".join(f"{o.get('side')} {o.get('symbol')}" for o in bad)
                  + " — these fill at the next auction at an unknown price",
                  orders=[{"ticker": o.get("symbol"), "side": o.get("side"),
                           "id": o.get("id")} for o in bad])
    return _r("unpriced_orders", "ok", f"no market orders resting ({session})")


def check_protective_stops(positions: list[dict], open_orders: list[dict],
                           skipped: list[dict] | None = None) -> dict:
    """Every position should have a resting stop, or a recorded reason not to."""
    if not positions:
        return _r("stops", "ok", "no positions")
    resting = {(o.get("symbol") or "").upper() for o in (open_orders or [])
               if o.get("type") == "stop" and o.get("side") == "sell"}
    excused = {(s.get("ticker") or "").upper() for s in (skipped or [])}
    held = {(p.get("symbol") or "").upper() for p in positions}
    missing = sorted(held - resting - excused)
    if missing:
        return _r("stops", "warn",
                  f"{len(missing)}/{len(held)} position(s) with no resting stop: "
                  + ", ".join(missing) + " — run `trader_cli sync-stops --apply`",
                  missing=missing)
    return _r("stops", "ok", f"{len(resting)} resting, {len(excused)} excused")


def check_tests(returncode: int | None, tail: str = "") -> dict:
    if returncode is None:
        return _r("tests", "warn", "test suite not run")
    if returncode == 0:
        return _r("tests", "ok", tail.strip()[-120:] or "suite passed")
    return _r("tests", "fail", f"test suite failing: {tail.strip()[-200:]}")


def check_git_clean(status_porcelain: str, ahead: int) -> dict:
    """Uncommitted or unpushed work means the cloud dashboard is behind."""
    dirty = [l for l in status_porcelain.splitlines() if l.strip()]
    if dirty and ahead:
        return _r("git", "warn", f"{len(dirty)} uncommitted file(s), {ahead} unpushed commit(s)")
    if dirty:
        return _r("git", "warn", f"{len(dirty)} uncommitted file(s)")
    if ahead:
        return _r("git", "warn", f"{ahead} commit(s) not pushed — cloud is behind")
    return _r("git", "ok", "clean and pushed")


# ── report assembly ───────────────────────────────────────────────────────────

def summarize(checks: list[dict]) -> dict:
    counts = {"ok": 0, "warn": 0, "fail": 0}
    for c in checks:
        counts[c["status"]] = counts.get(c["status"], 0) + 1
    overall = "fail" if counts["fail"] else ("warn" if counts["warn"] else "ok")
    return {"overall": overall, "counts": counts,
            "problems": [c for c in checks if c["status"] != "ok"]}


def render(report: dict) -> str:
    """One line per check, worst first — designed to be read in a notification."""
    icon = {"ok": "  ok  ", " warn ": " warn ", "warn": " warn ", "fail": " FAIL "}
    order = {"fail": 0, "warn": 1, "ok": 2}
    lines = [f"WATCHDOG {report['summary']['overall'].upper()} · "
             f"{report['generated_at']}"]
    for c in sorted(report["checks"], key=lambda x: order[x["status"]]):
        lines.append(f"{icon.get(c['status'], '  ?   ')} {c['check']:18} {c['detail']}")
    return "\n".join(lines)


def run(run_tests: bool = True) -> dict:
    """Gather everything and evaluate. Read-only: places and cancels nothing."""
    from src import broker, orders, paper

    now_et = datetime.now(ET)
    today = now_et.date().isoformat()
    checks: list[dict] = []

    snapshot = paper.load_snapshot()
    checks.append(check_snapshot_freshness(snapshot, today))
    checks.append(check_reconciliation(snapshot))
    checks.append(check_cadence((snapshot or {}).get("cadence"), today))

    open_orders: list[dict] = []
    positions: list[dict] = []
    session = "unknown"
    try:
        clock = broker.get_clock()
        session = orders.market_session(clock, now_et)
        open_orders = broker.get_orders("open")
        positions = broker.get_positions()
        checks.append(_r("broker", "ok", f"reachable, session={session}"))
    except Exception as e:  # noqa: BLE001
        checks.append(_r("broker", "fail", f"broker unreachable: {type(e).__name__}: {e}"))

    if session != "unknown":
        checks.append(check_unpriced_orders(open_orders, session))
        checks.append(check_stale_orders(open_orders, datetime.now(ET)))
        checks.append(check_protective_stops(positions, open_orders))

    if run_tests:
        proc = subprocess.run(
            ["python3", "-m", "pytest", "-q", "--timeout=300"],
            cwd=SCREENER_DIR, capture_output=True, text=True, timeout=900,
        )
        tail = (proc.stdout or "").strip().splitlines()
        checks.append(check_tests(proc.returncode, tail[-1] if tail else ""))

    try:
        st = subprocess.run(["git", "status", "--porcelain"], cwd=SCREENER_DIR,
                            capture_output=True, text=True, timeout=60).stdout
        ahead_out = subprocess.run(
            ["git", "rev-list", "--count", "@{u}..HEAD"], cwd=SCREENER_DIR,
            capture_output=True, text=True, timeout=60).stdout.strip()
        checks.append(check_git_clean(st, int(ahead_out or 0)))
    except Exception as e:  # noqa: BLE001
        checks.append(_r("git", "warn", f"git check failed: {e}"))

    report = {
        "generated_at": now_et.isoformat(timespec="seconds"),
        "session": session,
        "checks": checks,
    }
    report["summary"] = summarize(checks)
    return report


def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="watchdog", description=__doc__)
    p.add_argument("--json", action="store_true", help="emit the raw report")
    p.add_argument("--skip-tests", action="store_true", help="skip the pytest run")
    a = p.parse_args(argv)

    report = run(run_tests=not a.skip_tests)
    print(json.dumps(report, indent=2) if a.json else render(report))
    return 0 if report["summary"]["overall"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
