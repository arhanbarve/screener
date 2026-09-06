"""Plan / review / execute / journal-draft for the paper trader.

The engine (src.portfolio_engine) is pure. This module does the IO around it:

  gather_inputs()    Alpaca book, plans.json, screen_latest.json, regime,
                     earnings dates, tradable assets, engine state
  cmd_plan()         build the plan, write trading/plans/<target>.json,
                     persist peak/breaker/rank-history state
  cmd_review()       the LLM's per-leg verdict (APPROVE / SKIP / DOWNSIZE /
                     DEFER) — only on legs the engine marked overridable
  cmd_execute()      place the legs in order, crash-safe (the plan file is
                     rewritten after every order), honouring reviews and the
                     session (dust legs need regular hours)
  cmd_journal_draft() pre-fill trading/journal/<target>.md so the reviewer
                     writes analysis, not tables

Files:
  trading/plans/<target>.json          the plan, its reviews and its executions
  trading/engine_state.json            peak equity, breaker, rank history, trims
"""
from __future__ import annotations

import json
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from src import broker, portfolio_engine as pe

ET = ZoneInfo("America/New_York")
SCREENER_DIR = Path(__file__).parent.parent
PLANS_DIR = SCREENER_DIR / "trading" / "plans"
STATE_FILE = SCREENER_DIR / "trading" / "engine_state.json"
SCREEN_FILE = SCREENER_DIR / "output" / "screen_latest.json"
JOURNAL_DIR = SCREENER_DIR / "trading" / "journal"

REVIEW_DECISIONS = ("APPROVE", "SKIP", "DOWNSIZE", "DEFER")

# Scorecard reference points: the paper account's starting equity and the SPY
# close on the day it started, so the journal can show performance vs. SPY
# since inception. Fixed constants, not read from live state, since the
# account was never re-funded.
PAPER_ACCOUNT_BASELINE = 100_000.0
SPY_INCEPTION_PRICE = 738.93


class PlanError(RuntimeError):
    pass


# ── small file helpers ────────────────────────────────────────────────────────

def _read_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return default


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str))
    os.replace(tmp, path)


def plan_path(target_date: str) -> Path:
    return PLANS_DIR / f"{target_date}.json"


def load_state() -> dict:
    return _read_json(STATE_FILE, {})


def save_state(state: dict) -> None:
    _write_json(STATE_FILE, state)


def load_screen(path: Path | None = None) -> dict | None:
    return _read_json(path or SCREEN_FILE, None)


# ── sessions ──────────────────────────────────────────────────────────────────

def latest_completed_session(clock: dict, now: datetime, calendar_days: list[str]) -> str:
    """The most recent trading day whose close has happened. After 16:00 ET on
    a trading day that is today; otherwise the last trading day before today.
    `calendar_days` are exchange trading dates covering the last ~10 days."""
    today = now.date().isoformat()
    days = sorted(d for d in calendar_days if d <= today)
    if not days:
        return today
    if days[-1] == today and (now.hour, now.minute) >= (16, 0) and not clock.get("is_open"):
        return today
    before = [d for d in days if d < today]
    return before[-1] if before else days[-1]


def _calendar_days(now: datetime) -> list[str]:
    start = (now.date() - timedelta(days=12)).isoformat()
    try:
        cal = broker._get("/v2/calendar", {"start": start, "end": now.date().isoformat()})
        return [d["date"] for d in cal]
    except Exception:  # noqa: BLE001 — fall back to weekdays
        return [(now.date() - timedelta(days=i)).isoformat() for i in range(12)
                if (now.date() - timedelta(days=i)).weekday() < 5]


# ── inputs ────────────────────────────────────────────────────────────────────

def gather_inputs(target_date: str, now: datetime | None = None, cfg: dict | None = None) -> dict:
    """Everything build_plan needs, with every optional source failing soft."""
    from src import paper_stops
    from src.config import load_config
    from src.spy_analysis import compute_market_stress_overlay
    from src.finnhub_data import fetch_earnings_calendar, days_to_earnings
    from src.universe import load_tradable_assets

    now = now or datetime.now(ET)
    cfg = cfg or load_config()
    clock = broker.get_clock()
    book = {"account": broker.get_account(), "positions": broker.get_positions(),
            "open_orders": broker.get_orders("open")}
    plans = paper_stops.load_plans()
    screen = load_screen()
    try:
        regime = compute_market_stress_overlay()
    except Exception as e:  # noqa: BLE001
        regime = {"regime": "UNKNOWN", "reason": f"regime unavailable: {e!r}"}
    held = [str(p.get("symbol") or "").upper() for p in book["positions"]]
    try:
        cal = fetch_earnings_calendar()
        earnings = {s: days_to_earnings(s, cal, now.date()) for s in held}
    except Exception:  # noqa: BLE001
        earnings = {}
    assets = load_tradable_assets()
    session = latest_completed_session(clock, now, _calendar_days(now))
    return {"screen": screen, "book": book, "plans": plans, "regime": regime,
            "earnings": earnings, "assets": assets, "state": load_state(),
            "policy": pe.policy_from_config(cfg), "latest_session": session,
            "target_date": target_date, "today": now.date()}


# ── plan ──────────────────────────────────────────────────────────────────────

def cmd_plan(target_date: str, inputs: dict | None = None, write: bool = True) -> dict:
    """Build tonight's plan. Persists peak/breaker/rank-history state; the
    trims_executed record is committed only when the trim leg is executed."""
    inputs = inputs or gather_inputs(target_date)
    plan, new_state = pe.build_plan(
        inputs["screen"], inputs["book"], inputs["plans"], inputs["regime"], inputs["policy"],
        inputs["state"], inputs["today"], target_date, inputs["latest_session"],
        earnings=inputs.get("earnings"), assets=inputs.get("assets"))
    held_without_plan = [s for s in {str(p.get("symbol") or "").upper() for p in inputs["book"]["positions"]}
                         if s not in inputs["plans"] and s != inputs["policy"]["core_symbol"]]
    if held_without_plan:
        plan["warnings"].append(f"no exit plan for {', '.join(sorted(held_without_plan))} — run sync-stops")
    plan["reviews"] = {}
    plan["executed"] = {}
    plan["summary"] = pe.summarize(plan)
    if write:
        existing = _read_json(plan_path(target_date), None)
        if existing and existing.get("executed"):
            # A plan for this session already placed orders. Never overwrite it.
            raise PlanError(f"plan {target_date} already has executions; refusing to rebuild")
        _write_json(plan_path(target_date), plan)
        state = dict(inputs["state"])
        state.update({k: v for k, v in new_state.items() if k != "trims_executed"})
        save_state(state)
    return plan


# ── review ────────────────────────────────────────────────────────────────────

def cmd_review(target_date: str, leg_id: str, decision: str, reason: str,
               notional: float | None = None) -> dict:
    decision = decision.upper()
    if decision not in REVIEW_DECISIONS:
        raise PlanError(f"decision must be one of {REVIEW_DECISIONS}")
    if not reason or len(reason.strip()) < 10:
        raise PlanError("a review needs a reason (≥10 chars) — it goes in the journal")
    path = plan_path(target_date)
    plan = _read_json(path, None)
    if not plan:
        raise PlanError(f"no plan for {target_date}")
    leg = next((l for l in plan["legs"] if l["id"] == leg_id), None)
    if leg is None:
        raise PlanError(f"no leg {leg_id}")
    if leg_id in plan.get("executed", {}):
        raise PlanError(f"{leg_id} already executed")
    if decision != "APPROVE" and not leg["overridable"]:
        raise PlanError(f"{leg_id} ({leg['kind']}) is not overridable — only APPROVE is accepted")
    if decision == "DOWNSIZE":
        if leg["side"] != "buy" or notional is None or notional <= 0 or notional >= (leg["notional"] or 0):
            raise PlanError("DOWNSIZE needs --notional smaller than the leg's notional (buy legs only)")
    if decision == "DEFER" and not leg["risk_reducing"]:
        raise PlanError("DEFER is for risk-reducing legs (rotate_out); use SKIP for entries")
    plan.setdefault("reviews", {})[leg_id] = {
        "decision": decision, "reason": reason.strip(), "notional": notional,
        "at": datetime.now(ET).isoformat(timespec="seconds"),
    }
    _write_json(path, plan)
    return {"leg": leg_id, "review": plan["reviews"][leg_id]}


# ── execute ───────────────────────────────────────────────────────────────────

def _effective(leg: dict, review: dict | None) -> tuple[bool, dict, str]:
    """(execute?, leg-with-size-applied, note)."""
    if not review or review["decision"] == "APPROVE":
        return True, leg, ""
    if review["decision"] in ("SKIP", "DEFER"):
        return False, leg, f"{review['decision'].lower()}ed by reviewer: {review['reason']}"
    if review["decision"] == "DOWNSIZE":
        l = dict(leg)
        l["notional"] = float(review["notional"])
        return True, l, f"downsized to ${review['notional']:,.0f}: {review['reason']}"
    return True, leg, ""


def cmd_execute(target_date: str, legs: list[str] | None = None, risk_only: bool = False,
                dry_run: bool = False, session: str | None = None) -> dict:
    """Place the plan's legs in order. Idempotent: executed legs are skipped.

    risk_only: only legs flagged risk_reducing (the runner's fallback when the
    reviewer session died). dry_run: report what would be sent, send nothing.
    """
    from src import orders as orders_mod
    from src.trader_cli import cmd_order, cmd_cancel_stops

    path = plan_path(target_date)
    plan = _read_json(path, None)
    if not plan:
        raise PlanError(f"no plan for {target_date}")
    session = session or orders_mod.market_session(broker.get_clock())
    wanted = set(legs) if legs else None
    results = []
    state = load_state()

    for leg in plan["legs"]:
        lid = leg["id"]
        if wanted is not None and lid not in wanted:
            continue
        if lid in plan.get("executed", {}):
            results.append({"leg": lid, "status": "already_executed"})
            continue
        if risk_only and not leg["risk_reducing"]:
            results.append({"leg": lid, "status": "skipped", "note": "risk-only run"})
            continue
        go, eff, note = _effective(leg, (plan.get("reviews") or {}).get(lid))
        if not go:
            results.append({"leg": lid, "status": "skipped", "note": note})
            continue
        if leg.get("requires_rth") and session != "open":
            results.append({"leg": lid, "status": "deferred", "note": "needs regular hours (fractional market sell)"})
            continue
        entry = {"leg": lid, "status": "dry_run" if dry_run else "submitted", "note": note}
        if dry_run:
            entry["would_send"] = {k: eff.get(k) for k in ("kind", "symbol", "side", "qty", "notional", "cancel_stop_first")}
            results.append(entry)
            continue
        try:
            if eff.get("cancel_stop_first"):
                entry["cancelled_stops"] = cmd_cancel_stops(apply=True, symbol=eff["symbol"])["stops"]
            out = cmd_order(eff["symbol"], eff["side"], notional=eff.get("notional"), qty=eff.get("qty"))
            entry["order"] = {"id": (out.get("order") or {}).get("id"), "type": out["submitted"].get("order_type"),
                              "limit_price": out["submitted"].get("limit_price"),
                              "qty": out["submitted"].get("qty"), "notional": out["submitted"].get("notional"),
                              "status": (out.get("order") or {}).get("status")}
            entry["reason"] = out.get("reason")
            plan.setdefault("executed", {})[lid] = entry
            if leg["kind"] == "trim":
                te = state.setdefault("trims_executed", {})
                te[leg["symbol"]] = sorted(set(te.get(leg["symbol"]) or []) | set(leg["source"].get("rungs") or []))
                save_state(state)
        except broker.BrokerError as e:
            entry.update(status="failed", error=str(e))
            plan.setdefault("failures", {})[lid] = entry
        _write_json(path, plan)     # after EVERY order, so a crash loses nothing
        results.append(entry)

    plan["executed_at"] = datetime.now(ET).isoformat(timespec="seconds")
    _write_json(path, plan)
    return {"target_date": target_date, "session": session, "risk_only": risk_only,
            "dry_run": dry_run, "results": results,
            "submitted": sum(1 for r in results if r["status"] == "submitted"),
            "failed": sum(1 for r in results if r["status"] == "failed")}


# ── journal draft ─────────────────────────────────────────────────────────────

def _fmt_money(v) -> str:
    try:
        return f"${float(v):,.2f}"
    except (TypeError, ValueError):
        return "—"


def journal_text(plan: dict, book: dict, spy_price: float | None, spy_start: float = SPY_INCEPTION_PRICE,
                 baseline: float = PAPER_ACCOUNT_BASELINE) -> str:
    """Markdown skeleton the reviewer completes. Tables are filled from data;
    the analysis sections are left for the session to write."""
    acct = book.get("account") or {}
    equity = float(acct.get("equity") or 0)
    lines = [
        f"# Trading Journal — {plan['target_date']}", "",
        "**Status:** PLANNED — no orders placed yet", "",
        f"Drafted {plan['as_of']} for the {plan['target_date']} open by the portfolio engine; "
        f"reviewed by the session below. Regime **{plan['regime']['regime']}**, "
        f"alpha target {plan['targets']['alpha_pct']:.0%}"
        + (", **DRAWDOWN BREAKER ACTIVE**" if plan['breaker']['active'] else "") + ".", "",
        "## Snapshot (pre-decision)", "",
        "| | Value |", "|---|---|",
        f"| Equity | {_fmt_money(acct.get('equity'))} |",
        f"| Cash | {_fmt_money(acct.get('cash'))} |",
        f"| Prior equity | {_fmt_money(acct.get('last_equity'))} |", "",
        "| Ticker | Qty | Avg cost | Last | Mkt value | Unreal P&L | % |", "|---|---|---|---|---|---|---|",
    ]
    for p in sorted(book.get("positions") or [], key=lambda x: -float(x.get("market_value") or 0)):
        lines.append(f"| {p.get('symbol')} | {float(p.get('qty') or 0):g} | {_fmt_money(p.get('avg_entry_price'))} | "
                     f"{_fmt_money(p.get('current_price'))} | {_fmt_money(p.get('market_value'))} | "
                     f"{_fmt_money(p.get('unrealized_pl'))} | {float(p.get('unrealized_plpc') or 0):+.2%} |")
    lines += ["", "## Screen", "",
              f"`screen_latest.json` dated {plan.get('screen_date')} — usable: **{plan['screen_usable']}** ({plan['screen_note']}).", "",
              "## Market context", "", "_(session: 2–4 sentences from `trader_cli news` on holdings and legs)_", "",
              "## Engine plan", "", "```", plan.get("summary", pe.summarize(plan)), "```", "",
              "## Review", "", "| Leg | Kind | Symbol | Size | Decision | Reason |", "|---|---|---|---|---|---|"]
    for l in plan["legs"]:
        size = f"${l['notional']:,.0f}" if l.get("notional") else f"{l['qty']:g} sh"
        lines.append(f"| {l['id']} | {l['kind']} | {l['symbol']} | {size} | _pending_ | |")
    lines += ["", "## Orders placed", "", "_(filled in by the session from `execute-plan` output)_", "",
              "## Scorecard", "", "| | Value |", "|---|---|",
              f"| Equity | {_fmt_money(equity)} |", f"| Baseline | {_fmt_money(baseline)} |",
              f"| Account vs baseline | {(equity / baseline - 1):+.2%} |" if baseline else "| Account vs baseline | — |"]
    if spy_price:
        lines += [f"| SPY | {_fmt_money(spy_price)} |", f"| SPY vs start ({_fmt_money(spy_start)}) | {(spy_price / spy_start - 1):+.2%} |",
                  f"| Relative | {((equity / baseline - 1) - (spy_price / spy_start - 1)) * 100:+.2f} pp |"]
    lines += ["", "## Carry-forward", "", "1. "]
    return "\n".join(lines) + "\n"


def cmd_journal_draft(target_date: str, force: bool = False) -> dict:
    plan = _read_json(plan_path(target_date), None)
    if not plan:
        raise PlanError(f"no plan for {target_date}")
    path = JOURNAL_DIR / f"{target_date}.md"
    if path.exists() and not force:
        return {"path": str(path), "written": False, "note": "journal exists — supersede it in place"}
    book = {"account": broker.get_account(), "positions": broker.get_positions()}
    try:
        spy = broker.get_latest_trade("SPY")["price"]
    except Exception:  # noqa: BLE001
        spy = None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(journal_text(plan, book, spy))
    return {"path": str(path), "written": True}
