# Phase 2 — Portfolio Engine, Reviewer Session, Runner Fallback (Plan 03)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace discretionary LLM trading with a deterministic engine whose legs the LLM reviews within bounded powers, and make risk-reducing exits happen even when the LLM session dies.

**Architecture:** `portfolio_engine.build_plan` (pure) → `trader_plan` (IO: gather inputs, write `trading/plans/<target>.json`, reviews, crash-safe execution, journal draft) → `trader_cli` subcommands (`screen`, `news`, `plan`, `review`, `execute-plan`, `journal-draft`, `buy --replace-stop`) → `run_trader.sh` builds the plan before the session and executes `--risk-only` on the final failed attempt → `trading/PROMPT.md` becomes a reviewer prompt with a 40-turn cap. Policy = **aggressive**: 1.25% risk per trade at the floor, 12% name cap, 35% sector cap, max 8 alpha positions, alpha target 70/40/15% by regime with a SPY core sleeve above a 5% cash buffer, −15% breaker, ≤ 6% into earnings.

**Tech Stack:** dataclasses, JSON files, Alpaca REST via `src.broker`, Finnhub news.

**Live probe (2026-09-04 12:30 ET, no orders):** against the real paper book (SPY core sleeve plus a few real-money-scale alpha names, dust in two tiny legacy positions) the engine produced: two dust sells (RTH-only), one entry sized at 1.25% risk on a 2×ATR floor and capped by the sector cap because the 09-03 CSV has no sectors, one SPY core top-up (`cancel-stop` because a floor rests), and a run of sensible rejections (entry WAIT, news avoid, not tradable, sector-cap sized below the $500 minimum). Projected cash after: ≈5% of equity.

---

### Task 2.1: `src/portfolio_engine.py` — the pure engine

**Files:**
- Create: `src/portfolio_engine.py`
- Create: `tests/test_portfolio_engine.py`

Read the module docstring first; it states the design rules. `build_plan` returns `(plan, new_state)`; `validate_plan` enforces invariants (no entries from an unusable screen, projected cash ≥ −$1, fixed legs are not overridable); `summarize` renders the plan for the journal.

- [ ] **Step 1:** **Create `tests/test_portfolio_engine.py` with exactly this content:**

```python
"""Portfolio engine: sizing, sleeves, exits, breaker, and every rejection path."""
from datetime import date

import pytest

from src import portfolio_engine as pe

TODAY = date(2026, 9, 4)
TARGET = "2026-09-08"
SESSION = "2026-09-04"


def _row(ticker, rank, price=100.0, atr=2.0, entry="OK", signal="", fund=0.6, sector="Technology",
         dte=None, conviction=6, composite=1.0, alpaca_symbol=None):
    return {"ticker": ticker, "alpaca_symbol": alpaca_symbol or ticker, "rank": rank, "price": price,
            "atr_14": atr, "entry": entry, "entry_signal": signal, "fund_score": fund, "sector": sector,
            "days_to_earnings": dte, "conviction": conviction, "composite": composite,
            "composite_final": composite}


def _screen(rows, ok=True, d=SESSION, tail=None):
    return {"date": d, "health": {"ok": ok, "reasons": [] if ok else ["adv_survivors=53 below floor 800"]},
            "rows": rows, "ranking_tail": tail if tail is not None else
            [{"rank": r["rank"], "ticker": r["ticker"], "composite": r["composite"]} for r in rows]}


def _pos(sym, qty, price, avg=None, avail=None):
    return {"symbol": sym, "qty": str(qty), "market_value": str(qty * price), "current_price": str(price),
            "avg_entry_price": str(avg if avg is not None else price),
            "qty_available": str(avail if avail is not None else qty)}


def _book(equity=100_000.0, cash=100_000.0, positions=(), open_orders=()):
    return {"account": {"equity": str(equity), "cash": str(cash)},
            "positions": list(positions), "open_orders": list(open_orders)}


def _plan(entry, floor, level, verdict="HOLD", reason=None, trims=(), last=None, risk=None):
    return {"entry_price": entry, "stop_floor": floor, "stop_level": level, "verdict": verdict,
            "verdict_reason": reason, "trims_fired": list(trims), "last_close": last or entry,
            "risk_R": risk if risk is not None else entry - floor}


NORMAL = {"regime": "NORMAL", "reason": "all clear"}
P = dict(pe.DEFAULT_POLICY)


def build(screen, book, plans=None, regime=NORMAL, state=None, policy=P, earnings=None, assets=None):
    return pe.build_plan(screen, book, plans or {}, regime, policy, state or {}, TODAY, TARGET, SESSION,
                         earnings=earnings, assets=assets)


def legs(plan, kind=None):
    return [l for l in plan["legs"] if kind is None or l["kind"] == kind]


# ── policy / helpers ──────────────────────────────────────────────────────────

def test_policy_from_config_overlays_nested_and_flat():
    pol = pe.policy_from_config({"trader": {"name_cap": 0.10, "alpha_target": {"NORMAL": 0.8}}})
    assert pol["name_cap"] == 0.10 and pol["alpha_target"]["NORMAL"] == 0.8
    assert pol["alpha_target"]["STRESS"] == 0.15          # untouched keys survive
    assert pe.DEFAULT_POLICY["name_cap"] == 0.12          # defaults not mutated


def test_screen_usable_rules():
    ok, _ = pe.screen_usable(_screen([_row("A", 1)]), TARGET, SESSION)
    assert ok
    assert pe.screen_usable(_screen([_row("A", 1)], ok=False), TARGET, SESSION)[0] is False
    assert pe.screen_usable(_screen([_row("A", 1)], d="2026-09-03"), TARGET, SESSION)[0] is False
    assert pe.screen_usable(_screen([]), TARGET, SESSION)[0] is False
    assert pe.screen_usable(None, TARGET, SESSION)[0] is False


def test_breaker_hysteresis():
    pol = P
    b = pe.update_breaker({}, 100_000, pol)
    assert b["active"] is False and b["peak_equity"] == 100_000
    b = pe.update_breaker({"peak_equity": 100_000}, 84_000, pol)
    assert b["active"] is True and b["drawdown"] == pytest.approx(0.16)
    b = pe.update_breaker({"peak_equity": 100_000, "breaker_active": True}, 90_000, pol)
    assert b["active"] is True                            # 10% dd: still above the 7% recovery line
    b = pe.update_breaker({"peak_equity": 100_000, "breaker_active": True}, 93_500, pol)
    assert b["active"] is False
    b = pe.update_breaker({"peak_equity": 100_000}, 120_000, pol)
    assert b["peak_equity"] == 120_000


def test_stop_distance_fallback_and_cap():
    floor, dist, note = pe.stop_distance(100.0, 2.0, P)
    assert (floor, dist) == (96.0, 4.0) and "ATR" in note
    floor, dist, note = pe.stop_distance(100.0, None, P)
    assert dist == pytest.approx(8.0) and "fallback" in note
    floor, dist, note = pe.stop_distance(100.0, 20.0, P)      # 40 > 15% cap
    assert dist == pytest.approx(15.0) and "capped" in note


def test_consecutive_out_of_band():
    h = [{"rank": 5}, {"rank": 40}, {"rank": None}, {"rank": 36}]
    assert pe.consecutive_out_of_band(h, 35) == 3
    assert pe.consecutive_out_of_band([{"rank": 10}], 35) == 0
    assert pe.consecutive_out_of_band([], 35) == 0


def test_update_rank_history_dedupes_by_date_and_drops_unheld():
    st = {"rank_history": {"A": [{"date": "2026-09-03", "rank": 4}], "GONE": [{"date": "2026-09-03", "rank": 1}]}}
    h = pe.update_rank_history(st, ["A", "B"], {"A": 6}, "2026-09-04", keep=6)
    assert h["A"] == [{"date": "2026-09-03", "rank": 4}, {"date": "2026-09-04", "rank": 6}]
    assert h["B"] == [{"date": "2026-09-04", "rank": None}] and "GONE" not in h
    h2 = pe.update_rank_history({"rank_history": h}, ["A", "B"], {"A": 7}, "2026-09-04", keep=6)
    assert len(h2["A"]) == 2 and h2["A"][-1]["rank"] == 7   # same date replaced, not appended


# ── entries and sizing ────────────────────────────────────────────────────────

def test_empty_book_entries_sized_by_risk_and_core_fills_the_rest():
    screen = _screen([_row("AAA", 1, price=100, atr=2.0), _row("BBB", 2, price=50, atr=5.0)])
    plan, state = build(screen, _book())
    entries = legs(plan, "entry")
    assert [l["symbol"] for l in entries] == ["AAA", "BBB"]
    # risk 1.25% of 100k = $1,250 at a $4 stop → 312.5 sh → $31,250, capped by name cap 12% = $12,000
    assert entries[0]["notional"] == pytest.approx(12_000.0)
    assert "name cap" in entries[0]["reason"]
    # BBB: 2×ATR = $10 = 20% of price → capped at 15% = $7.50 → 166.67 sh → $8,333
    assert entries[1]["notional"] == pytest.approx(1_250 / 7.5 * 50, abs=0.01)
    assert "capped" in entries[1]["reason"]
    core = legs(plan, "core_buy")
    assert len(core) == 1 and core[0]["symbol"] == "SPY"
    # core = 95% of equity − alpha buys
    assert core[0]["notional"] == pytest.approx(95_000 - 12_000 - 1_250 / 7.5 * 50, abs=0.01)
    assert plan["exposure_after"]["cash"] == pytest.approx(5_000.0)
    assert [l["kind"] for l in plan["legs"]] == ["entry", "entry", "core_buy"]
    assert state["peak_equity"] == 100_000 and state["breaker_active"] is False


def test_alpha_budget_caps_total_entries_by_regime():
    rows = [_row(f"T{i}", i + 1, price=100, atr=10.0) for i in range(8)]   # each wants $6,250
    plan, _ = build(_screen(rows), _book(), regime={"regime": "STRESS"})
    total = sum(l["notional"] for l in legs(plan, "entry"))
    assert total <= 0.15 * 100_000 + 1e-6
    assert any("alpha budget" in r["reason"] or "sized below" in r["reason"] for r in plan["rejected"]) or total == pytest.approx(15_000)


def test_unknown_regime_treated_as_caution():
    plan, _ = build(_screen([_row("AAA", 1)]), _book(), regime={"regime": "WEIRD"})
    assert plan["regime"]["regime"] == "UNKNOWN" and plan["targets"]["alpha_pct"] == 0.40


def test_max_positions_counts_existing_and_new():
    held = [_pos(f"H{i}", 10, 100.0) for i in range(7)]
    rows = [_row("N1", 1), _row("N2", 2)]
    plan, _ = build(_screen(rows), _book(cash=90_000, positions=held))
    assert [l["symbol"] for l in legs(plan, "entry")] == ["N1"]
    assert any(r["symbol"] == "N2" and "max positions" in r["reason"] for r in plan["rejected"])


def test_sector_cap_binds_across_entries():
    rows = [_row("A", 1, atr=1.0, sector="Healthcare"), _row("B", 2, atr=1.0, sector="Healthcare"),
            _row("C", 3, atr=1.0, sector="Healthcare"), _row("D", 4, atr=1.0, sector="Healthcare")]
    plan, _ = build(_screen(rows), _book())
    hc = sum(l["notional"] for l in legs(plan, "entry"))
    assert hc <= 0.35 * 100_000 + 1e-6


def test_rejections_each_path():
    rows = [
        _row("WAIT", 1, entry="WAIT"),
        _row("AVOID", 2, signal="avoid"),
        _row("CHEAP", 3, price=3.0),
        _row("BADFUND", 4, fund=0.1),
        _row("OTC", 5, alpaca_symbol="OTCX"),
        _row("PEND", 6),
        _row("EARN", 7, dte=3),
        _row("TIGHT", 8, price=100.0, atr=0.5),          # 1% stop
        _row("OK1", 9),
    ]
    assets = {r["alpaca_symbol"]: {"fractionable": True} for r in rows if r["ticker"] != "OTC"}
    book = _book(open_orders=[{"symbol": "PEND", "type": "limit", "side": "buy"}])
    plan, _ = build(_screen(rows), book, assets=assets)
    reasons = {r["symbol"]: r["reason"] for r in plan["rejected"]}
    assert "entry grade WAIT" in reasons["WAIT"]
    assert "avoid" in reasons["AVOID"]
    assert "below $5" in reasons["CHEAP"]
    assert "fund_score" in reasons["BADFUND"]
    assert "not tradable" in reasons["OTC"]
    assert "already resting" in reasons["PEND"]
    assert "earnings in 3d" in reasons["EARN"]
    assert "stop too tight" in reasons["TIGHT"]
    assert [l["symbol"] for l in legs(plan, "entry")] == ["OK1"]


def test_nan_fund_score_passes_when_gate_was_skipped_upstream():
    plan, _ = build(_screen([_row("A", 1, fund=float("nan"))]), _book())
    assert legs(plan, "entry")


def test_non_fractionable_rounds_to_whole_shares():
    rows = [_row("A", 1, price=333.0, atr=33.3)]                # 20% stop → capped 15% → $50 dist → 25 sh → $8,325
    plan, _ = build(_screen(rows), _book(), assets={"A": {"fractionable": False}})
    e = legs(plan, "entry")[0]
    assert e["notional"] == pytest.approx(25 * 333.0)
    assert (e["notional"] / 333.0) == pytest.approx(round(e["notional"] / 333.0))


def test_entry_below_min_notional_rejected():
    plan, _ = build(_screen([_row("A", 1)]), _book(equity=100_000, cash=300.0))
    assert not legs(plan, "entry") and any("sized below" in r["reason"] for r in plan["rejected"])


def test_no_entries_from_unusable_screen_but_exits_still_happen():
    book = _book(cash=50_000, positions=[_pos("DYING", 100, 40.0, avg=50.0)])
    plans = {"DYING": _plan(50.0, 45.0, 45.0, verdict="SELL", reason="close 40 below floor 45", last=40.0)}
    plan, _ = build(_screen([_row("NEW", 1)], ok=False), book, plans)
    kinds = [l["kind"] for l in plan["legs"]]
    assert "entry" not in kinds and "exit" in kinds
    assert plan["screen_usable"] is False and any("no new entries" in w for w in plan["warnings"])


# ── exits, trims, earnings, rotation ──────────────────────────────────────────

def test_exit_leg_is_fixed_and_cancels_resting_stop_and_funds_buys():
    book = _book(cash=1_000, positions=[_pos("DYING", 100, 40.0, avg=50.0)],
                 open_orders=[{"symbol": "DYING", "type": "stop", "side": "sell"}])
    plans = {"DYING": _plan(50.0, 45.0, 45.0, verdict="SELL", reason="floor", last=40.0)}
    plan, _ = build(_screen([_row("NEW", 1, atr=1.0)]), book, plans)
    ex = legs(plan, "exit")[0]
    assert ex["overridable"] is False and ex["risk_reducing"] is True and ex["cancel_stop_first"] is True
    assert ex["qty"] == 100
    # proceeds (97% haircut of $4,000) fund the entry
    e = legs(plan, "entry")[0]
    assert e["notional"] <= 1_000 + 4_000 * 0.97 + 1e-6
    assert plan["legs"][0]["kind"] == "exit"                # sells first


def test_trim_fires_once_and_records_rung():
    book = _book(cash=50_000, positions=[_pos("WIN", 90, 130.0, avg=100.0)])
    plans = {"WIN": _plan(100.0, 90.0, 110.0, verdict="TRIM", reason="derisk", trims=["derisk"], last=130.0)}
    plan, state = build(_screen([]), book, plans)
    t = legs(plan, "trim")[0]
    assert t["qty"] == pytest.approx(30.0) and t["overridable"] is False
    assert state["trims_executed"] == {"WIN": ["derisk"]}
    plan2, _ = build(_screen([]), book, plans, state=state)
    assert not legs(plan2, "trim")                         # same rung never re-fires
    plans["WIN"]["trims_fired"] = ["derisk", "blowoff"]
    plan3, _ = build(_screen([]), book, plans, state=state)
    assert legs(plan3, "trim")[0]["source"]["rungs"] == ["blowoff"]


def test_earnings_trim_to_max_weight_only_when_oversized_and_imminent():
    book = _book(cash=20_000, positions=[_pos("BIG", 100, 100.0)])       # 10% of equity
    plan, _ = build(_screen([]), book, earnings={"BIG": 2})
    t = legs(plan, "earnings_trim")[0]
    assert t["qty"] == pytest.approx(40.0)                  # 10% → 6%
    assert t["overridable"] is False
    plan, _ = build(_screen([]), book, earnings={"BIG": 3})
    assert not legs(plan, "earnings_trim")
    small = _book(cash=20_000, positions=[_pos("SMALL", 50, 100.0)])     # 5%
    plan, _ = build(_screen([]), small, earnings={"SMALL": 1})
    assert not legs(plan, "earnings_trim")


def test_rotate_out_after_three_screens_below_band_and_deferrable():
    book = _book(cash=20_000, positions=[_pos("OLD", 100, 100.0)])
    state = {"rank_history": {"OLD": [{"date": "2026-09-02", "rank": 40}, {"date": "2026-09-03", "rank": None}]}}
    screen = _screen([_row("NEW", 1)], tail=[{"rank": 1, "ticker": "NEW", "composite": 1.0}])
    plan, new_state = build(screen, book, state=state)
    ro = legs(plan, "rotate_out")
    assert len(ro) == 1 and ro[0]["overridable"] is True and ro[0]["risk_reducing"] is True
    assert new_state["rank_history"]["OLD"][-1] == {"date": SESSION, "rank": None}
    # Two misses only → no rotation yet
    plan2, _ = build(screen, book, state={"rank_history": {"OLD": [{"date": "2026-09-03", "rank": None}]}})
    assert not legs(plan2, "rotate_out")


def test_rotate_out_not_evaluated_on_unusable_screen():
    book = _book(cash=20_000, positions=[_pos("OLD", 100, 100.0)])
    state = {"rank_history": {"OLD": [{"date": "2026-09-02", "rank": None}, {"date": "2026-09-03", "rank": None}]}}
    plan, new_state = build(_screen([_row("NEW", 1)], ok=False), book, state=state)
    assert not legs(plan, "rotate_out")
    assert len(new_state["rank_history"]["OLD"]) == 2        # nothing appended from a bad screen


def test_held_names_in_screen_are_not_re_entered_and_can_be_added_at_1R():
    book = _book(cash=50_000, positions=[_pos("WIN", 100, 120.0, avg=100.0)],
                 open_orders=[{"symbol": "WIN", "type": "stop", "side": "sell"}])
    plans = {"WIN": _plan(100.0, 90.0, 105.0, last=120.0)}          # +2R, trailing stop 105
    plan, _ = build(_screen([_row("WIN", 3, price=120.0, atr=2.0)]), book, plans)
    assert not legs(plan, "entry")
    # 100 sh × $120 = $12,000 = the whole 12% name cap → no room, no add
    assert not legs(plan, "add")
    book2 = _book(cash=50_000, positions=[_pos("WIN", 50, 120.0, avg=100.0)],
                  open_orders=[{"symbol": "WIN", "type": "stop", "side": "sell"}])
    plan2, _ = build(_screen([_row("WIN", 3, price=120.0, atr=2.0)]), book2, plans)
    add2 = legs(plan2, "add")[0]
    # half risk ($625) / $15 stop distance × $120 = $5,000; room = 12,000 − 6,000
    assert add2["notional"] == pytest.approx(5_000.0)
    assert add2["cancel_stop_first"] is True and add2["overridable"] is True


def test_add_blocked_below_1R_or_when_rank_slips_or_entry_wait():
    plans = {"WIN": _plan(100.0, 90.0, 95.0, last=105.0)}          # +0.5R
    book = _book(cash=50_000, positions=[_pos("WIN", 50, 105.0, avg=100.0)])
    assert not legs(build(_screen([_row("WIN", 3, price=105.0)]), book, plans)[0], "add")
    plans["WIN"]["last_close"] = 120.0
    assert not legs(build(_screen([_row("WIN", 25, price=120.0)]), book, plans)[0], "add")
    assert not legs(build(_screen([_row("WIN", 3, price=120.0, entry="WAIT")]), book, plans)[0], "add")


def test_dust_leg_requires_rth():
    book = _book(cash=99_990, positions=[_pos("EVC", 0.318, 8.28)])
    plan, _ = build(_screen([]), book)
    d = legs(plan, "dust")[0]
    assert d["requires_rth"] is True and d["overridable"] is True and d["qty"] == pytest.approx(0.318)


# ── core sleeve and cash ──────────────────────────────────────────────────────

def test_core_rebalances_only_outside_band():
    # 95% target; holding 93% SPY is within the 3% band → no leg
    book = _book(cash=7_000, positions=[_pos("SPY", 93, 1000.0)])
    plan, _ = build(_screen([]), book)
    assert not legs(plan, "core_buy") and not legs(plan, "core_sell")
    book = _book(cash=60_000, positions=[_pos("SPY", 40, 1000.0)])
    plan, _ = build(_screen([]), book)
    assert legs(plan, "core_buy")[0]["notional"] == pytest.approx(55_000)


def test_core_sells_down_when_alpha_needs_room():
    book = _book(cash=2_000, positions=[_pos("SPY", 98, 1000.0)])
    rows = [_row("A", 1, atr=1.0), _row("B", 2, atr=1.0)]          # $12k each (name cap)
    plan, _ = build(_screen(rows), book)
    # Entries limited by cash ($2k): the engine never assumes core proceeds
    # fund alpha the same night (core_sell is planned after entries)
    total_entries = sum(l["notional"] for l in legs(plan, "entry"))
    assert total_entries <= 2_000 + 1e-6
    assert plan["exposure_after"]["cash"] >= -1.0


def test_negative_cash_triggers_fixed_core_sell():
    book = _book(equity=100_000, cash=-3_000, positions=[_pos("SPY", 100, 1000.0)])
    plan, _ = build(_screen([]), book)
    cs = legs(plan, "core_sell")[0]
    assert cs["risk_reducing"] is True and cs["overridable"] is False
    assert cs["qty"] == pytest.approx(3_000 / (1_000 * 0.97))       # sized on haircut proceeds
    assert plan["exposure_after"]["cash"] == pytest.approx(0.0, abs=0.01)


def test_breaker_halves_alpha_and_makes_core_sell_fixed():
    state = {"peak_equity": 120_000}
    book = _book(equity=100_000, cash=10_000, positions=[_pos("SPY", 90, 1000.0)])
    plan, new_state = build(_screen([_row("A", 1, atr=1.0)]), book, state=state)
    assert plan["breaker"]["active"] is True and plan["targets"]["alpha_pct"] == pytest.approx(0.35)
    assert new_state["breaker_active"] is True
    assert any("breaker" in w for w in plan["warnings"])


# ── validation / summary ──────────────────────────────────────────────────────

def test_validate_rejects_bad_plans():
    good, _ = build(_screen([_row("A", 1)]), _book())
    pe.validate_plan(good)
    bad = dict(good); bad["screen_usable"] = False
    with pytest.raises(pe.EngineError):
        pe.validate_plan(bad)
    bad2 = dict(good); bad2["exposure_after"] = {"cash": -500.0, "alpha_mv": 0, "core_mv": 0}
    with pytest.raises(pe.EngineError):
        pe.validate_plan(bad2)


def test_missing_equity_raises():
    with pytest.raises(pe.EngineError):
        build(_screen([]), {"account": {}, "positions": []})


def test_summary_mentions_every_leg_and_hold_when_empty():
    plan, _ = build(_screen([_row("A", 1)]), _book())
    text = pe.summarize(plan)
    assert "L1" in text and "entry" in text and "core_buy" in text
    hold, _ = build(_screen([]), _book(cash=5_000, positions=[_pos("SPY", 95, 1000.0)]))
    assert "hold" in pe.summarize(hold)
```

- [ ] **Step 2:** Run:

```bash
/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m pytest tests/test_portfolio_engine.py -q
```

Expected: FAIL at import.

- [ ] **Step 3:** **Create `src/portfolio_engine.py` with exactly this content:**

```python
"""Deterministic portfolio engine for the Alpaca paper book.

Turns tonight's inputs into a list of order *legs* with reasons:

    screen   output/screen_latest.json (health, rows, ranking_tail)
    book     Alpaca account + positions + open orders
    plans    per-position exit plans (data/alpaca/plans.json, src.paper_stops)
    regime   src.spy_analysis.compute_market_stress_overlay()
    policy   config.yaml `trader` block (risk rules)
    state    trading/engine_state.json (peak equity, breaker, rank history)

Everything here is pure: no network, no clock reads, no files. src.trader_cli
gathers the inputs, calls build_plan, and executes the legs; the LLM reviewer
may SKIP or DOWNSIZE legs marked `overridable`, never the others.

Design rules (decided 2026-09-04, "aggressive" posture):

  * Two sleeves. Alpha = screener picks, sized by regime; core = SPY holds
    whatever equity the alpha sleeve does not, above a cash buffer. The book
    is never parked in cash by accident — the measured drag on this account
    was -4pp of absence from the tape, not bad entries.
  * Risk-budgeted sizing: a new position risks `risk_per_trade` of equity at
    its max-loss floor (entry - 2xATR14, the same floor the resting stop
    uses). A tight stop means a bigger position, capped by `name_cap`.
  * Exits come from the standing exit plan (SELL / TRIM verdicts), from
    falling out of the ranking (`exit_band` for `rotate_out_runs` screens),
    and from an earnings print inside `earnings_trim_days` on an oversized
    position. None of these can be skipped by the reviewer except the
    rotate-out, which may be deferred one session.
  * Drawdown breaker: equity below peak x (1 - breaker_drawdown) halves the
    alpha target until equity recovers to peak x (1 - breaker_recover).
  * Nothing new is bought from a screen whose health.ok is False or that is
    older than the session being decided for. Exits still happen.

Leg ordering is the execution order: sells first so their (haircut) proceeds
can fund the buys at the same open.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from datetime import date

DEFAULT_POLICY: dict = {
    "risk_per_trade": 0.0125,          # of equity, at the max-loss floor
    "name_cap": 0.12,                  # max weight of one alpha name
    "sector_cap": 0.35,                # max weight of one sector (alpha sleeve)
    "max_positions": 8,                # alpha positions (excludes core, dust)
    "alpha_target": {"NORMAL": 0.70, "CAUTION": 0.40, "STRESS": 0.15, "UNKNOWN": 0.40},
    "core_symbol": "SPY",
    "cash_buffer": 0.05,               # kept uninvested
    "core_band": 0.03,                 # of equity; rebalance core only outside it
    "breaker_drawdown": 0.15,
    "breaker_recover": 0.07,
    "earnings_no_entry_days": 5,
    "earnings_trim_days": 2,
    "earnings_max_weight": 0.06,
    "entry_band": 20,
    "exit_band": 35,
    "rotate_out_runs": 3,
    "min_entry_notional": 500.0,
    "min_stop_distance_pct": 0.02,     # a stop closer than this is noise
    "max_stop_distance_pct": 0.15,     # a stop further than this caps the risk unit
    "fallback_stop_pct": 0.08,         # when atr_14 is missing (matches exit_plan)
    "add_trigger_r": 1.0,
    "add_fraction": 0.5,               # of the original risk budget
    "dust_notional": 50.0,
    "min_price": 5.0,
    "entry_grades": ["OK", "STRONG"],
    "avoid_signals": ["avoid"],
    "fund_floor": 0.30,
    "sell_haircut": 0.97,              # fraction of sell proceeds assumed available
    "trim_fraction": 1 / 3,
    "rank_history_keep": 6,
}

# Execution order by kind. Sells first so their proceeds fund the buys.
KIND_ORDER = ["exit", "trim", "earnings_trim", "rotate_out", "dust", "core_sell",
              "entry", "add", "core_buy"]
RISK_REDUCING_KINDS = {"exit", "trim", "earnings_trim", "rotate_out", "core_sell"}
NON_OVERRIDABLE_KINDS = {"exit", "trim", "earnings_trim"}


@dataclass
class Leg:
    id: str
    kind: str
    symbol: str
    side: str
    qty: float | None
    notional: float | None
    reason: str
    risk_reducing: bool
    overridable: bool
    requires_rth: bool = False
    cancel_stop_first: bool = False
    source: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        if d["qty"] is not None:
            d["qty"] = round(d["qty"], 6)
        if d["notional"] is not None:
            d["notional"] = round(d["notional"], 2)
        return d


class EngineError(ValueError):
    """The inputs cannot produce a safe plan (e.g. no equity figure)."""


# ── helpers ───────────────────────────────────────────────────────────────────

def _f(v, default: float = 0.0) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(x) else x


def policy_from_config(cfg: dict) -> dict:
    """DEFAULT_POLICY overlaid with config.yaml's `trader` block."""
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in DEFAULT_POLICY.items()}
    for k, v in ((cfg or {}).get("trader") or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k].update(v)
        else:
            out[k] = v
    return out


def screen_usable(screen: dict | None, target_date: str, latest_session: str) -> tuple[bool, str]:
    """A screen may feed new entries only if it passed its health gate and is
    dated the latest completed session (the one whose close the evening
    session reads). `latest_session` is the most recent trading day ≤ today."""
    if not screen:
        return False, "no screen"
    health = screen.get("health") or {}
    if health.get("ok") is not True:
        return False, "screen health not ok: " + "; ".join(health.get("reasons") or ["unknown"])
    d = str(screen.get("date") or "")
    if d < latest_session:
        return False, f"screen dated {d} is older than the last session {latest_session}"
    if not screen.get("rows"):
        return False, "screen has no rows (stress regime or empty)"
    return True, "ok"


def rank_lookup(screen: dict | None) -> dict[str, int]:
    """ticker -> rank from ranking_tail (top ~100) falling back to rows."""
    out: dict[str, int] = {}
    for r in (screen or {}).get("ranking_tail") or []:
        t = str(r.get("ticker") or "").upper()
        if t and r.get("rank"):
            out[t] = int(r["rank"])
    for r in (screen or {}).get("rows") or []:
        t = str(r.get("ticker") or "").upper()
        if t and r.get("rank") and t not in out:
            out[t] = int(r["rank"])
    return out


def update_breaker(state: dict, equity: float, policy: dict) -> dict:
    """Peak-equity tracking with hysteresis. Returns the breaker block."""
    peak = max(_f(state.get("peak_equity"), 0.0), equity)
    dd = 0.0 if peak <= 0 else 1.0 - equity / peak
    active = bool(state.get("breaker_active", False))
    if active and dd <= policy["breaker_recover"]:
        active = False
    elif not active and dd >= policy["breaker_drawdown"]:
        active = True
    return {"active": active, "drawdown": round(dd, 6), "peak_equity": round(peak, 2)}


def update_rank_history(state: dict, held: list[str], ranks: dict[str, int],
                        screen_date: str | None, keep: int) -> dict:
    """Append today's rank (or None) for every held alpha name; one entry per
    screen date so a rerun does not double-count."""
    hist = dict(state.get("rank_history") or {})
    if not screen_date:
        return hist
    for sym in held:
        entries = [e for e in (hist.get(sym) or []) if e.get("date") != screen_date]
        entries.append({"date": screen_date, "rank": ranks.get(sym)})
        hist[sym] = entries[-keep:]
    # Forget names no longer held.
    return {s: v for s, v in hist.items() if s in held}


def consecutive_out_of_band(entries: list[dict], exit_band: int) -> int:
    """How many most-recent screens in a row the name ranked worse than
    exit_band (None = not in the top ~100 at all)."""
    n = 0
    for e in reversed(entries or []):
        r = e.get("rank")
        if r is None or int(r) > exit_band:
            n += 1
        else:
            break
    return n


def stop_distance(price: float, atr: float | None, policy: dict) -> tuple[float, float, str]:
    """(floor, distance, note) for a new entry: entry − 2×ATR14 clamped to
    [min, max] stop distance; percentage fallback without ATR."""
    if atr is None or not (atr == atr) or atr <= 0:
        dist = price * policy["fallback_stop_pct"]
        note = "fallback 8% stop (no ATR)"
    else:
        dist = 2.0 * atr
        note = "2×ATR14"
    max_d = price * policy["max_stop_distance_pct"]
    if dist > max_d:
        dist, note = max_d, note + f", capped at {policy['max_stop_distance_pct']:.0%}"
    return price - dist, dist, note


# ── the plan ──────────────────────────────────────────────────────────────────

def build_plan(
    screen: dict | None,
    book: dict,
    plans: dict,
    regime: dict | None,
    policy: dict,
    state: dict,
    today: date,
    target_date: str,
    latest_session: str,
    earnings: dict[str, int | None] | None = None,
    assets: dict[str, dict] | None = None,
) -> tuple[dict, dict]:
    """Return (plan, new_state). Pure.

    book: {"account": {...}, "positions": [...], "open_orders": [...]}
    plans: {ticker: {"entry_price", "risk_R", "stop_floor", "stop_level",
                     "verdict", "verdict_reason", "trims_fired", "last_close"}}
    earnings: {ticker: days_to_earnings or None} for held names
    assets: {alpaca_symbol: {"fractionable": bool}} or None
    """
    account = book.get("account") or {}
    equity = _f(account.get("equity"))
    cash = _f(account.get("cash"))
    if equity <= 0:
        raise EngineError("account equity missing or non-positive")
    core_sym = policy["core_symbol"].upper()
    earnings = earnings or {}
    legs: list[Leg] = []
    rejected: list[dict] = []
    warnings: list[str] = []
    counter = iter(range(1, 10_000))

    def new_leg(**kw) -> Leg:
        leg = Leg(id=f"L{next(counter)}", **kw)
        legs.append(leg)
        return leg

    # ── positions ────────────────────────────────────────────────────────────
    positions = {}
    for p in book.get("positions") or []:
        sym = str(p.get("symbol") or "").upper()
        if not sym:
            continue
        positions[sym] = {
            "symbol": sym,
            "qty": _f(p.get("qty")),
            "mv": _f(p.get("market_value")),
            "price": _f(p.get("current_price")),
            "avg_cost": _f(p.get("avg_entry_price")),
            "qty_available": _f(p.get("qty_available"), _f(p.get("qty"))),
        }
    core_pos = positions.get(core_sym)
    dust = {s: p for s, p in positions.items() if s != core_sym and p["mv"] < policy["dust_notional"]}
    alpha = {s: p for s, p in positions.items() if s != core_sym and s not in dust}

    # Open non-stop orders block re-entry in the same symbol tonight.
    pending_syms = {str(o.get("symbol") or "").upper() for o in (book.get("open_orders") or [])
                    if o.get("type") != "stop"}
    stop_syms = {str(o.get("symbol") or "").upper() for o in (book.get("open_orders") or [])
                 if o.get("type") == "stop"}

    # ── regime, breaker, targets ─────────────────────────────────────────────
    regime_name = str((regime or {}).get("regime") or "UNKNOWN").upper()
    if regime_name not in policy["alpha_target"]:
        regime_name = "UNKNOWN"
    breaker = update_breaker(state, equity, policy)
    alpha_target = float(policy["alpha_target"][regime_name])
    if breaker["active"]:
        alpha_target *= 0.5
        warnings.append(f"drawdown breaker active ({breaker['drawdown']:.1%} from peak): alpha target halved")

    # ── screen ───────────────────────────────────────────────────────────────
    usable, why = screen_usable(screen, target_date, latest_session)
    if not usable:
        warnings.append(f"no new entries: {why}")
    ranks = rank_lookup(screen)
    rows = {str(r.get("ticker") or "").upper(): r for r in (screen or {}).get("rows") or []}
    screen_date = (screen or {}).get("date") if screen else None
    rank_history = update_rank_history(state, sorted(alpha), ranks, screen_date if usable else None,
                                       int(policy["rank_history_keep"]))

    # ── exits from the standing exit plan ────────────────────────────────────
    exiting: set[str] = set()
    sell_proceeds = 0.0
    trims_executed = dict(state.get("trims_executed") or {})
    for sym, pos in sorted(alpha.items()):
        plan = plans.get(sym) or {}
        verdict = str(plan.get("verdict") or "HOLD").upper()
        if verdict == "SELL":
            new_leg(kind="exit", symbol=sym, side="sell", qty=pos["qty"], notional=None,
                    reason=f"exit plan SELL: {plan.get('verdict_reason') or 'stop or trend break'}",
                    risk_reducing=True, overridable=False, cancel_stop_first=sym in stop_syms,
                    source={"verdict": verdict, "stop_level": plan.get("stop_level"),
                            "stop_floor": plan.get("stop_floor"), "last_close": plan.get("last_close")})
            exiting.add(sym)
            sell_proceeds += pos["mv"]
            continue
        fired = [t for t in (plan.get("trims_fired") or []) if t not in (trims_executed.get(sym) or [])]
        if verdict == "TRIM" and fired:
            qty = pos["qty"] * float(policy["trim_fraction"])
            new_leg(kind="trim", symbol=sym, side="sell", qty=qty, notional=None,
                    reason=f"exit plan TRIM ({', '.join(fired)}): {plan.get('verdict_reason') or ''}".strip(),
                    risk_reducing=True, overridable=False, cancel_stop_first=sym in stop_syms,
                    source={"rungs": fired})
            sell_proceeds += qty * pos["price"]
            trims_executed[sym] = sorted(set(trims_executed.get(sym) or []) | set(fired))

    # ── earnings trims ───────────────────────────────────────────────────────
    for sym, pos in sorted(alpha.items()):
        if sym in exiting:
            continue
        dte = earnings.get(sym)
        if dte is None or dte > policy["earnings_trim_days"]:
            continue
        weight = pos["mv"] / equity
        if weight > policy["earnings_max_weight"]:
            target_mv = policy["earnings_max_weight"] * equity
            qty = pos["qty"] * (1 - target_mv / pos["mv"])
            new_leg(kind="earnings_trim", symbol=sym, side="sell", qty=qty, notional=None,
                    reason=(f"earnings in {dte}d at {weight:.1%} of equity; trim to "
                            f"{policy['earnings_max_weight']:.0%}"),
                    risk_reducing=True, overridable=False, cancel_stop_first=sym in stop_syms,
                    source={"days_to_earnings": dte, "weight": round(weight, 4)})
            sell_proceeds += qty * pos["price"]

    # ── rotate out of names the screener has dropped ─────────────────────────
    if usable:
        for sym, pos in sorted(alpha.items()):
            if sym in exiting:
                continue
            n_out = consecutive_out_of_band(rank_history.get(sym) or [], int(policy["exit_band"]))
            if n_out >= int(policy["rotate_out_runs"]):
                new_leg(kind="rotate_out", symbol=sym, side="sell", qty=pos["qty"], notional=None,
                        reason=f"ranked below {policy['exit_band']} (or unranked) for {n_out} consecutive screens",
                        risk_reducing=True, overridable=True, cancel_stop_first=sym in stop_syms,
                        source={"rank_history": rank_history.get(sym)})
                exiting.add(sym)
                sell_proceeds += pos["mv"]

    # ── dust ─────────────────────────────────────────────────────────────────
    for sym, pos in sorted(dust.items()):
        if pos["qty"] <= 0:
            continue
        new_leg(kind="dust", symbol=sym, side="sell", qty=pos["qty"], notional=None,
                reason=f"dust position (${pos['mv']:.2f}) — market sell in regular hours",
                risk_reducing=False, overridable=True, requires_rth=True,
                cancel_stop_first=sym in stop_syms)

    # ── exposure after sells ─────────────────────────────────────────────────
    alpha_mv_after = sum(p["mv"] for s, p in alpha.items() if s not in exiting)
    for leg in legs:
        if leg.kind in ("trim", "earnings_trim"):
            alpha_mv_after -= leg.qty * alpha[leg.symbol]["price"]
    n_alpha_after = len([s for s in alpha if s not in exiting])
    sector_mv: dict[str, float] = {}
    for s, p in alpha.items():
        if s in exiting:
            continue
        sec = str((rows.get(s) or {}).get("sector") or "Unknown")
        sector_mv[sec] = sector_mv.get(sec, 0.0) + p["mv"]

    cash_after = cash + sell_proceeds * float(policy["sell_haircut"])
    alpha_budget = alpha_target * equity - alpha_mv_after
    risk_dollars = equity * float(policy["risk_per_trade"])
    buys_total = 0.0

    def room_for(sym: str, notional: float, sector: str) -> tuple[float, str | None]:
        """Cap a buy by name cap, sector cap, alpha budget and cash."""
        held_mv = alpha.get(sym, {}).get("mv", 0.0) if sym not in exiting else 0.0
        caps = [
            (policy["name_cap"] * equity - held_mv, "name cap"),
            (policy["sector_cap"] * equity - sector_mv.get(sector, 0.0), f"sector cap ({sector})"),
            (alpha_budget - buys_total, "alpha budget"),
            (cash_after - buys_total, "cash"),
        ]
        binding = None
        for cap, label in caps:
            if cap < notional:
                notional, binding = cap, label
        return max(notional, 0.0), binding

    # ── entries ──────────────────────────────────────────────────────────────
    if usable:
        candidates = sorted(
            [r for r in rows.values() if int(r.get("rank") or 999) <= int(policy["entry_band"])],
            key=lambda r: int(r.get("rank") or 999))
        for r in candidates:
            sym = str(r.get("ticker") or "").upper()
            asym = str(r.get("alpaca_symbol") or sym).upper()
            price = _f(r.get("price"))
            reason_skip = None
            if sym in alpha or sym in exiting or sym == core_sym:
                continue                                   # adds handled below
            if str(r.get("entry") or "") not in policy["entry_grades"]:
                reason_skip = f"entry grade {r.get('entry')}"
            elif str(r.get("entry_signal") or "") in policy["avoid_signals"]:
                reason_skip = "news signal avoid"
            elif price < policy["min_price"]:
                reason_skip = f"price {price:.2f} below ${policy['min_price']:g}"
            elif r.get("fund_score") is not None and _f(r.get("fund_score"), float("nan")) == _f(r.get("fund_score"), float("nan")) \
                    and _f(r.get("fund_score")) < policy["fund_floor"]:
                reason_skip = f"fund_score {_f(r.get('fund_score')):.2f} below floor"
            elif assets is not None and asym not in assets:
                reason_skip = "not tradable on Alpaca"
            elif asym in pending_syms:
                reason_skip = "an order is already resting for this symbol"
            elif r.get("days_to_earnings") is not None and _f(r.get("days_to_earnings"), 999) <= policy["earnings_no_entry_days"]:
                reason_skip = f"earnings in {int(_f(r.get('days_to_earnings')))}d"
            elif n_alpha_after >= int(policy["max_positions"]):
                reason_skip = f"max positions ({policy['max_positions']}) reached"
            if reason_skip:
                rejected.append({"symbol": sym, "rank": r.get("rank"), "reason": reason_skip})
                continue

            floor, dist, note = stop_distance(price, r.get("atr_14"), policy)
            if dist / price < policy["min_stop_distance_pct"]:
                rejected.append({"symbol": sym, "rank": r.get("rank"), "reason": f"stop too tight ({dist / price:.1%})"})
                continue
            qty = risk_dollars / dist
            notional = qty * price
            sector = str(r.get("sector") or "Unknown")
            notional, binding = room_for(sym, notional, sector)
            fractionable = True if assets is None else bool((assets.get(asym) or {}).get("fractionable", False))
            if not fractionable:
                qty_whole = math.floor(notional / price)
                notional = qty_whole * price
            if notional < policy["min_entry_notional"]:
                rejected.append({"symbol": sym, "rank": r.get("rank"),
                                 "reason": f"sized below ${policy['min_entry_notional']:g}" + (f" ({binding})" if binding else "")})
                continue
            qty = notional / price
            new_leg(kind="entry", symbol=asym, side="buy", qty=None, notional=notional,
                    reason=(f"rank {r.get('rank')}, entry {r.get('entry')}, composite {_f(r.get('composite_final', r.get('composite'))):.3f}, "
                            f"fund {_f(r.get('fund_score'), float('nan')):.2f}; risk ${risk_dollars:,.0f} at floor "
                            f"{floor:.2f} ({note})" + (f"; capped by {binding}" if binding else "")),
                    risk_reducing=False, overridable=True,
                    source={"ticker": sym, "rank": r.get("rank"), "price": price, "floor": round(floor, 4),
                            "atr_14": r.get("atr_14"), "sector": sector, "qty_est": round(qty, 4),
                            "days_to_earnings": r.get("days_to_earnings"), "conviction": r.get("conviction")})
            buys_total += notional
            n_alpha_after += 1
            sector_mv[sector] = sector_mv.get(sector, 0.0) + notional

        # ── adds to winners ──────────────────────────────────────────────────
        for sym, pos in sorted(alpha.items()):
            if sym in exiting:
                continue
            plan = plans.get(sym) or {}
            r = rows.get(sym)
            if not r or int(r.get("rank") or 999) > int(policy["entry_band"]):
                continue
            if str(r.get("entry") or "") not in policy["entry_grades"]:
                continue
            risk_R = _f(plan.get("risk_R"))
            entry = _f(plan.get("entry_price"))
            last = _f(plan.get("last_close"), pos["price"])
            if risk_R <= 0 or entry <= 0:
                continue
            gain_R = (last - entry) / risk_R
            if gain_R < policy["add_trigger_r"]:
                continue
            if _f(r.get("days_to_earnings"), 999) <= policy["earnings_no_entry_days"]:
                continue
            stop = _f(plan.get("stop_level"))
            dist = last - stop if stop > 0 else 0.0
            if dist <= 0 or dist / last < policy["min_stop_distance_pct"]:
                continue
            notional = policy["add_fraction"] * risk_dollars / dist * last
            sector = str(r.get("sector") or "Unknown")
            notional, binding = room_for(sym, notional, sector)
            if notional < policy["min_entry_notional"]:
                continue
            asym = str(r.get("alpaca_symbol") or sym).upper()
            new_leg(kind="add", symbol=asym, side="buy", qty=None, notional=notional,
                    reason=(f"add: +{gain_R:.1f}R with rank {r.get('rank')} and entry {r.get('entry')}; "
                            f"half risk at trailing stop {stop:.2f}" + (f"; capped by {binding}" if binding else "")),
                    risk_reducing=False, overridable=True, cancel_stop_first=asym in stop_syms,
                    source={"gain_R": round(gain_R, 3), "stop_level": stop, "rank": r.get("rank")})
            buys_total += notional
            sector_mv[sector] = sector_mv.get(sector, 0.0) + notional

    # ── core sleeve ──────────────────────────────────────────────────────────
    core_mv = core_pos["mv"] if core_pos else 0.0
    core_target = max(0.0, equity * (1.0 - policy["cash_buffer"]) - (alpha_mv_after + buys_total))
    diff = core_target - core_mv
    band = max(policy["core_band"] * equity, policy["min_entry_notional"])
    cash_left = cash_after - buys_total
    haircut = float(policy["sell_haircut"])
    if cash_left < 0 and core_mv > 0:
        # Sells may not fill or cash was already negative: restore cash ≥ 0 first,
        # sized on haircut proceeds so the projection actually reaches zero.
        qty = min(core_pos["qty"], (-cash_left) / (core_pos["price"] * haircut)) if core_pos["price"] > 0 else 0.0
        if qty > 0:
            new_leg(kind="core_sell", symbol=core_sym, side="sell", qty=qty, notional=None,
                    reason=f"projected cash {cash_left:,.0f} < 0: sell core to restore",
                    risk_reducing=True, overridable=False, cancel_stop_first=core_sym in stop_syms)
    elif diff > band:
        notional = min(diff, max(cash_left, 0.0))
        if notional >= policy["min_entry_notional"]:
            new_leg(kind="core_buy", symbol=core_sym, side="buy", qty=None, notional=notional,
                    reason=(f"core sleeve {core_mv / equity:.1%} → target {core_target / equity:.1%} "
                            f"(alpha {(alpha_mv_after + buys_total) / equity:.1%} of {alpha_target:.0%} target)"),
                    risk_reducing=False, overridable=True, cancel_stop_first=core_sym in stop_syms)
    elif diff < -band and core_pos:
        qty = min(core_pos["qty"], (-diff) / core_pos["price"]) if core_pos["price"] > 0 else 0.0
        if qty * core_pos["price"] >= policy["min_entry_notional"]:
            new_leg(kind="core_sell", symbol=core_sym, side="sell", qty=qty, notional=None,
                    reason=f"core sleeve {core_mv / equity:.1%} above target {core_target / equity:.1%}",
                    risk_reducing=breaker["active"], overridable=not breaker["active"],
                    cancel_stop_first=core_sym in stop_syms)

    legs.sort(key=lambda l: KIND_ORDER.index(l.kind))
    core_price = core_pos["price"] if core_pos else 0.0
    core_buy_total = sum(l.notional or 0.0 for l in legs if l.kind == "core_buy")
    core_sell_qty = sum(l.qty or 0.0 for l in legs if l.kind == "core_sell")
    projected_cash = cash_after - buys_total - core_buy_total + core_sell_qty * core_price * haircut
    plan = {
        "as_of": today.isoformat(),
        "target_date": target_date,
        "screen_date": screen_date,
        "screen_usable": usable,
        "screen_note": why,
        "regime": {"regime": regime_name, "detail": (regime or {}).get("reason", "")},
        "equity": round(equity, 2),
        "cash": round(cash, 2),
        "targets": {"alpha_pct": round(alpha_target, 4),
                    "core_pct": round(max(0.0, 1.0 - alpha_target - policy["cash_buffer"]), 4),
                    "cash_pct": policy["cash_buffer"]},
        "breaker": breaker,
        "exposure_after": {"alpha_mv": round(alpha_mv_after + buys_total, 2),
                           "core_mv": round(core_mv + core_buy_total - core_sell_qty * core_price, 2),
                           "cash": round(projected_cash, 2)},
        "legs": [l.to_dict() for l in legs],
        "rejected": rejected,
        "warnings": warnings,
        "policy": {k: v for k, v in policy.items() if not isinstance(v, dict)} | {"alpha_target": policy["alpha_target"]},
    }
    new_state = {
        "peak_equity": breaker["peak_equity"],
        "breaker_active": breaker["active"],
        "rank_history": rank_history,
        "trims_executed": trims_executed,
        "last_plan_date": target_date,
    }
    validate_plan(plan)
    return plan, new_state


def validate_plan(plan: dict) -> None:
    """Invariants every plan must satisfy. Raises EngineError."""
    legs = plan.get("legs") or []
    buys = sum(_f(l.get("notional")) for l in legs if l["side"] == "buy")
    if not plan.get("screen_usable") and any(l["kind"] in ("entry", "add") for l in legs):
        raise EngineError("entries planned from an unusable screen")
    if plan["exposure_after"]["cash"] < -1.0:
        raise EngineError(f"plan leaves projected cash negative: {plan['exposure_after']['cash']}")
    for l in legs:
        if l["side"] == "buy" and (l.get("notional") or 0) <= 0:
            raise EngineError(f"buy leg {l['id']} has no notional")
        if l["side"] == "sell" and (l.get("qty") or 0) <= 0:
            raise EngineError(f"sell leg {l['id']} has no qty")
        if l["kind"] in NON_OVERRIDABLE_KINDS and l["overridable"]:
            raise EngineError(f"{l['kind']} leg {l['id']} must not be overridable")
    ids = [l["id"] for l in legs]
    if len(ids) != len(set(ids)):
        raise EngineError("duplicate leg ids")
    _ = buys


def summarize(plan: dict) -> str:
    """Human-readable plan for the journal and the log."""
    lines = [f"PLAN for {plan['target_date']} (as of {plan['as_of']}) — regime {plan['regime']['regime']}, "
             f"equity ${plan['equity']:,.0f}, alpha target {plan['targets']['alpha_pct']:.0%}"]
    if plan["breaker"]["active"]:
        lines.append(f"  BREAKER ACTIVE: {plan['breaker']['drawdown']:.1%} below peak ${plan['breaker']['peak_equity']:,.0f}")
    if not plan["screen_usable"]:
        lines.append(f"  screen unusable: {plan['screen_note']}")
    for w in plan.get("warnings") or []:
        lines.append(f"  ! {w}")
    if not plan["legs"]:
        lines.append("  no legs — hold")
    for l in plan["legs"]:
        size = f"${l['notional']:,.0f}" if l.get("notional") else f"{l['qty']:g} sh"
        flags = " ".join(f for f, on in (("RISK", l["risk_reducing"]), ("fixed", not l["overridable"]),
                                          ("RTH-only", l["requires_rth"]), ("cancel-stop", l["cancel_stop_first"])) if on)
        lines.append(f"  {l['id']:>4} {l['kind']:<13} {l['side']:<4} {l['symbol']:<6} {size:>10}  [{flags}]  {l['reason']}")
    if plan["rejected"]:
        lines.append("  rejected: " + "; ".join(f"{r['symbol']} ({r['reason']})" for r in plan["rejected"][:12]))
    e = plan["exposure_after"]
    lines.append(f"  after: alpha ${e['alpha_mv']:,.0f}, core ${e['core_mv']:,.0f}, cash ${e['cash']:,.0f}")
    return "\n".join(lines)
```

- [ ] **Step 4:** Run:

```bash
/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m pytest tests/test_portfolio_engine.py -q
```

Expected: 31 passed.

- [ ] **Step 5:** Commit as `feat(engine): deterministic portfolio engine with sleeves, risk-budgeted sizing, exits and breaker`. Per the user's global CLAUDE.md §6: run `git status` first and **ask the user before committing**. Never add a Co-Authored-By line.

**Edge cases this task must handle (each has a test):**
- Empty book: entries sized at $1,250 risk over the stop distance, capped at 12% ($12,000); a 20%-wide ATR stop is capped to 15%; core buys 95% − alpha; cash left = 5%.
- STRESS regime caps total entries at 15%; unknown regime → CAUTION (40%).
- Max positions counts held + new; sector cap binds across entries; each rejection path yields a reason (WAIT, avoid, price < $5, fund_score below floor, not tradable, resting order, earnings ≤ 5d, stop too tight, sized below $500).
- NaN fund_score passes (gate skipped upstream); non-fractionable rounds to whole shares.
- Unusable screen: no entries/adds, exits still planned, warning recorded, rank history untouched.
- SELL verdict → fixed exit leg with `cancel_stop_first` when a stop rests; proceeds (×0.97) fund entries; sells come first.
- TRIM fires once per rung; `trims_executed` in the returned state prevents a repeat; a new rung fires again.
- Earnings trim only when ≤ 2 sessions away AND weight > 6%, trims to exactly 6%.
- Rotate-out after 3 consecutive screens below rank 35/unranked; deferrable; not evaluated on an unusable screen; rank history dedupes by screen date.
- Held names are not re-entered; adds at ≥ +1R with rank ≤ 20 and OK/STRONG, sized at half risk over the trailing-stop distance, within name-cap room; blocked below 1R, rank slip, or WAIT.
- Dust (< $50) → RTH-only market sell.
- Core: no trade inside a 3%-of-equity band; buys the shortfall; sells the excess; negative projected cash → fixed core sell sized on haircut proceeds so cash projects to 0.
- Breaker at −15% halves alpha and makes core sells fixed; clears at −7%.
- `validate_plan` rejects entries from an unusable screen and negative projected cash; missing equity raises.


### Task 2.2: `paper_stops` persists trim rungs and verdict reasons

**Files:**
- Modify: `src/paper_stops.py`

The engine needs `trims_fired`, `verdict_reason`, `below_50d_streak` and `last_eval` in `data/alpaca/plans.json`; `build_plans` used to drop them.

- [ ] **Step 1:** Apply:

**Apply this unified diff to `src/paper_stops.py`** (from the reference patch; `patch -p1 < docs/superpowers/plans/2026-09-04-overhaul-reference.patch` applies all of them at once):

```diff
--- a/src/paper_stops.py	2026-08-04 19:15:46
+++ b/src/paper_stops.py	2026-09-04 12:40:35
@@ -109,7 +109,13 @@
             "risk_R": plan.get("risk_R"),
             "peak_close": plan.get("peak_close"),
             "verdict": plan.get("verdict"),
+            # The portfolio engine turns these into orders, so it needs the
+            # reason and the rung list, not just the verdict word.
+            "verdict_reason": plan.get("verdict_reason"),
+            "trims_fired": list(plan.get("trims_fired") or []),
+            "below_50d_streak": plan.get("below_50d_streak"),
             "last_close": plan.get("last_close"),
+            "last_eval": plan.get("last_eval"),
         }
     return plans
 
```

- [ ] **Step 2:** Run:

```bash
/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m pytest tests/test_paper.py -q
```

Expected: all pass (no test asserts on the plan shape beyond stop_floor).

- [ ] **Step 3:** Regenerate the plans file so tonight's plan has the fields: `set -a; source .env; set +a; /Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m src.trader_cli sync-stops` (report-only; it rewrites `data/alpaca/plans.json`). Expected: JSON with `trims_fired` keys.


### Task 2.3: `src/trader_plan.py` — gather, plan, review, execute, journal draft

**Files:**
- Create: `src/trader_plan.py`
- Create: `tests/test_trader_plan.py`

The IO wrapper. `cmd_execute` rewrites the plan file after every order (crash-safe), skips executed legs (idempotent), honours reviews, defers RTH-only legs outside regular hours, records failures as retryable, and commits `trims_executed` only when a trim order is actually submitted.

- [ ] **Step 1:** **Create `tests/test_trader_plan.py` with exactly this content:**

```python
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
```

- [ ] **Step 2:** Run:

```bash
/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m pytest tests/test_trader_plan.py -q
```

Expected: FAIL at import.

- [ ] **Step 3:** **Create `src/trader_plan.py` with exactly this content:**

```python
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


def journal_text(plan: dict, book: dict, spy_price: float | None, spy_start: float = 738.93,
                 baseline: float = 100_000.0) -> str:
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
```

- [ ] **Step 4:** Run:

```bash
/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m pytest tests/test_trader_plan.py -q
```

Expected: 10 passed (after Task 2.4's trader_cli changes — the tests patch `src.trader_cli.broker`).

- [ ] **Step 5:** Commit as `feat(trader): plan/review/execute/journal-draft around the engine`. Per the user's global CLAUDE.md §6: run `git status` first and **ask the user before committing**. Never add a Co-Authored-By line.

**Edge cases this task must handle (each has a test):**
- `latest_completed_session`: after 16:00 on a trading day → today; morning/midday → previous trading day; weekend → last trading day.
- `cmd_plan` refuses to overwrite a plan with executions; warns about held names without an exit plan; persists state without `trims_executed`.
- Reviews: non-overridable → APPROVE only; reason ≥ 10 chars; DOWNSIZE needs a smaller notional on a buy leg; DEFER only on risk-reducing legs; unknown leg rejected.
- Execute: sells before buys, stop cancelled before a sell/add that needs it, downsized notional applied, skipped legs skipped, dust deferred outside RTH, second run places nothing, failures recorded and retryable, `--risk-only` skips entries, `--dry-run` sends nothing.
- Journal draft written once; contains status PLANNED, engine plan block, review table, scorecard with SPY vs start.


### Task 2.4: `trader_cli` subcommands and `--replace-stop`

**Files:**
- Modify: `src/trader_cli.py`

Thin dispatch to `trader_plan`; `screen` gives the LLM a compact JSON view (no CSV parsing); `news` gives it Finnhub headlines because WebSearch/WebFetch are denied headless. Engine/plan errors print JSON to stderr and return 1, like broker errors.

- [ ] **Step 1:** Apply:

**Apply this unified diff to `src/trader_cli.py`** (from the reference patch; `patch -p1 < docs/superpowers/plans/2026-09-04-overhaul-reference.patch` applies all of them at once):

```diff
--- a/src/trader_cli.py	2026-08-05 02:27:01
+++ b/src/trader_cli.py	2026-09-04 12:45:16
@@ -13,6 +13,12 @@
   cancel-all          cancel every open order
   cancel-stops [SYM]  cancel resting protective stops (one symbol, or all)
   sync-stops          rest protective stops at each position's max-loss floor
+  screen              health + top rows of output/screen_latest.json
+  news SYM            recent company news (Finnhub) — works headless
+  plan                build tonight's plan (src.portfolio_engine) → trading/plans/<target>.json
+  review LEG DECISION --reason  APPROVE | SKIP | DOWNSIZE --notional N | DEFER
+  execute-plan        place the plan's legs in order (--risk-only, --legs L1,L2, --dry-run)
+  journal-draft       pre-fill trading/journal/<target>.md from the plan
 
 Order types. buy/sell default to --auto, which picks the order that fits the
 current session: market inside regular hours, otherwise a marketable limit that
@@ -154,14 +160,20 @@
     stop_price: float | None = None,
     buffer_bps: float | None = None,
     allow_extended: bool = False,
+    replace_stop: bool = False,
 ) -> dict:
     """Place an order, choosing the type from the session unless told otherwise.
 
     "auto" is the default because the failure this guards against is not a bad
     limit price, it is an unpriced market order resting overnight.
+
+    replace_stop: cancel this symbol's resting protective stop first. Alpaca
+    rejects a buy while a sell stop rests on the same symbol as a wash trade,
+    and a sell needs the shares the stop holds. sync-stops re-arms the floor.
     """
     from src import orders as orders_mod
 
+    cancelled = cmd_cancel_stops(apply=True, symbol=symbol)["stops"] if replace_stop else []
     session = orders_mod.market_session(broker.get_clock())
 
     if order_type == "auto":
@@ -188,7 +200,11 @@
     why = plan.pop("_why", "")
     plan = {k: v for k, v in plan.items() if v is not None}
     result = broker.submit_order(**plan)
-    return {"session": session, "reason": why, "submitted": plan, "order": result}
+    out = {"session": session, "reason": why, "submitted": plan, "order": result}
+    if replace_stop:
+        out["cancelled_stops"] = cancelled
+        out["note"] = "protective stop cancelled; run sync-stops --apply after the fill"
+    return out
 
 
 def cmd_cancel_stops(apply: bool = True, symbol: str | None = None) -> dict:
@@ -283,6 +299,50 @@
     return result
 
 
+def cmd_screen(path=None, top: int = 20) -> dict:
+    """Compact view of screen_latest.json: health first, then the rows the
+    engine can act on. The LLM reads this instead of parsing the CSV."""
+    from src.trader_plan import load_screen
+    from pathlib import Path
+
+    screen = load_screen(Path(path) if path else None)
+    if not screen:
+        return {"available": False, "health": {"ok": False, "reasons": ["screen_latest.json missing"]}}
+    keep = ["rank", "ticker", "alpaca_symbol", "name", "sector", "composite_final", "composite",
+            "conviction", "fund_score", "entry", "entry_signal", "price", "atr_14",
+            "days_to_earnings", "eps_rev_30d", "rsi_14", "macd", "adx", "pct_from_high", "news_reasoning"]
+    rows = [{k: r.get(k) for k in keep if k in r} for r in (screen.get("rows") or [])[:top]]
+    return {"available": True, "date": screen.get("date"), "generated_at": screen.get("generated_at"),
+            "health": screen.get("health"), "regime": screen.get("regime"), "rows": rows,
+            "ranking_tail": screen.get("ranking_tail", [])[:40]}
+
+
+def cmd_news(symbol: str, days: int = 7, limit: int = 10) -> dict:
+    """Recent company news from Finnhub. Exists because WebSearch/WebFetch are
+    denied to the headless session; without this the reviewer flies blind."""
+    import os
+    from datetime import datetime, timedelta
+
+    import finnhub
+
+    fh = finnhub.Client(api_key=os.environ.get("FINNHUB_API_KEY", ""))
+    end = datetime.now().strftime("%Y-%m-%d")
+    start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
+    try:
+        items = fh.company_news(symbol.upper().replace("-", "."), _from=start, to=end) or []
+    except Exception as e:  # noqa: BLE001
+        return {"symbol": symbol.upper(), "error": f"{type(e).__name__}: {e}", "articles": []}
+    out = []
+    for a in items[:limit]:
+        try:
+            when = datetime.fromtimestamp(int(a.get("datetime") or 0)).strftime("%Y-%m-%d %H:%M")
+        except (TypeError, ValueError, OSError):
+            when = ""
+        out.append({"when": when, "source": a.get("source"), "headline": a.get("headline"),
+                    "summary": (a.get("summary") or "")[:400], "url": a.get("url")})
+    return {"symbol": symbol.upper(), "days": days, "count": len(items), "articles": out}
+
+
 def _with_held_since(positions: list[dict]) -> list[dict]:
     """Attach FIFO held_since from the paper snapshot when available.
 
@@ -326,6 +386,8 @@
                         help="max slippage for an auto limit (default 50bp buy / 200bp sell)")
         sp.add_argument("--extended", dest="allow_extended", action="store_true",
                         help="allow filling in pre/post-market (thin books)")
+        sp.add_argument("--replace-stop", dest="replace_stop", action="store_true",
+                        help="cancel this symbol's protective stop first (adds/sells on a stopped position)")
     sp = sub.add_parser("close")
     sp.add_argument("symbol")
     sp = sub.add_parser("orders")
@@ -342,8 +404,35 @@
     sp = sub.add_parser("sync-stops")
     sp.add_argument("--apply", action="store_true",
                     help="actually place/cancel stops (default reports only)")
+    sp = sub.add_parser("screen")
+    sp.add_argument("--path", default=None)
+    sp.add_argument("--top", type=int, default=20)
+    sp = sub.add_parser("news")
+    sp.add_argument("symbol")
+    sp.add_argument("--days", type=int, default=7)
+    sp.add_argument("--limit", type=int, default=10)
+    sp = sub.add_parser("plan")
+    sp.add_argument("--target", default=None, help="session being decided for (default: gate's target_date)")
+    sp.add_argument("--no-write", action="store_true")
+    sp = sub.add_parser("review")
+    sp.add_argument("leg")
+    sp.add_argument("decision", choices=["APPROVE", "SKIP", "DOWNSIZE", "DEFER"])
+    sp.add_argument("--reason", required=True)
+    sp.add_argument("--notional", type=float, default=None)
+    sp.add_argument("--target", default=None)
+    sp = sub.add_parser("execute-plan")
+    sp.add_argument("--target", default=None)
+    sp.add_argument("--legs", default=None, help="comma-separated leg ids; default all")
+    sp.add_argument("--risk-only", action="store_true")
+    sp.add_argument("--dry-run", action="store_true")
+    sp = sub.add_parser("journal-draft")
+    sp.add_argument("--target", default=None)
+    sp.add_argument("--force", action="store_true")
     args = p.parse_args(argv)
 
+    def _target():
+        return args.target or cmd_gate()["target_date"]
+
     try:
         if args.cmd == "status":
             out = cmd_status()
@@ -356,7 +445,8 @@
                             limit_price=args.limit_price,
                             stop_price=args.stop_price,
                             buffer_bps=args.buffer_bps,
-                            allow_extended=args.allow_extended)
+                            allow_extended=args.allow_extended,
+                            replace_stop=args.replace_stop)
         elif args.cmd == "close":
             out = broker.close_position(args.symbol.upper())
         elif args.cmd == "activity-today":
@@ -371,11 +461,36 @@
             out = cmd_cancel_stops(apply=not args.report_only, symbol=args.symbol)
         elif args.cmd == "sync-stops":
             out = cmd_sync_stops(apply=args.apply)
+        elif args.cmd == "screen":
+            out = cmd_screen(args.path, args.top)
+        elif args.cmd == "news":
+            out = cmd_news(args.symbol, args.days, args.limit)
+        elif args.cmd == "plan":
+            from src import trader_plan
+            out = trader_plan.cmd_plan(_target(), write=not args.no_write)
+            print(out["summary"], file=sys.stderr)
+        elif args.cmd == "review":
+            from src import trader_plan
+            out = trader_plan.cmd_review(_target(), args.leg, args.decision, args.reason, args.notional)
+        elif args.cmd == "execute-plan":
+            from src import trader_plan
+            legs = [l.strip() for l in args.legs.split(",")] if args.legs else None
+            out = trader_plan.cmd_execute(_target(), legs=legs, risk_only=args.risk_only, dry_run=args.dry_run)
+        elif args.cmd == "journal-draft":
+            from src import trader_plan
+            out = trader_plan.cmd_journal_draft(_target(), force=args.force)
         else:  # orders
             out = broker.get_orders(args.status)
     except broker.BrokerError as e:
         print(json.dumps({"error": str(e)}), file=sys.stderr)
         return 1
+    except Exception as e:  # noqa: BLE001 — PlanError, EngineError: same contract, JSON on stderr
+        from src.portfolio_engine import EngineError
+        from src.trader_plan import PlanError
+        if isinstance(e, (EngineError, PlanError)):
+            print(json.dumps({"error": f"{type(e).__name__}: {e}"}), file=sys.stderr)
+            return 1
+        raise
     print(json.dumps(out, indent=2))
     return 0
 
```

- [ ] **Step 2:** Run:

```bash
/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m pytest tests/test_trader_cli.py tests/test_trader_plan.py -q
```

Expected: all pass (39 + 10).

- [ ] **Step 3:** Live, read-only checks:

```bash
set -a; source .env; set +a
/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m src.trader_cli screen --top 5 | head -40
/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m src.trader_cli news VLO --limit 3
/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m src.trader_cli plan --no-write 2>&1 | tail -25
/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m src.trader_cli execute-plan --dry-run --target $(/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m src.trader_cli gate | /Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -c "import json,sys;print(json.load(sys.stdin)['target_date'])") 2>&1 | head -30
```

Expected: `screen` shows `health.ok` and rows; `news` returns headlines; `plan --no-write` prints a PLAN summary with legs and rejections (nothing written); `execute-plan --dry-run` fails with `no plan for …` unless a plan was written — that is correct (nothing sent).

- [ ] **Step 4:** Commit as `feat(trader_cli): screen, news, plan, review, execute-plan, journal-draft; --replace-stop`. Per the user's global CLAUDE.md §6: run `git status` first and **ask the user before committing**. Never add a Co-Authored-By line.


### Task 2.5: Reviewer prompt

**Files:**
- Modify: `trading/PROMPT.md` (full replacement below)

The LLM's role is now: verify health, read news, decide per leg within allowed decisions, journal. Do NOT deploy this file until Task 2.8's gate passes; keep the old file until then (`git stash` or copy aside).

- [ ] **Step 1:** **Create `trading/PROMPT.md` with exactly this content:**

```markdown
# Daily Paper-Trading Session — Reviewer

You review and journal a plan built by a deterministic portfolio engine for an
Alpaca PAPER account. You do not invent trades. The engine owns sizing, sleeves,
exits and risk rules (`config.yaml` → `trader`); you own judgment on the legs it
marked reviewable, the news read, and the written record.

## Hard boundaries

- PAPER account only. Every order goes through `PY -m src.trader_cli` where
  `PY = /Library/Frameworks/Python.framework/Versions/3.14/bin/python3`.
  Never `buy`/`sell`/`close` directly in a session — orders come from
  `execute-plan`. (`sync-stops --apply` is the one exception, step 8.)
- You may `review` a leg only with a decision the engine allows: legs marked
  `overridable: false` accept APPROVE only. You may never add a symbol, upsize a
  leg, or bypass the drawdown breaker.
- NEVER touch `positions.json`; edit nothing outside `trading/`.
- If `screen` reports `health.ok: false`, or `plan` errors, or `status` errors:
  no `execute-plan` except `--risk-only`. Write the journal saying what failed.
- **Journal on disk before `execute-plan`.** `journal-draft` writes it; your
  review decisions must be in it before any order is sent.

## Why it is built this way

Six weeks of full discretion produced 2 winners in 13 closed trades, a book that
was 35% pre-revenue biotech into a rate shock, a 7.9% position carried into a
binary earnings print, and the same operational lessons re-learned three
sessions running. The engine encodes those lessons so they cannot be re-argued
at 17:15 with the market closed. Your value is the part it cannot do: read the
news, notice when the data is wrong, and say so in writing.

## Windows

Evening (16:15–23:59 ET) is primary: the screener finishes ~17:00 and the plan
targets the next open, with every order resting as a priced limit. Morning
(08:30–15:45) is the fallback. `gate` tells you the `target_date`; every command
below defaults to it.

## Procedure

1. `PY -m src.trader_cli screen` — read `health` first. If `ok` is false the
   universe collapsed or a floor was breached: note it and skip to step 5 with
   the plan's risk legs only. Otherwise read the top rows: rank, entry,
   entry_signal, fund_score, days_to_earnings, eps_rev_30d, news_reasoning.
2. `PY -m src.trader_cli status` — equity, cash, positions, resting stops.
3. `PY -m src.trader_cli plan` — builds `trading/plans/<target>.json` and prints
   the summary. Read every leg: kind, size, `risk_reducing`, `overridable`,
   reason. Read `rejected` and `warnings` too — "no exit plan for X" means run
   `sync-stops --apply` before anything else.
4. `PY -m src.trader_cli news SYMBOL` for every leg symbol and every holding
   (≤ 12 calls). You are looking for: a scheduled print the calendar missed, a
   dilution/offering, guidance change, regulatory action, M&A, or index event.
   Price-recap articles are noise.
5. `PY -m src.trader_cli journal-draft` — creates `trading/journal/<target>.md`
   with the snapshot, plan and scorecard filled in. Then EDIT it: write
   **Market context** (2–4 sentences from the news you read) and fill the
   **Review** table with a decision and a one-line reason per leg.
6. Record each decision: `PY -m src.trader_cli review L3 SKIP --reason "..."`.
   Decisions: `APPROVE` (default — you need not record it), `SKIP` (entries,
   adds, core adjustments, dust), `DOWNSIZE --notional N` (buy legs; N below the
   engine's size), `DEFER` (a `rotate_out` only, one session). Reasons ≥ 10
   characters; they are the journal.
   Good reasons to SKIP an entry: an offering or print the engine did not see; a
   thesis-breaking headline; the same sector already dominates the book after
   tonight's fills. Bad reasons: "RSI looks high", "I would rather wait for a
   pullback", "the market feels risky" — the engine already priced regime and
   technicals, and discretionary hesitation cost this account −4pp of absence.
7. `PY -m src.trader_cli execute-plan` — places every non-skipped leg in order
   (sells first). Read the output: `submitted`, `skipped`, `deferred` (dust legs
   wait for regular hours), `failed` (with the broker's message). A failed
   risk-reducing leg is the one thing worth a second attempt: fix the cause
   (usually `cancel-stops SYMBOL` then `execute-plan --legs L1`) and retry once.
8. `PY -m src.trader_cli sync-stops --apply` — re-arms the max-loss floor under
   every position. Idempotent. Record what it placed.
9. Update the journal: status line → `SUBMITTED` (evening) or `EXECUTED`, paste
   the `execute-plan` result into **Orders placed**, note anything that failed,
   and write **Carry-forward** (max 5 items — things the NEXT session must act
   on, not lessons).
10. Fridays: `trading/journal/weekly/YYYY-Www.md` — equity vs SPY for the week,
    best and worst leg, one lesson about the ENGINE's rules (a proposal for
    `config.yaml`), current book.

## Journal shape

```
# Trading Journal — YYYY-MM-DD
**Status:** PLANNED | SUBMITTED | EXECUTED | PARTIAL | NO TRADES
## Snapshot (pre-decision)       <- drafted
## Screen                        <- drafted (health, date, usable)
## Market context                <- you
## Engine plan                   <- drafted (summary block)
## Review                        <- you: one row per leg, decision + reason
## Orders placed                 <- execute-plan output
## Scorecard                     <- drafted
## Carry-forward                 <- you
```

Keep the whole session under 40 turns. The engine did the arithmetic; do not
redo it. If you run out of turns after `execute-plan`, the plan file has every
order id and the runner commits `trading/` regardless.
```

- [ ] **Step 2:** Commit as `feat(trader): reviewer prompt for the engine-driven session`. Per the user's global CLAUDE.md §6: run `git status` first and **ask the user before committing**. Never add a Co-Authored-By line.


### Task 2.6: Runner — plan before the session, risk-only fallback, reviewer tool set

**Files:**
- Modify: `run_trader.sh`
- Modify: `tests/test_run_trader.sh`

Also fixes a pre-existing defect in the shell suite: its `passed/failed` summary and exit status were printed mid-file, so the last five cases never counted. The suite now ends with the summary and runs 36 cases.

- [ ] **Step 1:** Apply the shell-suite diff:

**Apply this unified diff to `tests/test_run_trader.sh`** (from the reference patch; `patch -p1 < docs/superpowers/plans/2026-09-04-overhaul-reference.patch` applies all of them at once):

```diff
--- a/tests/test_run_trader.sh	2026-08-04 20:27:13
+++ b/tests/test_run_trader.sh	2026-09-04 12:50:37
@@ -36,10 +36,15 @@
     # (the paper-snapshot refresh is a -c invocation and must be a no-op here).
     cat > "$SANDBOX/bin/python3" <<'STUB'
 #!/bin/bash
+# Records every trader_cli subcommand it is asked for, so tests can assert on
+# what the runner invoked (plan, execute-plan --risk-only, sync-stops).
 for a in "$@"; do
     case "$a" in
         gate)           echo "{\"run\": ${GATE_RUN:-true}, \"window\": \"${GATE_WINDOW:-evening}\", \"target_date\": \"${GATE_TARGET:-2099-01-01}\", \"reason\": \"stub\"}"; exit 0 ;;
         activity-today) echo "{\"count\": ${ACT_COUNT:-0}, \"safe_to_retry\": ${ACT_SAFE:-true}}"; exit 0 ;;
+        plan)           echo "plan $*" >> "${CALLS_FILE:-/dev/null}"; [ -n "${PLAN_FAIL:-}" ] && exit 1; mkdir -p trading/plans; echo '{}' > "trading/plans/${GATE_TARGET:-2099-01-01}.json"; exit 0 ;;
+        execute-plan)   echo "execute-plan $*" >> "${CALLS_FILE:-/dev/null}"; exit 0 ;;
+        sync-stops)     echo "sync-stops $*" >> "${CALLS_FILE:-/dev/null}"; exit 0 ;;
     esac
 done
 exit 0
@@ -89,9 +94,10 @@
 # env, not a bare "VAR=x" prefix: "$@" expands after parsing, so bash would treat
 # the first expanded word as the command name instead of an assignment.
 trader() {
-    ( cd "$SANDBOX" && env PATH="$SANDBOX/bin:$PATH" "$@" \
+    ( cd "$SANDBOX" && env PATH="$SANDBOX/bin:$PATH" CALLS_FILE="$SANDBOX/calls.txt" "$@" \
       bash "$SANDBOX/run_trader.sh" >/dev/null 2>&1 )
 }
+CALLS() { cat "$SANDBOX/calls.txt" 2>/dev/null || echo ""; }
 
 TODAY=$(date +%Y-%m-%d)
 TARGET="2099-01-01"   # what the stubbed gate reports as target_date
@@ -336,9 +342,6 @@
     bad "allowedTools has Edit(trading/**)" "Edit rule missing — journal writes would be denied"
 fi
 
-echo
-echo "passed $PASS, failed $FAIL"
-[ "$FAIL" -eq 0 ]
 
 # ── the evening session must not be blocked by today's stamp ───────────────────
 # This is the bug that would have stopped tonight from deciding tomorrow: the
@@ -390,4 +393,77 @@
 [ "$(STAMP)" = "$TODAY" ] \
     && ok "missing target_date falls back to the calendar date" \
     || bad "missing target_date falls back to the calendar date" "stamp=$(STAMP)"
+teardown
+
+# ── 14. the plan is built before the session, and a plan failure does not block it ──
+setup
+trader CLAUDE_RC=0
+if CALLS | grep -q "^plan" && LOGTXT | grep -q "Plan built for $TARGET" && LOGTXT | grep -q "stub claude ran"; then
+    ok "plan is built before the reviewer session"
+else
+    bad "plan is built before the reviewer session" "calls=$(CALLS)"
+fi
+teardown
+setup
+trader CLAUDE_RC=0 PLAN_FAIL=1
+if LOGTXT | grep -q "Plan build FAILED" && LOGTXT | grep -q "stub claude ran" && [ "$(STAMP)" = "$TARGET" ]; then
+    ok "a plan failure is logged and the session still runs"
+else
+    bad "a plan failure is logged and the session still runs" "$(LOGTXT | tail -5)"
+fi
+teardown
+
+# ── 15. crash + untouched book on the FINAL attempt runs the risk-only fallback ──
+setup
+echo "$TODAY $((5 - 1))" > "$SANDBOX/logs/trader_attempts"     # this is attempt 5 of 5
+trader CLAUDE_RC=1 ACT_SAFE=true ACT_COUNT=0
+if CALLS | grep -q "execute-plan.*--risk-only" && CALLS | grep -q "sync-stops.*--apply" && [ "$(STAMP)" = "$TARGET" ]; then
+    ok "final failed attempt executes risk-only legs and stamps"
+else
+    bad "final failed attempt executes risk-only legs and stamps" "calls=$(CALLS) stamp=$(STAMP)"
+fi
+teardown
+
+# ── 16. crash + untouched book late in the evening (>= 21:30 ET) also falls back ──
+setup
+trader CLAUDE_RC=1 ACT_SAFE=true ACT_COUNT=0 TEST_NOW_ET=2140
+if CALLS | grep -q "execute-plan.*--risk-only" && [ "$(STAMP)" = "$TARGET" ]; then
+    ok "late-evening crash falls back to risk-only execution"
+else
+    bad "late-evening crash falls back to risk-only execution" "calls=$(CALLS) stamp=$(STAMP)"
+fi
 teardown
+
+# ── 17. an early crash with retries left does NOT fall back (a retry will review properly) ──
+setup
+trader CLAUDE_RC=1 ACT_SAFE=true ACT_COUNT=0
+if ! CALLS | grep -q "execute-plan" && [ "$(STAMP)" = "<none>" ]; then
+    ok "early crash with retries left leaves execution to the retry"
+else
+    bad "early crash with retries left leaves execution to the retry" "calls=$(CALLS) stamp=$(STAMP)"
+fi
+teardown
+
+# ── 18. crash AFTER orders were placed never runs the fallback (book already touched) ──
+setup
+echo "$TODAY $((5 - 1))" > "$SANDBOX/logs/trader_attempts"
+trader CLAUDE_RC=1 ACT_SAFE=false ACT_COUNT=2
+if ! CALLS | grep -q "execute-plan" && [ "$(STAMP)" = "$TARGET" ]; then
+    ok "crash after fills never double-executes via the fallback"
+else
+    bad "crash after fills never double-executes via the fallback" "calls=$(CALLS)"
+fi
+teardown
+
+# ── 19. WebSearch/WebFetch are no longer granted; turn cap is 40 ──
+setup
+if grep -q -- '--max-turns 40' "$SANDBOX/run_trader.sh" && ! grep 'allowedTools' "$SANDBOX/run_trader.sh" | grep -q 'WebSearch'; then
+    ok "reviewer session has no web tools and a 40-turn cap"
+else
+    bad "reviewer session has no web tools and a 40-turn cap" ""
+fi
+teardown
+
+echo
+echo "passed $PASS, failed $FAIL"
+[ "$FAIL" -eq 0 ]
```

- [ ] **Step 2:** Run:

```bash
bash tests/test_run_trader.sh | tail -3
```

Expected: new cases FAIL (plan not invoked, no fallback) — `passed 30, failed 6` or similar.

- [ ] **Step 3:** Apply the runner diff:

**Apply this unified diff to `run_trader.sh`** (from the reference patch; `patch -p1 < docs/superpowers/plans/2026-09-04-overhaul-reference.patch` applies all of them at once):

```diff
--- a/run_trader.sh	2026-09-03 18:01:23
+++ b/run_trader.sh	2026-09-04 12:48:48
@@ -130,15 +130,43 @@
 # Write(trading/**) is intentionally absent: Claude Code reports it as unmatched
 # by file permission checks, and Edit(trading/**) already covers every
 # file-editing tool. Listing both only produced a warning on every run.
+# Build the engine's plan BEFORE the reviewer session. If the session then dies
+# the plan file still exists, and the fallback below can execute its
+# risk-reducing legs unattended. A plan failure is logged, not fatal: the
+# session reads the error and journals it (PROMPT.md: no plan → no trades).
+if "$PY" -m src.trader_cli plan --target "$TARGET" >> "$LOG_FILE" 2>&1; then
+    echo "=== Plan built for $TARGET ===" >> "$LOG_FILE"
+else
+    echo "=== Plan build FAILED for $TARGET (session will journal it) ===" >> "$LOG_FILE"
+fi
+
+# WebSearch/WebFetch are deliberately absent: they are permission-denied under
+# `claude -p` anyway, and the session now has `trader_cli news` for headlines.
 SESSION_RC=0
 caffeinate -imsu "$CLAUDE" -p "$(cat trading/PROMPT.md)" \
     --model opus \
-    --allowedTools "Bash($PY -m src.trader_cli:*),Read,Glob,Grep,WebSearch,WebFetch,Edit(trading/**)" \
-    --max-turns 120 \
+    --allowedTools "Bash($PY -m src.trader_cli:*),Read,Glob,Grep,Edit(trading/**)" \
+    --max-turns 40 \
     >> "$LOG_FILE" 2>&1 || SESSION_RC=$?
 
 echo "=== Trader session finished: $(date) ===" >> "$LOG_FILE"
 
+# Final-attempt fallback: the reviewer never traded and no retry is coming
+# (attempt cap reached, or it is past 21:30 ET). Exits, trims and earnings
+# trims must not be skipped because an LLM session crashed, so the engine's
+# risk-reducing legs are executed unattended. Entries are never placed this way.
+run_risk_only_fallback() {
+    if [ -f "trading/plans/$TARGET.json" ]; then
+        echo "=== Fallback: executing risk-reducing legs of plan $TARGET ===" >> "$LOG_FILE"
+        "$PY" -m src.trader_cli execute-plan --target "$TARGET" --risk-only >> "$LOG_FILE" 2>&1 \
+            || echo "=== Fallback execute-plan failed ===" >> "$LOG_FILE"
+        "$PY" -m src.trader_cli sync-stops --apply >> "$LOG_FILE" 2>&1 \
+            || echo "=== Fallback sync-stops failed ===" >> "$LOG_FILE"
+    else
+        echo "=== Fallback: no plan file for $TARGET, nothing to execute ===" >> "$LOG_FILE"
+    fi
+}
+
 if [ "$SESSION_RC" -eq 0 ]; then
     echo "$TARGET" > "$STAMP_FILE"
 else
@@ -153,7 +181,13 @@
         echo "$TARGET" > "$STAMP_FILE"
         echo "=== Could not verify order activity — stamping conservatively, will NOT retry ===" >> "$LOG_FILE"
     elif echo "$ACTIVITY" | grep -q '"safe_to_retry": true'; then
-        echo "=== No orders placed today — leaving unstamped, will retry ===" >> "$LOG_FILE"
+        if [ "$((ATTEMPTS + 1))" -ge "$MAX_ATTEMPTS" ] || [ "$NOW_ET" -ge 2130 ]; then
+            echo "=== No orders placed and no retry left — running risk-only fallback ===" >> "$LOG_FILE"
+            run_risk_only_fallback
+            echo "$TARGET" > "$STAMP_FILE"
+        else
+            echo "=== No orders placed today — leaving unstamped, will retry ===" >> "$LOG_FILE"
+        fi
     else
         echo "$TARGET" > "$STAMP_FILE"
         echo "=== Orders already placed today — stamping, will NOT retry ===" >> "$LOG_FILE"
```

- [ ] **Step 4:** Run:

```bash
bash tests/test_run_trader.sh | tail -1 && /Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m pytest tests/test_run_trader_sh.py -q
```

Expected: `passed 36, failed 0` and 1 passed.

- [ ] **Step 5:** Commit as `feat(runner): build the engine plan before the session; risk-only fallback on the final attempt`. Per the user's global CLAUDE.md §6: run `git status` first and **ask the user before committing**. Never add a Co-Authored-By line.

**Edge cases this task must handle (each has a test):**
- Plan built before the session; plan failure logged and the session still runs.
- Final attempt (5/5) or ≥ 21:30 ET with an untouched book → `execute-plan --risk-only` + `sync-stops --apply` + stamp.
- Early crash with retries left → no fallback, unstamped.
- Crash after fills → never runs the fallback (book touched).
- `--max-turns 40`; the `allowedTools` line has no WebSearch/WebFetch.


### Task 2.7: Keep plan files out of the public repo; publish them privately

**Files:**
- Modify: `.gitignore`
- Modify: `scripts/publish_data.sh`

`trading/plans/*.json` and `trading/engine_state.json` carry live position sizes. `run_trader.sh` does `git add trading/`, so they must be gitignored; the private data repo gets them (and `screen_latest.json`, `plans.json`) so the Paper page can render them.

- [ ] **Step 1:** Apply:

**Apply this unified diff to `.gitignore`** (from the reference patch; `patch -p1 < docs/superpowers/plans/2026-09-04-overhaul-reference.patch` applies all of them at once):

```diff
--- a/.gitignore	2026-09-04 12:10:55
+++ b/.gitignore	2026-09-04 12:51:03
@@ -31,6 +31,11 @@
 docs/handoffs/
 handoff-thesis-pivot-*.md
 
+# Engine plans and state carry live position sizes — published to the private
+# data repo by scripts/publish_data.sh, never committed here.
+trading/plans/
+trading/engine_state.json
+
 # Scratch
 trading/_scratch_*.py
 trading/.tmp_*.py
```

- [ ] **Step 2:** Apply:

**Apply this unified diff to `scripts/publish_data.sh`** (from the reference patch; `patch -p1 < docs/superpowers/plans/2026-09-04-overhaul-reference.patch` applies all of them at once):

```diff
--- a/scripts/publish_data.sh	2026-09-03 18:00:43
+++ b/scripts/publish_data.sh	2026-09-04 12:51:03
@@ -48,8 +48,17 @@
 copy run_status.json
 copy data/fidelity/positions_data.json
 copy data/alpaca/portfolio.json
+copy data/alpaca/plans.json
+copy trading/engine_state.json
+copy output/screen_latest.json
 copy logs/fidelity_sync_status.json
 
+# Engine plans: one JSON per target session (legs, reviews, executions).
+mkdir -p "$DATA_REPO_DIR/trading/plans"
+for f in "$SCREENER_DIR"/trading/plans/*.json; do
+    [ -f "$f" ] && cp "$f" "$DATA_REPO_DIR/trading/plans/"
+done
+
 # Screener results: the dashboard lists every date, so publish them all. This is
 # ~1.3 MB of CSV/markdown and grows by a few KB per trading day.
 mkdir -p "$DATA_REPO_DIR/output"
```

- [ ] **Step 3:** Run:

```bash
git check-ignore -v trading/plans/x.json trading/engine_state.json
```

Expected: both lines print a matching .gitignore rule.

- [ ] **Step 4:** Commit as `chore: keep engine plans private; publish them to the data repo`. Per the user's global CLAUDE.md §6: run `git status` first and **ask the user before committing**. Never add a Co-Authored-By line.


### Task 2.8: Rollout gate for the reviewer session (no code)

- [ ] **Step 1:** With the OLD `trading/PROMPT.md` still in place, let the runner build a plan tonight (it does so before the session). After the session, inspect `trading/plans/<target>.json`: legs, reasons, rejections, `screen_usable`. Confirm with the user that each fixed leg (exit/trim/earnings_trim) is one they would have wanted and that no entry violates the policy.

- [ ] **Step 2:** Dry-run the reviewer prompt manually once (interactive, not launchd):

```bash
set -a; source .env; set +a; export TRADER_DRY_RUN=1
/opt/homebrew/bin/claude -p "$(cat trading/PROMPT.md)" --model opus --allowedTools "Bash(/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m src.trader_cli:*),Read,Glob,Grep,Edit(trading/**)" --max-turns 40
```
`TRADER_DRY_RUN=1` makes every order a no-op (`src/broker.py`). Expected: the session runs `screen`, `status`, `plan`, `news`, `journal-draft`, records reviews, runs `execute-plan` (all dry), `sync-stops --apply` (dry), and leaves a journal with a filled Review table.

- [ ] **Step 3:** Read the journal with the user. Only after they approve, deploy the new `trading/PROMPT.md` (Task 2.5) for the next evening session. Unset `TRADER_DRY_RUN` in `.env` if it was set (it is 1 in `.env.template` by design — check the real `.env`).

- [ ] **Step 4:** After the first live session, verify: `trading/plans/<target>.json` has `executed` entries with order ids; `trader_cli orders --status all` shows them; the 09:40 stop sync re-armed floors; the Paper page renders the plan.
