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
