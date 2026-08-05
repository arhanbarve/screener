"""Order-type intelligence.

The property that matters: an order must never be left unpriced when it cannot
fill immediately. On 2026-08-04 five market orders were queued 14 hours ahead of
the open, which is the exposure this module removes.
"""
from datetime import datetime

import pytest

from src.orders import (
    ET,
    ENTRY_BUFFER_BPS,
    EXIT_BUFFER_BPS,
    OrderPlanError,
    limit_price_for,
    market_session,
    plan_order,
    plan_stop,
    qty_for_notional,
    reconcile_stops,
    round_tick,
)


def _clock(is_open, next_open="2026-08-05T09:30:00-04:00"):
    return {"is_open": is_open, "next_open": next_open,
            "next_close": "2026-08-05T16:00:00-04:00"}


# ── session classification ────────────────────────────────────────────────────

def test_session_open_when_clock_says_so():
    assert market_session(_clock(True)) == "open"


def test_session_extended_post_market():
    """19:15 ET on a trading day is post-market: limit orders only."""
    now = datetime(2026, 8, 4, 19, 15, tzinfo=ET)
    assert market_session(_clock(False), now) == "extended"


def test_session_extended_pre_market():
    now = datetime(2026, 8, 5, 8, 30, tzinfo=ET)
    assert market_session(_clock(False, "2026-08-05T09:30:00-04:00"), now) == "extended"


def test_session_closed_overnight():
    now = datetime(2026, 8, 5, 2, 0, tzinfo=ET)
    assert market_session(_clock(False, "2026-08-05T09:30:00-04:00"), now) == "closed"


def test_session_closed_after_post_market_ends():
    now = datetime(2026, 8, 4, 20, 30, tzinfo=ET)
    assert market_session(_clock(False, "2026-08-05T09:30:00-04:00"), now) == "closed"


def test_saturday_evening_is_not_post_market():
    """Guards the bug where any evening looked like post-market."""
    now = datetime(2026, 8, 8, 18, 0, tzinfo=ET)  # Saturday
    assert market_session(_clock(False, "2026-08-10T09:30:00-04:00"), now) == "closed"


def test_naive_datetime_treated_as_et():
    now = datetime(2026, 8, 4, 19, 15)
    assert market_session(_clock(False), now) == "extended"


# ── sizing ────────────────────────────────────────────────────────────────────

def test_qty_for_notional_nine_decimals():
    assert qty_for_notional(4000, 89.21) == pytest.approx(44.838022643, abs=1e-9)
    # Round-trips back to the dollar amount asked for.
    assert 44.838022643 * 89.21 == pytest.approx(4000, abs=0.01)


def test_qty_for_notional_rejects_bad_price():
    with pytest.raises(OrderPlanError):
        qty_for_notional(4000, 0)


def test_qty_for_notional_rejects_zero_notional():
    with pytest.raises(OrderPlanError):
        qty_for_notional(0, 100)


# ── tick rounding (Alpaca's sub-penny rule) ───────────────────────────────────

def test_round_tick_two_decimals_at_or_above_a_dollar():
    assert round_tick(89.2137) == 89.21
    assert round_tick(1.005) == 1.0


def test_round_tick_four_decimals_below_a_dollar():
    assert round_tick(0.98765) == 0.9877


def test_round_tick_rejects_nonpositive():
    with pytest.raises(OrderPlanError):
        round_tick(0)


# ── limit pricing ─────────────────────────────────────────────────────────────

def test_buy_limit_sits_above_last():
    assert limit_price_for("buy", 100.0, 50) == 100.5


def test_sell_limit_sits_below_last():
    assert limit_price_for("sell", 100.0, 200) == 98.0


def test_zero_buffer_is_last_price():
    assert limit_price_for("buy", 100.0, 0) == 100.0


def test_limit_rejects_bad_side_and_price():
    with pytest.raises(OrderPlanError):
        limit_price_for("short", 100.0, 50)
    with pytest.raises(OrderPlanError):
        limit_price_for("buy", 0, 50)
    with pytest.raises(OrderPlanError):
        limit_price_for("buy", 100.0, -5)


# ── the core behaviour ────────────────────────────────────────────────────────

def test_open_market_uses_market_order_with_notional():
    p = plan_order("VSXY", "buy", "open", last=89.21, notional=4000)
    assert p["order_type"] == "market"
    assert p["notional"] == 4000.0
    assert "qty" not in p


def test_closed_market_never_sends_a_market_order():
    """The 2026-08-04 regression, stated as a test."""
    p = plan_order("VSXY", "buy", "closed", last=89.21, notional=4000)
    assert p["order_type"] == "limit"
    assert p["limit_price"] == pytest.approx(89.66, abs=0.01)
    assert p["qty"] == pytest.approx(4000 / p["limit_price"], abs=1e-6)
    assert p.get("extended_hours") is None


@pytest.mark.parametrize("session", ["closed", "extended"])
@pytest.mark.parametrize("side", ["buy", "sell"])
def test_no_market_orders_outside_regular_hours(session, side):
    p = plan_order("SPY", side, session, last=771.70, notional=8000)
    assert p["order_type"] == "limit", f"{session}/{side} produced {p['order_type']}"


def test_extended_hours_is_opt_in():
    off = plan_order("SPY", "buy", "extended", last=771.70, notional=8000)
    on = plan_order("SPY", "buy", "extended", last=771.70, notional=8000,
                    allow_extended=True)
    assert off.get("extended_hours") is None
    assert on["extended_hours"] is True


def test_sell_limit_uses_the_wider_exit_buffer():
    """Exits default to a wider buffer: not filling an exit is the worse error."""
    p = plan_order("HUT", "sell", "closed", last=101.21, qty=36.022964697)
    assert p["limit_price"] == pytest.approx(101.21 * (1 - EXIT_BUFFER_BPS / 10_000), abs=0.01)
    assert p["limit_price"] < 101.21
    assert p["qty"] == 36.022964697


def test_buy_default_buffer_is_the_entry_buffer():
    p = plan_order("VSXY", "buy", "closed", last=100.0, notional=1000)
    assert p["limit_price"] == pytest.approx(100.0 * (1 + ENTRY_BUFFER_BPS / 10_000))


def test_explicit_buffer_overrides_default():
    p = plan_order("VSXY", "buy", "closed", last=100.0, notional=1000, buffer_bps=10)
    assert p["limit_price"] == 100.1


def test_plan_order_requires_exactly_one_size():
    with pytest.raises(OrderPlanError):
        plan_order("SPY", "buy", "open", last=100.0)
    with pytest.raises(OrderPlanError):
        plan_order("SPY", "buy", "open", last=100.0, notional=100, qty=1)


def test_every_plan_explains_itself():
    """An order type nobody can explain is one nobody should send."""
    for session in ("open", "extended", "closed"):
        p = plan_order("SPY", "buy", session, last=771.70, notional=8000)
        assert p["_why"] and len(p["_why"]) > 10


# ── protective stops ──────────────────────────────────────────────────────────

def test_plan_stop_shape():
    s = plan_stop("MYE", 229.357511467, 31.4567)
    assert s["symbol"] == "MYE"
    assert s["side"] == "sell"
    assert s["order_type"] == "stop"
    assert s["stop_price"] == 31.46
    # Whole shares + GTC, so the stop outlives a day with no session.
    assert s["qty"] == 229.0
    assert s["time_in_force"] == "gtc"


def test_plan_stop_rejects_empty_position():
    with pytest.raises(OrderPlanError):
        plan_stop("MYE", 0, 31.45)


# ── stop reconciliation: must be idempotent ───────────────────────────────────

def _pos(sym, qty, available=None):
    return {"symbol": sym, "qty": str(qty),
            "qty_available": str(qty if available is None else available)}


def _stop(sym, qty, price, oid="o1", tif="gtc"):
    return {"id": oid, "symbol": sym, "side": "sell", "type": "stop",
            "qty": str(qty), "stop_price": str(price), "time_in_force": tif}


def test_places_stop_when_none_exists():
    r = reconcile_stops([_pos("MYE", 100)], {"MYE": 31.45}, [])
    assert [p["symbol"] for p in r["place"]] == ["MYE"]
    assert r["cancel"] == []


def test_second_run_is_a_noop():
    """Without this, the retry path stacks duplicate stops until they oversell."""
    floors = {"MYE": 31.45}
    first = reconcile_stops([_pos("MYE", 100)], floors, [])
    placed = first["place"][0]
    resting = _stop("MYE", placed["qty"], placed["stop_price"])
    second = reconcile_stops([_pos("MYE", 100)], floors, [resting])
    assert second["place"] == []
    assert second["cancel"] == []
    assert len(second["keep"]) == 1


def test_replaces_stop_when_floor_moves():
    resting = _stop("MYE", 100, 31.45)
    r = reconcile_stops([_pos("MYE", 100)], {"MYE": 33.00}, [resting])
    assert r["cancel"] == [resting]
    assert r["place"][0]["stop_price"] == 33.0


def test_replaces_stop_when_position_size_changes():
    resting = _stop("MYE", 100, 31.45)
    r = reconcile_stops([_pos("MYE", 150)], {"MYE": 31.45}, [resting])
    assert r["cancel"] == [resting]
    assert r["place"][0]["qty"] == 150.0


def test_cancels_stop_for_a_position_no_longer_held():
    resting = _stop("HUT", 36.0, 100.0)
    r = reconcile_stops([_pos("MYE", 100)], {"MYE": 31.45}, [resting])
    assert resting in r["cancel"]


def test_skips_symbol_with_no_floor_and_clears_its_stops():
    resting = _stop("GTX", 128.0, 28.0)
    r = reconcile_stops([_pos("GTX", 128.0)], {}, [resting])
    assert r["place"] == []
    assert resting in r["cancel"]
    assert r["skip"][0]["ticker"] == "GTX"


def test_ignores_non_protective_orders():
    """A resting buy limit must not be mistaken for a stop and cancelled."""
    buy = {"id": "b1", "symbol": "MYE", "side": "buy", "type": "limit",
           "qty": "10", "limit_price": "35.00"}
    r = reconcile_stops([_pos("MYE", 100)], {"MYE": 31.45}, [buy])
    assert buy not in r["cancel"]
    assert len(r["place"]) == 1


def test_reconcile_handles_empty_book():
    r = reconcile_stops([], {}, [])
    assert r == {"place": [], "cancel": [], "keep": [], "skip": []}


# ── free-share accounting: a stop must never oversell ─────────────────────────
# HUT on 2026-08-04 had qty 36.02 and qty_available 0 — a market close was already
# resting against every share. Sizing off qty would have doubled the sell.

def test_skips_when_all_shares_committed_elsewhere():
    r = reconcile_stops([_pos("HUT", 36.022964697, available=0)], {"HUT": 89.76}, [])
    assert r["place"] == []
    assert r["skip"][0]["ticker"] == "HUT"
    assert "committed to another open order" in r["skip"][0]["reason"]


def test_sizes_stop_to_free_shares_only():
    """Half the position is committed; the stop covers the other half."""
    r = reconcile_stops([_pos("MYE", 200, available=80)], {"MYE": 31.45}, [])
    assert r["place"][0]["qty"] == 80.0


def test_existing_stop_shares_are_added_back_so_replacement_works():
    """A resting stop makes qty_available 0; that must not block replacing it."""
    resting = _stop("MYE", 100, 31.45)
    r = reconcile_stops([_pos("MYE", 100, available=0)], {"MYE": 33.00}, [resting])
    assert resting in r["cancel"]
    assert r["place"][0]["qty"] == 100.0
    assert r["place"][0]["stop_price"] == 33.0


def test_idempotent_with_resting_stop_holding_all_shares():
    resting = _stop("MYE", 100, 31.45)
    r = reconcile_stops([_pos("MYE", 100, available=0)], {"MYE": 31.45}, [resting])
    assert r["place"] == [] and r["cancel"] == [] and len(r["keep"]) == 1


def test_missing_qty_available_falls_back_to_full_size():
    r = reconcile_stops([{"symbol": "MYE", "qty": "100"}], {"MYE": 31.45}, [])
    assert r["place"][0]["qty"] == 100.0


# ── session-start cleanup must be surgical ────────────────────────────────────

def test_protective_stop_ids_selects_only_stops():
    """cancel-all at session start would take out deliberate pending entries."""
    orders = [
        _stop("MYE", 100, 31.45, oid="stop1"),
        {"id": "entry1", "symbol": "ORKA", "side": "buy", "type": "limit",
         "qty": "40", "limit_price": "98.50"},
        {"id": "exit1", "symbol": "HUT", "side": "sell", "type": "market",
         "qty": "36"},
    ]
    from src.orders import protective_stop_ids
    assert protective_stop_ids(orders) == ["stop1"]


def test_protective_stop_ids_empty():
    from src.orders import protective_stop_ids
    assert protective_stop_ids([]) == []


# ── stops must outlive a day with no session ───────────────────────────────────
# Alpaca allows fractional qty only with time_in_force=day, so a stop on
# 36.022964697 shares expires at the next close — defeating the whole purpose,
# since the scenario it covers is the day nobody re-places it.

def test_fractional_position_gets_whole_share_gtc_stop():
    s = plan_stop("HUT", 36.022964697, 89.7586)
    assert s["time_in_force"] == "gtc"
    assert s["qty"] == 36.0
    assert s["stop_price"] == 89.76
    assert s["_coverage"] == pytest.approx(36.0 / 36.022964697, abs=1e-6)
    assert "uncovered" in s["_why"]


def test_whole_share_position_is_fully_covered():
    s = plan_stop("MYE", 229.0, 31.76)
    assert s["time_in_force"] == "gtc"
    assert s["qty"] == 229.0
    assert s["_coverage"] == 1.0
    assert "uncovered" not in s["_why"]


def test_sub_one_share_position_falls_back_to_day_and_says_so():
    s = plan_stop("FIX", 0.5, 1502.89)
    assert s["time_in_force"] == "day"
    assert s["qty"] == 0.5
    assert "expires at the next close" in s["_why"]


def test_reconcile_is_idempotent_with_whole_share_gtc():
    """The trap: matching on the fractional free qty would replace forever."""
    floors = {"HUT": 89.7586}
    pos = [_pos("HUT", 36.022964697)]
    first = reconcile_stops(pos, floors, [])
    want = first["place"][0]
    assert want["qty"] == 36.0 and want["time_in_force"] == "gtc"

    resting = {"id": "s1", "symbol": "HUT", "side": "sell", "type": "stop",
               "qty": "36", "stop_price": "89.76", "time_in_force": "gtc"}
    second = reconcile_stops([_pos("HUT", 36.022964697, available=0.022964697)],
                             floors, [resting])
    assert second["place"] == [], f"replaced a matching stop: {second['place']}"
    assert second["cancel"] == []
    assert len(second["keep"]) == 1


def test_existing_day_stop_is_replaced_with_gtc():
    resting = {"id": "s1", "symbol": "HUT", "side": "sell", "type": "stop",
               "qty": "36.022964697", "stop_price": "89.76", "time_in_force": "day"}
    r = reconcile_stops([_pos("HUT", 36.022964697, available=0)], {"HUT": 89.7586},
                        [resting])
    assert resting in r["cancel"]
    assert r["place"][0]["time_in_force"] == "gtc"
