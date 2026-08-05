"""Alpaca paper-portfolio snapshot maths.

The load-bearing test here is test_reconciles_against_equity: FIFO lot matching
is only correct if realized + unrealized equals the account's equity change, and
that invariant is what proved average-cost accounting wrong (off by $241 on the
real book). Fixtures below are the real fill sequence from the paper account.
"""
import pytest

from src.paper import (
    SnapshotError,
    align_on_dates,
    assemble,
    closed_trades,
    equity_by_date,
    first_fill_date,
    inception_baseline,
    live_view,
    load_snapshot,
    merge_live,
    match_fifo,
    open_lot_summary,
    pct_series,
    realized_total,
    reconcile,
    save_snapshot,
    vs_spy,
    _fills,
)


def _order(sym, side, qty, price, day, status="filled"):
    return {
        "symbol": sym, "side": side, "filled_qty": str(qty),
        "filled_avg_price": str(price), "filled_at": f"{day}T14:30:00Z",
        "status": status,
    }


# The actual fill history of the account, in order.
REAL_ORDERS = [
    _order("TSEM", "buy",  33.582360842, 238.22,     "2026-07-24"),
    _order("FIX",  "buy",   4.679807191, 1709.47,    "2026-07-24"),
    _order("STX",  "buy",   4.68344515,  854.07,     "2026-07-24"),
    _order("LQDA", "buy",  45.740308747, 87.45,      "2026-07-24"),
    _order("HUT",  "buy",  36.022964697, 111.04,     "2026-07-24"),
    _order("TSEM", "sell", 14.968219916, 200.423966, "2026-07-29"),
    _order("LQDA", "sell", 45.740308747, 85.56,      "2026-07-29"),
    _order("STX",  "buy",   5.314400733, 752.67,     "2026-07-29"),
    _order("VSXY", "buy",  89.887528089, 89.00,      "2026-07-31"),
    _order("VLO",  "buy",  26.054355968, 307.05,     "2026-07-31"),
    _order("EVC",  "buy", 362.317934782, 11.04,      "2026-07-31"),
    _order("TSEM", "sell", 18.614140926, 227.47,     "2026-08-03"),
    _order("STX",  "sell",  4.963813707, 805.83,     "2026-08-03"),
    _order("VSXY", "sell", 45.325665722, 88.25,      "2026-08-03"),
    _order("MYE",  "buy", 229.357511467, 34.88,      "2026-08-03"),
    _order("GTX",  "buy", 128.369383825, 31.16,      "2026-08-03"),
    _order("SPY",  "buy",  19.860171062, 755.28,     "2026-08-03"),
]

# Unrealized P&L as the broker reports it, same moment the fills were read.
REAL_UNREALIZED = {
    "EVC": 213.767582, "FIX": 269.074406, "GTX": 88.022886, "HUT": -307.636119,
    "MYE": 403.66922, "SPY": 332.06206, "STX": 426.281845, "VLO": 31.004684,
    "VSXY": 9.357991,
}
REAL_EQUITY = 100368.22
REAL_DEPOSITS = 100000.0


@pytest.fixture
def books():
    return match_fifo(_fills(REAL_ORDERS))


# ── fill parsing ──────────────────────────────────────────────────────────────

def test_unfilled_orders_ignored():
    orders = REAL_ORDERS + [
        {"symbol": "AAPL", "side": "buy", "status": "new", "filled_at": None,
         "filled_qty": "0", "filled_avg_price": None},
    ]
    assert not any(f["symbol"] == "AAPL" for f in _fills(orders))


def test_zero_price_fill_ignored():
    orders = [_order("ZZZ", "buy", 10, 0, "2026-08-01")]
    assert _fills(orders) == []


# ── FIFO correctness ──────────────────────────────────────────────────────────

def test_fully_closed_symbol_realizes_whole_loss(books):
    """TSEM: bought 33.58 @ 238.22, sold in two tranches, nothing left."""
    tsem = books["TSEM"]
    assert tsem["closed"] is True
    assert tsem["open_lots"] == []
    # $8,000 in, $7,234.15 out across the two tranches.
    assert tsem["realized"] == pytest.approx(-765.84, abs=0.02)


def test_single_lot_round_trip(books):
    lqda = books["LQDA"]
    assert lqda["closed"] is True
    assert lqda["realized"] == pytest.approx(-86.45, abs=0.02)


def test_fifo_retires_oldest_lot_first(books):
    """STX is the case average-cost accounting gets wrong.

    Two buys (854.07 then 752.67), one sale. FIFO closes the expensive first lot,
    leaving the cheaper one — which is why Alpaca reports avg_entry_price 752.67
    and not a blended 800.17.
    """
    stx = open_lot_summary(books["STX"])
    assert stx["qty"] == pytest.approx(5.034032176, abs=1e-6)
    assert stx["avg_cost"] == pytest.approx(752.67, abs=0.01)
    assert books["STX"]["realized"] == pytest.approx(-211.03, abs=0.05)


def test_average_cost_would_disagree(books):
    """Guards the method choice itself, so nobody 'simplifies' it back."""
    blended = (4.68344515 * 854.07 + 5.314400733 * 752.67) / 9.997845883
    avg_cost_realized = 4.963813707 * (805.83 - blended)
    assert avg_cost_realized == pytest.approx(28.02, abs=0.1)
    # FIFO gives a materially different, and correct, answer.
    assert abs(books["STX"]["realized"] - avg_cost_realized) > 200


def test_partial_trim_stays_open(books):
    """VSXY was halved, not closed — it must not appear as a finished trade."""
    vsxy = books["VSXY"]
    assert vsxy["closed"] is False
    assert open_lot_summary(vsxy)["qty"] == pytest.approx(44.561862367, abs=1e-6)
    assert vsxy["realized"] == pytest.approx(-33.99, abs=0.02)


def test_held_since_is_oldest_remaining_lot_not_first_fill(books):
    """STX first filled Jul 24, but FIFO retired that lot — held since Jul 29."""
    assert open_lot_summary(books["STX"])["held_since"] == "2026-07-29"
    assert open_lot_summary(books["HUT"])["held_since"] == "2026-07-24"


def test_oversell_does_not_go_negative():
    """Defensive: a sell larger than the book must not produce phantom lots."""
    orders = [
        _order("ZZZ", "buy", 10, 100, "2026-08-01"),
        _order("ZZZ", "sell", 25, 110, "2026-08-02"),
    ]
    b = match_fifo(_fills(orders))["ZZZ"]
    assert b["open_lots"] == []
    assert b["realized"] == pytest.approx(10 * 10)  # only the 10 held are matched


# ── the invariant ─────────────────────────────────────────────────────────────

def test_reconciles_against_equity(books):
    """realized + unrealized == equity - deposits. This is why FIFO, not average."""
    realized = realized_total(books)
    unrealized = round(sum(REAL_UNREALIZED.values()), 2)
    rec = reconcile(realized, unrealized, REAL_EQUITY, REAL_DEPOSITS)
    assert rec["ok"] is True, rec
    assert abs(rec["drift"]) < 1.0


def test_reconcile_flags_drift():
    rec = reconcile(realized=-1000.0, unrealized=1465.60, equity=100368.22, deposits=100000.0)
    assert rec["ok"] is False
    assert rec["drift"] == pytest.approx(97.38, abs=0.01)


# ── closed trades ─────────────────────────────────────────────────────────────

def test_closed_trades_lists_only_fully_closed(books):
    rows = closed_trades(books, _fills(REAL_ORDERS))
    tickers = {r["ticker"] for r in rows}
    assert tickers == {"TSEM", "LQDA"}
    assert "VSXY" not in tickers  # trimmed, still open
    assert "STX" not in tickers


def test_closed_trades_sorted_most_recent_first(books):
    rows = closed_trades(books, _fills(REAL_ORDERS))
    assert [r["ticker"] for r in rows] == ["TSEM", "LQDA"]
    assert rows[0]["exit_date"] == "2026-08-03"


def test_closed_trade_held_days_and_pct(books):
    rows = {r["ticker"]: r for r in closed_trades(books, _fills(REAL_ORDERS))}
    lqda = rows["LQDA"]
    assert lqda["entry_date"] == "2026-07-24"
    assert lqda["exit_date"] == "2026-07-29"
    assert lqda["held_days"] == 5
    assert lqda["realized_pct"] == pytest.approx(-86.45 / 4000.0, abs=1e-4)


# ── series maths ──────────────────────────────────────────────────────────────

def test_pct_series_from_first_value():
    assert pct_series([100.0, 110.0, 90.0]) == [0.0, 10.0, -10.0]


def test_pct_series_skips_leading_zeros_for_base():
    """Alpaca pads history with 0.0 before the account was funded."""
    out = pct_series([0.0, 0.0, 100.0, 105.0])
    assert out[0] is None and out[1] is None
    assert out[2] == 0.0
    assert out[3] == pytest.approx(5.0)


def test_pct_series_all_zero_is_all_none():
    assert pct_series([0.0, 0.0]) == [None, None]


def test_equity_by_date_uses_exchange_date_not_utc():
    """Alpaca stamps UTC midnight *after* the session, so UTC reads a day late.

    These are the real epochs for the Jul 27 and Aug 3 sessions. Read as UTC they
    land on Jul 25 and Aug 1 — weekends — and get dropped from the curve.
    """
    hist = {"timestamp": [1785196800, 1785801600], "equity": [99082.94, 100042.31]}
    assert equity_by_date(hist) == {"2026-07-27": 99082.94, "2026-08-03": 100042.31}


def test_equity_by_date_skips_prefunding_zeros():
    hist = {"timestamp": [1784764800, 1784851200], "equity": [0.0, 100000.0]}
    assert equity_by_date(hist) == {"2026-07-23": 100000.0}


def test_equity_by_date_empty():
    assert equity_by_date({}) == {}


def test_curve_keeps_every_market_day_after_tz_fix():
    """The regression: 9 market days must yield 9 points, not 7."""
    epochs = [1784851200, 1784937600, 1785196800, 1785283200, 1785369600,
              1785456000, 1785542400, 1785801600]
    equities = [100000.0, 99844.81, 99082.94, 97542.26, 97417.65, 99204.85,
                99722.60, 100042.31]
    account = equity_by_date({"timestamp": epochs, "equity": equities})
    account["2026-08-04"] = 100373.77

    market = ["2026-07-23", "2026-07-24", "2026-07-27", "2026-07-28", "2026-07-29",
              "2026-07-30", "2026-07-31", "2026-08-03", "2026-08-04"]
    spy = {d: 700.0 + i for i, d in enumerate(market)}

    c = align_on_dates(account, spy, market)
    assert c["dates"] == market
    assert len(c["account_pct"]) == 9


def test_inception_baseline_is_day_before_first_fill():
    """Alpaca backfills equity before the account traded; that padding must not
    hand SPY market move the account was in cash for."""
    market = ["2026-07-22", "2026-07-23", "2026-07-24", "2026-07-27"]
    assert inception_baseline("2026-07-24", market) == "2026-07-23"


def test_inception_baseline_falls_back_to_first_fill():
    assert inception_baseline("2026-07-22", ["2026-07-22", "2026-07-23"]) == "2026-07-22"


def test_inception_baseline_without_fills_uses_first_market_day():
    assert inception_baseline(None, ["2026-07-22", "2026-07-23"]) == "2026-07-22"


def test_inception_baseline_no_market_days():
    assert inception_baseline(None, []) is None


def test_first_fill_date_is_earliest():
    assert first_fill_date(_fills(REAL_ORDERS)) == "2026-07-24"


def test_first_fill_date_empty():
    assert first_fill_date([]) is None


def test_baseline_trims_curve_and_changes_vs_spy():
    """The correctness fix: two dead cash days understated the gap by 1.3 points."""
    account = {"2026-07-22": 100000.0, "2026-07-23": 100000.0, "2026-08-04": 100373.77}
    spy = {"2026-07-22": 747.41, "2026-07-23": 738.18, "2026-08-04": 771.33}
    market = ["2026-07-22", "2026-07-23", "2026-08-04"]

    unbaselined = align_on_dates(account, spy, market)
    baselined = align_on_dates(account, spy, market, baseline="2026-07-23")

    assert unbaselined["dates"][0] == "2026-07-22"
    assert baselined["dates"][0] == "2026-07-23"
    # Same account return either way; SPY's changes, and so does the gap.
    assert vs_spy(baselined) == pytest.approx(-4.12, abs=0.05)
    assert vs_spy(unbaselined) == pytest.approx(-2.82, abs=0.05)


def test_align_intersects_on_market_days():
    """Alpaca labels history in UTC; alignment must be by date, not by position."""
    account = {"2026-08-01": 100.0, "2026-08-03": 110.0, "2026-08-04": 120.0}
    spy = {"2026-08-03": 700.0, "2026-08-04": 714.0, "2026-08-05": 720.0}
    market = ["2026-08-03", "2026-08-04", "2026-08-05"]
    c = align_on_dates(account, spy, market)
    # Aug 1 is not a market day, Aug 5 has no account point — both dropped.
    assert c["dates"] == ["2026-08-03", "2026-08-04"]
    assert c["account_pct"] == [0.0, pytest.approx(9.0909, abs=1e-3)]
    assert c["spy_pct"] == [0.0, pytest.approx(2.0)]


def test_vs_spy_is_point_difference():
    c = align_on_dates(
        {"2026-08-03": 100.0, "2026-08-04": 100.37},
        {"2026-08-03": 700.0, "2026-08-04": 731.43},
        ["2026-08-03", "2026-08-04"],
    )
    assert vs_spy(c) == pytest.approx(0.37 - 4.49, abs=0.02)


def test_vs_spy_none_when_series_empty():
    assert vs_spy({"account_pct": [], "spy_pct": []}) is None


# ── assemble ──────────────────────────────────────────────────────────────────

def _raw_positions():
    return [
        {"symbol": "HUT", "qty": "36.022964697", "avg_entry_price": "111.04",
         "current_price": "102.50", "market_value": "3692.353881",
         "cost_basis": "4000.0", "unrealized_pl": "-307.636119",
         "unrealized_plpc": "-0.07691", "unrealized_intraday_pl": "-345.100002",
         "unrealized_intraday_plpc": "-0.0324", "lastday_price": "112.08"},
        {"symbol": "STX", "qty": "5.034032176", "avg_entry_price": "752.67",
         "current_price": "837.35", "market_value": "4215.246843",
         "cost_basis": "3788.96", "unrealized_pl": "426.281845",
         "unrealized_plpc": "0.11251", "unrealized_intraday_pl": "31.664063",
         "unrealized_intraday_plpc": "0.0075", "lastday_price": "831.06"},
    ]


def _assembled():
    curve = align_on_dates(
        {"2026-08-03": 100000.0, "2026-08-04": 100368.22},
        {"2026-08-03": 757.67, "2026-08-04": 771.33},
        ["2026-08-03", "2026-08-04"],
    )
    return assemble(
        account={"equity": "100368.22", "last_equity": "100042.31",
                 "cash": "40147.71", "long_market_value": "60220.51"},
        raw_positions=_raw_positions(),
        orders=REAL_ORDERS,
        history={"HUT": [{"date": "2026-08-04", "close": 102.5}]},
        curve=curve,
        cadence={"days": [], "summary": {}},
        deposits=100000.0,
        synced_at="2026-08-04T17:30",
    )


def test_assemble_sorts_worst_first():
    snap = _assembled()
    assert [p["ticker"] for p in snap["positions"]] == ["HUT", "STX"]


def test_assemble_splits_realized_and_unrealized():
    a = _assembled()["account"]
    assert a["unrealized"] == pytest.approx(118.65, abs=0.01)   # -307.64 + 426.28
    assert a["realized"] < 0
    assert a["total_pl"] == pytest.approx(a["realized"] + a["unrealized"], abs=0.01)


def test_assemble_computes_deployed_pct():
    a = _assembled()["account"]
    assert a["deployed_pct"] == pytest.approx(0.6, abs=0.001)


def test_assemble_today_from_equity_delta():
    a = _assembled()["account"]
    assert a["today_dollar"] == pytest.approx(325.91, abs=0.01)


def test_assemble_carries_history_onto_position():
    snap = _assembled()
    hut = next(p for p in snap["positions"] if p["ticker"] == "HUT")
    assert hut["history"] == [{"date": "2026-08-04", "close": 102.5}]
    stx = next(p for p in snap["positions"] if p["ticker"] == "STX")
    assert stx["history"] == []  # absent history must be empty, not None


def test_assemble_open_orders_excludes_filled():
    assert _assembled()["open_orders"] == []


def test_assemble_open_orders_includes_pending():
    snap = assemble(
        account={"equity": "100000", "last_equity": "100000", "cash": "0",
                 "long_market_value": "0"},
        raw_positions=[],
        orders=[{"symbol": "AAPL", "side": "buy", "status": "new", "qty": "5",
                 "submitted_at": "2026-08-04T13:00:00Z"}],
        history={}, curve={"account_pct": [], "spy_pct": []},
        cadence={}, deposits=100000.0, synced_at="2026-08-04T17:30",
    )
    assert snap["open_orders"] == [
        {"ticker": "AAPL", "side": "buy", "qty": "5", "notional": None,
         "submitted_at": "2026-08-04T13:00:00"}
    ]


def test_assemble_no_account_identifiers_anywhere():
    """The snapshot is committed to git. Nothing account-identifying may leak."""
    import json
    blob = json.dumps(_assembled()).lower()
    for banned in ["account_number", "accountnumber", "account_id", "accountid"]:
        assert banned not in blob


# ── snapshot IO ───────────────────────────────────────────────────────────────

def test_save_load_roundtrip(tmp_path):
    p = tmp_path / "portfolio.json"
    snap = _assembled()
    save_snapshot(snap, p)
    assert load_snapshot(p) == snap


def test_save_leaves_no_tmp_file(tmp_path):
    p = tmp_path / "portfolio.json"
    save_snapshot(_assembled(), p)
    assert list(tmp_path.glob("*.tmp")) == []


def test_load_missing_returns_none(tmp_path):
    assert load_snapshot(tmp_path / "nope.json") is None


def test_load_corrupt_returns_none(tmp_path):
    p = tmp_path / "portfolio.json"
    p.write_text("{not json")
    assert load_snapshot(p) is None


def test_refresh_failure_leaves_previous_snapshot(tmp_path, monkeypatch):
    """All-or-nothing: a fetch failure must not touch the good snapshot."""
    import src.paper as paper

    p = tmp_path / "portfolio.json"
    good = _assembled()
    save_snapshot(good, p)
    monkeypatch.setattr(paper, "SNAPSHOT_FILE", p)

    import src.broker as broker
    monkeypatch.setattr(broker, "get_account", lambda: (_ for _ in ()).throw(RuntimeError("network down")))

    with pytest.raises(SnapshotError, match="Alpaca fetch failed"):
        paper.refresh()
    assert load_snapshot(p) == good


# ── live overlay: fast parts live, slow parts carried ─────────────────────────
# The page fetches on load, so this has to be quick. Account/positions/orders are
# three fast Alpaca calls; the curve, cadence and per-position history need a
# yfinance download and barely move intraday, so they come from the snapshot.

def _live_account():
    return {"equity": "100600.00", "last_equity": "100042.31",
            "cash": "40147.79", "long_market_value": "60452.21"}


def _live_positions():
    return [
        {"symbol": "HUT", "qty": "36.022964697", "avg_entry_price": "111.04",
         "current_price": "95.00", "market_value": "3422.18", "cost_basis": "4000.0",
         "unrealized_pl": "-577.82", "unrealized_plpc": "-0.1445",
         "unrealized_intraday_pl": "-221.90", "unrealized_intraday_plpc": "-0.0609",
         "lastday_price": "101.16"},
        {"symbol": "MYE", "qty": "229.357511467", "avg_entry_price": "34.88",
         "current_price": "37.50", "market_value": "8600.91", "cost_basis": "8000.0",
         "unrealized_pl": "600.91", "unrealized_plpc": "0.0751",
         "unrealized_intraday_pl": "197.25", "unrealized_intraday_plpc": "0.0234",
         "lastday_price": "36.64"},
    ]


def _stored():
    snap = _assembled()
    snap["positions"] = [
        {"ticker": "HUT", "held_since": "2026-07-24", "realized_to_date": 0.0,
         "history": [{"date": "2026-08-04", "close": 101.16}], "total_gl_pct": 0.0},
        {"ticker": "MYE", "held_since": "2026-08-03", "realized_to_date": 0.0,
         "history": [{"date": "2026-08-04", "close": 36.64}], "total_gl_pct": 0.0},
    ]
    snap["account"]["realized"] = -1097.31
    snap["account"]["deposits"] = 100000.0
    snap["cadence"] = {"days": [{"date": "2026-08-04"}], "summary": {"completed": 1}}
    return snap


def test_merge_live_uses_live_prices_and_pnl():
    out = merge_live(_stored(), _live_account(), _live_positions(), [], "2026-08-05T10:00")
    hut = next(p for p in out["positions"] if p["ticker"] == "HUT")
    assert hut["last_price"] == 95.0
    assert hut["total_gl_dollar"] == pytest.approx(-577.82)
    assert out["account"]["equity"] == 100600.0


def test_merge_live_carries_the_expensive_fields():
    """history/held_since come from FIFO + a price download; never refetched here."""
    out = merge_live(_stored(), _live_account(), _live_positions(), [], "2026-08-05T10:00")
    hut = next(p for p in out["positions"] if p["ticker"] == "HUT")
    assert hut["held_since"] == "2026-07-24"
    assert hut["history"] == [{"date": "2026-08-04", "close": 101.16}]
    assert out["cadence"]["summary"]["completed"] == 1


def test_merge_live_keeps_worst_first_ordering():
    out = merge_live(_stored(), _live_account(), _live_positions(), [], "2026-08-05T10:00")
    assert [p["ticker"] for p in out["positions"]] == ["HUT", "MYE"]


def test_merge_live_recomputes_reconciliation_against_live_equity():
    out = merge_live(_stored(), _live_account(), _live_positions(), [], "2026-08-05T10:00")
    a = out["account"]
    assert a["unrealized"] == pytest.approx(23.09, abs=0.01)   # -577.82 + 600.91
    assert a["total_pl"] == pytest.approx(a["realized"] + a["unrealized"], abs=0.01)
    assert out["reconciliation"]["expected"] == pytest.approx(600.0, abs=0.01)


def test_merge_live_today_dollar_from_equity_delta():
    out = merge_live(_stored(), _live_account(), _live_positions(), [], "2026-08-05T10:00")
    assert out["account"]["today_dollar"] == pytest.approx(557.69, abs=0.01)


def test_merge_live_extends_the_curve_to_now():
    out = merge_live(_stored(), _live_account(), _live_positions(), [], "2026-08-05T10:00")
    assert out["curve"]["dates"][-1] == "2026-08-05"
    assert out["curve"]["account_level"][-1] == 100600.0


def test_merge_live_replaces_todays_curve_point_rather_than_appending():
    """Reruns must not stack a new point every 20 seconds."""
    first = merge_live(_stored(), _live_account(), _live_positions(), [], "2026-08-05T10:00")
    n = len(first["curve"]["dates"])
    second = merge_live(first, _live_account(), _live_positions(), [], "2026-08-05T10:20")
    assert len(second["curve"]["dates"]) == n


def test_merge_live_carries_open_orders_with_type():
    orders = [{"symbol": "SPY", "side": "buy", "qty": "10", "type": "limit",
               "time_in_force": "day", "submitted_at": "2026-08-05T01:00:00Z"}]
    out = merge_live(_stored(), _live_account(), _live_positions(), orders, "2026-08-05T10:00")
    assert out["open_orders"][0]["type"] == "limit"


def test_live_view_falls_back_to_snapshot_when_broker_unreachable(monkeypatch, tmp_path):
    """No keys on the host must degrade to the snapshot, not to an exception."""
    import src.broker as broker
    monkeypatch.setattr(broker, "get_account",
                        lambda: (_ for _ in ()).throw(broker.BrokerError("Missing ALPACA_API_KEY")))
    data, source, err = live_view(_stored())
    assert source == "snapshot"
    assert "ALPACA_API_KEY" in err
    assert data["positions"]  # still renders


def test_live_view_reports_live_on_success(monkeypatch):
    import src.broker as broker
    monkeypatch.setattr(broker, "get_account", lambda: _live_account())
    monkeypatch.setattr(broker, "get_positions", lambda: _live_positions())
    monkeypatch.setattr(broker, "get_orders", lambda status="open": [])
    data, source, err = live_view(_stored())
    assert source == "live" and err is None
    assert data["account"]["equity"] == 100600.0


def test_live_view_with_no_snapshot_says_so():
    data, source, err = live_view({})
    assert source == "none" and "no snapshot" in err
