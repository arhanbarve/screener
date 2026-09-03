"""Positions-page gain/loss math.

Regression cover for the Jul-2026 outage: a zeroed cost basis made
_live_fid_metrics compute qty * (price - 0), so every card's Total G/L equalled
the whole position value while the percentage read +0.0%, and the summary strip's
TOTAL G/L exactly equalled PORTFOLIO VALUE. Unknown must render as "—", never as
a number.

Fixtures are synthetic round numbers, not real holdings — this repo is public.
"""
import pytest

from app_shared import _effective_avg_cost, _fmt_gl, _live_fid_metrics


def _fid(**over) -> dict:
    """A well-formed Fidelity snapshot row; override fields per test.

    10 shares @ 100.00 = 1000.00 value against a 1500.00 basis (150.00/sh),
    so Total G/L is -500.00 (-33.33%). Prior close 90.00 makes today +100.00.
    """
    base = {
        "ticker": "AAA", "quantity": 10.0, "last_price": 100.00,
        "last_price_chg": 10.00, "current_value": 1000.00,
        "today_gl_dollar": 100.00, "today_gl_pct": 0.1111,
        "total_gl_dollar": -500.00, "total_gl_pct": -0.3333,
        "pct_of_account": 0.2000, "cost_basis_total": 1500.00, "avg_cost": 150.00,
    }
    return {**base, **over}


# ── _effective_avg_cost ───────────────────────────────────────────────────────

def test_avg_cost_used_when_present():
    assert _effective_avg_cost(_fid()) == 150.00


def test_avg_cost_derived_from_cost_basis_total():
    avg = _effective_avg_cost(_fid(avg_cost=0.0))
    assert avg == pytest.approx(1500.00 / 10.0)


def test_avg_cost_none_when_nothing_usable():
    """None, not 0.0 — 0.0 is what turned market value into 'gain'."""
    assert _effective_avg_cost(_fid(avg_cost=0.0, cost_basis_total=0.0)) is None


# ── the exact bug ─────────────────────────────────────────────────────────────

def test_zero_cost_basis_does_not_report_market_value_as_gain():
    corrupt = _fid(avg_cost=0.0, cost_basis_total=0.0, total_gl_dollar=0.0,
                   total_gl_pct=0.0, current_value=0.0)
    m = _live_fid_metrics(corrupt, live_price=100.00, prev_close=90.00)

    market_value = 10.0 * 100.00
    assert m["total_gl_d"] is not market_value
    assert m["total_gl_d"] is None
    assert m["total_gl_p"] is None
    # Today's G/L is independent of cost basis, so it stays real.
    assert m["today_gl_d"] == pytest.approx(10.0 * (100.00 - 90.00))


def test_zero_cost_basis_snapshot_path_reports_unknown_not_zero():
    """No live quote AND no cost basis: the snapshot's own G/L is untrustworthy."""
    corrupt = _fid(avg_cost=0.0, cost_basis_total=0.0, total_gl_dollar=0.0,
                   total_gl_pct=0.0, today_gl_dollar=0.0, today_gl_pct=0.0)
    m = _live_fid_metrics(corrupt, live_price=None, prev_close=None)
    assert m["total_gl_d"] is None
    assert m["today_gl_d"] is None


def test_good_data_produces_correct_total_gl():
    m = _live_fid_metrics(_fid(), live_price=100.00, prev_close=90.00)
    assert m["total_gl_d"] == pytest.approx(10.0 * (100.00 - 150.00))
    assert m["total_gl_p"] == pytest.approx((100.00 - 150.00) / 150.00)
    assert m["total_gl_d"] < 0  # a loser must read as a loser


def test_summary_total_never_equals_portfolio_value_on_corrupt_data():
    """The tell from the bug report: TOTAL G/L identical to PORTFOLIO VALUE."""
    corrupt = [_fid(ticker=t, quantity=q, avg_cost=0.0, cost_basis_total=0.0,
                    current_value=0.0, total_gl_dollar=0.0, total_gl_pct=0.0)
               for t, q in [("AAA", 10.0), ("BBB", 20.0), ("CCC", 30.0)]]
    quotes = {"AAA": 100.00, "BBB": 20.00, "CCC": 50.00}

    total_value = 0.0
    total_overall = 0.0
    unknown = 0
    for f in corrupt:
        m = _live_fid_metrics(f, quotes[f["ticker"]], None)
        total_value += m["current_value"]
        if m["total_gl_d"] is None:
            unknown += 1
        else:
            total_overall += m["total_gl_d"]

    assert unknown == len(corrupt)
    assert total_overall == 0.0
    assert total_value > 0
    assert total_overall != total_value


# ── _fmt_gl ───────────────────────────────────────────────────────────────────

def test_fmt_gl_none_renders_dash():
    assert _fmt_gl(None, None) == "—"


def test_fmt_gl_negative_puts_sign_before_dollar():
    """Was rendering '$-500.00'."""
    assert _fmt_gl(-500.00, -0.3333) == "-$500.00 (-33.3%)"


def test_fmt_gl_positive_unchanged():
    assert _fmt_gl(100.00, 0.1111) == "+$100.00 (+11.1%)"


def test_fmt_gl_known_dollar_unknown_pct():
    assert _fmt_gl(100.00, None) == "+$100.00 (—)"
