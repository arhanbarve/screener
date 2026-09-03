"""Fidelity CSV parse tests.

Regression cover for the Jul-2026 outage: Fidelity renamed the positions CSV
headers from Title Case to Sentence case ("Last Price" -> "Last price"), and
because csv.DictReader keys are exact, every money field silently parsed as
0.0 for five days. Quantity survived (single-word header), so the file looked
plausible while Total G/L on the Positions page equalled the whole portfolio
value. These tests pin both dialects and make a future rename fail loudly.

Fixtures are synthetic. Account numbers are REDACTED and every quantity, price
and cost basis below is invented round-number data chosen so the arithmetic
balances — this repo is public and must not carry real holdings.
"""
import pytest

from src.fidelity_sync import (
    FidelityCsvSchemaError,
    _looks_corrupt,
    _parse_fidelity_csv,
)

# Data rows are byte-identical between the two fixtures; only the header differs.
# AAA: 10 @ 100.00 = 1000.00 value, basis 10 @ 150.00 = 1500.00, so G/L -500.00
# (-33.33%); prior close 90.00, so today +10.00/sh = +100.00 (+11.11%).
# BBB: 20 @ 20.00 = 400.00 value, basis 20 @ 25.00 = 500.00, so G/L -100.00
# (-20.00%); prior close 22.00, so today -2.00/sh = -40.00 (-9.09%).
_ROWS = (
    "REDACTED,Individual,SPAXX**,HELD IN MONEY MARKET,,,,$500.00,,,,,10.00%,,,Cash,\n"
    "REDACTED,Individual,AAA,ALPHA EXAMPLE CORP COM,10,$100.00,+$10.00,"
    "$1000.00,+$100.00,+11.11%,-$500.00,-33.33%,20.00%,$1500.00,$150.00,Cash,\n"
    "REDACTED,Individual,BBB,BETA EXAMPLE INC COM,20,$20.00,-$2.00,"
    "$400.00,-$40.00,-9.09%,-$100.00,-20.00%,8.00%,$500.00,$25.00,Cash,\n"
)

HEADER_TITLE_CASE = (
    "Account Number,Account Name,Symbol,Description,Quantity,Last Price,"
    "Last Price Change,Current Value,Today's Gain/Loss Dollar,"
    "Today's Gain/Loss Percent,Total Gain/Loss Dollar,Total Gain/Loss Percent,"
    "Percent Of Account,Cost Basis Total,Average Cost Basis,Type\n"
)

HEADER_SENTENCE_CASE = (
    "Account number,Account name,Symbol,Description,Quantity,Last price,"
    "Last price change,Current value,Today's gain/loss dollar,"
    "Today's gain/loss percent,Total gain/loss dollar,Total gain/loss percent,"
    "Percent of account,Cost basis total,Average cost basis,Type\n"
)

CSV_TITLE_CASE = HEADER_TITLE_CASE + _ROWS
CSV_SENTENCE_CASE = HEADER_SENTENCE_CASE + _ROWS


# ── header-dialect tolerance ──────────────────────────────────────────────────

@pytest.mark.parametrize(
    "content", [CSV_TITLE_CASE, CSV_SENTENCE_CASE], ids=["title_case", "sentence_case"]
)
def test_money_fields_parse_in_both_header_dialects(content):
    """The exact regression: Sentence-case headers must not zero the money fields."""
    holdings = _parse_fidelity_csv(content)
    aaa = next(h for h in holdings if h["ticker"] == "AAA")

    assert aaa["quantity"] == 10.0
    assert aaa["last_price"] == 100.00
    assert aaa["avg_cost"] == 150.00
    assert aaa["cost_basis_total"] == 1500.00
    assert aaa["current_value"] == 1000.00
    assert aaa["today_gl_dollar"] == 100.00
    assert aaa["today_gl_pct"] == pytest.approx(0.1111)
    assert aaa["total_gl_dollar"] == -500.00
    assert aaa["total_gl_pct"] == pytest.approx(-0.3333)
    assert aaa["pct_of_account"] == pytest.approx(0.2000)


def test_both_header_dialects_produce_identical_output():
    assert _parse_fidelity_csv(CSV_TITLE_CASE) == _parse_fidelity_csv(CSV_SENTENCE_CASE)


def test_screaming_case_and_extra_whitespace_tolerated():
    """Normalisation is on alphanumerics, so spacing/case churn cannot break it."""
    header = (
        "ACCOUNT NUMBER,ACCOUNT NAME,SYMBOL, Description ,QUANTITY, LAST PRICE ,"
        "LAST PRICE CHANGE,CURRENT VALUE,TODAYS GAIN/LOSS DOLLAR,"
        "TODAYS GAIN/LOSS PERCENT,TOTAL GAIN/LOSS DOLLAR,TOTAL GAIN/LOSS PERCENT,"
        "PERCENT OF ACCOUNT,COST BASIS TOTAL,AVERAGE COST BASIS,TYPE\n"
    )
    assert _parse_fidelity_csv(header + _ROWS) == _parse_fidelity_csv(CSV_TITLE_CASE)


def test_bom_prefixed_header_parses():
    """The response-interception path can hand us a BOM-prefixed body."""
    assert _parse_fidelity_csv("﻿" + CSV_SENTENCE_CASE) == _parse_fidelity_csv(
        CSV_TITLE_CASE
    )


# ── fail loud, never persist zeros ────────────────────────────────────────────

def test_missing_money_column_raises_instead_of_zeroing():
    """A dropped/renamed column must abort the sync, not write 0.0 silently."""
    header = HEADER_TITLE_CASE.replace("Average Cost Basis", "Avg Unit Cost")
    with pytest.raises(FidelityCsvSchemaError) as exc:
        _parse_fidelity_csv(header + _ROWS)
    assert "Average Cost Basis" in str(exc.value)


def test_schema_error_names_every_missing_column():
    header = HEADER_TITLE_CASE.replace("Last Price Change", "Last Price Delta").replace(
        "Cost Basis Total", "Basis"
    )
    with pytest.raises(FidelityCsvSchemaError) as exc:
        _parse_fidelity_csv(header + _ROWS)
    msg = str(exc.value)
    # Last Price Change is optional; Cost Basis Total is not.
    assert "Cost Basis Total" in msg
    assert "Last Price Change" not in msg


def test_unparseable_header_raises():
    with pytest.raises(FidelityCsvSchemaError):
        _parse_fidelity_csv("this file is not a positions export\n1,2,3\n")


def test_optional_column_absent_is_tolerated():
    header = HEADER_TITLE_CASE.replace("Last Price Change,", "")
    rows = (_ROWS.replace(",+$10.00,", ",")
                 .replace(",-$2.00,", ",")
                 .replace(",,,,$500.00", ",,,$500.00"))
    holdings = _parse_fidelity_csv(header + rows)
    aaa = next(h for h in holdings if h["ticker"] == "AAA")
    assert aaa["last_price"] == 100.00
    assert aaa["last_price_chg"] == 0.0


# ── row filtering (pre-existing behaviour, pinned) ─────────────────────────────

def test_cash_and_money_market_rows_skipped():
    tickers = [h["ticker"] for h in _parse_fidelity_csv(CSV_SENTENCE_CASE)]
    assert tickers == ["AAA", "BBB"]
    assert "SPAXX" not in tickers


# ── _looks_corrupt: values unusable despite an intact header ───────────────────

def test_good_holdings_not_flagged_corrupt():
    assert _looks_corrupt(_parse_fidelity_csv(CSV_SENTENCE_CASE)) is False


def test_all_zero_holdings_flagged_corrupt():
    """What the outage actually wrote to disk for five days."""
    zeroed = [
        {"ticker": "AAA", "quantity": 10.0, "last_price": 0.0,
         "avg_cost": 0.0, "cost_basis_total": 0.0},
        {"ticker": "BBB", "quantity": 20.0, "last_price": 0.0,
         "avg_cost": 0.0, "cost_basis_total": 0.0},
    ]
    assert _looks_corrupt(zeroed) is True


def test_missing_only_cost_basis_flagged_corrupt():
    holdings = [{"ticker": "AAA", "quantity": 10.0, "last_price": 100.00,
                 "avg_cost": 0.0, "cost_basis_total": 0.0}]
    assert _looks_corrupt(holdings) is True


def test_cost_basis_total_alone_is_enough():
    holdings = [{"ticker": "AAA", "quantity": 10.0, "last_price": 100.00,
                 "avg_cost": 0.0, "cost_basis_total": 1500.00}]
    assert _looks_corrupt(holdings) is False


def test_empty_holdings_not_flagged_corrupt():
    """Empty is 'no holdings', a separate status — not corruption."""
    assert _looks_corrupt([]) is False
