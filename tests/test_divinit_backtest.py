import pandas as pd

from src.divinit_backtest import (
    detect_candidates,
    find_declaration_via_submissions,
    match_fts_declaration,
    resolve_declarations,
)


def _payments(pairs):
    return [(pd.Timestamp(d), amt) for d, amt in pairs]


def test_first_ever_payment_is_initiation():
    payments = {"AAA": _payments([("2026-01-01", 0.10)])}
    out = detect_candidates(payments, "2025-01-01", "2026-12-31")
    assert len(out) == 1
    assert out.iloc[0]["category"] == "div_initiation"
    assert out.iloc[0]["gap_years"] is None


def test_gap_over_three_years_is_initiation():
    payments = {"AAA": _payments([("2020-01-01", 0.10), ("2026-01-01", 0.10)])}
    out = detect_candidates(payments, "2025-01-01", "2026-12-31")
    assert len(out) == 1
    assert out.iloc[0]["category"] == "div_initiation"


def test_gap_one_to_three_years_is_resumption():
    payments = {"AAA": _payments([("2024-01-01", 0.10), ("2026-01-01", 0.10)])}
    out = detect_candidates(payments, "2025-01-01", "2026-12-31")
    assert len(out) == 1
    assert out.iloc[0]["category"] == "div_resumption"


def test_routine_quarterly_gap_not_a_candidate():
    # 3 prior payments establish a routine cadence; the 2026-01-01 payment
    # (a normal 3-month gap after the previous one) should not be flagged —
    # only the FIRST payment in a ticker's history is a "first ever" case.
    payments = {"AAA": _payments([
        ("2025-01-01", 0.10), ("2025-04-01", 0.10), ("2025-07-01", 0.10),
        ("2025-10-01", 0.10), ("2026-01-01", 0.10),
    ])}
    out = detect_candidates(payments, "2026-01-01", "2026-12-31")
    assert out.empty


def test_payment_outside_window_excluded():
    payments = {"AAA": _payments([("2020-01-01", 0.10)])}
    out = detect_candidates(payments, "2025-01-01", "2026-12-31")
    assert out.empty


def test_match_fts_declaration_finds_latest_hit_in_window():
    candidates = pd.DataFrame([{"ticker": "AAA", "cik": "0000000001", "ex_date": pd.Timestamp("2026-02-01")}])
    fts_hits = pd.DataFrame([
        {"cik": "0000000001", "file_date": pd.Timestamp("2026-01-10")},
        {"cik": "0000000001", "file_date": pd.Timestamp("2026-01-20")},
    ])
    out = match_fts_declaration(candidates, fts_hits)
    assert out[("AAA", pd.Timestamp("2026-02-01"))] == pd.Timestamp("2026-01-20")


def test_match_fts_declaration_ignores_hits_outside_window():
    candidates = pd.DataFrame([{"ticker": "AAA", "cik": "0000000001", "ex_date": pd.Timestamp("2026-02-01")}])
    fts_hits = pd.DataFrame([{"cik": "0000000001", "file_date": pd.Timestamp("2025-01-01")}])
    out = match_fts_declaration(candidates, fts_hits)
    assert out == {}


def test_find_declaration_via_submissions_confirms_regex_match(monkeypatch):
    monkeypatch.setattr("src.divinit_backtest.fetch_submissions", lambda cik, db_path: {"stub": True})
    monkeypatch.setattr("src.divinit_backtest.parse_submissions", lambda data: [
        {"form": "8-K", "accession": "acc1", "filing_date": "2026-01-15",
         "report_date": "", "primary_doc": "doc.htm"},
    ])
    monkeypatch.setattr("src.divinit_backtest.fetch_filing_doc", lambda cik, acc, doc, db_path: "<html>x</html>")
    monkeypatch.setattr("src.divinit_backtest.plain_text", lambda html: "The board declared an inaugural dividend.")
    decl_date, confirmed = find_declaration_via_submissions("0000000001", pd.Timestamp("2026-02-01"), "db")
    assert confirmed is True
    assert decl_date == pd.Timestamp("2026-01-15")


def test_find_declaration_via_submissions_falls_back_when_no_regex_match(monkeypatch):
    monkeypatch.setattr("src.divinit_backtest.fetch_submissions", lambda cik, db_path: {"stub": True})
    monkeypatch.setattr("src.divinit_backtest.parse_submissions", lambda data: [
        {"form": "8-K", "accession": "acc1", "filing_date": "2026-01-15",
         "report_date": "", "primary_doc": "doc.htm"},
    ])
    monkeypatch.setattr("src.divinit_backtest.fetch_filing_doc", lambda cik, acc, doc, db_path: "<html>x</html>")
    monkeypatch.setattr("src.divinit_backtest.plain_text", lambda html: "Unrelated 8-K content.")
    decl_date, confirmed = find_declaration_via_submissions("0000000001", pd.Timestamp("2026-02-01"), "db")
    assert confirmed is False
    assert decl_date == pd.Timestamp("2026-01-18")  # ex_date - 14 days


def test_find_declaration_via_submissions_drops_when_no_8k_in_window(monkeypatch):
    monkeypatch.setattr("src.divinit_backtest.fetch_submissions", lambda cik, db_path: {"stub": True})
    monkeypatch.setattr("src.divinit_backtest.parse_submissions", lambda data: [
        {"form": "8-K", "accession": "acc1", "filing_date": "2020-01-15",
         "report_date": "", "primary_doc": "doc.htm"},
    ])
    decl_date, confirmed = find_declaration_via_submissions("0000000001", pd.Timestamp("2026-02-01"), "db")
    assert decl_date is None
    assert confirmed is False


def test_resolve_declarations_prefers_fts_over_submissions_fallback(monkeypatch):
    candidates = pd.DataFrame([{
        "ticker": "AAA", "cik": "0000000001", "ex_date": pd.Timestamp("2026-02-01"),
        "amount": 0.10, "gap_years": None, "category": "div_initiation",
    }])
    fts_hits = pd.DataFrame([{"cik": "0000000001", "file_date": pd.Timestamp("2026-01-20")}])

    def _boom(*a, **k):
        raise AssertionError("should not fall back to submissions when FTS already matched")
    monkeypatch.setattr("src.divinit_backtest.fetch_submissions", _boom)

    out = resolve_declarations(candidates, fts_hits)
    assert len(out) == 1
    assert out.iloc[0]["file_date"] == pd.Timestamp("2026-01-20")


def test_resolve_declarations_drops_unresolved_candidate(monkeypatch):
    candidates = pd.DataFrame([{
        "ticker": "AAA", "cik": "0000000001", "ex_date": pd.Timestamp("2026-02-01"),
        "amount": 0.10, "gap_years": None, "category": "div_initiation",
    }])
    monkeypatch.setattr("src.divinit_backtest.fetch_submissions", lambda cik, db_path: None)
    out = resolve_declarations(candidates, pd.DataFrame())
    assert out.empty
