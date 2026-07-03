import pandas as pd

from src.divhike_backtest import detect_hike_candidates, resolve_declarations


def _payments(pairs):
    return [(pd.Timestamp(d), amt) for d, amt in pairs]


def test_large_hike_classified_correctly():
    payments = {"AAA": _payments([("2026-01-01", 0.40), ("2026-04-01", 0.50)])}  # 1.25x exactly
    out = detect_hike_candidates(payments, "2026-01-01", "2026-12-31")
    assert len(out) == 1
    assert out.iloc[0]["category"] == "div_hike_large"


def test_small_hike_classified_correctly():
    payments = {"AAA": _payments([("2026-01-01", 0.50), ("2026-04-01", 0.53)])}  # 1.06x
    out = detect_hike_candidates(payments, "2026-01-01", "2026-12-31")
    assert len(out) == 1
    assert out.iloc[0]["category"] == "div_hike_small"


def test_cut_classified_correctly():
    payments = {"AAA": _payments([("2026-01-01", 0.50), ("2026-04-01", 0.30)])}  # 0.6x
    out = detect_hike_candidates(payments, "2026-01-01", "2026-12-31")
    assert len(out) == 1
    assert out.iloc[0]["category"] == "div_cut"


def test_ambiguous_zone_not_a_candidate():
    payments = {"AAA": _payments([("2026-01-01", 0.50), ("2026-04-01", 0.55)])}  # 1.10x, boundary — in small
    out = detect_hike_candidates(payments, "2026-01-01", "2026-12-31")
    assert len(out) == 1  # 1.10 is included (<=1.10)
    payments2 = {"AAA": _payments([("2026-01-01", 0.50), ("2026-04-01", 0.57)])}  # 1.14x, in the gap
    out2 = detect_hike_candidates(payments2, "2026-01-01", "2026-12-31")
    assert out2.empty


def test_first_payment_has_no_predecessor_and_is_skipped():
    payments = {"AAA": _payments([("2026-01-01", 0.50)])}
    out = detect_hike_candidates(payments, "2026-01-01", "2026-12-31")
    assert out.empty


def test_payment_outside_window_excluded():
    payments = {"AAA": _payments([("2020-01-01", 0.40), ("2020-04-01", 0.50)])}
    out = detect_hike_candidates(payments, "2026-01-01", "2026-12-31")
    assert out.empty


def test_resolve_declarations_drops_candidates_with_no_cik(monkeypatch):
    candidates = pd.DataFrame([{"ticker": "ZZZ", "ex_date": pd.Timestamp("2026-02-01"),
                               "amount": 0.5, "ratio": 1.3, "category": "div_hike_large"}])
    uni_cik = pd.DataFrame([{"ticker": "AAA", "cik": "0000000001"}])  # ZZZ not in universe map

    def _boom(*a, **k):
        raise AssertionError("should not attempt declaration lookup without a CIK")
    monkeypatch.setattr("src.divhike_backtest.find_declaration_via_submissions", _boom)

    out = resolve_declarations(candidates, uni_cik)
    assert out.empty


def test_resolve_declarations_attaches_file_date(monkeypatch):
    candidates = pd.DataFrame([{"ticker": "AAA", "ex_date": pd.Timestamp("2026-02-01"),
                               "amount": 0.5, "ratio": 1.3, "category": "div_hike_large"}])
    uni_cik = pd.DataFrame([{"ticker": "AAA", "cik": "0000000001"}])
    monkeypatch.setattr("src.divhike_backtest.find_declaration_via_submissions",
                        lambda cik, ex_date, db_path: (pd.Timestamp("2026-01-15"), True))
    out = resolve_declarations(candidates, uni_cik)
    assert len(out) == 1
    assert out.iloc[0]["file_date"] == pd.Timestamp("2026-01-15")
