import pandas as pd

from src.specialdiv_backtest import confirm_special_events, find_regular_increase_events


def _hit(ticker, file_date, cik="0000000001", adsh="x"):
    return {"ticker": ticker, "cik": cik, "file_date": pd.Timestamp(file_date), "adsh": adsh}


def _payments(pairs):
    return [(pd.Timestamp(d), amt) for d, amt in pairs]


def test_outsized_payment_confirmed_as_special():
    hits = pd.DataFrame([_hit("AAA", "2026-01-05")])
    payments = _payments([
        ("2025-01-01", 0.50), ("2025-04-01", 0.50), ("2025-07-01", 0.50), ("2025-10-01", 0.50),
        ("2026-02-01", 2.00),  # 4x baseline, within 90d of the hit
    ])
    out = confirm_special_events(hits, {"AAA": payments})
    assert len(out) == 1
    assert out.iloc[0]["category"] == "special_div"
    assert out.iloc[0]["amount"] == 2.00


def test_routine_payment_not_confirmed():
    hits = pd.DataFrame([_hit("AAA", "2026-01-05")])
    payments = _payments([
        ("2025-01-01", 0.50), ("2025-04-01", 0.50), ("2025-07-01", 0.50), ("2025-10-01", 0.50),
        ("2026-02-01", 0.52),  # routine, ~1.04x baseline — not special
    ])
    out = confirm_special_events(hits, {"AAA": payments})
    assert out.empty


def test_no_matching_hit_within_window_drops_payment():
    hits = pd.DataFrame([_hit("AAA", "2020-01-01")])  # far before the payment
    payments = _payments([
        ("2025-01-01", 0.50), ("2025-04-01", 0.50), ("2025-07-01", 0.50), ("2025-10-01", 0.50),
        ("2026-02-01", 2.00),
    ])
    out = confirm_special_events(hits, {"AAA": payments})
    assert out.empty


def test_insufficient_payment_history_dropped():
    hits = pd.DataFrame([_hit("AAA", "2026-01-05")])
    payments = _payments([("2025-10-01", 0.50), ("2026-02-01", 2.00)])  # only 2 payments, need 5
    out = confirm_special_events(hits, {"AAA": payments})
    assert out.empty


def test_find_regular_increase_events_matches_modest_raise():
    payments = {"AAA": _payments([("2026-01-01", 0.50), ("2026-04-01", 0.55)])}  # 1.10x, in range
    out = find_regular_increase_events(payments, "2026-01-01", "2026-12-31")
    assert len(out) == 1
    assert out.iloc[0]["category"] == "regular_div_increase"


def test_find_regular_increase_events_excludes_large_jump():
    payments = {"AAA": _payments([("2026-01-01", 0.50), ("2026-04-01", 2.00)])}  # 4x, too big
    out = find_regular_increase_events(payments, "2026-01-01", "2026-12-31")
    assert out.empty


def test_find_regular_increase_events_excludes_decrease():
    payments = {"AAA": _payments([("2026-01-01", 0.50), ("2026-04-01", 0.40)])}  # decrease
    out = find_regular_increase_events(payments, "2026-01-01", "2026-12-31")
    assert out.empty


def test_find_regular_increase_events_respects_date_range():
    payments = {"AAA": _payments([("2020-01-01", 0.50), ("2020-04-01", 0.55)])}  # outside range
    out = find_regular_increase_events(payments, "2026-01-01", "2026-12-31")
    assert out.empty
