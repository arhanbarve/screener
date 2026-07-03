from src.edgar_fts import month_windows, parse_ticker_hint


def test_parse_ticker_hint_single_ticker():
    name = "PHIBRO ANIMAL HEALTH CORP  (PAHC)  (CIK 0001069899)"
    assert parse_ticker_hint(name) == "PAHC"


def test_parse_ticker_hint_multiple_tickers_takes_first():
    name = "SIM Acquisition Corp. I  (SIMA, SIMAU, SIMAW)  (CIK 0002014982)"
    assert parse_ticker_hint(name) == "SIMA"


def test_parse_ticker_hint_no_ticker_for_private_filer():
    name = "BFI Co., LLC  (CIK 0001601607)"
    assert parse_ticker_hint(name) is None


def test_parse_ticker_hint_empty_string():
    assert parse_ticker_hint("") is None


def test_month_windows_single_month():
    assert month_windows("2026-05-01", "2026-05-31") == [("2026-05-01", "2026-05-31")]


def test_month_windows_partial_month_clips_to_range():
    assert month_windows("2026-05-15", "2026-05-20") == [("2026-05-15", "2026-05-20")]


def test_month_windows_spans_multiple_months():
    windows = month_windows("2026-05-15", "2026-07-10")
    assert windows == [
        ("2026-05-15", "2026-05-31"),
        ("2026-06-01", "2026-06-30"),
        ("2026-07-01", "2026-07-10"),
    ]
