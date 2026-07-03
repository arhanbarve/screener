import pandas as pd

from src.sc13d_backtest import (
    dedupe_events,
    drop_earnings_contamination,
    fetch_raw_events,
    filter_to_universe,
)


def _events(rows):
    df = pd.DataFrame(rows)
    df["file_date"] = pd.to_datetime(df["file_date"])
    return df


def test_filter_to_universe_drops_non_universe_ciks():
    events = pd.DataFrame([{"cik": "0000000001"}, {"cik": "0000000002"}])
    uni = pd.DataFrame([{"cik": "0000000001", "ticker": "AAA"}])
    out = filter_to_universe(events, uni)
    assert list(out["ticker"]) == ["AAA"]


def test_dedupe_keeps_first_within_window():
    events = _events([
        {"ticker": "AAA", "file_date": "2026-01-05"},
        {"ticker": "AAA", "file_date": "2026-01-15"},  # 10d gap, dropped
    ])
    out = dedupe_events(events, dedupe_days=20)
    assert len(out) == 1
    assert out.iloc[0]["file_date"] == pd.Timestamp("2026-01-05")


def test_dedupe_keeps_events_past_window():
    events = _events([
        {"ticker": "AAA", "file_date": "2026-01-05"},
        {"ticker": "AAA", "file_date": "2026-02-10"},  # 36d gap, kept
    ])
    out = dedupe_events(events, dedupe_days=20)
    assert len(out) == 2


def test_dedupe_is_per_ticker():
    events = _events([
        {"ticker": "AAA", "file_date": "2026-01-05"},
        {"ticker": "BBB", "file_date": "2026-01-06"},
    ])
    out = dedupe_events(events, dedupe_days=20)
    assert len(out) == 2


def test_drop_earnings_contamination_drops_within_window():
    events = pd.DataFrame([
        {"ticker": "AAA", "event_date": "2026-01-10"},   # 2d from earnings
        {"ticker": "AAA", "event_date": "2026-06-01"},   # far from earnings
    ])
    earnings = pd.DataFrame([{"ticker": "AAA", "announce_date": "2026-01-08"}])
    out = drop_earnings_contamination(events, earnings, window_days=3)
    assert list(out["event_date"]) == ["2026-06-01"]


def test_drop_earnings_contamination_no_earnings_data_keeps_event():
    events = pd.DataFrame([{"ticker": "ZZZ", "event_date": "2026-01-10"}])
    earnings = pd.DataFrame([{"ticker": "AAA", "announce_date": "2026-01-08"}])
    out = drop_earnings_contamination(events, earnings, window_days=3)
    assert len(out) == 1


def test_fetch_raw_events_output_feeds_dedupe_without_crashing(monkeypatch):
    # Regression: search() returns file_date as a plain string; dedupe_events
    # does datetime arithmetic on it. fetch_raw_events must convert.
    hits = pd.DataFrame([
        {"cik": "0000000001", "ticker_hint": "AAA", "file_date": "2026-01-05",
         "adsh": "x", "form": "SCHEDULE 13D"},
    ])
    monkeypatch.setattr("src.sc13d_backtest.search", lambda *a, **k: hits)
    raw = fetch_raw_events("Schedule 13D", "SCHEDULE 13D", "SCHEDULE 13D",
                           "2026-01-01", "2026-01-31", "sc13d_new")
    assert pd.api.types.is_datetime64_any_dtype(raw["file_date"])
    uni = pd.DataFrame([{"cik": "0000000001", "ticker": "AAA"}])
    out = dedupe_events(filter_to_universe(raw, uni))
    assert len(out) == 1


def test_fetch_raw_events_excludes_amendments(monkeypatch):
    hits = pd.DataFrame([
        {"cik": "0000000001", "ticker_hint": "AAA", "file_date": "2026-01-05",
         "adsh": "x", "form": "SCHEDULE 13D"},
        {"cik": "0000000002", "ticker_hint": "BBB", "file_date": "2026-01-06",
         "adsh": "y", "form": "SCHEDULE 13D/A"},
    ])
    monkeypatch.setattr("src.sc13d_backtest.search", lambda *a, **k: hits)
    out = fetch_raw_events("Schedule 13D", "SCHEDULE 13D", "SCHEDULE 13D",
                           "2026-01-01", "2026-01-31", "sc13d_new")
    assert list(out["cik"]) == ["0000000001"]
    assert (out["category"] == "sc13d_new").all()
