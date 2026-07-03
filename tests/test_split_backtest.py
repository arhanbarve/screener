import pandas as pd

from src.split_backtest import build_events, match_hits_to_splits


def _hit(ticker, file_date, cik="0000000001", adsh="x"):
    return {"ticker": ticker, "cik": cik, "file_date": pd.Timestamp(file_date), "adsh": adsh}


def test_forward_split_classified_correctly():
    hits = pd.DataFrame([_hit("AAA", "2026-01-05")])
    splits = {"AAA": [(pd.Timestamp("2026-02-01"), 4.0)]}
    out = match_hits_to_splits(hits, splits)
    assert len(out) == 1
    assert out.iloc[0]["category"] == "split_fwd_announce"
    assert out.iloc[0]["factor"] == 4.0


def test_reverse_split_classified_correctly():
    hits = pd.DataFrame([_hit("AAA", "2026-01-05")])
    splits = {"AAA": [(pd.Timestamp("2026-02-01"), 0.1)]}
    out = match_hits_to_splits(hits, splits)
    assert len(out) == 1
    assert out.iloc[0]["category"] == "split_reverse_announce"


def test_hit_with_no_matching_real_split_is_dropped():
    hits = pd.DataFrame([_hit("AAA", "2026-01-05")])
    out = match_hits_to_splits(hits, {"AAA": []})
    assert out.empty


def test_real_split_outside_window_not_matched():
    hits = pd.DataFrame([_hit("AAA", "2026-01-05")])
    # split 200 days after the hit — outside the 90d match window
    splits = {"AAA": [(pd.Timestamp("2026-07-24"), 2.0)]}
    out = match_hits_to_splits(hits, splits, window_days=90)
    assert out.empty


def test_earliest_hit_wins_as_announcement():
    hits = pd.DataFrame([
        _hit("AAA", "2026-01-05", adsh="first"),
        _hit("AAA", "2026-01-20", adsh="second"),
    ])
    splits = {"AAA": [(pd.Timestamp("2026-02-01"), 2.0)]}
    out = match_hits_to_splits(hits, splits)
    assert len(out) == 1
    assert out.iloc[0]["adsh"] == "first"


def test_factor_of_one_is_ignored():
    hits = pd.DataFrame([_hit("AAA", "2026-01-05")])
    splits = {"AAA": [(pd.Timestamp("2026-02-01"), 1.0)]}
    out = match_hits_to_splits(hits, splits)
    assert out.empty


def test_multiple_real_splits_produce_multiple_events():
    hits = pd.DataFrame([
        _hit("AAA", "2026-01-05", adsh="a1"),
        _hit("AAA", "2026-06-01", adsh="a2"),
    ])
    splits = {"AAA": [(pd.Timestamp("2026-02-01"), 2.0), (pd.Timestamp("2026-07-01"), 3.0)]}
    out = match_hits_to_splits(hits, splits)
    assert len(out) == 2


def test_build_events_sets_event_date_and_applies_universe_filter():
    # search() output has no `ticker` column (only `cik`) — filter_to_universe
    # is what joins in the ticker, so raw hits here mirror that shape.
    raw_hits = pd.DataFrame([
        {"cik": "0000000001", "file_date": pd.Timestamp("2026-01-05"), "adsh": "x"},
        {"cik": "0000000002", "file_date": pd.Timestamp("2026-01-05"), "adsh": "y"},
    ])
    uni = pd.DataFrame([{"cik": "0000000001", "ticker": "AAA"}])
    splits = {"AAA": [(pd.Timestamp("2026-02-01"), 4.0)]}
    earnings = pd.DataFrame(columns=["ticker", "announce_date"])
    out = build_events(raw_hits, uni, splits, earnings)
    assert list(out["ticker"]) == ["AAA"]
    assert out.iloc[0]["event_date"] == "2026-01-06"
