import pandas as pd
import pytest

from src.insider_backtest import detect_events, build_events


def _rec(ticker, owner, filing_date, code="P", value=50_000):
    return {"ticker": ticker, "RPTOWNERCIK": owner,
            "filing_date": pd.Timestamp(filing_date),
            "TRANS_CODE": code, "value_usd": float(value)}


def test_two_owners_within_window_is_cluster():
    df = pd.DataFrame([_rec("AAA", 1, "2026-01-05"), _rec("AAA", 2, "2026-01-10")])
    events = detect_events(df, "P", "insider_cluster_buy", "insider_single_buy")
    clusters = [e for e in events if e["category"] == "insider_cluster_buy"]
    assert len(clusters) == 1
    # fires on the filing that completes the cluster, not the first buy
    assert clusters[0]["filing_date"] == pd.Timestamp("2026-01-10")
    assert clusters[0]["n_owners"] == 2
    assert clusters[0]["total_usd"] == 100_000


def test_same_owner_twice_is_not_a_cluster():
    df = pd.DataFrame([_rec("AAA", 1, "2026-01-05"), _rec("AAA", 1, "2026-01-08")])
    events = detect_events(df, "P", "insider_cluster_buy", "insider_single_buy")
    assert all(e["category"] == "insider_single_buy" for e in events)


def test_owners_outside_window_do_not_cluster():
    df = pd.DataFrame([_rec("AAA", 1, "2026-01-05"), _rec("AAA", 2, "2026-01-25")])
    events = detect_events(df, "P", "insider_cluster_buy", "insider_single_buy")
    assert all(e["category"] == "insider_single_buy" for e in events)
    assert len(events) == 2


def test_dedupe_suppresses_repeat_cluster_events():
    df = pd.DataFrame([
        _rec("AAA", 1, "2026-01-05"), _rec("AAA", 2, "2026-01-08"),
        _rec("AAA", 3, "2026-01-12"),  # still within 20d of first event
    ])
    events = detect_events(df, "P", "insider_cluster_buy", None)
    assert len(events) == 1


def test_new_cluster_after_dedupe_gap_fires():
    df = pd.DataFrame([
        _rec("AAA", 1, "2026-01-05"), _rec("AAA", 2, "2026-01-08"),
        _rec("AAA", 3, "2026-03-01"), _rec("AAA", 4, "2026-03-05"),
    ])
    events = detect_events(df, "P", "insider_cluster_buy", None)
    assert len(events) == 2


def test_single_buy_pending_cluster_not_double_counted():
    # owner1's buy would be a "single" at 01-05 but owner2 arrives 01-10 —
    # must emit exactly one cluster event, not a single AND a cluster
    df = pd.DataFrame([_rec("AAA", 1, "2026-01-05"), _rec("AAA", 2, "2026-01-10")])
    events = detect_events(df, "P", "insider_cluster_buy", "insider_single_buy")
    assert len(events) == 1
    assert events[0]["category"] == "insider_cluster_buy"


def test_sales_do_not_mix_with_buys():
    df = pd.DataFrame([
        _rec("AAA", 1, "2026-01-05", code="P"),
        _rec("AAA", 2, "2026-01-08", code="S"),
    ])
    buy_events = detect_events(df, "P", "insider_cluster_buy", "insider_single_buy")
    assert all(e["category"] == "insider_single_buy" for e in buy_events)


def test_build_events_filters_universe_and_sets_event_date():
    df = pd.DataFrame([
        _rec("AAA", 1, "2026-01-05"), _rec("AAA", 2, "2026-01-10"),
        _rec("ZZZ", 1, "2026-01-05"), _rec("ZZZ", 2, "2026-01-10"),
    ])
    events = build_events(df, universe_tickers={"AAA"})
    assert set(events["ticker"]) == {"AAA"}
    cluster = events[events["category"] == "insider_cluster_buy"].iloc[0]
    assert cluster["event_date"] == "2026-01-11"  # filing +1 day


def test_build_events_empty_universe_returns_empty():
    df = pd.DataFrame([_rec("AAA", 1, "2026-01-05")])
    assert build_events(df, universe_tickers=set()).empty
