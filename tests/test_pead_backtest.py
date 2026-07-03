import sqlite3
from datetime import date, datetime, timedelta

import pandas as pd
import pytest

from src.pead_backtest import (
    classify_surprise, event_date_from_announcement, load_events,
    _init_earnings_table, _already_fetched,
)


# ── classify_surprise ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("pct,expected", [
    (25.0, "pead_beat_large"),
    (15.0, "pead_beat_large"),   # boundary: >=15 is large
    (14.99, "pead_beat_mid"),
    (5.0, "pead_beat_mid"),      # boundary: >=5 is mid
    (4.99, "pead_beat_small"),
    (0.0, "pead_beat_small"),    # exact meet counts as small beat, not miss
    (-0.01, "pead_miss"),
    (-50.0, "pead_miss"),
])
def test_classify_surprise_buckets(pct, expected):
    assert classify_surprise(pct) == expected


def test_classify_surprise_nan_returns_none():
    assert classify_surprise(float("nan")) is None
    assert classify_surprise(None) is None


# ── event date shift ──────────────────────────────────────────────────────────

def test_event_date_is_day_after_announcement():
    assert event_date_from_announcement("2026-05-05") == "2026-05-06"


def test_event_date_crosses_month_boundary():
    assert event_date_from_announcement("2026-01-31") == "2026-02-01"


# ── load_events filtering ─────────────────────────────────────────────────────

def _seed_db(tmp_path, rows):
    db = str(tmp_path / "cache.db")
    _init_earnings_table(db)
    con = sqlite3.connect(db)
    for r in rows:
        con.execute("INSERT OR REPLACE INTO pead_earnings VALUES (?,?,?,?,?,?)",
                    (r["ticker"], r["announce_date"], r["eps_estimate"],
                     r["eps_actual"], r["surprise_pct"], datetime.now().isoformat()))
    con.commit()
    con.close()
    return db


def test_load_events_filters_tiny_estimates(tmp_path):
    recent = (date.today() - timedelta(days=30)).isoformat()
    db = _seed_db(tmp_path, [
        {"ticker": "AAA", "announce_date": recent, "eps_estimate": 0.01,
         "eps_actual": 0.05, "surprise_pct": 400.0},
        {"ticker": "BBB", "announce_date": recent, "eps_estimate": 1.00,
         "eps_actual": 1.10, "surprise_pct": 10.0},
    ])
    events = load_events(db_path=db, lookback_days=730)
    assert list(events["ticker"]) == ["BBB"]


def test_load_events_respects_lookback(tmp_path):
    old = (date.today() - timedelta(days=800)).isoformat()
    recent = (date.today() - timedelta(days=30)).isoformat()
    db = _seed_db(tmp_path, [
        {"ticker": "OLD", "announce_date": old, "eps_estimate": 1.0,
         "eps_actual": 1.2, "surprise_pct": 20.0},
        {"ticker": "NEW", "announce_date": recent, "eps_estimate": 1.0,
         "eps_actual": 1.2, "surprise_pct": 20.0},
    ])
    events = load_events(db_path=db, lookback_days=730)
    assert list(events["ticker"]) == ["NEW"]


def test_load_events_excludes_future_announcements(tmp_path):
    future = (date.today() + timedelta(days=5)).isoformat()
    db = _seed_db(tmp_path, [
        {"ticker": "FUT", "announce_date": future, "eps_estimate": 1.0,
         "eps_actual": 1.2, "surprise_pct": 20.0},
    ])
    events = load_events(db_path=db, lookback_days=730)
    assert events.empty


def test_load_events_assigns_category_and_event_date(tmp_path):
    recent = (date.today() - timedelta(days=30)).isoformat()
    db = _seed_db(tmp_path, [
        {"ticker": "AAA", "announce_date": recent, "eps_estimate": 1.0,
         "eps_actual": 1.2, "surprise_pct": 20.0},
    ])
    events = load_events(db_path=db, lookback_days=730)
    assert events.iloc[0]["category"] == "pead_beat_large"
    assert events.iloc[0]["event_date"] == event_date_from_announcement(recent)


# ── fetch log TTL ─────────────────────────────────────────────────────────────

def test_already_fetched_respects_ttl(tmp_path):
    db = str(tmp_path / "cache.db")
    _init_earnings_table(db)
    con = sqlite3.connect(db)
    fresh = datetime.now().isoformat()
    stale = (datetime.now() - timedelta(days=30)).isoformat()
    con.execute("INSERT INTO pead_fetch_log VALUES (?,?,?)", ("FRESH", fresh, 8))
    con.execute("INSERT INTO pead_fetch_log VALUES (?,?,?)", ("STALE", stale, 8))
    con.commit()
    con.close()
    assert _already_fetched(db, ttl_days=7) == {"FRESH"}
