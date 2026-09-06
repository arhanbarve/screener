# tests/test_cache.py
import os
import tempfile
import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from src.cache import (
    init_db, put_prices, get_prices, put_fundamentals, get_fundamentals, put_edgar, get_edgar,
    archive_universe_snapshot, put_backtest_prices, get_backtest_prices,
    get_ticker_profile, put_ticker_profile,
)

def make_tmp_db():
    f = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    path = f.name
    f.close()
    return path

def test_init_creates_tables():
    db = make_tmp_db()
    init_db(db)
    import sqlite3
    conn = sqlite3.connect(db)
    c = conn.cursor()
    c.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {r[0] for r in c.fetchall()}
    conn.close()
    assert {"prices", "fundamentals", "edgar"} <= tables
    os.unlink(db)

def test_prices_roundtrip():
    db = make_tmp_db()
    init_db(db)
    df = pd.DataFrame({
        "open": [100.0], "high": [105.0], "low": [99.0],
        "close": [103.0], "volume": [1000000]
    }, index=pd.to_datetime(["2024-01-02"]))
    df.index.name = "date"
    put_prices(db, "AAPL", df)
    result = get_prices(db, "AAPL", ttl_hours=18)
    assert result is not None
    assert len(result) == 1
    assert abs(result["close"].iloc[0] - 103.0) < 1e-6
    os.unlink(db)

def test_prices_expired_returns_none():
    db = make_tmp_db()
    init_db(db)
    df = pd.DataFrame({
        "open": [100.0], "high": [105.0], "low": [99.0],
        "close": [103.0], "volume": [1000000]
    }, index=pd.to_datetime(["2024-01-02"]))
    df.index.name = "date"
    put_prices(db, "AAPL", df)
    # TTL of 0 hours means immediately expired
    result = get_prices(db, "AAPL", ttl_hours=0)
    assert result is None
    os.unlink(db)

def test_backtest_prices_roundtrip():
    db = make_tmp_db()
    init_db(db)
    df = pd.DataFrame({
        "open": [10.0, 10.5], "high": [10.2, 10.8], "low": [9.9, 10.4],
        "close": [10.1, 10.6], "volume": [50000, 60000]
    }, index=pd.to_datetime(["2024-01-02", "2024-01-03"]))
    df.index.name = "date"
    put_backtest_prices(db, "SMCP", df)
    result = get_backtest_prices(db, "SMCP", ttl_days=30)
    assert result is not None
    assert len(result) == 2
    assert abs(result["close"].iloc[1] - 10.6) < 1e-6
    os.unlink(db)

def test_backtest_prices_expired_returns_none():
    db = make_tmp_db()
    init_db(db)
    df = pd.DataFrame({
        "open": [10.0], "high": [10.2], "low": [9.9],
        "close": [10.1], "volume": [50000]
    }, index=pd.to_datetime(["2024-01-02"]))
    df.index.name = "date"
    put_backtest_prices(db, "SMCP", df)
    result = get_backtest_prices(db, "SMCP", ttl_days=0)
    assert result is None
    os.unlink(db)

def test_fundamentals_roundtrip():
    db = make_tmp_db()
    init_db(db)
    payload = {"eps": [1.2, 1.3], "estimate": [1.1, 1.25]}
    put_fundamentals(db, "MSFT", payload)
    result = get_fundamentals(db, "MSFT", ttl_days=7)
    assert result is not None
    assert result["eps"] == [1.2, 1.3]
    os.unlink(db)

def test_edgar_roundtrip():
    db = make_tmp_db()
    init_db(db)
    put_edgar(db, "0000320193", gp_assets=0.35, revenue=400e9, cogs=200e9, assets=350e9)
    result = get_edgar(db, "0000320193", ttl_days=30)
    assert result is not None
    assert abs(result["gp_assets"] - 0.35) < 1e-9
    os.unlink(db)

def test_archive_universe_snapshot_nan_cik():
    import sqlite3
    db = make_tmp_db()
    init_db(db)
    df = pd.DataFrame({"ticker": ["AAPL", "MSFT"], "cik": [np.nan, "0000789019"]})
    archive_universe_snapshot(db, "2026-06-30", df)
    conn = sqlite3.connect(db)
    rows = conn.execute(
        "SELECT ticker, cik FROM universe_snapshots ORDER BY ticker"
    ).fetchall()
    conn.close()
    assert rows[0] == ("AAPL", "")
    assert rows[1] == ("MSFT", "0000789019")
    os.unlink(db)

def test_ticker_profile_roundtrip():
    db = make_tmp_db()
    init_db(db)
    assert get_ticker_profile(db, "AAA") is None
    put_ticker_profile(db, "AAA", "Technology", "Semiconductors")
    prof = get_ticker_profile(db, "AAA")
    assert prof == {"sector": "Technology", "industry": "Semiconductors"}
    os.unlink(db)

def test_ticker_profile_stores_empty_strings():
    db = make_tmp_db()
    init_db(db)
    put_ticker_profile(db, "BBB", None, None)  # failed lookups cache as empty
    assert get_ticker_profile(db, "BBB") == {"sector": "", "industry": ""}
    os.unlink(db)


def test_fh_metrics_roundtrip_and_bulk():
    from src.cache import put_fh_metrics, get_fh_metrics, get_fh_metrics_bulk
    db = make_tmp_db()
    init_db(db)
    assert get_fh_metrics(db, "AAPL", ttl_days=7) is None
    put_fh_metrics(db, "aapl", {"peTTM": 30.0})
    put_fh_metrics(db, "NONE", {})
    assert get_fh_metrics(db, "AAPL", ttl_days=7) == {"peTTM": 30.0}
    assert get_fh_metrics(db, "NONE", ttl_days=7) == {}          # cached "nothing", not None
    assert get_fh_metrics(db, "AAPL", ttl_days=0) is None        # expired
    bulk = get_fh_metrics_bulk(db, ttl_days=7)
    assert set(bulk) == {"AAPL", "NONE"}
    os.unlink(db)


def test_fh_profile_roundtrip_and_bulk():
    from src.cache import put_fh_profile, get_fh_profile, get_fh_profiles_bulk
    db = make_tmp_db()
    init_db(db)
    assert get_fh_profile(db, "AAPL") is None
    put_fh_profile(db, "AAPL", "Technology", 3.0e12, 1.5e10)
    put_fh_profile(db, "X", None, None, None)
    p = get_fh_profile(db, "AAPL")
    assert p["industry"] == "Technology" and p["market_cap"] == 3.0e12 and p["fetched_at"]
    assert get_fh_profile(db, "X")["industry"] == ""
    assert set(get_fh_profiles_bulk(db)) == {"AAPL", "X"}
    os.unlink(db)


def test_failed_ticker_default_ttl_is_seven_days_and_purge():
    from src.cache import put_failed_ticker, is_failed_ticker, purge_failed_tickers, count_failed_tickers
    import sqlite3
    db = make_tmp_db()
    init_db(db)
    put_failed_ticker(db, "AAPL", "no_data")
    assert is_failed_ticker(db, "AAPL")
    # Backdate the row 8 days: outside the default 7-day quarantine.
    conn = sqlite3.connect(db)
    conn.execute("UPDATE failed_tickers SET fetched_at=? WHERE ticker='AAPL'",
                 ((datetime.utcnow() - timedelta(days=8)).isoformat(),))
    conn.commit(); conn.close()
    assert not is_failed_ticker(db, "AAPL")
    assert is_failed_ticker(db, "AAPL", ttl_days=30)
    put_failed_ticker(db, "MSFT", "no_data")
    assert count_failed_tickers(db) == 2
    assert purge_failed_tickers(db, before=(datetime.utcnow() - timedelta(days=1)).isoformat()) == 1
    assert count_failed_tickers(db) == 1
    assert purge_failed_tickers(db) == 1
    assert count_failed_tickers(db) == 0
    os.unlink(db)
