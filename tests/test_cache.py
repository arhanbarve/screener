# tests/test_cache.py
import os
import tempfile
import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from src.cache import init_db, put_prices, get_prices, put_fundamentals, get_fundamentals, put_edgar, get_edgar, archive_universe_snapshot

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
