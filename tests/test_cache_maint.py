"""Cache maintenance CLI: quarantine purge and Finnhub warm-up ordering."""
import json
import os
import sqlite3
import tempfile

import pandas as pd

from src.cache import init_db, put_failed_ticker, put_prices
from src import cache_maint


def _db():
    f = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    f.close()
    init_db(f.name)
    return f.name


def _prices(close, volume, n=25):
    idx = pd.date_range(end="2026-09-03", periods=n, freq="B")
    return pd.DataFrame({"open": close, "high": close, "low": close, "close": close, "volume": volume}, index=idx)


def test_purge_failed_reports_counts(capsys):
    db = _db()
    put_failed_ticker(db, "AAPL"); put_failed_ticker(db, "MSFT")
    rc = cache_maint.main(["--db", db, "purge-failed"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out == {"before": 2, "deleted": 2, "after": 0}
    os.unlink(db)


def test_adv_ranked_tickers_liquid_first_then_rest():
    db = _db()
    put_prices(db, "BIG", _prices(100.0, 1_000_000))     # ADV $100M
    put_prices(db, "MID", _prices(10.0, 1_000_000))      # ADV $10M
    put_prices(db, "TINY", _prices(1.0, 10_000))         # ADV $10k
    order = cache_maint._adv_ranked_tickers(db, min_adv=5_000_000)
    assert order == ["BIG", "MID", "TINY"]
    os.unlink(db)


def test_stats_lists_tables(capsys):
    db = _db()
    assert cache_maint.main(["--db", db, "stats"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["failed_tickers"] == 0 and out["fh_metrics"] == 0 and "distinct_price_tickers" in out
    os.unlink(db)
