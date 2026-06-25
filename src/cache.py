import sqlite3
import json
import pandas as pd
from datetime import datetime, timedelta

def init_db(db_path: str):
    conn = sqlite3.connect(db_path, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS prices (
            ticker TEXT, date TEXT, open REAL, high REAL, low REAL,
            close REAL, volume INTEGER, fetched_at TEXT,
            PRIMARY KEY(ticker, date)
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS fundamentals (
            ticker TEXT PRIMARY KEY, payload TEXT, fetched_at TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS edgar (
            cik TEXT PRIMARY KEY, gp_assets REAL, revenue REAL,
            cogs REAL, assets REAL, fetched_at TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS market_cap (
            ticker TEXT PRIMARY KEY, value REAL, fetched_at TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS news_sentiment (
            ticker TEXT,
            as_of_date TEXT,
            entry_signal TEXT,
            catalyst TEXT,
            priced_in INTEGER,
            duration TEXT,
            thesis_consistency TEXT,
            conviction_delta INTEGER,
            reasoning TEXT,
            payload TEXT,
            fetched_at TEXT,
            PRIMARY KEY(ticker, as_of_date)
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS fundamentals_history (
            ticker TEXT,
            snapshot_date TEXT,
            payload TEXT,
            fetched_at TEXT,
            PRIMARY KEY(ticker, snapshot_date)
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS universe_snapshots (
            snapshot_date TEXT,
            ticker TEXT,
            cik TEXT,
            PRIMARY KEY(snapshot_date, ticker)
        )
    """)
    # --- Filing-edge screen ("Lazy Prices") ---
    c.execute("""
        CREATE TABLE IF NOT EXISTS submissions (
            cik TEXT PRIMARY KEY, payload TEXT, fetched_at TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS filings (
            accession TEXT PRIMARY KEY, cik TEXT, form TEXT,
            html TEXT, fetched_at TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS filing_similarity (
            accession TEXT PRIMARY KEY, cik TEXT, report_date TEXT,
            text_stability REAL, payload TEXT, fetched_at TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS filing_analysis (
            accession TEXT PRIMARY KEY, payload TEXT, fetched_at TEXT
        )
    """)
    # Point-in-time archive for filing-edge backtest validation.
    # Each run persists the ranked output so forward returns can be measured later.
    c.execute("""
        CREATE TABLE IF NOT EXISTS filing_edge_snapshots (
            snapshot_date TEXT,
            ticker TEXT,
            list_type TEXT,          -- 'long' or 'watch'
            rank INTEGER,
            composite REAL,
            text_stability REAL,
            conviction INTEGER,
            market_cap REAL,
            avg_dollar_vol_20d REAL,
            accession TEXT,
            fetched_at TEXT,
            PRIMARY KEY (snapshot_date, ticker, list_type)
        )
    """)
    conn.commit()
    conn.close()


def archive_fundamentals_snapshot(db_path: str, date_str: str):
    """Copy today's fundamentals into point-in-time history table.
    After 6 months of daily archiving, enables backtesting fundamental factors.
    """
    conn = sqlite3.connect(db_path, timeout=30)
    conn.execute("""
        INSERT OR IGNORE INTO fundamentals_history (ticker, snapshot_date, payload, fetched_at)
        SELECT ticker, ?, payload, fetched_at FROM fundamentals
    """, (date_str,))
    conn.commit()
    conn.close()


def archive_universe_snapshot(db_path: str, date_str: str, tickers_df):
    """Save today's liquidity-gate-surviving universe for survivorship-bias-aware backtesting."""
    conn = sqlite3.connect(db_path, timeout=30)
    rows = [(date_str, str(row.get("ticker", "")), str(row.get("cik", "")))
            for _, row in tickers_df.iterrows()]
    conn.executemany(
        "INSERT OR IGNORE INTO universe_snapshots VALUES (?,?,?)", rows
    )
    conn.commit()
    conn.close()


def _now_iso() -> str:
    return datetime.utcnow().isoformat()

def put_prices(db_path: str, ticker: str, df: pd.DataFrame):
    conn = sqlite3.connect(db_path, timeout=30)
    c = conn.cursor()
    now = _now_iso()
    rows = []
    for dt, row in df.iterrows():
        rows.append((
            ticker,
            str(dt.date()),
            float(row["open"]),
            float(row["high"]),
            float(row["low"]),
            float(row["close"]),
            int(row["volume"]),
            now,
        ))
    c.executemany(
        "INSERT OR REPLACE INTO prices VALUES (?,?,?,?,?,?,?,?)", rows
    )
    conn.commit()
    conn.close()

def get_prices(db_path: str, ticker: str, ttl_hours: int) -> pd.DataFrame | None:
    cutoff = (datetime.utcnow() - timedelta(hours=ttl_hours)).isoformat()
    conn = sqlite3.connect(db_path, timeout=30)
    c = conn.cursor()
    c.execute(
        "SELECT date, open, high, low, close, volume FROM prices "
        "WHERE ticker=? AND fetched_at > ? ORDER BY date",
        (ticker, cutoff),
    )
    rows = c.fetchall()
    conn.close()
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    return df

def put_fundamentals(db_path: str, ticker: str, payload: dict):
    conn = sqlite3.connect(db_path, timeout=30)
    c = conn.cursor()
    c.execute(
        "INSERT OR REPLACE INTO fundamentals VALUES (?,?,?)",
        (ticker, json.dumps(payload), _now_iso()),
    )
    conn.commit()
    conn.close()

def get_fundamentals(db_path: str, ticker: str, ttl_days: int) -> dict | None:
    cutoff = (datetime.utcnow() - timedelta(days=ttl_days)).isoformat()
    conn = sqlite3.connect(db_path, timeout=30)
    c = conn.cursor()
    c.execute(
        "SELECT payload FROM fundamentals WHERE ticker=? AND fetched_at > ?",
        (ticker, cutoff),
    )
    row = c.fetchone()
    conn.close()
    if row is None:
        return None
    return json.loads(row[0])

def put_market_cap(db_path: str, ticker: str, value: float):
    conn = sqlite3.connect(db_path, timeout=30)
    c = conn.cursor()
    c.execute(
        "INSERT OR REPLACE INTO market_cap VALUES (?,?,?)",
        (ticker, value, _now_iso()),
    )
    conn.commit()
    conn.close()

def get_market_cap(db_path: str, ticker: str, ttl_hours: int) -> float | None:
    cutoff = (datetime.utcnow() - timedelta(hours=ttl_hours)).isoformat()
    conn = sqlite3.connect(db_path, timeout=30)
    c = conn.cursor()
    c.execute(
        "SELECT value FROM market_cap WHERE ticker=? AND fetched_at > ?",
        (ticker, cutoff),
    )
    row = c.fetchone()
    conn.close()
    return row[0] if row else None

def put_news_sentiment(db_path: str, ticker: str, payload: dict):
    as_of = datetime.utcnow().strftime("%Y-%m-%d")
    conn = sqlite3.connect(db_path, timeout=30)
    c = conn.cursor()
    c.execute(
        "INSERT OR REPLACE INTO news_sentiment VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (
            ticker, as_of,
            payload.get("entry_signal", "wait"),
            payload.get("catalyst", "none"),
            int(bool(payload.get("priced_in", False))),
            payload.get("duration", "noise"),
            payload.get("thesis_consistency", "neutral"),
            int(payload.get("conviction_delta", 0)),
            payload.get("reasoning", ""),
            json.dumps(payload),
            _now_iso(),
        ),
    )
    conn.commit()
    conn.close()


def get_news_sentiment(db_path: str, ticker: str, ttl_hours: int) -> dict | None:
    cutoff = (datetime.utcnow() - timedelta(hours=ttl_hours)).isoformat()
    conn = sqlite3.connect(db_path, timeout=30)
    c = conn.cursor()
    c.execute(
        "SELECT payload FROM news_sentiment WHERE ticker=? AND fetched_at > ? ORDER BY fetched_at DESC LIMIT 1",
        (ticker, cutoff),
    )
    row = c.fetchone()
    conn.close()
    if row is None:
        return None
    try:
        return json.loads(row[0])
    except Exception:
        return None


# --- Filing-edge screen accessors ---

def archive_filing_edge_snapshot(
    db_path: str,
    date_str: str,
    longs_df,
    watch_df,
):
    """Persist today's filing-edge ranked output for future forward-return validation."""
    conn = sqlite3.connect(db_path, timeout=30)
    rows = []
    now = _now_iso()
    for rank, (_, row) in enumerate(longs_df.iterrows(), 1):
        rows.append((
            date_str, str(row.get("ticker", "")), "long", rank,
            float(row.get("composite", 0) or 0),
            float(row.get("text_stability", 0) or 0),
            int(row.get("conviction", 0) or 0),
            float(row.get("market_cap", 0) or 0),
            float(row.get("avg_dollar_vol_20d", 0) or 0),
            str(row.get("accession", "")),
            now,
        ))
    for rank, (_, row) in enumerate(watch_df.iterrows(), 1):
        rows.append((
            date_str, str(row.get("ticker", "")), "watch", rank,
            float(row.get("composite", 0) or 0),
            float(row.get("text_stability", 0) or 0),
            int(row.get("conviction", 0) or 0),
            float(row.get("market_cap", 0) or 0),
            float(row.get("avg_dollar_vol_20d", 0) or 0),
            str(row.get("accession", "")),
            now,
        ))
    conn.executemany(
        "INSERT OR REPLACE INTO filing_edge_snapshots VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows
    )
    conn.commit()
    conn.close()


def put_submissions(db_path: str, cik: str, payload: dict):
    conn = sqlite3.connect(db_path, timeout=30)
    conn.execute(
        "INSERT OR REPLACE INTO submissions VALUES (?,?,?)",
        (cik, json.dumps(payload), _now_iso()),
    )
    conn.commit()
    conn.close()


def get_submissions(db_path: str, cik: str, ttl_hours: int) -> dict | None:
    cutoff = (datetime.utcnow() - timedelta(hours=ttl_hours)).isoformat()
    conn = sqlite3.connect(db_path, timeout=30)
    c = conn.cursor()
    c.execute(
        "SELECT payload FROM submissions WHERE cik=? AND fetched_at > ?",
        (cik, cutoff),
    )
    row = c.fetchone()
    conn.close()
    return json.loads(row[0]) if row else None


def put_filing_doc(db_path: str, accession: str, cik: str, form: str, html: str):
    """Filings are immutable once filed — stored permanently (no TTL)."""
    conn = sqlite3.connect(db_path, timeout=30)
    conn.execute(
        "INSERT OR REPLACE INTO filings VALUES (?,?,?,?,?)",
        (accession, cik, form, html, _now_iso()),
    )
    conn.commit()
    conn.close()


def get_filing_doc(db_path: str, accession: str) -> str | None:
    conn = sqlite3.connect(db_path, timeout=30)
    c = conn.cursor()
    c.execute("SELECT html FROM filings WHERE accession=?", (accession,))
    row = c.fetchone()
    conn.close()
    return row[0] if row else None


def put_filing_similarity(db_path: str, cik: str, result: dict):
    """Memoized by the current filing's accession (an immutable pair)."""
    conn = sqlite3.connect(db_path, timeout=30)
    conn.execute(
        "INSERT OR REPLACE INTO filing_similarity VALUES (?,?,?,?,?,?)",
        (
            result["accession"], cik, result.get("report_date", ""),
            float(result.get("text_stability", 0.0)),
            json.dumps(result), _now_iso(),
        ),
    )
    conn.commit()
    conn.close()


def get_filing_similarity(db_path: str, accession: str) -> dict | None:
    conn = sqlite3.connect(db_path, timeout=30)
    c = conn.cursor()
    c.execute("SELECT payload FROM filing_similarity WHERE accession=?", (accession,))
    row = c.fetchone()
    conn.close()
    if row is None:
        return None
    try:
        return json.loads(row[0])
    except Exception:
        return None


def put_filing_analysis(db_path: str, accession: str, payload: dict):
    conn = sqlite3.connect(db_path, timeout=30)
    conn.execute(
        "INSERT OR REPLACE INTO filing_analysis VALUES (?,?,?)",
        (accession, json.dumps(payload), _now_iso()),
    )
    conn.commit()
    conn.close()


def get_filing_analysis(db_path: str, accession: str) -> dict | None:
    conn = sqlite3.connect(db_path, timeout=30)
    c = conn.cursor()
    c.execute("SELECT payload FROM filing_analysis WHERE accession=?", (accession,))
    row = c.fetchone()
    conn.close()
    if row is None:
        return None
    try:
        return json.loads(row[0])
    except Exception:
        return None


def put_edgar(db_path: str, cik: str, gp_assets: float, revenue: float, cogs: float, assets: float):
    conn = sqlite3.connect(db_path, timeout=30)
    c = conn.cursor()
    c.execute(
        "INSERT OR REPLACE INTO edgar VALUES (?,?,?,?,?,?)",
        (cik, gp_assets, revenue, cogs, assets, _now_iso()),
    )
    conn.commit()
    conn.close()

def get_edgar(db_path: str, cik: str, ttl_days: int) -> dict | None:
    cutoff = (datetime.utcnow() - timedelta(days=ttl_days)).isoformat()
    conn = sqlite3.connect(db_path, timeout=30)
    c = conn.cursor()
    c.execute(
        "SELECT gp_assets, revenue, cogs, assets FROM edgar WHERE cik=? AND fetched_at > ?",
        (cik, cutoff),
    )
    row = c.fetchone()
    conn.close()
    if row is None:
        return None
    return {"gp_assets": row[0], "revenue": row[1], "cogs": row[2], "assets": row[3]}
