import sqlite3
import json
import pandas as pd
from datetime import datetime, timedelta

def init_db(db_path: str):
    conn = sqlite3.connect(db_path)
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
    conn.commit()
    conn.close()

def _now_iso() -> str:
    return datetime.utcnow().isoformat()

def put_prices(db_path: str, ticker: str, df: pd.DataFrame):
    conn = sqlite3.connect(db_path)
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
    conn = sqlite3.connect(db_path)
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
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute(
        "INSERT OR REPLACE INTO fundamentals VALUES (?,?,?)",
        (ticker, json.dumps(payload), _now_iso()),
    )
    conn.commit()
    conn.close()

def get_fundamentals(db_path: str, ticker: str, ttl_days: int) -> dict | None:
    cutoff = (datetime.utcnow() - timedelta(days=ttl_days)).isoformat()
    conn = sqlite3.connect(db_path)
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
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute(
        "INSERT OR REPLACE INTO market_cap VALUES (?,?,?)",
        (ticker, value, _now_iso()),
    )
    conn.commit()
    conn.close()

def get_market_cap(db_path: str, ticker: str, ttl_hours: int) -> float | None:
    cutoff = (datetime.utcnow() - timedelta(hours=ttl_hours)).isoformat()
    conn = sqlite3.connect(db_path)
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
    conn = sqlite3.connect(db_path)
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
    conn = sqlite3.connect(db_path)
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


def put_edgar(db_path: str, cik: str, gp_assets: float, revenue: float, cogs: float, assets: float):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute(
        "INSERT OR REPLACE INTO edgar VALUES (?,?,?,?,?,?)",
        (cik, gp_assets, revenue, cogs, assets, _now_iso()),
    )
    conn.commit()
    conn.close()

def get_edgar(db_path: str, cik: str, ttl_days: int) -> dict | None:
    cutoff = (datetime.utcnow() - timedelta(days=ttl_days)).isoformat()
    conn = sqlite3.connect(db_path)
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
