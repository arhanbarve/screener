import sqlite3
import math
import json
import pandas as pd
from datetime import datetime, timedelta


def _str_or_empty(v) -> str:
    try:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return ""
    except (TypeError, ValueError):
        pass
    return str(v)


_conn_cache: dict[str, sqlite3.Connection] = {}

def _get_conn(db_path: str) -> sqlite3.Connection:
    if db_path not in _conn_cache:
        conn = sqlite3.connect(db_path, timeout=60, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA wal_autocheckpoint=500")
        _conn_cache[db_path] = conn
    return _conn_cache[db_path]

def init_db(db_path: str):
    conn = _get_conn(db_path)
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
        CREATE TABLE IF NOT EXISTS failed_tickers (
            ticker TEXT PRIMARY KEY, reason TEXT, fetched_at TEXT
        )
    """)
    # Multi-year OHLCV history for event-backtest forward returns. Separate
    # from `prices` (which holds only a rolling ~420-day window keyed to the
    # live screener's TTL) so backtest fetches never collide with it.
    c.execute("""
        CREATE TABLE IF NOT EXISTS event_backtest_prices (
            ticker TEXT PRIMARY KEY, payload TEXT, fetched_at TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS edgar_facts (
            cik TEXT PRIMARY KEY, payload TEXT, fetched_at TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS fh_metrics (
            ticker TEXT PRIMARY KEY, payload TEXT, fetched_at TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS fh_profile (
            ticker TEXT PRIMARY KEY, industry TEXT, market_cap REAL,
            shares_out REAL, fetched_at TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS ticker_profile_cache (
            ticker TEXT PRIMARY KEY,
            sector TEXT,
            industry TEXT,
            fetched_at TEXT
        )
    """)
    conn.commit()


def archive_fundamentals_snapshot(db_path: str, date_str: str):
    """Copy today's fundamentals into point-in-time history table.
    After 6 months of daily archiving, enables backtesting fundamental factors.
    """
    conn = _get_conn(db_path)
    conn.execute("""
        INSERT OR IGNORE INTO fundamentals_history (ticker, snapshot_date, payload, fetched_at)
        SELECT ticker, ?, payload, fetched_at FROM fundamentals
    """, (date_str,))
    conn.commit()



def archive_universe_snapshot(db_path: str, date_str: str, tickers_df):
    """Save today's liquidity-gate-surviving universe for survivorship-bias-aware backtesting."""
    conn = _get_conn(db_path)
    rows = [(date_str, _str_or_empty(row.get("ticker", "")), _str_or_empty(row.get("cik", "")))
            for _, row in tickers_df.iterrows()]
    conn.executemany(
        "INSERT OR IGNORE INTO universe_snapshots VALUES (?,?,?)", rows
    )
    conn.commit()



def _now_iso() -> str:
    return datetime.utcnow().isoformat()

def put_prices(db_path: str, ticker: str, df: pd.DataFrame):
    conn = _get_conn(db_path)
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


def get_prices(db_path: str, ticker: str, ttl_hours: int) -> pd.DataFrame | None:
    cutoff = (datetime.utcnow() - timedelta(hours=ttl_hours)).isoformat()
    conn = _get_conn(db_path)
    c = conn.cursor()
    c.execute(
        "SELECT date, open, high, low, close, volume FROM prices "
        "WHERE ticker=? AND fetched_at > ? ORDER BY date",
        (ticker, cutoff),
    )
    rows = c.fetchall()

    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    return df

def put_backtest_prices(db_path: str, ticker: str, df: pd.DataFrame):
    """Full-history OHLCV for one ticker, stored as a JSON payload (not row-per-day
    like `prices`) since the backtest wants years of history, not a rolling window."""
    conn = _get_conn(db_path)
    payload = df.reset_index().rename(columns={df.index.name or "index": "date"})
    payload["date"] = payload["date"].astype(str)
    conn.execute(
        "INSERT OR REPLACE INTO event_backtest_prices VALUES (?,?,?)",
        (ticker, payload.to_json(orient="records"), _now_iso()),
    )
    conn.commit()


def get_backtest_prices(db_path: str, ticker: str, ttl_days: int = 30) -> pd.DataFrame | None:
    cutoff = (datetime.utcnow() - timedelta(days=ttl_days)).isoformat()
    conn = _get_conn(db_path)
    c = conn.cursor()
    c.execute(
        "SELECT payload FROM event_backtest_prices WHERE ticker=? AND fetched_at > ?",
        (ticker, cutoff),
    )
    row = c.fetchone()
    if row is None:
        return None
    df = pd.DataFrame(json.loads(row[0]))
    if df.empty:
        return None
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()


def put_fundamentals(db_path: str, ticker: str, payload: dict):
    conn = _get_conn(db_path)
    c = conn.cursor()
    c.execute(
        "INSERT OR REPLACE INTO fundamentals VALUES (?,?,?)",
        (ticker, json.dumps(payload), _now_iso()),
    )
    conn.commit()


def get_fundamentals(db_path: str, ticker: str, ttl_days: int) -> dict | None:
    cutoff = (datetime.utcnow() - timedelta(days=ttl_days)).isoformat()
    conn = _get_conn(db_path)
    c = conn.cursor()
    c.execute(
        "SELECT payload FROM fundamentals WHERE ticker=? AND fetched_at > ?",
        (ticker, cutoff),
    )
    row = c.fetchone()

    if row is None:
        return None
    return json.loads(row[0])

def put_market_cap(db_path: str, ticker: str, value: float):
    conn = _get_conn(db_path)
    c = conn.cursor()
    c.execute(
        "INSERT OR REPLACE INTO market_cap VALUES (?,?,?)",
        (ticker, value, _now_iso()),
    )
    conn.commit()


def get_market_cap(db_path: str, ticker: str, ttl_hours: int) -> float | None:
    cutoff = (datetime.utcnow() - timedelta(hours=ttl_hours)).isoformat()
    conn = _get_conn(db_path)
    c = conn.cursor()
    c.execute(
        "SELECT value FROM market_cap WHERE ticker=? AND fetched_at > ?",
        (ticker, cutoff),
    )
    row = c.fetchone()
    return row[0] if row else None


def put_ticker_profile(db_path: str, ticker: str, sector, industry):
    conn = _get_conn(db_path)
    c = conn.cursor()
    c.execute(
        "INSERT OR REPLACE INTO ticker_profile_cache VALUES (?,?,?,?)",
        (ticker, _str_or_empty(sector), _str_or_empty(industry), _now_iso()),
    )
    conn.commit()


def get_ticker_profile(db_path: str, ticker: str) -> dict | None:
    """No TTL: sector/industry are effectively static. None = never fetched;
    empty strings = fetched but yfinance had no data (don't refetch)."""
    conn = _get_conn(db_path)
    c = conn.cursor()
    c.execute(
        "SELECT sector, industry FROM ticker_profile_cache WHERE ticker=?",
        (ticker,),
    )
    row = c.fetchone()
    if row is None:
        return None
    return {"sector": row[0], "industry": row[1]}


def get_market_cap_stale(db_path: str, ticker: str) -> float | None:
    """Return any cached market cap regardless of age (stale fallback)."""
    conn = _get_conn(db_path)
    c = conn.cursor()
    c.execute("SELECT value FROM market_cap WHERE ticker=? ORDER BY fetched_at DESC LIMIT 1", (ticker,))
    row = c.fetchone()
    return row[0] if row else None

def put_news_sentiment(db_path: str, ticker: str, payload: dict):
    as_of = datetime.utcnow().strftime("%Y-%m-%d")
    conn = _get_conn(db_path)
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



def get_news_sentiment(db_path: str, ticker: str, ttl_hours: int) -> dict | None:
    cutoff = (datetime.utcnow() - timedelta(hours=ttl_hours)).isoformat()
    conn = _get_conn(db_path)
    c = conn.cursor()
    c.execute(
        "SELECT payload FROM news_sentiment WHERE ticker=? AND fetched_at > ? ORDER BY fetched_at DESC LIMIT 1",
        (ticker, cutoff),
    )
    row = c.fetchone()

    if row is None:
        return None
    try:
        return json.loads(row[0])
    except Exception:
        return None


# --- SEC filing fetch cache (shared by backtest modules) ---

def put_submissions(db_path: str, cik: str, payload: dict):
    conn = _get_conn(db_path)
    conn.execute(
        "INSERT OR REPLACE INTO submissions VALUES (?,?,?)",
        (cik, json.dumps(payload), _now_iso()),
    )
    conn.commit()



def get_submissions(db_path: str, cik: str, ttl_hours: int) -> dict | None:
    cutoff = (datetime.utcnow() - timedelta(hours=ttl_hours)).isoformat()
    conn = _get_conn(db_path)
    c = conn.cursor()
    c.execute(
        "SELECT payload FROM submissions WHERE cik=? AND fetched_at > ?",
        (cik, cutoff),
    )
    row = c.fetchone()

    return json.loads(row[0]) if row else None


def put_filing_doc(db_path: str, accession: str, cik: str, form: str, html: str):
    """Filings are immutable once filed — stored permanently (no TTL)."""
    conn = _get_conn(db_path)
    conn.execute(
        "INSERT OR REPLACE INTO filings VALUES (?,?,?,?,?)",
        (accession, cik, form, html, _now_iso()),
    )
    conn.commit()



def get_filing_doc(db_path: str, accession: str) -> str | None:
    conn = _get_conn(db_path)
    c = conn.cursor()
    c.execute("SELECT html FROM filings WHERE accession=?", (accession,))
    row = c.fetchone()

    return row[0] if row else None


def put_failed_ticker(db_path: str, ticker: str, reason: str = "no_data"):
    conn = _get_conn(db_path)
    conn.execute(
        "INSERT OR REPLACE INTO failed_tickers VALUES (?,?,?)",
        (ticker, reason, _now_iso()),
    )
    conn.commit()


def is_failed_ticker(db_path: str, ticker: str, ttl_days: int = 7) -> bool:
    """A ticker quarantined within the last `ttl_days` is skipped by the price
    fetch. 7 days, not 30: on 2026-08-24 a batch-level yfinance exception
    quarantined 4,344 real names for a month (see purge_failed_tickers)."""
    cutoff = (datetime.utcnow() - timedelta(days=ttl_days)).isoformat()
    conn = _get_conn(db_path)
    c = conn.cursor()
    c.execute(
        "SELECT 1 FROM failed_tickers WHERE ticker=? AND fetched_at > ?",
        (ticker, cutoff),
    )
    return c.fetchone() is not None


def purge_failed_tickers(db_path: str, before: str | None = None) -> int:
    """Delete quarantine rows. `before` (ISO date/time) limits the purge to rows
    stamped earlier than that; None purges everything. Returns rows deleted.

    Cheap to be generous: a genuinely dead ticker costs one batch slot to
    re-discover, while a wrongly quarantined one costs its whole screen."""
    conn = _get_conn(db_path)
    if before is None:
        cur = conn.execute("DELETE FROM failed_tickers")
    else:
        cur = conn.execute("DELETE FROM failed_tickers WHERE fetched_at < ?", (before,))
    conn.commit()
    return cur.rowcount


def count_failed_tickers(db_path: str) -> int:
    conn = _get_conn(db_path)
    return int(conn.execute("SELECT COUNT(*) FROM failed_tickers").fetchone()[0])


# --- Finnhub fundamentals / profile cache (src/finnhub_data.py) ---

def put_fh_metrics(db_path: str, ticker: str, metrics: dict):
    """Store the `metric` dict from company_basic_financials. An empty dict is
    stored too, so a symbol Finnhub does not know is not re-requested daily."""
    conn = _get_conn(db_path)
    conn.execute(
        "INSERT OR REPLACE INTO fh_metrics VALUES (?,?,?)",
        (ticker.upper(), json.dumps(metrics or {}), _now_iso()),
    )
    conn.commit()


def get_fh_metrics(db_path: str, ticker: str, ttl_days: int) -> dict | None:
    """Cached metrics newer than ttl, else None. Returns {} for a cached
    'Finnhub has nothing' answer — callers treat {} as no data, None as unfetched."""
    cutoff = (datetime.utcnow() - timedelta(days=ttl_days)).isoformat()
    row = _get_conn(db_path).execute(
        "SELECT payload FROM fh_metrics WHERE ticker=? AND fetched_at > ?",
        (ticker.upper(), cutoff),
    ).fetchone()
    return json.loads(row[0]) if row else None


def get_fh_metrics_bulk(db_path: str, ttl_days: int) -> dict[str, dict]:
    """Every cached metrics payload newer than ttl, keyed by ticker. One query
    instead of thousands when a run starts."""
    cutoff = (datetime.utcnow() - timedelta(days=ttl_days)).isoformat()
    rows = _get_conn(db_path).execute(
        "SELECT ticker, payload FROM fh_metrics WHERE fetched_at > ?", (cutoff,)
    ).fetchall()
    out = {}
    for t, payload in rows:
        try:
            out[t] = json.loads(payload)
        except json.JSONDecodeError:
            continue
    return out


def put_fh_profile(db_path: str, ticker: str, industry: str | None,
                   market_cap: float | None, shares_out: float | None):
    conn = _get_conn(db_path)
    conn.execute(
        "INSERT OR REPLACE INTO fh_profile VALUES (?,?,?,?,?)",
        (ticker.upper(), _str_or_empty(industry), market_cap, shares_out, _now_iso()),
    )
    conn.commit()


def get_fh_profile(db_path: str, ticker: str) -> dict | None:
    """Profile row regardless of age (industry is effectively static). None if
    never fetched. Includes fetched_at so callers can decide cap staleness."""
    row = _get_conn(db_path).execute(
        "SELECT industry, market_cap, shares_out, fetched_at FROM fh_profile WHERE ticker=?",
        (ticker.upper(),),
    ).fetchone()
    if row is None:
        return None
    return {"industry": row[0], "market_cap": row[1], "shares_out": row[2], "fetched_at": row[3]}


def get_fh_profiles_bulk(db_path: str) -> dict[str, dict]:
    rows = _get_conn(db_path).execute(
        "SELECT ticker, industry, market_cap, shares_out, fetched_at FROM fh_profile"
    ).fetchall()
    return {t: {"industry": i, "market_cap": mc, "shares_out": so, "fetched_at": f}
            for t, i, mc, so, f in rows}


def put_edgar_facts(db_path: str, cik: str, facts: dict):
    """Extended EDGAR facts (src.fundamentals.parse_edgar_facts). NaN is stored
    as null so the JSON round-trips; get_edgar_facts restores NaN."""
    clean = {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in facts.items()}
    conn = _get_conn(db_path)
    conn.execute("INSERT OR REPLACE INTO edgar_facts VALUES (?,?,?)",
                 (cik, json.dumps(clean), _now_iso()))
    conn.commit()


def get_edgar_facts(db_path: str, cik: str, ttl_days: int) -> dict | None:
    cutoff = (datetime.utcnow() - timedelta(days=ttl_days)).isoformat()
    row = _get_conn(db_path).execute(
        "SELECT payload FROM edgar_facts WHERE cik=? AND fetched_at > ?", (cik, cutoff)
    ).fetchone()
    if row is None:
        return None
    facts = json.loads(row[0])
    return {k: (float("nan") if v is None and k != "n_resolved" else v) for k, v in facts.items()}


def put_edgar(db_path: str, cik: str, gp_assets: float, revenue: float, cogs: float, assets: float):
    conn = _get_conn(db_path)
    c = conn.cursor()
    c.execute(
        "INSERT OR REPLACE INTO edgar VALUES (?,?,?,?,?,?)",
        (cik, gp_assets, revenue, cogs, assets, _now_iso()),
    )
    conn.commit()


def get_edgar(db_path: str, cik: str, ttl_days: int) -> dict | None:
    cutoff = (datetime.utcnow() - timedelta(days=ttl_days)).isoformat()
    conn = _get_conn(db_path)
    c = conn.cursor()
    c.execute(
        "SELECT gp_assets, revenue, cogs, assets FROM edgar WHERE cik=? AND fetched_at > ?",
        (cik, cutoff),
    )
    row = c.fetchone()

    if row is None:
        return None
    return {"gp_assets": row[0], "revenue": row[1], "cogs": row[2], "assets": row[3]}
