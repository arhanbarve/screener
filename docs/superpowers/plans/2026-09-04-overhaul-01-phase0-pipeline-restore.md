# Phase 0 — Pipeline Restore (Plan 01)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The screener ranks the real liquid US universe every day or fails loudly, before tonight's 16:30 ET run.

**Architecture:** Universe = SEC list ∩ Alpaca tradable equities (dot/dash symbol mapping). Prices via yfinance with split-and-retry, never quarantining on a batch exception. Dollar-volume gate first, then Finnhub basic-financials/profile for market cap + industry, then the cap gate. Health floors after each stage; `screen_latest.json` written on every run (failed runs write `health.ok: false` and no rows); stage counts flow into `run_status.json.stats` and a watchdog check.

**Tech Stack:** pandas, sqlite3, yfinance 1.4.1, finnhub-python, requests.

**Root cause being fixed (from the vetting report):** on 2026-08-24 one yfinance batch exception made `_fetch_batch_yfinance` return `{}`, and the caller marked all 4,344 tickers in the affected batches as `failed_tickers` for 30 days (AAPL, MSFT, NVDA, the book's own VLO/TXG). Runs 08-24…08-28 then crashed with `KeyError: 'market_cap'` (no factor rows) and from 08-31 "succeeded" with 48–53 liquidity survivors, ranking a top 20 from 38 names. Separately, `market_cap` has not refreshed since 2026-06-27 (`fast_info` with a custom `requests.Session` fails in yfinance 1.x and Yahoo rate-limits), `sector` is empty for every row (same `get_info` path), and the quality gate's cutoff is −999.

---

### Task 0.1: Cache — 7-day quarantine, purge, Finnhub and EDGAR-facts tables

**Files:**
- Modify: `src/cache.py`
- Modify: `tests/test_cache.py`

Adds the storage every later task needs and shortens the quarantine. `edgar_facts` is created here too (used by Phase 1) so the schema lands once.

- [ ] **Step 1:** Append the new tests to `tests/test_cache.py`:

**Apply this unified diff to `tests/test_cache.py`** (from the reference patch; `patch -p1 < docs/superpowers/plans/2026-09-04-overhaul-reference.patch` applies all of them at once):

```diff
--- a/tests/test_cache.py	2026-07-03 16:41:35
+++ b/tests/test_cache.py	2026-09-04 12:27:22
@@ -135,3 +135,55 @@
     put_ticker_profile(db, "BBB", None, None)  # failed lookups cache as empty
     assert get_ticker_profile(db, "BBB") == {"sector": "", "industry": ""}
     os.unlink(db)
+
+
+def test_fh_metrics_roundtrip_and_bulk():
+    from src.cache import put_fh_metrics, get_fh_metrics, get_fh_metrics_bulk
+    db = make_tmp_db()
+    init_db(db)
+    assert get_fh_metrics(db, "AAPL", ttl_days=7) is None
+    put_fh_metrics(db, "aapl", {"peTTM": 30.0})
+    put_fh_metrics(db, "NONE", {})
+    assert get_fh_metrics(db, "AAPL", ttl_days=7) == {"peTTM": 30.0}
+    assert get_fh_metrics(db, "NONE", ttl_days=7) == {}          # cached "nothing", not None
+    assert get_fh_metrics(db, "AAPL", ttl_days=0) is None        # expired
+    bulk = get_fh_metrics_bulk(db, ttl_days=7)
+    assert set(bulk) == {"AAPL", "NONE"}
+    os.unlink(db)
+
+
+def test_fh_profile_roundtrip_and_bulk():
+    from src.cache import put_fh_profile, get_fh_profile, get_fh_profiles_bulk
+    db = make_tmp_db()
+    init_db(db)
+    assert get_fh_profile(db, "AAPL") is None
+    put_fh_profile(db, "AAPL", "Technology", 3.0e12, 1.5e10)
+    put_fh_profile(db, "X", None, None, None)
+    p = get_fh_profile(db, "AAPL")
+    assert p["industry"] == "Technology" and p["market_cap"] == 3.0e12 and p["fetched_at"]
+    assert get_fh_profile(db, "X")["industry"] == ""
+    assert set(get_fh_profiles_bulk(db)) == {"AAPL", "X"}
+    os.unlink(db)
+
+
+def test_failed_ticker_default_ttl_is_seven_days_and_purge():
+    from src.cache import put_failed_ticker, is_failed_ticker, purge_failed_tickers, count_failed_tickers
+    import sqlite3
+    db = make_tmp_db()
+    init_db(db)
+    put_failed_ticker(db, "AAPL", "no_data")
+    assert is_failed_ticker(db, "AAPL")
+    # Backdate the row 8 days: outside the default 7-day quarantine.
+    conn = sqlite3.connect(db)
+    conn.execute("UPDATE failed_tickers SET fetched_at=? WHERE ticker='AAPL'",
+                 ((datetime.utcnow() - timedelta(days=8)).isoformat(),))
+    conn.commit(); conn.close()
+    assert not is_failed_ticker(db, "AAPL")
+    assert is_failed_ticker(db, "AAPL", ttl_days=30)
+    put_failed_ticker(db, "MSFT", "no_data")
+    assert count_failed_tickers(db) == 2
+    assert purge_failed_tickers(db, before=(datetime.utcnow() - timedelta(days=1)).isoformat()) == 1
+    assert count_failed_tickers(db) == 1
+    assert purge_failed_tickers(db) == 1
+    assert count_failed_tickers(db) == 0
+    os.unlink(db)
```

- [ ] **Step 2:** Run:

```bash
/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m pytest tests/test_cache.py -q
```

Expected: 3 new tests FAIL with `ImportError`/`AttributeError` (put_fh_metrics, purge_failed_tickers … not defined).

- [ ] **Step 3:** Apply the cache changes:

**Apply this unified diff to `src/cache.py`** (from the reference patch; `patch -p1 < docs/superpowers/plans/2026-09-04-overhaul-reference.patch` applies all of them at once):

```diff
--- a/src/cache.py	2026-07-08 19:11:50
+++ b/src/cache.py	2026-09-04 12:31:25
@@ -106,10 +106,26 @@
     # live screener's TTL) so backtest fetches never collide with it.
     c.execute("""
         CREATE TABLE IF NOT EXISTS event_backtest_prices (
+            ticker TEXT PRIMARY KEY, payload TEXT, fetched_at TEXT
+        )
+    """)
+    c.execute("""
+        CREATE TABLE IF NOT EXISTS edgar_facts (
+            cik TEXT PRIMARY KEY, payload TEXT, fetched_at TEXT
+        )
+    """)
+    c.execute("""
+        CREATE TABLE IF NOT EXISTS fh_metrics (
             ticker TEXT PRIMARY KEY, payload TEXT, fetched_at TEXT
         )
     """)
     c.execute("""
+        CREATE TABLE IF NOT EXISTS fh_profile (
+            ticker TEXT PRIMARY KEY, industry TEXT, market_cap REAL,
+            shares_out REAL, fetched_at TEXT
+        )
+    """)
+    c.execute("""
         CREATE TABLE IF NOT EXISTS ticker_profile_cache (
             ticker TEXT PRIMARY KEY,
             sector TEXT,
@@ -393,7 +409,10 @@
     conn.commit()
 
 
-def is_failed_ticker(db_path: str, ticker: str, ttl_days: int = 30) -> bool:
+def is_failed_ticker(db_path: str, ticker: str, ttl_days: int = 7) -> bool:
+    """A ticker quarantined within the last `ttl_days` is skipped by the price
+    fetch. 7 days, not 30: on 2026-08-24 a batch-level yfinance exception
+    quarantined 4,344 real names for a month (see purge_failed_tickers)."""
     cutoff = (datetime.utcnow() - timedelta(days=ttl_days)).isoformat()
     conn = _get_conn(db_path)
     c = conn.cursor()
@@ -402,8 +421,119 @@
         (ticker, cutoff),
     )
     return c.fetchone() is not None
+
+
+def purge_failed_tickers(db_path: str, before: str | None = None) -> int:
+    """Delete quarantine rows. `before` (ISO date/time) limits the purge to rows
+    stamped earlier than that; None purges everything. Returns rows deleted.
+
+    Cheap to be generous: a genuinely dead ticker costs one batch slot to
+    re-discover, while a wrongly quarantined one costs its whole screen."""
+    conn = _get_conn(db_path)
+    if before is None:
+        cur = conn.execute("DELETE FROM failed_tickers")
+    else:
+        cur = conn.execute("DELETE FROM failed_tickers WHERE fetched_at < ?", (before,))
+    conn.commit()
+    return cur.rowcount
+
+
+def count_failed_tickers(db_path: str) -> int:
+    conn = _get_conn(db_path)
+    return int(conn.execute("SELECT COUNT(*) FROM failed_tickers").fetchone()[0])
+
+
+# --- Finnhub fundamentals / profile cache (src/finnhub_data.py) ---
+
+def put_fh_metrics(db_path: str, ticker: str, metrics: dict):
+    """Store the `metric` dict from company_basic_financials. An empty dict is
+    stored too, so a symbol Finnhub does not know is not re-requested daily."""
+    conn = _get_conn(db_path)
+    conn.execute(
+        "INSERT OR REPLACE INTO fh_metrics VALUES (?,?,?)",
+        (ticker.upper(), json.dumps(metrics or {}), _now_iso()),
+    )
+    conn.commit()
+
+
+def get_fh_metrics(db_path: str, ticker: str, ttl_days: int) -> dict | None:
+    """Cached metrics newer than ttl, else None. Returns {} for a cached
+    'Finnhub has nothing' answer — callers treat {} as no data, None as unfetched."""
+    cutoff = (datetime.utcnow() - timedelta(days=ttl_days)).isoformat()
+    row = _get_conn(db_path).execute(
+        "SELECT payload FROM fh_metrics WHERE ticker=? AND fetched_at > ?",
+        (ticker.upper(), cutoff),
+    ).fetchone()
+    return json.loads(row[0]) if row else None
+
+
+def get_fh_metrics_bulk(db_path: str, ttl_days: int) -> dict[str, dict]:
+    """Every cached metrics payload newer than ttl, keyed by ticker. One query
+    instead of thousands when a run starts."""
+    cutoff = (datetime.utcnow() - timedelta(days=ttl_days)).isoformat()
+    rows = _get_conn(db_path).execute(
+        "SELECT ticker, payload FROM fh_metrics WHERE fetched_at > ?", (cutoff,)
+    ).fetchall()
+    out = {}
+    for t, payload in rows:
+        try:
+            out[t] = json.loads(payload)
+        except json.JSONDecodeError:
+            continue
+    return out
 
 
+def put_fh_profile(db_path: str, ticker: str, industry: str | None,
+                   market_cap: float | None, shares_out: float | None):
+    conn = _get_conn(db_path)
+    conn.execute(
+        "INSERT OR REPLACE INTO fh_profile VALUES (?,?,?,?,?)",
+        (ticker.upper(), _str_or_empty(industry), market_cap, shares_out, _now_iso()),
+    )
+    conn.commit()
+
+
+def get_fh_profile(db_path: str, ticker: str) -> dict | None:
+    """Profile row regardless of age (industry is effectively static). None if
+    never fetched. Includes fetched_at so callers can decide cap staleness."""
+    row = _get_conn(db_path).execute(
+        "SELECT industry, market_cap, shares_out, fetched_at FROM fh_profile WHERE ticker=?",
+        (ticker.upper(),),
+    ).fetchone()
+    if row is None:
+        return None
+    return {"industry": row[0], "market_cap": row[1], "shares_out": row[2], "fetched_at": row[3]}
+
+
+def get_fh_profiles_bulk(db_path: str) -> dict[str, dict]:
+    rows = _get_conn(db_path).execute(
+        "SELECT ticker, industry, market_cap, shares_out, fetched_at FROM fh_profile"
+    ).fetchall()
+    return {t: {"industry": i, "market_cap": mc, "shares_out": so, "fetched_at": f}
+            for t, i, mc, so, f in rows}
+
+
+def put_edgar_facts(db_path: str, cik: str, facts: dict):
+    """Extended EDGAR facts (src.fundamentals.parse_edgar_facts). NaN is stored
+    as null so the JSON round-trips; get_edgar_facts restores NaN."""
+    clean = {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in facts.items()}
+    conn = _get_conn(db_path)
+    conn.execute("INSERT OR REPLACE INTO edgar_facts VALUES (?,?,?)",
+                 (cik, json.dumps(clean), _now_iso()))
+    conn.commit()
+
+
+def get_edgar_facts(db_path: str, cik: str, ttl_days: int) -> dict | None:
+    cutoff = (datetime.utcnow() - timedelta(days=ttl_days)).isoformat()
+    row = _get_conn(db_path).execute(
+        "SELECT payload FROM edgar_facts WHERE cik=? AND fetched_at > ?", (cik, cutoff)
+    ).fetchone()
+    if row is None:
+        return None
+    facts = json.loads(row[0])
+    return {k: (float("nan") if v is None and k != "n_resolved" else v) for k, v in facts.items()}
+
+
 def put_edgar(db_path: str, cik: str, gp_assets: float, revenue: float, cogs: float, assets: float):
     conn = _get_conn(db_path)
     c = conn.cursor()
```

- [ ] **Step 4:** Run:

```bash
/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m pytest tests/test_cache.py -q
```

Expected: all pass (13 tests).

- [ ] **Step 5:** Commit as `feat(cache): 7-day quarantine with purge; Finnhub metrics/profile and EDGAR facts tables`. Per the user's global CLAUDE.md §6: run `git status` first and **ask the user before committing**. Never add a Co-Authored-By line.

**Edge cases this task must handle (each has a test):**
- `is_failed_ticker` default TTL is 7 days (was 30); a row backdated 8 days is not quarantined.
- `purge_failed_tickers(db)` deletes all rows; `purge_failed_tickers(db, before=ISO)` only older ones; both return the count.
- `get_fh_metrics` returns `{{}}` for a cached 'Finnhub has nothing' answer and `None` for never-fetched — callers distinguish them.
- `put_edgar_facts`/`get_edgar_facts` round-trip NaN as JSON null and back.


### Task 0.2: `src/finnhub_data.py` — market cap, industry, ratios, earnings calendar

**Files:**
- Create: `src/finnhub_data.py`
- Create: `tests/test_finnhub_data.py`

Replaces the dead yfinance `fast_info`/`get_info` path. One Finnhub call per ticker gives market cap and ~130 ratios (verified live: AAPL cap 4.81e12, industry 'Technology'; JPM/BRK-B/O flagged financial). Units: Finnhub reports caps and shares in **millions**; percent metrics are in percent units.

- [ ] **Step 1:** **Create `tests/test_finnhub_data.py` with exactly this content:**

```python
"""Finnhub-sourced caps, industries, ratios and earnings dates."""
import json
import os
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from src.cache import init_db, put_fh_metrics, put_fh_profile, put_market_cap
from src import finnhub_data as fd


def _db():
    f = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    f.close()
    init_db(f.name)
    return f.name


class _APIError(Exception):
    def __init__(self, status_code):
        super().__init__(f"status {status_code}")
        self.status_code = status_code


class FakeClient:
    """Scripted Finnhub client. `metrics[sym]` is the metric dict to return,
    `errors[sym]` an HTTP status to raise (once, unless persistent)."""

    def __init__(self, metrics=None, profiles=None, errors=None, persistent=False, calendar=None):
        self.metrics = metrics or {}
        self.profiles = profiles or {}
        self.errors = dict(errors or {})
        self.persistent = persistent
        self.calendar = calendar or {"earningsCalendar": []}
        self.calls = []

    def _maybe_raise(self, sym):
        if sym in self.errors:
            code = self.errors[sym]
            if not self.persistent:
                del self.errors[sym]
            raise _APIError(code)

    def company_basic_financials(self, sym, metric):
        self.calls.append(("metrics", sym))
        self._maybe_raise(sym)
        return {"metric": self.metrics.get(sym, {})} if sym in self.metrics else {}

    def company_profile2(self, symbol):
        self.calls.append(("profile", symbol))
        self._maybe_raise(symbol)
        return self.profiles.get(symbol, {})

    def earnings_calendar(self, _from, to, symbol):
        self.calls.append(("calendar", _from, to))
        return self.calendar


# ── symbols / sectors ─────────────────────────────────────────────────────────

def test_to_finnhub_symbol_maps_dash_to_dot():
    assert fd.to_finnhub_symbol("brk-b") == "BRK.B"
    assert fd.to_finnhub_symbol("AAPL") == "AAPL"


def test_sector_for_maps_known_and_passes_through_unknown():
    assert fd.sector_for("Biotechnology") == "Healthcare"
    assert fd.sector_for("Banking") == "Financials"
    assert fd.sector_for("Weird New Industry") == "Weird New Industry"
    assert fd.sector_for("") == "Unknown"
    assert fd.sector_for(None) == "Unknown"


def test_is_financial():
    assert fd.is_financial("Banking") and fd.is_financial("Real Estate")
    assert not fd.is_financial("Technology") and not fd.is_financial(None)


# ── fetch_metrics ─────────────────────────────────────────────────────────────

def test_fetch_metrics_caches_and_reuses():
    db = _db()
    client = FakeClient(metrics={"AAPL": {"peTTM": 30.0, "marketCapitalization": 3_000_000}})
    out, stats = fd.fetch_metrics(["AAPL"], db, client=client, sleep=lambda s: None)
    assert out["AAPL"]["peTTM"] == 30.0
    assert stats["fetched"] == 1 and stats["cached"] == 0
    out2, stats2 = fd.fetch_metrics(["AAPL"], db, client=client, sleep=lambda s: None)
    assert out2["AAPL"]["peTTM"] == 30.0
    assert stats2["cached"] == 1 and stats2["fetched"] == 0
    assert len([c for c in client.calls if c[0] == "metrics"]) == 1
    os.unlink(db)


def test_fetch_metrics_unknown_symbol_cached_as_empty():
    """Finnhub answers {} for an OTC line; cache that so it is not re-asked daily."""
    db = _db()
    client = FakeClient(metrics={})
    out, stats = fd.fetch_metrics(["WCPRF"], db, client=client, sleep=lambda s: None)
    assert out["WCPRF"] == {}
    _, stats2 = fd.fetch_metrics(["WCPRF"], db, client=client, sleep=lambda s: None)
    assert stats2["cached"] == 1 and stats2["fetched"] == 0
    os.unlink(db)


def test_fetch_metrics_respects_max_fetch():
    db = _db()
    client = FakeClient(metrics={t: {"peTTM": 1.0} for t in "ABCDE"})
    out, stats = fd.fetch_metrics(list("ABCDE"), db, max_fetch=2, client=client, sleep=lambda s: None)
    assert stats["fetched"] == 2 and stats["unfetched"] == 3
    assert set(out) == {"A", "B"}
    os.unlink(db)


def test_fetch_metrics_retries_once_on_429_then_succeeds():
    db = _db()
    slept = []
    client = FakeClient(metrics={"AAPL": {"peTTM": 30.0}}, errors={"AAPL": 429})
    out, stats = fd.fetch_metrics(["AAPL"], db, client=client, sleep=slept.append)
    assert out["AAPL"]["peTTM"] == 30.0
    assert 60.0 in slept
    os.unlink(db)


def test_fetch_metrics_403_stops_the_loop_without_poisoning_cache():
    db = _db()
    client = FakeClient(metrics={"AAPL": {}, "MSFT": {}}, errors={"AAPL": 403}, persistent=True)
    out, stats = fd.fetch_metrics(["AAPL", "MSFT"], db, client=client, sleep=lambda s: None)
    assert stats["forbidden"] is True
    assert "AAPL" not in out and "MSFT" not in out
    assert len(client.calls) == 1  # stopped after the 403
    os.unlink(db)


def test_fetch_metrics_other_error_counts_failed_and_continues():
    db = _db()
    client = FakeClient(metrics={"MSFT": {"peTTM": 20.0}}, errors={"AAPL": 500}, persistent=True)
    out, stats = fd.fetch_metrics(["AAPL", "MSFT"], db, client=client, sleep=lambda s: None)
    assert stats["failed"] == 1 and out["MSFT"]["peTTM"] == 20.0
    os.unlink(db)


def test_fetch_metrics_uses_finnhub_symbol_for_share_classes():
    db = _db()
    client = FakeClient(metrics={"BRK.B": {"peTTM": 10.0}})
    out, _ = fd.fetch_metrics(["BRK-B"], db, client=client, sleep=lambda s: None)
    assert out["BRK-B"]["peTTM"] == 10.0  # keyed by our ticker, asked with Finnhub's
    os.unlink(db)


# ── fetch_profiles ────────────────────────────────────────────────────────────

def test_fetch_profiles_converts_millions_and_keeps_industry():
    db = _db()
    client = FakeClient(profiles={"AAPL": {"finnhubIndustry": "Technology",
                                          "marketCapitalization": 3_000_000.0,   # $3.0T in millions
                                          "shareOutstanding": 15_000.0}})        # 15B shares in millions
    out, stats = fd.fetch_profiles(["AAPL"], db, client=client, sleep=lambda s: None)
    p = out["AAPL"]
    assert p["industry"] == "Technology"
    assert p["market_cap"] == pytest.approx(3.0e12)
    assert p["shares_out"] == pytest.approx(1.5e10)
    assert stats["fetched"] == 1
    os.unlink(db)


def test_fetch_profiles_refreshes_only_stale_caps():
    db = _db()
    put_fh_profile(db, "AAPL", "Technology", 2.0e12, 1.5e10)   # fresh row
    client = FakeClient(profiles={"AAPL": {"finnhubIndustry": "Technology", "marketCapitalization": 3_000_000.0}})
    out, stats = fd.fetch_profiles(["AAPL"], db, cap_ttl_days=7, client=client, sleep=lambda s: None)
    assert stats["fetched"] == 0 and out["AAPL"]["market_cap"] == pytest.approx(2.0e12)
    os.unlink(db)


def test_fetch_profiles_stale_row_still_returned_when_budget_exhausted():
    db = _db()
    put_fh_profile(db, "AAPL", "Technology", 2.0e12, None)
    client = FakeClient(profiles={})
    out, stats = fd.fetch_profiles(["AAPL"], db, cap_ttl_days=0, max_fetch=0, client=client, sleep=lambda s: None)
    assert out["AAPL"]["industry"] == "Technology"
    assert stats["unfetched"] == 1
    os.unlink(db)


# ── market cap resolution ─────────────────────────────────────────────────────

def test_resolve_market_cap_precedence_metrics_first():
    cap, src = fd.resolve_market_cap(
        "AAPL", 200.0,
        metrics={"AAPL": {"marketCapitalization": 3_000_000.0}},
        profiles={"AAPL": {"market_cap": 2.0e12, "shares_out": 1.5e10}})
    assert cap == pytest.approx(3.0e12) and src == "fh_metrics"


def test_resolve_market_cap_falls_to_profile_then_shares_times_price():
    cap, src = fd.resolve_market_cap("X", 10.0, metrics={}, profiles={"X": {"market_cap": 5.0e8, "shares_out": None}})
    assert cap == 5.0e8 and src == "fh_profile"
    cap, src = fd.resolve_market_cap("X", 10.0, metrics={}, profiles={"X": {"market_cap": None, "shares_out": 1.0e8}})
    assert cap == pytest.approx(1.0e9) and src == "shares_x_price"


def test_resolve_market_cap_legacy_cache_last_resort_and_none():
    db = _db()
    put_market_cap(db, "OLD", 7.5e8)
    cap, src = fd.resolve_market_cap("OLD", 10.0, {}, {}, db_path=db)
    assert cap == 7.5e8 and src == "legacy_cache"
    cap, src = fd.resolve_market_cap("NEW", 10.0, {}, {}, db_path=db)
    assert cap is None and src == "none"
    os.unlink(db)


def test_resolve_market_cap_rejects_implausible_values():
    # 3e18 is not a market cap; a wrong-unit source must not pass the gate.
    cap, src = fd.resolve_market_cap("X", 10.0, metrics={"X": {"marketCapitalization": 3e12}}, profiles={})
    assert cap is None and src == "none"
    cap, src = fd.resolve_market_cap("X", 10.0, metrics={"X": {"marketCapitalization": 0}}, profiles={})
    assert cap is None


def test_attach_market_caps_adds_columns_and_counts():
    df = pd.DataFrame({"ticker": ["AAPL", "NOCAP", "BANK"], "price": [200.0, 10.0, 50.0]})
    metrics = {"AAPL": {"marketCapitalization": 3_000_000.0}}
    profiles = {"AAPL": {"industry": "Technology", "market_cap": None, "shares_out": None},
                "BANK": {"industry": "Banking", "market_cap": 8.0e9, "shares_out": None}}
    out, stats = fd.attach_market_caps(df, metrics, profiles)
    assert out.loc[0, "market_cap"] == pytest.approx(3.0e12)
    assert pd.isna(out.loc[1, "market_cap"]) and out.loc[1, "cap_source"] == "none"
    assert out.loc[2, "sector"] == "Financials" and bool(out.loc[2, "is_financial"]) is True
    assert out.loc[0, "sector"] == "Technology" and bool(out.loc[0, "is_financial"]) is False
    assert stats["dropped_no_cap"] == 1 and stats["fh_metrics"] == 1 and stats["fh_profile"] == 1
    assert out.loc[1, "sector"] == "Unknown"


# ── earnings calendar ─────────────────────────────────────────────────────────

def test_fetch_earnings_calendar_keeps_earliest_date_and_normalises_symbols(tmp_path):
    client = FakeClient(calendar={"earningsCalendar": [
        {"symbol": "AAPL", "date": "2026-10-29"},
        {"symbol": "AAPL", "date": "2026-10-28"},
        {"symbol": "BRK.B", "date": "2026-11-01"},
        {"symbol": "", "date": "2026-11-01"},
        {"symbol": "BAD", "date": "2026-1-1"},
    ]})
    cal = fd.fetch_earnings_calendar(cache_path=tmp_path / "cal.json", client=client, today=date(2026, 9, 4))
    assert cal == {"AAPL": "2026-10-28", "BRK-B": "2026-11-01"}
    # Second call is served from the cache: no new client call.
    n = len(client.calls)
    fd.fetch_earnings_calendar(cache_path=tmp_path / "cal.json", client=client, today=date(2026, 9, 4))
    assert len(client.calls) == n


def test_fetch_earnings_calendar_falls_back_to_stale_cache_on_error(tmp_path):
    path = tmp_path / "cal.json"
    path.write_text(json.dumps({"fetched_at": (datetime.utcnow() - timedelta(days=3)).isoformat(),
                                "dates": {"AAPL": "2026-10-28"}}))

    class Broken(FakeClient):
        def earnings_calendar(self, _from, to, symbol):
            raise _APIError(500)

    cal = fd.fetch_earnings_calendar(cache_path=path, client=Broken(), today=date(2026, 9, 4))
    assert cal == {"AAPL": "2026-10-28"}


def test_days_to_earnings():
    cal = {"AAPL": "2026-09-10", "OLD": "2026-09-01"}
    assert fd.days_to_earnings("aapl", cal, today=date(2026, 9, 4)) == 6
    assert fd.days_to_earnings("OLD", cal, today=date(2026, 9, 4)) is None   # stale, never negative
    assert fd.days_to_earnings("MSFT", cal, today=date(2026, 9, 4)) is None
```

- [ ] **Step 2:** Run:

```bash
/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m pytest tests/test_finnhub_data.py -q
```

Expected: FAIL at import: `No module named src.finnhub_data`.

- [ ] **Step 3:** **Create `src/finnhub_data.py` with exactly this content:**

```python
"""Finnhub-sourced universe data: market cap, industry, fundamental ratios,
earnings dates.

Why this module exists. Market cap and sector used to come from yfinance
`fast_info` / `get_info`, one call per ticker. yfinance 1.x rejects the custom
requests.Session the price fetcher passes, and Yahoo rate-limits per-ticker
calls at universe scale, so by 2026-09 the market_cap cache had not refreshed
since June and `sector` was empty for every ranked row. Finnhub's free tier
answers both in one call per ticker (`company_basic_financials`, ~130 ratios
including marketCapitalization) at 60 calls/minute, and the result is good
for a week.

Rate discipline. A run may fetch at most `max_fetch` missing tickers, so a
cold cache degrades a run (fewer names have caps) instead of stalling it; the
rest are filled by `python -m src.cache_maint warm-finnhub` or by later runs.
Everything cached is read in one bulk query.

Units. Finnhub reports marketCapitalization and shareOutstanding in MILLIONS.
Percent metrics (roeTTM, revenueGrowthTTMYoy, grossMarginTTM) are in percent
units (33.2 means 33.2%). Ratios (peTTM, pb, currentRatioQuarterly) are raw.
"""
import json
import logging
import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path

from src.cache import (
    get_fh_metrics_bulk, put_fh_metrics,
    get_fh_profiles_bulk, put_fh_profile,
    get_market_cap_stale,
)

logger = logging.getLogger(__name__)

MILLION = 1_000_000.0
# Sanity bounds for a market cap in dollars. Outside these the source is
# reporting in the wrong unit or is garbage, and the name is treated as capless.
CAP_MIN = 1_000_000.0          # $1M
CAP_MAX = 20_000_000_000_000.0 # $20T

# Finnhub `finnhubIndustry` strings for which gross-profitability, accruals and
# leverage ratios are meaningless. Checked with `is_financial()`.
FINANCIAL_INDUSTRIES = {
    "Banking", "Financial Services", "Insurance", "Real Estate",
}

# Coarse sector for the composite's sector demeaning and the sizing sector cap.
# Finnhub industries not listed map to themselves — a specific bucket beats
# "Unknown", which is what made the sector cap unenforceable.
_SECTOR_MAP = {
    "Technology": "Technology", "Semiconductors": "Technology",
    "Communications": "Communication Services", "Media": "Communication Services",
    "Telecommunication": "Communication Services",
    "Health Care": "Healthcare", "Biotechnology": "Healthcare",
    "Pharmaceuticals": "Healthcare", "Life Sciences Tools & Services": "Healthcare",
    "Medical Devices": "Healthcare", "Health Care Providers & Services": "Healthcare",
    "Banking": "Financials", "Financial Services": "Financials", "Insurance": "Financials",
    "Real Estate": "Real Estate",
    "Energy": "Energy", "Oil & Gas": "Energy",
    "Utilities": "Utilities", "Electrical Equipment": "Industrials",
    "Industrials": "Industrials", "Machinery": "Industrials", "Aerospace & Defense": "Industrials",
    "Airlines": "Industrials", "Building": "Industrials", "Construction": "Industrials",
    "Logistics & Transportation": "Industrials", "Road & Rail": "Industrials",
    "Trading Companies & Distributors": "Industrials", "Commercial Services & Supplies": "Industrials",
    "Professional Services": "Industrials",
    "Retail": "Consumer Cyclical", "Consumer products": "Consumer Defensive",
    "Textiles, Apparel & Luxury Goods": "Consumer Cyclical", "Hotels, Restaurants & Leisure": "Consumer Cyclical",
    "Automobiles": "Consumer Cyclical", "Auto Components": "Consumer Cyclical",
    "Leisure Products": "Consumer Cyclical", "Distributors": "Consumer Cyclical",
    "Beverages": "Consumer Defensive", "Food Products": "Consumer Defensive",
    "Tobacco": "Consumer Defensive", "Food & Staples Retailing": "Consumer Defensive",
    "Household Products": "Consumer Defensive", "Personal Products": "Consumer Defensive",
    "Chemicals": "Materials", "Metals & Mining": "Materials", "Packaging": "Materials",
    "Paper & Forest": "Materials", "Construction Materials": "Materials",
}


def to_finnhub_symbol(ticker: str) -> str:
    """SEC/yfinance write share classes with a dash (BRK-B); Finnhub and Alpaca
    use a dot (BRK.B)."""
    return ticker.upper().replace("-", ".")


def sector_for(industry: str | None) -> str:
    ind = (industry or "").strip()
    if not ind:
        return "Unknown"
    return _SECTOR_MAP.get(ind, ind)


def is_financial(industry: str | None) -> bool:
    return (industry or "").strip() in FINANCIAL_INDUSTRIES


class _Bucket:
    """Token bucket at `rate` calls per minute (same shape as fundamentals.py)."""

    def __init__(self, rate: int, sleep=time.sleep):
        self._rate = max(1, int(rate))
        self._tokens = float(self._rate)
        self._last = time.monotonic()
        self._sleep = sleep

    def consume(self):
        now = time.monotonic()
        self._tokens = min(self._rate, self._tokens + (now - self._last) * self._rate / 60.0)
        self._last = now
        if self._tokens < 1:
            self._sleep((1 - self._tokens) * 60.0 / self._rate)
            self._tokens = 0.0
        else:
            self._tokens -= 1


def _client():
    import finnhub
    return finnhub.Client(api_key=os.environ.get("FINNHUB_API_KEY", ""))


def _status_of(exc: Exception) -> int | None:
    """HTTP status carried by a finnhub.FinnhubAPIException, else None."""
    code = getattr(exc, "status_code", None)
    if code is not None:
        try:
            return int(code)
        except (TypeError, ValueError):
            return None
    return None


def _call(fn, *args, sleep=time.sleep, **kwargs):
    """One Finnhub call with a single retry on 429. Returns (payload, error)
    where error is None, "forbidden" (403 — stop asking), or "error"."""
    for attempt in range(2):
        try:
            return fn(*args, **kwargs), None
        except Exception as e:  # noqa: BLE001 — every failure is one ticker, never the run
            status = _status_of(e)
            if status == 429 and attempt == 0:
                sleep(60.0)
                continue
            if status == 403:
                return None, "forbidden"
            return None, "error"
    return None, "error"


def fetch_metrics(
    tickers: list[str],
    db_path: str,
    ttl_days: int = 7,
    max_fetch: int = 600,
    calls_per_minute: int = 60,
    client=None,
    sleep=time.sleep,
) -> tuple[dict[str, dict], dict]:
    """Basic-financials `metric` dict per ticker.

    Returns (metrics, stats). `metrics[t]` is {} when Finnhub has no data for
    the symbol (cached so it is not re-asked daily). Tickers beyond `max_fetch`
    that were not cached are simply absent — counted in stats["unfetched"].
    """
    cached = get_fh_metrics_bulk(db_path, ttl_days)
    wanted = [t.upper() for t in tickers]
    out = {t: cached[t] for t in wanted if t in cached}
    missing = [t for t in wanted if t not in cached]
    stats = {"cached": len(out), "fetched": 0, "failed": 0,
             "unfetched": max(0, len(missing) - max_fetch), "forbidden": False}
    if not missing or max_fetch <= 0:
        return out, stats

    fh = client or _client()
    bucket = _Bucket(calls_per_minute, sleep=sleep)
    for t in missing[:max_fetch]:
        bucket.consume()
        payload, err = _call(fh.company_basic_financials, to_finnhub_symbol(t), "all", sleep=sleep)
        if err == "forbidden":
            stats["forbidden"] = True
            logger.warning("[finnhub] 403 on basic financials — key lacks access, stopping")
            break
        if err:
            stats["failed"] += 1
            continue
        metric = (payload or {}).get("metric") or {}
        put_fh_metrics(db_path, t, metric)
        out[t] = metric
        stats["fetched"] += 1
    return out, stats


def fetch_profiles(
    tickers: list[str],
    db_path: str,
    cap_ttl_days: int = 7,
    max_fetch: int = 600,
    calls_per_minute: int = 60,
    client=None,
    sleep=time.sleep,
) -> tuple[dict[str, dict], dict]:
    """Industry + market cap + shares outstanding per ticker via company_profile2.

    Industry never expires. The cap is refreshed when older than cap_ttl_days,
    but a stale row is still returned (with its fetched_at) if the refresh
    budget runs out — a week-old cap is fine for a $300M gate.
    """
    cached = get_fh_profiles_bulk(db_path)
    wanted = [t.upper() for t in tickers]
    cutoff = (datetime.utcnow() - timedelta(days=cap_ttl_days)).isoformat()
    need = [t for t in wanted if t not in cached or (cached[t].get("fetched_at") or "") < cutoff]
    out = {t: cached[t] for t in wanted if t in cached}
    stats = {"cached": len(out), "fetched": 0, "failed": 0,
             "unfetched": max(0, len(need) - max_fetch), "forbidden": False}
    if not need or max_fetch <= 0:
        return out, stats

    fh = client or _client()
    bucket = _Bucket(calls_per_minute, sleep=sleep)
    for t in need[:max_fetch]:
        bucket.consume()
        payload, err = _call(fh.company_profile2, sleep=sleep, symbol=to_finnhub_symbol(t))
        if err == "forbidden":
            stats["forbidden"] = True
            break
        if err:
            stats["failed"] += 1
            continue
        p = payload or {}
        industry = p.get("finnhubIndustry") or ""
        cap = _dollars(p.get("marketCapitalization"))
        shares = _f(p.get("shareOutstanding"))
        shares = shares * MILLION if shares and shares > 0 else None
        put_fh_profile(db_path, t, industry, cap, shares)
        out[t] = {"industry": industry, "market_cap": cap, "shares_out": shares,
                  "fetched_at": datetime.utcnow().isoformat()}
        stats["fetched"] += 1
    return out, stats


def _f(v) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if x == x else None  # NaN -> None


def _dollars(cap_millions) -> float | None:
    """Finnhub millions -> dollars, or None when absent/implausible."""
    x = _f(cap_millions)
    if x is None or x <= 0:
        return None
    dollars = x * MILLION
    if not (CAP_MIN <= dollars <= CAP_MAX):
        return None
    return dollars


def resolve_market_cap(
    ticker: str,
    price: float | None,
    metrics: dict[str, dict],
    profiles: dict[str, dict],
    db_path: str | None = None,
) -> tuple[float | None, str]:
    """Best available cap in dollars and where it came from.

    Precedence: basic-financials cap -> profile cap -> profile shares × price ->
    legacy `market_cap` table (any age) -> None. Every source is bounds-checked.
    """
    t = ticker.upper()
    m = metrics.get(t) or {}
    cap = _dollars(m.get("marketCapitalization"))
    if cap is not None:
        return cap, "fh_metrics"
    p = profiles.get(t) or {}
    cap = p.get("market_cap")
    if cap is not None and CAP_MIN <= float(cap) <= CAP_MAX:
        return float(cap), "fh_profile"
    shares = p.get("shares_out")
    if shares and price and price > 0:
        est = float(shares) * float(price)
        if CAP_MIN <= est <= CAP_MAX:
            return est, "shares_x_price"
    if db_path:
        legacy = get_market_cap_stale(db_path, t)
        if legacy is not None and CAP_MIN <= float(legacy) <= CAP_MAX:
            return float(legacy), "legacy_cache"
    return None, "none"


def attach_market_caps(df, metrics: dict[str, dict], profiles: dict[str, dict],
                       db_path: str | None = None):
    """Add `market_cap`, `cap_source`, `industry`, `sector`, `is_financial`
    columns to a factors frame (must have `ticker` and `price`). Returns
    (df, stats) where stats counts names per cap source."""
    import pandas as pd

    caps, sources, inds, secs, fins = [], [], [], [], []
    for _, row in df.iterrows():
        t = str(row["ticker"]).upper()
        price = row.get("price")
        cap, src = resolve_market_cap(t, price, metrics, profiles, db_path)
        caps.append(cap if cap is not None else float("nan"))
        sources.append(src)
        industry = (profiles.get(t) or {}).get("industry") or ""
        inds.append(industry)
        secs.append(sector_for(industry))
        fins.append(is_financial(industry))
    out = df.copy()
    out["market_cap"] = pd.Series(caps, index=out.index, dtype="float64")
    out["cap_source"] = sources
    out["industry"] = inds
    out["sector"] = secs
    out["is_financial"] = fins
    stats = {s: sources.count(s) for s in set(sources)}
    stats["dropped_no_cap"] = sources.count("none")
    return out, stats


# ── earnings calendar ────────────────────────────────────────────────────────

EARNINGS_CACHE = Path("data") / "earnings_calendar.json"


def fetch_earnings_calendar(
    days_ahead: int = 45,
    cache_path: Path | None = None,
    ttl_hours: int = 20,
    client=None,
    today: date | None = None,
) -> dict[str, str]:
    """{ticker: 'YYYY-MM-DD'} for every US name reporting within days_ahead.

    One call for the whole market (Finnhub's calendar endpoint is free and
    accepts an empty symbol). Cached on disk because every consumer — the
    screener finalists, the portfolio engine, the exit plan — wants it on the
    same evening. Fail-soft: on any error the stale cache is returned, or {}.
    """
    today = today or date.today()
    path = cache_path or EARNINGS_CACHE
    if path.exists():
        try:
            blob = json.loads(path.read_text())
            fetched = datetime.fromisoformat(blob.get("fetched_at", "1970-01-01"))
            if datetime.utcnow() - fetched < timedelta(hours=ttl_hours):
                return blob.get("dates", {})
        except (OSError, ValueError, json.JSONDecodeError):
            blob = {}
    else:
        blob = {}

    fh = client or _client()
    payload, err = _call(fh.earnings_calendar, _from=today.isoformat(),
                         to=(today + timedelta(days=days_ahead)).isoformat(), symbol="")
    if err:
        logger.warning(f"[finnhub] earnings calendar unavailable ({err}); using stale cache")
        return blob.get("dates", {}) if isinstance(blob, dict) else {}

    dates: dict[str, str] = {}
    for e in (payload or {}).get("earningsCalendar", []) or []:
        sym = str(e.get("symbol") or "").upper().replace(".", "-")
        d = str(e.get("date") or "")
        if not sym or len(d) != 10:
            continue
        # Keep the EARLIEST upcoming date per symbol.
        if sym not in dates or d < dates[sym]:
            dates[sym] = d
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"fetched_at": datetime.utcnow().isoformat(), "dates": dates}))
    except OSError:
        pass
    return dates


def days_to_earnings(ticker: str, calendar: dict[str, str], today: date | None = None) -> int | None:
    """Calendar days until the ticker's next report, or None if not scheduled
    inside the calendar window. Never negative: a date before today is stale."""
    today = today or date.today()
    d = calendar.get(ticker.upper())
    if not d:
        return None
    try:
        delta = (date.fromisoformat(d) - today).days
    except ValueError:
        return None
    return delta if delta >= 0 else None
```

- [ ] **Step 4:** Run:

```bash
/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m pytest tests/test_finnhub_data.py -q
```

Expected: 21 passed.

- [ ] **Step 5:** Live probe (read-only, uses your Finnhub key):

```bash
set -a; source .env; set +a
/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 - <<'EOF'
from src.cache import init_db
from src.finnhub_data import fetch_metrics, fetch_profiles, attach_market_caps
import pandas as pd
init_db('data/cache.db')
sample=['AAPL','JPM','BRK-B','GEF-B','TSM','SPY']
m,ms=fetch_metrics(sample,'data/cache.db',max_fetch=10); p,ps=fetch_profiles(sample,'data/cache.db',max_fetch=10)
print(ms, ps)
df,cs=attach_market_caps(pd.DataFrame({'ticker':sample,'price':[1.0]*6}),m,p,'data/cache.db')
print(df[['ticker','market_cap','cap_source','industry','sector','is_financial']]); print(cs)
EOF
```

Expected: caps in dollars (AAPL ≈ 4.8e12), JPM/BRK-B `is_financial True`, SPY `cap_source none` (an ETF has no company facts — correct). Note: TSM (an ADR) may resolve via `shares_x_price`; that source is only a gate fallback.

- [ ] **Step 6:** Commit as `feat(finnhub): market cap, industry and ratios from Finnhub basic financials`. Per the user's global CLAUDE.md §6: run `git status` first and **ask the user before committing**. Never add a Co-Authored-By line.

**Edge cases this task must handle (each has a test):**
- Unknown symbol → `{{}}` cached; second call is a cache hit.
- `max_fetch` bounds the calls per run; the rest are `unfetched` in stats.
- HTTP 429 sleeps 60 s and retries once; 403 sets `forbidden` and stops the loop without poisoning the cache; other errors count `failed` and continue.
- `BRK-B` is asked as `BRK.B` and stored under `BRK-B`.
- Caps outside [$1M, $20T] are rejected; 0/None → NaN.
- Earnings calendar keeps the earliest date per symbol, normalises `.`→`-`, drops malformed rows, serves from the disk cache, and falls back to the stale cache on error; `days_to_earnings` is never negative.


### Task 0.3: Universe — Alpaca tradability filter, suffix exclusions, weekly rebuild

**Files:**
- Modify: `src/universe.py` (full rewrite below)
- Modify: `src/broker.py` (add `get_assets`)
- Create: `tests/test_universe_tradable.py`

Alpaca's free `/v2/assets` lists 13,393 tradable equities with exchange and fractionable flags. Intersecting the SEC list with it drops OTC/ADR lines the trader cannot buy (BHPLF, WCPRF, PBR-A was fine as PBR.A) and roughly halves yfinance load. Verified live: 9,619 SEC → 9,275 after suffix filter → 6,320 tradable (NASDAQ 4,089, NYSE 1,940, AMEX 238, ARCA 48, BATS 5).

- [ ] **Step 1:** **Create `tests/test_universe_tradable.py` with exactly this content:**

```python
"""Alpaca tradability filter, SEC suffix exclusions, universe staleness."""
import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from src import universe as U


def _sec_df(*tickers):
    return pd.DataFrame({"ticker": list(tickers), "cik": ["0000000001"] * len(tickers),
                         "name": ["Some Company Inc"] * len(tickers)})


def test_filter_universe_drops_preferred_warrant_unit_right_suffixes():
    df = _sec_df("ACR-PC", "AGM-PI", "CLBR-WS", "ABLV-WT", "BEBE-U", "GFR-R", "GEF-B", "BRK-B", "AAPL")
    out = U.filter_universe(df)
    assert sorted(out["ticker"]) == ["AAPL", "BRK-B", "GEF-B"]


def test_to_alpaca_symbol():
    assert U.to_alpaca_symbol("GEF-B") == "GEF.B"
    assert U.to_alpaca_symbol("aapl") == "AAPL"


def _assets_raw():
    return [
        {"symbol": "AAPL", "exchange": "NASDAQ", "tradable": True, "fractionable": True, "shortable": True},
        {"symbol": "GEF.B", "exchange": "NYSE", "tradable": True, "fractionable": True, "shortable": False},
        {"symbol": "WCPRF", "exchange": "OTC", "tradable": True, "fractionable": False, "shortable": False},
        {"symbol": "DEAD", "exchange": "NYSE", "tradable": False, "fractionable": False, "shortable": False},
        {"symbol": "BAC.PRN", "exchange": "NYSE", "tradable": True, "fractionable": False, "shortable": False},
        {"symbol": "CLBR.WS", "exchange": "NYSE", "tradable": True, "fractionable": False, "shortable": False},
        {"symbol": "GLED.U", "exchange": "NASDAQ", "tradable": True, "fractionable": False, "shortable": False},
        {"symbol": "GFR.RT", "exchange": "NYSE", "tradable": True, "fractionable": False, "shortable": False},
        {"symbol": "HEI.A", "exchange": "NYSE", "tradable": True, "fractionable": True, "shortable": True},
    ]


def test_load_tradable_assets_filters_and_caches(tmp_path):
    cache = tmp_path / "assets.json"
    assets = U.load_tradable_assets(fetch=_assets_raw, cache_path=cache)
    assert set(assets) == {"AAPL", "GEF.B", "HEI.A"}
    assert assets["AAPL"]["fractionable"] is True and assets["AAPL"]["exchange"] == "NASDAQ"
    # Cached: a second call with a failing fetch still returns the same set.
    def boom():
        raise RuntimeError("down")
    assert set(U.load_tradable_assets(fetch=boom, cache_path=cache)) == {"AAPL", "GEF.B", "HEI.A"}


def test_load_tradable_assets_returns_none_when_unavailable_and_no_cache(tmp_path):
    def boom():
        raise RuntimeError("down")
    assert U.load_tradable_assets(fetch=boom, cache_path=tmp_path / "none.json") is None


def test_load_tradable_assets_uses_stale_cache_when_fetch_fails(tmp_path):
    cache = tmp_path / "assets.json"
    cache.write_text(json.dumps({"fetched_at": (datetime.utcnow() - timedelta(days=3)).isoformat(),
                                 "assets": {"AAPL": {"exchange": "NASDAQ", "fractionable": True, "shortable": True}}}))
    def boom():
        raise RuntimeError("down")
    assert set(U.load_tradable_assets(fetch=boom, cache_path=cache, ttl_hours=24)) == {"AAPL"}


def test_load_tradable_assets_without_credentials_returns_none(tmp_path, monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    assert U.load_tradable_assets(cache_path=tmp_path / "none.json") is None


def test_filter_tradable_keeps_listed_and_maps_symbols():
    df = _sec_df("AAPL", "GEF-B", "WCPRF", "PBR-A", "BHPLF")
    assets = {"AAPL": {"exchange": "NASDAQ", "fractionable": True},
              "GEF.B": {"exchange": "NYSE", "fractionable": True},
              "PBR.A": {"exchange": "NYSE", "fractionable": True}}
    out = U.filter_tradable(df, assets)
    assert list(out["ticker"]) == ["AAPL", "GEF-B", "PBR-A"]
    assert list(out["alpaca_symbol"]) == ["AAPL", "GEF.B", "PBR.A"]
    assert list(out["exchange"]) == ["NASDAQ", "NYSE", "NYSE"]
    assert out["fractionable"].all()


def test_filter_tradable_with_none_drops_nothing():
    df = _sec_df("AAPL", "WCPRF")
    out = U.filter_tradable(df, None)
    assert list(out["ticker"]) == ["AAPL", "WCPRF"]
    assert list(out["alpaca_symbol"]) == ["AAPL", "WCPRF"]
    assert "fractionable" in out.columns


def test_load_or_build_universe_uses_fresh_cache(tmp_path, monkeypatch):
    path = tmp_path / "u.parquet"
    _sec_df("AAPL").to_parquet(path, index=False)
    monkeypatch.setattr(U, "build_universe", lambda cfg, p: (_ for _ in ()).throw(AssertionError("must not rebuild")))
    out = U.load_or_build_universe({"universe": {}}, str(path), max_age_days=7)
    assert list(out["ticker"]) == ["AAPL"]


def test_load_or_build_universe_rebuilds_when_stale(tmp_path, monkeypatch):
    path = tmp_path / "u.parquet"
    _sec_df("OLD").to_parquet(path, index=False)
    old = time.time() - 10 * 86400
    os.utime(path, (old, old))
    monkeypatch.setattr(U, "build_universe", lambda cfg, p: _sec_df("NEW"))
    out = U.load_or_build_universe({"universe": {}}, str(path), max_age_days=7)
    assert list(out["ticker"]) == ["NEW"]


def test_load_or_build_universe_stale_rebuild_failure_falls_back(tmp_path, monkeypatch):
    path = tmp_path / "u.parquet"
    _sec_df("OLD").to_parquet(path, index=False)
    old = time.time() - 10 * 86400
    os.utime(path, (old, old))
    def boom(cfg, p):
        raise RuntimeError("SEC down")
    monkeypatch.setattr(U, "build_universe", boom)
    out = U.load_or_build_universe({"universe": {}}, str(path), max_age_days=7)
    assert list(out["ticker"]) == ["OLD"]


def test_load_or_build_universe_missing_and_failing_raises(tmp_path, monkeypatch):
    def boom(cfg, p):
        raise RuntimeError("SEC down")
    monkeypatch.setattr(U, "build_universe", boom)
    with pytest.raises(RuntimeError):
        U.load_or_build_universe({"universe": {}}, str(tmp_path / "missing.parquet"))
```

- [ ] **Step 2:** Run:

```bash
/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m pytest tests/test_universe_tradable.py -q
```

Expected: FAIL: `filter_tradable`, `load_tradable_assets`, `to_alpaca_symbol`, `load_or_build_universe` not defined.

- [ ] **Step 3:** Add `get_assets` to the broker:

**Apply this unified diff to `src/broker.py`** (from the reference patch; `patch -p1 < docs/superpowers/plans/2026-09-04-overhaul-reference.patch` applies all of them at once):

```diff
--- a/src/broker.py	2026-08-04 19:12:57
+++ b/src/broker.py	2026-09-04 12:22:32
@@ -76,6 +76,12 @@
     return _get("/v2/orders", params={"status": status, "limit": 100})
 
 
+def get_assets() -> list:
+    """Every active US equity Alpaca knows about (~14k rows). Used to drop
+    untradeable SEC tickers from the screener universe."""
+    return _get("/v2/assets", params={"status": "active", "asset_class": "us_equity"})
+
+
 def submit_order(
     symbol: str,
     side: str,
```

- [ ] **Step 4:** Replace `src/universe.py` entirely:

**Create `src/universe.py` with exactly this content:**

```python
import json
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path

import requests
import pandas as pd
from src.config import get_env

logger = logging.getLogger(__name__)

SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

_EXCLUDE_TERMS = [
    " warrant", "-wt", " unit", " pfd", " preferred",
    "proshares", "ishares", "invesco", "direxion",
    " etf", "trust", "fund", " lp",
]

# SEC writes non-common lines with a dash suffix: -PA/-PC preferred series,
# -WT/-WS warrants, -U units, -R rights. Share classes (-A, -B) are kept.
_SEC_NONCOMMON_SUFFIX_RE = re.compile(r"-(?:P[A-Z]?|W[ST]?[A-Z]?|U|R)$")
# Alpaca spells the same things with a dot: .PRA preferred, .WS warrants,
# .U units, .RT rights.
_ALPACA_NONCOMMON_SUFFIX_RE = re.compile(r"\.(?:PR[A-Z]?|WS[A-Z]?|U|RT)$")

ALPACA_ASSETS_CACHE = Path("data") / "alpaca_assets.json"
ALPACA_EXCHANGES_OK = {"NYSE", "NASDAQ", "AMEX", "ARCA", "BATS"}
UNIVERSE_MAX_AGE_DAYS = 7


def _user_agent() -> str:
    return get_env("SEC_USER_AGENT")


def fetch_sec_tickers() -> dict:
    headers = {"User-Agent": _user_agent()}
    resp = requests.get(SEC_TICKERS_URL, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()


def parse_sec_tickers(raw: dict) -> pd.DataFrame:
    rows = []
    for _, entry in raw.items():
        cik = f"{int(entry['cik_str']):010d}"
        rows.append({
            "ticker": entry["ticker"].upper().strip(),
            "cik": cik,
            "name": entry["title"],
        })
    return pd.DataFrame(rows)


def filter_universe(df: pd.DataFrame) -> pd.DataFrame:
    name_lower = df["name"].str.lower()
    ticker_lower = df["ticker"].str.lower()
    mask_warrant = ticker_lower.str.contains(r"-wt$|\+$|\.wt$", regex=True)
    mask_exclude = name_lower.apply(
        lambda n: any(term in n for term in _EXCLUDE_TERMS)
    )
    mask_suffix = df["ticker"].str.upper().str.contains(_SEC_NONCOMMON_SUFFIX_RE, regex=True)
    return df[~mask_warrant & ~mask_exclude & ~mask_suffix].reset_index(drop=True)


def build_universe(cfg: dict, out_path: str = "data/universe.parquet") -> pd.DataFrame:
    raw = fetch_sec_tickers()
    df = parse_sec_tickers(raw)
    if cfg["universe"].get("exclude_etfs", True):
        df = filter_universe(df)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"[universe] {len(df)} tickers written to {out_path}")
    return df


def load_or_build_universe(cfg: dict, path: str = "data/universe.parquet",
                           force: bool = False, max_age_days: int = UNIVERSE_MAX_AGE_DAYS) -> pd.DataFrame:
    """Cached SEC universe, rebuilt when missing, forced, or older than
    max_age_days. The old code only rebuilt when the file was missing, so a
    universe from June was still in use in September."""
    if not force and os.path.exists(path):
        age_days = (time.time() - os.path.getmtime(path)) / 86400.0
        if age_days <= max_age_days:
            df = pd.read_parquet(path)
            print(f"[universe] loaded {len(df)} tickers from cache ({age_days:.1f}d old)")
            return df
        print(f"[universe] cache is {age_days:.1f}d old — rebuilding")
    try:
        return build_universe(cfg, path)
    except Exception as e:  # noqa: BLE001 — a SEC outage must not kill the run if a cache exists
        if os.path.exists(path):
            logger.warning(f"[universe] rebuild failed ({e!r}); using stale cache")
            return pd.read_parquet(path)
        raise


# ── Alpaca tradability filter ────────────────────────────────────────────────

def to_alpaca_symbol(ticker: str) -> str:
    """SEC/yfinance BRK-B -> Alpaca BRK.B."""
    return ticker.upper().replace("-", ".")


def load_tradable_assets(fetch=None, cache_path: Path | None = None,
                         ttl_hours: int = 24) -> dict[str, dict] | None:
    """{alpaca_symbol: asset} for active, tradable, exchange-listed US equities.

    Cached on disk for a day. Returns None — meaning "do not filter" — when
    there are no Alpaca credentials or the fetch fails with no cache, so a
    public clone without a brokerage still screens. The universe then keeps
    the untradeable OTC/ADR lines it always had; nothing is worse than before.
    """
    path = cache_path or ALPACA_ASSETS_CACHE
    if path.exists():
        try:
            blob = json.loads(path.read_text())
            fetched = datetime.fromisoformat(blob["fetched_at"])
            if (datetime.utcnow() - fetched).total_seconds() < ttl_hours * 3600:
                return blob["assets"]
        except (OSError, KeyError, ValueError, json.JSONDecodeError):
            blob = None
    else:
        blob = None

    if fetch is None:
        if not (os.environ.get("ALPACA_API_KEY") and os.environ.get("ALPACA_SECRET_KEY")):
            return blob["assets"] if blob else None
        from src import broker
        fetch = broker.get_assets
    try:
        raw = fetch()
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[universe] Alpaca assets unavailable ({e!r})")
        return blob["assets"] if blob else None

    assets = {}
    for a in raw or []:
        sym = str(a.get("symbol") or "").upper()
        if not sym or not a.get("tradable"):
            continue
        if a.get("exchange") not in ALPACA_EXCHANGES_OK:
            continue
        if _ALPACA_NONCOMMON_SUFFIX_RE.search(sym):
            continue
        assets[sym] = {"exchange": a.get("exchange"),
                       "fractionable": bool(a.get("fractionable")),
                       "shortable": bool(a.get("shortable"))}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"fetched_at": datetime.utcnow().isoformat(), "assets": assets}))
    except OSError:
        pass
    return assets


def filter_tradable(df: pd.DataFrame, assets: dict[str, dict] | None) -> pd.DataFrame:
    """Keep only names Alpaca can trade on a listed exchange. Adds
    `alpaca_symbol`, `exchange`, `fractionable`. With assets=None the frame is
    returned with alpaca_symbol filled and nothing dropped."""
    out = df.copy()
    out["alpaca_symbol"] = out["ticker"].map(to_alpaca_symbol)
    if assets is None:
        out["exchange"] = ""
        out["fractionable"] = True
        return out.reset_index(drop=True)
    keep = out["alpaca_symbol"].isin(assets.keys())
    out = out[keep].copy()
    out["exchange"] = out["alpaca_symbol"].map(lambda s: assets[s]["exchange"])
    out["fractionable"] = out["alpaca_symbol"].map(lambda s: assets[s]["fractionable"])
    dropped = int((~keep).sum())
    logger.info(f"[universe] tradability filter: {len(df)} → {len(out)} ({dropped} not tradable on Alpaca)")
    print(f"[universe] tradability filter: {len(df)} → {len(out)} tradable")
    return out.reset_index(drop=True)


def apply_neglect_gate(factors_df: pd.DataFrame, lp_cfg: dict, min_price: float = 0.0) -> pd.DataFrame:
    """Capacity-constrained band gate ($50M-$2B market cap, ADV floor) used by
    event_backtest.py's neglected-universe construction."""
    min_cap = lp_cfg["min_market_cap"]
    max_cap = lp_cfg["max_market_cap"]
    min_vol = lp_cfg["min_avg_dollar_vol_20d"]
    before = len(factors_df)
    result = factors_df[
        (factors_df["market_cap"] >= min_cap) &
        (factors_df["market_cap"] <= max_cap) &
        (factors_df["avg_dollar_vol_20d"] >= min_vol) &
        (factors_df["price"] >= min_price)
    ].reset_index(drop=True)
    logger.info(f"[neglect_gate] {before} → {len(result)} in $50M–$2B band (ADV≥$200K, price≥${min_price})")
    return result
```

- [ ] **Step 5:** Run:

```bash
/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m pytest tests/test_universe_tradable.py tests/test_universe.py -q
```

Expected: 17 passed (5 existing + 12 new).

- [ ] **Step 6:** Commit as `feat(universe): intersect SEC list with Alpaca tradable equities; rebuild weekly`. Per the user's global CLAUDE.md §6: run `git status` first and **ask the user before committing**. Never add a Co-Authored-By line.

**Edge cases this task must handle (each has a test):**
- SEC `-PA/-PC/-WT/-WS/-U/-R` suffixes dropped; `-A/-B` share classes kept and mapped to Alpaca `.A/.B`.
- Alpaca `.PRx/.WS/.U/.RT` and OTC exchange excluded; `tradable: false` excluded.
- No credentials, or fetch failure with no cache → `None` → filter skipped (public clones keep working); stale cache is used when the fetch fails.
- Universe parquet older than 7 days is rebuilt; a SEC outage falls back to the stale file; missing file + outage raises.


### Task 0.4: Prices — split-and-retry batches, no quarantine on exceptions, ADV gate, ATR/SMA200

**Files:**
- Modify: `src/prices.py` (full rewrite below)
- Modify: `tests/test_prices.py`

The bug fix. `_fetch_batch_yfinance` now raises `BatchFetchError`; `fetch_with_split` halves and retries down to 25-ticker batches and returns `(fetched, no_data, unfetched)` — only `no_data` (yfinance answered, ticker had no rows) is quarantined. Market cap leaves this module; `apply_adv_gate` runs before any per-name lookup. `sma_200` becomes a real 200-bar mean (it was 252) and `atr_14` is added for the engine's sizing.

- [ ] **Step 1:** Replace the obsolete `_get_market_cap` test and add the new ones in `tests/test_prices.py`:

**Apply this unified diff to `tests/test_prices.py`** (from the reference patch; `patch -p1 < docs/superpowers/plans/2026-09-04-overhaul-reference.patch` applies all of them at once):

```diff
--- a/tests/test_prices.py	2026-06-30 12:52:40
+++ b/tests/test_prices.py	2026-09-04 12:27:22
@@ -41,21 +41,99 @@
     result = apply_liquidity_gate(df, cfg)
     assert list(result["ticker"]) == ["A"]
 
-def test_get_market_cap_nan_returns_none():
-    import math
-    from unittest.mock import patch, MagicMock
-    from src.prices import _get_market_cap
+def test_compute_price_factors_has_atr_and_true_sma200():
+    df = make_ohlcv(300)
+    spy = make_ohlcv(300, start_price=450.0)
+    result = compute_price_factors("AAPL", df, spy)
+    assert result["market_cap"] is None            # attached later from Finnhub
+    assert result["atr_14"] > 0
+    assert abs(result["sma_200"] - float(df["close"].iloc[-200:].mean())) < 1e-9
 
-    mock_info = MagicMock()
-    mock_info.market_cap = math.nan
-    mock_ticker = MagicMock()
-    mock_ticker.fast_info = mock_info
 
-    with patch("src.prices.yf.Ticker", return_value=mock_ticker), \
-         patch("src.prices.get_market_cap", return_value=None), \
-         patch("src.prices.get_market_cap_stale", return_value=None), \
-         patch("src.prices.put_market_cap") as mock_put:
-        result = _get_market_cap("FAKE", ":memory:", 1)
+# ── batch fetch: exceptions split and retry, never quarantine ─────────────────
+from src.prices import fetch_with_split, BatchFetchError, apply_adv_gate
 
-    assert result is None
-    mock_put.assert_not_called()
+
+def _scripted_fetch(fail_if):
+    """fetch(tickers) raises BatchFetchError when fail_if(tickers) is True,
+    otherwise returns a frame for every ticker except those named 'DEAD*'."""
+    calls = []
+    def fetch(tickers):
+        calls.append(list(tickers))
+        if fail_if(tickers):
+            raise BatchFetchError("boom")
+        return {t: pd.DataFrame({"close": [1.0]}) for t in tickers if not t.startswith("DEAD")}
+    fetch.calls = calls
+    return fetch
+
+
+def test_fetch_with_split_clean_batch_reports_no_data_only_for_missing():
+    fetch = _scripted_fetch(lambda ts: False)
+    got, no_data, unfetched = fetch_with_split(["A", "DEAD1", "B"], fetch=fetch, min_batch=1, sleep=lambda s: None)
+    assert set(got) == {"A", "B"} and no_data == ["DEAD1"] and unfetched == []
+
+
+def test_fetch_with_split_halves_on_exception_and_recovers():
+    """A batch of 8 raises; each half of 4 succeeds. Nothing is quarantined."""
+    fetch = _scripted_fetch(lambda ts: len(ts) > 4)
+    tickers = [f"T{i}" for i in range(8)]
+    got, no_data, unfetched = fetch_with_split(tickers, fetch=fetch, min_batch=2, sleep=lambda s: None)
+    assert set(got) == set(tickers) and no_data == [] and unfetched == []
+    assert [len(c) for c in fetch.calls] == [8, 4, 4]
+
+
+def test_fetch_with_split_gives_up_below_min_batch_without_quarantine():
+    """Persistent failure: the tickers come back as unfetched, NOT as no_data.
+    This is the 2026-08-24 bug — 4,344 names were marked dead for 30 days."""
+    fetch = _scripted_fetch(lambda ts: True)
+    tickers = [f"T{i}" for i in range(8)]
+    got, no_data, unfetched = fetch_with_split(tickers, fetch=fetch, min_batch=4, sleep=lambda s: None)
+    assert got == {} and no_data == []
+    assert sorted(unfetched) == sorted(tickers)
+
+
+def test_fetch_with_split_mixed_outcome():
+    """Left half fails persistently, right half succeeds: only the right half is
+    fetched, the left half is unfetched, and a dead ticker on the right is no_data."""
+    def fail_if(ts):
+        return any(t.startswith("L") for t in ts)
+    fetch = _scripted_fetch(fail_if)
+    tickers = ["L1", "L2", "R1", "DEADR"]
+    got, no_data, unfetched = fetch_with_split(tickers, fetch=fetch, min_batch=2, sleep=lambda s: None)
+    assert set(got) == {"R1"} and no_data == ["DEADR"] and sorted(unfetched) == ["L1", "L2"]
+
+
+def test_fetch_with_split_sleeps_between_retries():
+    slept = []
+    fetch = _scripted_fetch(lambda ts: len(ts) > 2)
+    fetch_with_split(["A", "B", "C", "D"], fetch=fetch, min_batch=1, sleep=slept.append)
+    assert slept and all(s > 0 for s in slept)
+
+
+def test_apply_adv_gate_uses_volume_and_min_price():
+    df = pd.DataFrame({
+        "ticker": ["A", "B", "C"],
+        "avg_dollar_vol_20d": [10e6, 2e6, 10e6],
+        "price": [50.0, 50.0, 3.0],
+    })
+    cfg = {"liquidity_gate": {"min_market_cap": 300e6, "min_avg_dollar_vol_20d": 5e6},
+           "universe": {"min_price": 5.0}}
+    assert list(apply_adv_gate(df, cfg)["ticker"]) == ["A"]
+
+
+def test_apply_adv_gate_empty_frame():
+    cfg = {"liquidity_gate": {"min_market_cap": 300e6, "min_avg_dollar_vol_20d": 5e6}, "universe": {}}
+    assert len(apply_adv_gate(pd.DataFrame(), cfg)) == 0
+
+
+def test_apply_liquidity_gate_nan_cap_fails():
+    df = pd.DataFrame({"ticker": ["A", "B"], "market_cap": [400e6, float("nan")],
+                       "avg_dollar_vol_20d": [10e6, 10e6]})
+    cfg = {"liquidity_gate": {"min_market_cap": 300e6, "min_avg_dollar_vol_20d": 5e6}}
+    assert list(apply_liquidity_gate(df, cfg)["ticker"]) == ["A"]
+
+
+def test_apply_liquidity_gate_without_cap_column_returns_empty():
+    df = pd.DataFrame({"ticker": ["A"], "avg_dollar_vol_20d": [10e6]})
+    cfg = {"liquidity_gate": {"min_market_cap": 300e6, "min_avg_dollar_vol_20d": 5e6}}
+    assert len(apply_liquidity_gate(df, cfg)) == 0
```

- [ ] **Step 2:** Run:

```bash
/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m pytest tests/test_prices.py -q
```

Expected: FAIL: `fetch_with_split`, `BatchFetchError`, `apply_adv_gate` not importable; `atr_14` missing.

- [ ] **Step 3:** Replace `src/prices.py` entirely:

**Create `src/prices.py` with exactly this content:**

```python
import os
import math
import time
import logging
import requests
import pandas as pd
import yfinance as yf
from requests.adapters import HTTPAdapter
from typing import Optional
from src.factors import (
    mom_12_1, mom_1m, rs_vs_spy, rs_slope, residual_momentum,
    pct_from_52w_high, breakout_flag, avg_dollar_vol,
    rsi_14, macd_state, vol_surge_ratio, entry_grade,
    stochastic, bollinger_pct_b, adx_14, mfi_14, atr_14,
    obv_slope, parabolic_sar,
)
from src.cache import (
    get_prices, put_prices,
    put_failed_ticker, is_failed_ticker,
)


_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
yf.set_tz_cache_location(_DATA_DIR)


class _TimeoutAdapter(HTTPAdapter):
    def send(self, *args, **kwargs):
        kwargs.setdefault("timeout", 8)
        return super().send(*args, **kwargs)

_yf_session = requests.Session()
_yf_session.mount("https://", _TimeoutAdapter())
_yf_session.mount("http://", _TimeoutAdapter())

logger = logging.getLogger(__name__)

BATCH_SIZE = 200
# A batch that raises is split in half and retried, down to this size. Below
# it the tickers are skipped for THIS RUN ONLY — never quarantined — because a
# batch-level exception says nothing about any individual ticker.
MIN_SPLIT_BATCH = 25
BATCH_RETRY_SLEEP = 5.0
HISTORY_DAYS = 420
SPY_FETCH_RETRIES = 3
SPY_FETCH_RETRY_DELAY = 5.0
# How stale a cached SPY series may be before it's useless for relative strength.
SPY_STALE_CACHE_MAX_HOURS = 240
FAILED_TICKER_TTL_DAYS = 7


class BatchFetchError(RuntimeError):
    """yfinance raised for a whole batch. Distinct from 'ticker had no rows'."""


def _fetch_batch_yfinance(tickers: list[str], start: str | None = None, end: str | None = None) -> dict[str, pd.DataFrame]:
    """Batched yfinance download. Defaults to the rolling 420-day window used by
    the live screener; pass start/end for an explicit historical range (e.g.
    multi-year backtests) instead.

    Raises BatchFetchError when yfinance itself fails. Returns {} only when
    yfinance answered and had no rows for any ticker — the two cases used to be
    conflated, and on 2026-08-24 one exception quarantined 4,344 tickers.
    """
    joined = " ".join(tickers)
    try:
        if start is not None:
            raw = yf.download(
                joined, start=start, end=end, interval="1d",
                auto_adjust=True, progress=False, group_by="ticker", threads=True,
            )
        else:
            raw = yf.download(
                joined,
                period="420d",
                interval="1d",
                auto_adjust=True,
                progress=False,
                group_by="ticker",
                threads=True,
            )
    except Exception as e:
        logger.warning(f"yfinance batch failed: {e}")
        raise BatchFetchError(str(e)) from e

    if raw is None or raw.empty:
        return {}

    result = {}
    # yfinance 1.x always returns MultiIndex columns (Price, Ticker)
    if isinstance(raw.columns, pd.MultiIndex):
        l0 = set(raw.columns.get_level_values(0))
        l1 = set(raw.columns.get_level_values(1))
        ticker_set = {t.upper() for t in tickers}
        if ticker_set & l1:
            # (Price, Ticker) format
            for t in tickers:
                tu = t.upper()
                cols = [(p, tu) for p in l0 if (p, tu) in raw.columns]
                if not cols:
                    continue
                df = raw[[c for c in raw.columns if c[1] == tu]].copy()
                df.columns = [c[0].lower() for c in df.columns]
                df = df.dropna(how="all")
                if len(df) > 0:
                    result[t] = df
        else:
            # (Ticker, Price) format
            for t in tickers:
                tu = t.upper()
                if tu not in l0:
                    continue
                df = raw[tu].copy()
                df.columns = [c.lower() for c in df.columns]
                df = df.dropna(how="all")
                if len(df) > 0:
                    result[t] = df
    else:
        # Flat columns — single ticker fallback
        t = tickers[0]
        df = raw.copy()
        df.columns = [c.lower() for c in df.columns]
        df = df.dropna(how="all")
        if len(df) > 0:
            result[t] = df
    return result


def fetch_with_split(tickers: list[str], fetch=_fetch_batch_yfinance,
                     min_batch: int = MIN_SPLIT_BATCH, sleep=time.sleep) -> tuple[dict[str, pd.DataFrame], list[str], list[str]]:
    """Fetch a batch, halving and retrying on BatchFetchError.

    Returns (fetched, no_data, unfetched):
      fetched   — ticker -> OHLCV
      no_data   — yfinance answered and had no rows for these (quarantine-worthy)
      unfetched — every retry raised; skip this run, do NOT quarantine
    """
    try:
        got = fetch(tickers)
    except BatchFetchError:
        if len(tickers) <= min_batch:
            return {}, [], list(tickers)
        sleep(BATCH_RETRY_SLEEP)
        mid = len(tickers) // 2
        left = fetch_with_split(tickers[:mid], fetch, min_batch, sleep)
        right = fetch_with_split(tickers[mid:], fetch, min_batch, sleep)
        return ({**left[0], **right[0]}, left[1] + right[1], left[2] + right[2])
    no_data = [t for t in tickers if t not in got]
    return got, no_data, []


def compute_price_factors(
    ticker: str,
    df: pd.DataFrame,
    spy_df: pd.DataFrame,
    market_cap: Optional[float] = None,
) -> Optional[dict]:
    """Price-derived factors for one ticker. `market_cap` is passed through
    when the caller already knows it; the live pipeline leaves it None and
    attaches caps afterwards (src.finnhub_data.attach_market_caps)."""
    if len(df) < 252:
        return None
    close  = df["close"]
    volume = df["volume"]
    spy_close = spy_df["close"]

    try:
        high   = df["high"]
        low    = df["low"]

        price_now   = float(close.iloc[-1])
        sma20       = float(close.iloc[-20:].mean())
        sma50       = float(close.iloc[-50:].mean())
        sma200      = float(close.iloc[-200:].mean())
        rsi_val     = rsi_14(close)
        macd        = macd_state(close)
        vol_ratio   = vol_surge_ratio(volume)
        above_sma20 = price_now >= sma20

        stoch_k, stoch_d, stoch_cross = stochastic(high, low, close)
        bb_pct_b, bb_width            = bollinger_pct_b(close)
        adx_val                       = adx_14(high, low, close)
        mfi_val                       = mfi_14(high, low, close, volume)
        obv_slope_val                 = obv_slope(close, volume)
        _, sar_bullish, _             = parabolic_sar(high, low)
        atr_val                       = atr_14(high, low, close)

        return {
            "ticker": ticker,
            "market_cap": market_cap,
            "mom_12_1": mom_12_1(close),
            "mom_1m": mom_1m(close),
            "residual_mom": residual_momentum(close, spy_close),
            "rs_6m": rs_vs_spy(close, spy_close, window=126),
            "rs_3m": rs_vs_spy(close, spy_close, window=63),
            "rs_slope": rs_slope(close, spy_close),
            "pct_from_high": pct_from_52w_high(close),
            "breakout": breakout_flag(close, volume),
            "avg_dollar_vol_20d": avg_dollar_vol(close, volume, window=20),
            "price": price_now,
            "sma_20": sma20,
            "sma_50": sma50,
            "sma_200": sma200,
            "above_sma20": above_sma20,
            "above_sma50": price_now >= sma50,
            "atr_14": atr_val,
            "rsi_14": rsi_val,
            "macd": macd,
            "vol_surge": vol_ratio,
            "stoch_k": stoch_k,
            "stoch_d": stoch_d,
            "stoch_cross": stoch_cross,
            "bb_pct_b": bb_pct_b,
            "bb_width": bb_width,
            "adx": adx_val,
            "mfi": mfi_val,
            "entry": entry_grade(
                rsi_val, macd, vol_ratio, above_sma20,
                adx_val, stoch_k, stoch_d, stoch_cross, bb_pct_b, mfi_val,
                obv_slope_val, sar_bullish,
            ),
            "close_series": close,
        }
    except Exception as e:
        logger.warning(f"[prices] factor error for {ticker}: {e}")
        return None


def apply_adv_gate(factors_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Dollar-volume gate only. Runs BEFORE market caps are fetched so the cap
    lookups are spent on liquid names."""
    min_vol = cfg["liquidity_gate"]["min_avg_dollar_vol_20d"]
    min_price = float(cfg.get("universe", {}).get("min_price", 0.0) or 0.0)
    before = len(factors_df)
    if before == 0:
        return factors_df
    mask = factors_df["avg_dollar_vol_20d"] >= min_vol
    if min_price > 0:
        mask &= factors_df["price"] >= min_price
    result = factors_df[mask].reset_index(drop=True)
    logger.info(f"[adv_gate] {before} → {len(result)} survivors")
    print(f"[adv_gate] {before} in → {len(result)} survivors (ADV ≥ ${min_vol:,.0f}, price ≥ ${min_price:g})")
    return result


def apply_liquidity_gate(factors_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Market-cap + dollar-volume gate. Names with a NaN cap fail (NaN >= x is
    False) — callers must count them separately (see attach_market_caps)."""
    gate = cfg["liquidity_gate"]
    min_mcap = gate["min_market_cap"]
    min_vol  = gate["min_avg_dollar_vol_20d"]
    before = len(factors_df)
    if before == 0 or "market_cap" not in factors_df.columns:
        return factors_df.iloc[0:0] if before else factors_df
    result = factors_df[
        (factors_df["market_cap"] >= min_mcap) &
        (factors_df["avg_dollar_vol_20d"] >= min_vol)
    ].reset_index(drop=True)
    logger.info(f"[liquidity_gate] {before} → {len(result)} survivors")
    print(f"[liquidity_gate] {before} in → {len(result)} survivors")
    return result


def fetch_all_prices(
    universe_df: pd.DataFrame,
    cfg: dict,
    db_path: str,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict]:
    """OHLCV for the universe plus price factors for every name with ≥252 bars.

    Returns (price_store, factors_df, stats). factors_df has NOT been through
    the cap gate — run.py attaches caps from Finnhub and gates afterwards.
    stats: universe, from_cache, fetched, no_data (quarantined this run),
    unfetched (batch errors; skipped, not quarantined), priced (factor rows).
    """
    ttl = cfg["cache"]["price_ttl_hours"]
    tickers = universe_df["ticker"].tolist()

    spy_cached = get_prices(db_path, "SPY", ttl_hours=ttl)
    if spy_cached is not None and len(spy_cached) >= 252:
        spy_df = spy_cached
    else:
        spy_df = pd.DataFrame()
        for attempt in range(SPY_FETCH_RETRIES):
            try:
                spy_data = _fetch_batch_yfinance(["SPY"])
            except BatchFetchError:
                spy_data = {}
            spy_df = spy_data.get("SPY", pd.DataFrame())
            if not spy_df.empty:
                break
            if attempt < SPY_FETCH_RETRIES - 1:
                logger.warning(f"[prices] SPY fetch attempt {attempt+1}/{SPY_FETCH_RETRIES} failed — retrying")
                time.sleep(SPY_FETCH_RETRY_DELAY)
        if not spy_df.empty:
            put_prices(db_path, "SPY", spy_df)
        else:
            stale = get_prices(db_path, "SPY", ttl_hours=SPY_STALE_CACHE_MAX_HOURS)
            if stale is not None and len(stale) >= 252:
                logger.warning(
                    f"[prices] SPY live fetch failed after retries — using stale cache "
                    f"(last row {stale.index[-1].date()})"
                )
                spy_df = stale
    if spy_df is None or spy_df.empty:
        raise RuntimeError("Failed to fetch SPY — cannot compute relative strength")

    price_store: dict[str, pd.DataFrame] = {"SPY": spy_df}
    factor_rows = []
    stats = {"universe": len(tickers), "from_cache": 0, "fetched": 0,
             "no_data": 0, "unfetched": 0, "skipped_quarantined": 0}
    names = universe_df.set_index("ticker")["name"].to_dict() if "name" in universe_df.columns else {}
    ciks  = universe_df.set_index("ticker")["cik"].to_dict() if "cik" in universe_df.columns else {}

    batches = [tickers[i:i+BATCH_SIZE] for i in range(0, len(tickers), BATCH_SIZE)]
    for batch_idx, batch in enumerate(batches):
        logger.info(f"[prices] batch {batch_idx+1}/{len(batches)} ({len(batch)} tickers)")

        to_fetch = []
        for t in batch:
            if is_failed_ticker(db_path, t, ttl_days=FAILED_TICKER_TTL_DAYS):
                stats["skipped_quarantined"] += 1
                continue
            cached = get_prices(db_path, t, ttl_hours=ttl)
            if cached is not None and len(cached) >= 252:
                price_store[t] = cached
                stats["from_cache"] += 1
            else:
                to_fetch.append(t)

        if to_fetch:
            fetched, no_data, unfetched = fetch_with_split(to_fetch)
            for t, df in fetched.items():
                put_prices(db_path, t, df)
                price_store[t] = df
            stats["fetched"] += len(fetched)
            for t in no_data:
                put_failed_ticker(db_path, t, "no_data")
                logger.info(f"[prices] marked {t} as failed (no data returned)")
            stats["no_data"] += len(no_data)
            if unfetched:
                stats["unfetched"] += len(unfetched)
                logger.warning(f"[prices] batch {batch_idx+1}: {len(unfetched)} tickers unfetched "
                               f"after split retries (not quarantined)")

        for t in batch:
            df = price_store.get(t)
            if df is None or len(df) < 252:
                continue
            row = compute_price_factors(t, df, spy_df, None)
            if row is not None:
                row["name"] = names.get(t, "") or ""
                row["cik"]  = ciks.get(t, "") or ""
                factor_rows.append(row)

        if batch_idx < len(batches) - 1:
            time.sleep(1.0)

    factors_df = pd.DataFrame([{k: v for k, v in r.items() if k != "close_series"} for r in factor_rows])
    stats["priced"] = len(factors_df)
    print(f"[prices] {len(tickers)} universe → {len(factors_df)} priced "
          f"(cache {stats['from_cache']}, fetched {stats['fetched']}, no_data {stats['no_data']}, "
          f"unfetched {stats['unfetched']}, quarantined {stats['skipped_quarantined']})")
    return price_store, factors_df, stats
```

- [ ] **Step 4:** Run:

```bash
/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m pytest tests/test_prices.py -q
```

Expected: 13 passed.

- [ ] **Step 5:** Commit as `fix(prices): never quarantine on a batch exception; split-and-retry; ADV gate before cap lookups`. Per the user's global CLAUDE.md §6: run `git status` first and **ask the user before committing**. Never add a Co-Authored-By line.

**Edge cases this task must handle (each has a test):**
- Batch exception → halves → succeeds: nothing quarantined; calls are 8, 4, 4.
- Persistent exception → all tickers `unfetched`, none `no_data`.
- Mixed: failing half unfetched, dead ticker in the good half is `no_data`.
- `apply_adv_gate` applies `universe.min_price`; empty frame returns empty.
- `apply_liquidity_gate` fails NaN caps and returns empty when the column is missing (the old KeyError path).


### Task 0.5: Fundamentals loop — drop per-survivor yfinance calls; extended EDGAR parser

**Files:**
- Modify: `src/fundamentals.py`
- Modify: `tests/test_fundamentals.py`

The survivors loop called `yf.Ticker(t).get_info()` per name for sector and short interest — the rate-limited call that blanked `sector`. Sector now arrives from Finnhub before this stage; short interest moves to the finalists (Phase 1). This diff also contains the Phase 1 EDGAR parser (`parse_edgar_facts`, recency-based tag resolution, duration filter) because both live in one file — it is safe to land now; nothing consumes the new columns until Phase 1.

- [ ] **Step 1:** Update tests:

**Apply this unified diff to `tests/test_fundamentals.py`** (from the reference patch; `patch -p1 < docs/superpowers/plans/2026-09-04-overhaul-reference.patch` applies all of them at once):

```diff
--- a/tests/test_fundamentals.py	2026-06-24 09:25:33
+++ b/tests/test_fundamentals.py	2026-09-04 12:37:40
@@ -14,18 +14,18 @@
         "us-gaap": {
             "Revenues": {
                 "units": {"USD": [
-                    {"end": "2023-09-30", "val": 383285000000, "form": "10-K", "filed": "2023-11-03"},
-                    {"end": "2022-09-24", "val": 394328000000, "form": "10-K", "filed": "2022-10-28"},
+                    {"end": "2025-09-27", "val": 383285000000, "form": "10-K", "filed": "2025-11-03"},
+                    {"end": "2024-09-28", "val": 394328000000, "form": "10-K", "filed": "2024-10-28"},
                 ]}
             },
             "CostOfGoodsAndServicesSold": {
                 "units": {"USD": [
-                    {"end": "2023-09-30", "val": 214137000000, "form": "10-K", "filed": "2023-11-03"},
+                    {"end": "2025-09-27", "val": 214137000000, "form": "10-K", "filed": "2025-11-03"},
                 ]}
             },
             "Assets": {
                 "units": {"USD": [
-                    {"end": "2023-09-30", "val": 352583000000, "form": "10-K", "filed": "2023-11-03"},
+                    {"end": "2025-09-27", "val": 352583000000, "form": "10-K", "filed": "2025-11-03"},
                 ]}
             },
         }
@@ -75,3 +75,133 @@
     assert result["insider_buys_90d"] == 2   # 2 distinct insiders with purchases
     assert result["exec_buys_90d"] == 2      # both are executives (CEO + CFO)
     assert result["insider_buy_value"] > 0   # dollar value captured
+
+
+# ── extended EDGAR parser ─────────────────────────────────────────────────────
+from src.fundamentals import parse_edgar_facts, _annual_series, _latest_two
+
+
+def _dur(start, end, val, form="10-K", filed=None):
+    return {"start": start, "end": end, "val": val, "form": form, "filed": filed or end}
+
+
+def _inst(end, val, form="10-K", filed=None):
+    return {"end": end, "val": val, "form": form, "filed": filed or end}
+
+
+FULL_EDGAR = {"facts": {"us-gaap": {
+    "Revenues": {"units": {"USD": [
+        _dur("2025-01-01", "2025-12-31", 1000.0),
+        _dur("2025-10-01", "2025-12-31", 300.0),        # Q4 stub inside the 10-K — must be ignored
+        _dur("2024-01-01", "2024-12-31", 800.0),
+        _dur("2025-01-01", "2025-03-31", 200.0, form="10-Q"),
+    ]}},
+    "CostOfRevenue": {"units": {"USD": [_dur("2025-01-01", "2025-12-31", 400.0),
+                                         _dur("2024-01-01", "2024-12-31", 350.0)]}},
+    "NetIncomeLoss": {"units": {"USD": [_dur("2025-01-01", "2025-12-31", 120.0),
+                                         _dur("2025-01-01", "2025-12-31", 118.0, filed="2025-02-01"),  # earlier filing, superseded
+                                         _dur("2024-01-01", "2024-12-31", 90.0)]}},
+    "NetCashProvidedByUsedInOperatingActivities": {"units": {"USD": [_dur("2025-01-01", "2025-12-31", 150.0)]}},
+    "Assets": {"units": {"USD": [_inst("2025-12-31", 2000.0), _inst("2024-12-31", 1600.0),
+                                  _inst("2025-06-30", 1900.0, form="10-Q")]}},
+    "Liabilities": {"units": {"USD": [_inst("2025-12-31", 1200.0)]}},
+    "StockholdersEquity": {"units": {"USD": [_inst("2025-12-31", 800.0)]}},
+    "WeightedAverageNumberOfDilutedSharesOutstanding": {"units": {"shares": [
+        _dur("2025-01-01", "2025-12-31", 110.0), _dur("2024-01-01", "2024-12-31", 100.0)]}},
+}}}
+
+
+def test_annual_series_filters_forms_durations_and_dedupes_by_latest_filing():
+    ni = FULL_EDGAR["facts"]["us-gaap"]["NetIncomeLoss"]["units"]["USD"]
+    assert _annual_series(ni, duration=True) == [("2025-12-31", 120.0), ("2024-12-31", 90.0)]
+    rev = FULL_EDGAR["facts"]["us-gaap"]["Revenues"]["units"]["USD"]
+    assert _annual_series(rev, duration=True) == [("2025-12-31", 1000.0), ("2024-12-31", 800.0)]
+    assets = FULL_EDGAR["facts"]["us-gaap"]["Assets"]["units"]["USD"]
+    assert _annual_series(assets, duration=False) == [("2025-12-31", 2000.0), ("2024-12-31", 1600.0)]
+
+
+def test_latest_two_requires_a_real_prior_year():
+    assert _latest_two([("2024-12-31", 1.0), ("2023-12-31", 2.0)]) == (1.0, 2.0)
+    assert _latest_two([("2024-12-31", 1.0), ("2024-06-30", 2.0)]) == (1.0, None)   # stub, not a year
+    assert _latest_two([("2024-12-31", 1.0)]) == (1.0, None)
+    assert _latest_two([]) == (None, None)
+
+
+def test_parse_edgar_facts_derives_ratios():
+    f = parse_edgar_facts(FULL_EDGAR)
+    assert f["gp_assets"] == pytest.approx((1000 - 400) / 2000)
+    assert f["accruals"] == pytest.approx((120 - 150) / 2000)
+    assert f["asset_growth"] == pytest.approx(2000 / 1600 - 1)
+    assert f["net_issuance"] == pytest.approx(0.10)
+    assert f["roa"] == pytest.approx(120 / 2000)
+    assert f["leverage"] == pytest.approx(0.6)
+    assert f["n_resolved"] == 10
+
+
+def test_parse_edgar_facts_partial_data_is_nan_not_error():
+    only_assets = {"facts": {"us-gaap": {"Assets": {"units": {"USD": [_inst("2025-12-31", 2000.0)]}}}}}
+    f = parse_edgar_facts(only_assets)
+    assert f["assets"] == 2000.0 and f["n_resolved"] == 1
+    for k in ("gp_assets", "accruals", "asset_growth", "net_issuance", "roa", "leverage"):
+        assert f[k] != f[k]   # NaN
+
+
+def test_parse_edgar_facts_no_usgaap_returns_zero_resolved():
+    assert parse_edgar_facts({"facts": {"ifrs-full": {}}})["n_resolved"] == 0
+    assert parse_edgar_facts({})["n_resolved"] == 0
+
+
+def test_parse_edgar_gp_ignores_q4_stub_with_same_end_date():
+    gp, rev, cogs, assets = parse_edgar_gp(FULL_EDGAR)
+    assert rev == 1000.0 and cogs == 400.0 and assets == 2000.0
+
+
+def test_edgar_facts_cache_roundtrip_preserves_nan(tmp_path):
+    from src.cache import init_db, put_edgar_facts, get_edgar_facts
+    db = str(tmp_path / "c.db"); init_db(db)
+    put_edgar_facts(db, "0000000001", {"gp_assets": 0.3, "accruals": float("nan"), "n_resolved": 3})
+    back = get_edgar_facts(db, "0000000001", ttl_days=30)
+    assert back["gp_assets"] == 0.3 and back["accruals"] != back["accruals"] and back["n_resolved"] == 3
+    assert get_edgar_facts(db, "0000000001", ttl_days=0) is None
+
+
+def test_resolve_pair_prefers_most_recent_tag_not_list_order():
+    """MSFT: `Revenues` stopped in FY2010, the newer tag carries 2026. Priority
+    order used to return the 2010 number."""
+    from src.fundamentals import _resolve_pair
+    from datetime import date
+    facts = {"us-gaap": {
+        "Revenues": {"units": {"USD": [_dur("2009-07-01", "2010-06-30", 62.0)]}},
+        "RevenueFromContractWithCustomerExcludingAssessedTax": {"units": {"USD": [
+            _dur("2025-07-01", "2026-06-30", 331.0), _dur("2024-07-01", "2025-06-30", 281.0)]}},
+    }}
+    latest, prev = _resolve_pair(facts, ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax"],
+                                 duration=True, today=date(2026, 9, 4))
+    assert (latest, prev) == (331.0, 281.0)
+
+
+def test_resolve_pair_tie_on_end_takes_the_larger_total():
+    from src.fundamentals import _resolve_pair
+    from datetime import date
+    facts = {"us-gaap": {
+        "CostOfGoodsSold": {"units": {"USD": [_dur("2025-01-01", "2025-12-31", 15.0)]}},
+        "CostOfRevenue": {"units": {"USD": [_dur("2025-01-01", "2025-12-31", 34.0)]}},
+    }}
+    latest, _ = _resolve_pair(facts, ["CostOfGoodsSold", "CostOfRevenue"], duration=True, today=date(2026, 9, 4))
+    assert latest == 34.0
+
+
+def test_resolve_pair_discards_stale_filers():
+    from src.fundamentals import _resolve_pair
+    from datetime import date
+    facts = {"us-gaap": {"Assets": {"units": {"USD": [_inst("2023-12-31", 100.0)]}}}}
+    assert _resolve_pair(facts, ["Assets"], duration=False, today=date(2026, 9, 4)) == (None, None)
+    assert _resolve_pair(facts, ["Assets"], duration=False, today=date(2024, 9, 4)) == (100.0, None)
+
+
+def test_parse_edgar_facts_gross_profit_fallback_when_no_cogs():
+    facts = {"facts": {"us-gaap": {
+        "GrossProfit": {"units": {"USD": [_dur("2025-01-01", "2025-12-31", 600.0)]}},
+        "Assets": {"units": {"USD": [_inst("2025-12-31", 2000.0)]}},
+    }}}
+    assert parse_edgar_facts(facts)["gp_assets"] == pytest.approx(0.3)
```

- [ ] **Step 2:** Run:

```bash
/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m pytest tests/test_fundamentals.py -q
```

Expected: FAIL: `parse_edgar_facts`, `_annual_series`, `_latest_two`, `_resolve_pair` not importable.

- [ ] **Step 3:** Apply:

**Apply this unified diff to `src/fundamentals.py`** (from the reference patch; `patch -p1 < docs/superpowers/plans/2026-09-04-overhaul-reference.patch` applies all of them at once):

```diff
--- a/src/fundamentals.py	2026-07-06 00:13:10
+++ b/src/fundamentals.py	2026-09-04 12:37:40
@@ -2,12 +2,12 @@
 import logging
 import requests
 import pandas as pd
-from datetime import datetime, timedelta
+from datetime import date, datetime, timedelta
 from typing import Optional
 import finnhub
-import yfinance as yf
 from src.config import get_env
-from src.cache import get_fundamentals, put_fundamentals, get_edgar, put_edgar
+from src.cache import (get_fundamentals, put_fundamentals, get_edgar, put_edgar,
+                       get_edgar_facts, put_edgar_facts)
 from src.factors import compute_sue
 
 logger = logging.getLogger(__name__)
@@ -27,36 +27,138 @@
     "CostOfSales",
 ]
 ASSETS_TAGS = ["Assets"]
+NET_INCOME_TAGS = ["NetIncomeLoss", "ProfitLoss", "NetIncomeLossAvailableToCommonStockholdersBasic"]
+OCF_TAGS = ["NetCashProvidedByUsedInOperatingActivities",
+            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"]
+LIABILITIES_TAGS = ["Liabilities"]
+EQUITY_TAGS = ["StockholdersEquity",
+               "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"]
+SHARES_DURATION_TAGS = ["WeightedAverageNumberOfDilutedSharesOutstanding",
+                        "WeightedAverageNumberOfSharesOutstandingBasic"]
+SHARES_INSTANT_TAGS = ["CommonStockSharesOutstanding"]
+GROSS_PROFIT_TAGS = ["GrossProfit"]
 
+ANNUAL_FORMS = ("10-K", "20-F", "40-F")
+# A 10-K reports the fiscal year (≈365 days) and often the fourth quarter
+# (≈90 days) under the SAME end date. Flow items must be filtered to the
+# annual duration or a quarterly value can be read as the year's.
+ANNUAL_DURATION_DAYS = (340, 390)
+# Two consecutive fiscal years are ~365 days apart; allow for 52/53-week years.
+YEAR_GAP_DAYS = (300, 430)
 
+
 def _sec_headers() -> dict:
     return {"User-Agent": get_env("SEC_USER_AGENT")}
 
 
-def _latest_annual(entries: list) -> Optional[float]:
-    annual = [e for e in entries if e.get("form") in ("10-K", "20-F")]
-    if not annual:
+def _is_annual_form(form: str) -> bool:
+    return any(str(form or "").startswith(f) for f in ANNUAL_FORMS)
+
+
+def _days_between(a: str, b: str) -> int | None:
+    try:
+        return (datetime.strptime(b, "%Y-%m-%d") - datetime.strptime(a, "%Y-%m-%d")).days
+    except (TypeError, ValueError):
         return None
-    annual.sort(key=lambda e: e.get("end", ""), reverse=True)
-    return float(annual[0]["val"])
 
 
-def _resolve_tag(facts: dict, tag_list: list) -> Optional[float]:
+def _annual_series(entries: list, duration: bool) -> list[tuple[str, float]]:
+    """[(end, val)] newest first, one value per fiscal-year end, from annual
+    filings only. `duration=True` keeps only ≈12-month periods (flow items);
+    False is for balance-sheet instants (no `start`)."""
+    best: dict[str, tuple[str, float]] = {}   # end -> (filed, val)
+    for e in entries or []:
+        if not _is_annual_form(e.get("form")):
+            continue
+        end = e.get("end")
+        if not end or e.get("val") is None:
+            continue
+        if duration:
+            start = e.get("start")
+            if start:
+                d = _days_between(start, end)
+                if d is None or not (ANNUAL_DURATION_DAYS[0] <= d <= ANNUAL_DURATION_DAYS[1]):
+                    continue
+            # No `start` on a flow item: cannot check the duration, keep it.
+        elif e.get("start"):
+            continue
+        filed = e.get("filed", "")
+        if end not in best or filed >= best[end][0]:
+            best[end] = (filed, float(e["val"]))
+    return sorted(((end, v) for end, (_, v) in best.items()), key=lambda x: x[0], reverse=True)
+
+
+def _latest_two(series: list[tuple[str, float]]) -> tuple[float | None, float | None]:
+    """(latest, one year earlier) — the earlier value must be a real prior
+    fiscal year, not a restated duplicate or a stub period."""
+    if not series:
+        return None, None
+    latest_end, latest_val = series[0]
+    prev_val = None
+    for end, val in series[1:]:
+        gap = _days_between(end, latest_end)
+        if gap is not None and YEAR_GAP_DAYS[0] <= gap <= YEAR_GAP_DAYS[1]:
+            prev_val = val
+            break
+    return latest_val, prev_val
+
+
+# A filer whose newest annual value is older than this is not reporting (or
+# has moved to a tag we do not know); its facts are treated as missing rather
+# than compared against fresh prices and caps.
+MAX_FACT_AGE_DAYS = 550
+
+
+def _resolve_pair(facts: dict, tag_list: list, duration: bool, units: str = "USD",
+                  today: date | None = None) -> tuple[float | None, float | None]:
+    """(latest, prior-year) for the concept, choosing among synonym tags by
+    RECENCY of the latest fiscal year, not by list order.
+
+    Tag priority was the old rule and it is wrong: Microsoft last used
+    `Revenues` in FY2010 and `RevenueFromContractWithCustomer…` since, so
+    priority returned 2010 revenue against 2026 assets and a negative gross
+    profitability. Ties on end date go to the larger value (a total, not a
+    component). Anything older than MAX_FACT_AGE_DAYS is discarded.
+    """
     usgaap = facts.get("us-gaap", {})
+    today = today or date.today()
+    best: tuple[str, float, float | None] | None = None   # (end, latest, prev)
     for tag in tag_list:
-        if tag in usgaap:
-            entries = usgaap[tag].get("units", {}).get("USD", [])
-            val = _latest_annual(entries)
-            if val is not None:
-                return val
-    return None
+        if tag not in usgaap:
+            continue
+        series = _annual_series(usgaap[tag].get("units", {}).get(units, []), duration)
+        if not series:
+            continue
+        end = series[0][0]
+        age = _days_between(end, today.isoformat())
+        if age is None or age > MAX_FACT_AGE_DAYS:
+            continue
+        latest, prev = _latest_two(series)
+        if best is None or end > best[0] or (end == best[0] and latest > best[1]):
+            best = (end, latest, prev)
+    return (best[1], best[2]) if best else (None, None)
 
 
+def _latest_annual(entries: list) -> Optional[float]:
+    """Kept for callers/tests of the old API: latest annual FLOW value."""
+    latest, _ = _latest_two(_annual_series(entries, duration=True))
+    if latest is None:
+        latest, _ = _latest_two(_annual_series(entries, duration=False))
+    return latest
+
+
+def _resolve_tag(facts: dict, tag_list: list) -> Optional[float]:
+    latest, _ = _resolve_pair(facts, tag_list, duration=True)
+    if latest is None:
+        latest, _ = _resolve_pair(facts, tag_list, duration=False)
+    return latest
+
+
 def parse_edgar_gp(data: dict) -> tuple[float, float, float, float]:
     facts = data["facts"]
-    revenue = _resolve_tag(facts, REVENUE_TAGS)
-    cogs    = _resolve_tag(facts, COGS_TAGS)
-    assets  = _resolve_tag(facts, ASSETS_TAGS)
+    revenue, _ = _resolve_pair(facts, REVENUE_TAGS, duration=True)
+    cogs, _    = _resolve_pair(facts, COGS_TAGS, duration=True)
+    assets, _  = _resolve_pair(facts, ASSETS_TAGS, duration=False)
     if revenue is None or cogs is None or assets is None:
         raise KeyError("Could not resolve revenue/cogs/assets XBRL tags")
     if assets <= 0:
@@ -65,8 +167,69 @@
     return gp, revenue, cogs, assets
 
 
+def _safe_div(a, b) -> float:
+    if a is None or b is None or b == 0:
+        return float("nan")
+    return float(a) / float(b)
+
+
+def parse_edgar_facts(data: dict) -> dict:
+    """Everything the fundamentals block wants from one companyfacts JSON.
+
+    Returns raw values plus derived ratios; any item that cannot be resolved
+    is NaN (never raises — a filer with no COGS tag still has accruals). A
+    filer with no usable us-gaap facts at all returns a dict of NaNs plus
+    "n_resolved": 0, which callers treat as "no EDGAR data".
+    """
+    facts = data.get("facts", {}) or {}
+    revenue, _            = _resolve_pair(facts, REVENUE_TAGS, duration=True)
+    cogs, _               = _resolve_pair(facts, COGS_TAGS, duration=True)
+    assets, assets_prev   = _resolve_pair(facts, ASSETS_TAGS, duration=False)
+    net_income, _         = _resolve_pair(facts, NET_INCOME_TAGS, duration=True)
+    ocf, _                = _resolve_pair(facts, OCF_TAGS, duration=True)
+    liabilities, _        = _resolve_pair(facts, LIABILITIES_TAGS, duration=False)
+    equity, _             = _resolve_pair(facts, EQUITY_TAGS, duration=False)
+    shares, shares_prev   = _resolve_pair(facts, SHARES_DURATION_TAGS, duration=True, units="shares")
+    if shares is None:
+        shares, shares_prev = _resolve_pair(facts, SHARES_INSTANT_TAGS, duration=False, units="shares")
+
+    gp_assets = float("nan")
+    if revenue is not None and cogs is not None and assets and assets > 0:
+        gp_assets = (revenue - cogs) / assets
+    elif assets and assets > 0:
+        gross, _ = _resolve_pair(facts, GROSS_PROFIT_TAGS, duration=True)
+        if gross is not None:
+            gp_assets = gross / assets
+    accruals = float("nan")
+    if net_income is not None and ocf is not None and assets and assets > 0:
+        accruals = (net_income - ocf) / assets
+    asset_growth = float("nan")
+    if assets and assets_prev and assets_prev > 0:
+        asset_growth = assets / assets_prev - 1.0
+    net_issuance = float("nan")
+    if shares and shares_prev and shares_prev > 0:
+        net_issuance = shares / shares_prev - 1.0
+    roa = _safe_div(net_income, assets) if assets and assets > 0 else float("nan")
+    leverage = _safe_div(liabilities, assets) if assets and assets > 0 else float("nan")
+
+    raw = {"revenue": revenue, "cogs": cogs, "assets": assets, "assets_prev": assets_prev,
+           "net_income": net_income, "ocf": ocf, "liabilities": liabilities, "equity": equity,
+           "shares": shares, "shares_prev": shares_prev}
+    n_resolved = sum(1 for v in raw.values() if v is not None)
+    return {**{k: (float("nan") if v is None else float(v)) for k, v in raw.items()},
+            "gp_assets": gp_assets, "accruals": accruals, "asset_growth": asset_growth,
+            "net_issuance": net_issuance, "roa": roa, "leverage": leverage,
+            "n_resolved": n_resolved}
+
+
+EDGAR_FACT_COLUMNS = ["gp_assets", "accruals", "asset_growth", "net_issuance", "roa", "leverage"]
+
+
 def fetch_edgar(cik: str, db_path: str, ttl_days: int) -> Optional[dict]:
-    cached = get_edgar(db_path, cik, ttl_days=ttl_days)
+    """Extended EDGAR facts for one CIK, cached in `edgar_facts` (payload JSON).
+    Returns None when the fetch fails or nothing resolves. Also refreshes the
+    legacy `edgar` row so older readers keep working."""
+    cached = get_edgar_facts(db_path, cik, ttl_days=ttl_days)
     if cached is not None:
         return cached
     url = f"{EDGAR_BASE}/CIK{cik}.json"
@@ -74,12 +237,19 @@
         resp = requests.get(url, headers=_sec_headers(), timeout=30)
         resp.raise_for_status()
         data = resp.json()
-        gp, rev, cogs, assets = parse_edgar_gp(data)
-        put_edgar(db_path, cik, gp_assets=gp, revenue=rev, cogs=cogs, assets=assets)
-        return {"gp_assets": gp, "revenue": rev, "cogs": cogs, "assets": assets}
     except Exception as e:
         logger.warning(f"[edgar] failed for CIK {cik}: {e}")
         return None
+    facts = parse_edgar_facts(data)
+    if facts["n_resolved"] == 0:
+        # Cache the miss too (IFRS filers, shells) so it is not re-requested daily.
+        put_edgar_facts(db_path, cik, facts)
+        return None
+    put_edgar_facts(db_path, cik, facts)
+    if facts["gp_assets"] == facts["gp_assets"]:   # not NaN
+        put_edgar(db_path, cik, gp_assets=facts["gp_assets"], revenue=facts["revenue"],
+                  cogs=facts["cogs"], assets=facts["assets"])
+    return facts
 
 
 def parse_finnhub_surprise(earnings: list) -> tuple[list, list]:
@@ -213,24 +383,15 @@
 
         row = {"ticker": ticker}
 
-        if cik:
-            edgar = fetch_edgar(cik, db_path, ttl_days=ttl_edgar)
-            row["gp_assets"] = edgar["gp_assets"] if edgar else float("nan")
-        else:
-            row["gp_assets"] = float("nan")
+        edgar = fetch_edgar(cik, db_path, ttl_days=ttl_edgar) if cik else None
+        for col in EDGAR_FACT_COLUMNS:
+            row[col] = edgar[col] if edgar and edgar.get("n_resolved", 0) > 0 else float("nan")
 
         cached_fund = get_fundamentals(db_path, ticker, ttl_days=ttl_fund)
         if cached_fund:
-            # Back-fill empty sector in stale cache entries without full re-fetch
-            if not cached_fund.get("sector"):
-                try:
-                    info = yf.Ticker(ticker).get_info()
-                    sector = (info.get("sector") or info.get("industry") or "").strip()
-                    if sector:
-                        cached_fund["sector"] = sector
-                        put_fundamentals(db_path, ticker, cached_fund)
-                except Exception:
-                    pass
+            # Sector now comes from Finnhub (src.finnhub_data) and is already
+            # on the survivors frame; an old cached payload must not shadow it.
+            cached_fund.pop("sector", None)
             row.update(cached_fund)
         else:
             fund = {}
@@ -278,20 +439,12 @@
                 fund["insider_buy_value"] = 0.0
                 fund["insider_flag"]      = False
 
-            try:
-                # Use get_info() not .info property — the property bypasses
-                # yfinance's cookie/crumb handling with custom sessions,
-                # causing silent rate-limit failures that blank sector.
-                info = yf.Ticker(ticker).get_info()
-                sf, dtc = parse_short_interest(info)
-                fund["short_float"]   = sf
-                fund["days_to_cover"] = dtc
-                sector = (info.get("sector") or info.get("industry") or "").strip()
-                fund["sector"] = sector
-            except Exception:
-                fund["short_float"]   = float("nan")
-                fund["days_to_cover"] = float("nan")
-                fund["sector"]        = ""
+            # Short interest used to come from a per-ticker yfinance get_info()
+            # call here — one call per survivor, rate-limited at universe scale,
+            # and the same call that silently blanked `sector`. It now happens
+            # only for the finalists (src.enrich, Phase 1). Left NaN here.
+            fund["short_float"]   = float("nan")
+            fund["days_to_cover"] = float("nan")
 
             put_fundamentals(db_path, ticker, fund)
             row.update(fund)
```

- [ ] **Step 4:** Run:

```bash
/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m pytest tests/test_fundamentals.py -q
```

Expected: 17 passed.

- [ ] **Step 5:** Live check of the parser (fixes a production bug — MSFT's `Revenues` tag stopped in FY2010 and tag priority returned it):

```bash
set -a; source .env; set +a
/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 - <<'EOF'
import requests, os
from src.fundamentals import parse_edgar_facts
h={{'User-Agent': os.environ['SEC_USER_AGENT']}}
d=requests.get('https://data.sec.gov/api/xbrl/companyfacts/CIK0000789019.json',headers=h,timeout=30).json()
f=parse_edgar_facts(d); print({{k: f[k] for k in ('revenue','gp_assets','accruals','asset_growth','net_issuance','roa','n_resolved')}})
EOF
```

Expected: revenue ≈ 3.3e11 (not 6.2e10), gp_assets ≈ 0.30 (not negative), n_resolved 10.

- [ ] **Step 6:** Commit as `fix(fundamentals): resolve EDGAR tags by recency, filter to annual durations; drop per-survivor yfinance calls`. Per the user's global CLAUDE.md §6: run `git status` first and **ask the user before committing**. Never add a Co-Authored-By line.

**Edge cases this task must handle (each has a test):**
- A 10-K's Q4 stub with the same `end` as the fiscal year is ignored (duration filter); entries without `start` are kept.
- Synonym tags chosen by latest fiscal-year end; ties go to the larger value; facts older than 550 days discarded.
- Prior-year value must be 300–430 days earlier (not a restatement or stub).
- Partial data → NaN ratios, never an exception; IFRS-only → `n_resolved 0`.
- `GrossProfit` is the fallback when COGS is absent.
- Old cached fundamentals payloads carry a `sector` key — it is popped so it cannot shadow the Finnhub sector on merge.


### Task 0.6: Health floors, `screen_latest.json`, `run_status.stats`, watchdog screen check

**Files:**
- Modify: `src/output.py`
- Modify: `src/run_status.py`
- Modify: `src/watchdog.py`
- Modify: `src/compose.py` (attrs only in this task; the rest is Phase 1)
- Create: `tests/test_pipeline_health.py`
- Modify: `tests/test_watchdog.py`

Makes a collapsed universe impossible to miss: floors checked after each stage, a failed run writes `screen_latest.json` with `health.ok: false` and no rows (the trader reads that, not yesterday's file), `run_status.json` carries the stage counts, and the watchdog fails on a failed run or breached floor.

- [ ] **Step 1:** **Create `tests/test_pipeline_health.py` with exactly this content:**

```python
"""Health floors, stats file, and the machine-readable screen_latest.json."""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import run as run_mod
from src.output import write_latest_json
from src.run_status import build_status, load_stats

CFG = {"health": {"min_adv_survivors": 800, "min_cap_survivors": 500, "min_ranked": 150}}


def test_evaluate_health_passes_when_all_floors_met():
    ok, reasons = run_mod.evaluate_health({"adv_survivors": 2400, "cap_survivors": 1900, "ranked": 600}, CFG)
    assert ok and reasons == []


def test_evaluate_health_reports_every_breach():
    ok, reasons = run_mod.evaluate_health({"adv_survivors": 53, "cap_survivors": 53, "ranked": 38}, CFG)
    assert not ok
    assert len(reasons) == 3
    assert "adv_survivors=53 below floor 800" in reasons


def test_evaluate_health_only_checks_stages_that_ran():
    """Called after stage 2 the ranked count does not exist yet — that is not a breach."""
    ok, reasons = run_mod.evaluate_health({"adv_survivors": 2400}, CFG)
    assert ok and reasons == []


def test_evaluate_health_without_config_never_fails():
    ok, reasons = run_mod.evaluate_health({"adv_survivors": 1}, {})
    assert ok


def test_gate_raises_and_writes_failed_screen(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_mod, "STATS_PATH", tmp_path / "data" / "last_run_stats.json")
    monkeypatch.setattr(run_mod, "OUTPUT_DIR", str(tmp_path / "output"))
    stats = {"date": "2026-09-04", "adv_survivors": 53}
    with pytest.raises(run_mod.PipelineHealthError) as exc:
        run_mod._gate(stats, CFG, allow_degraded=False, today="2026-09-04")
    assert "adv_survivors=53" in str(exc.value)
    written = json.loads((tmp_path / "data" / "last_run_stats.json").read_text())
    assert written["ok"] is False and written["reasons"]
    latest = json.loads((tmp_path / "output" / "screen_latest.json").read_text())
    assert latest["health"]["ok"] is False and latest["rows"] == []
    # No dated CSV must exist: a failed run publishes no ranking.
    assert not list((tmp_path / "output").glob("screen_2026-*.csv"))


def test_gate_allow_degraded_continues_and_flags(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_mod, "STATS_PATH", tmp_path / "data" / "last_run_stats.json")
    monkeypatch.setattr(run_mod, "OUTPUT_DIR", str(tmp_path / "output"))
    stats = {"date": "2026-09-04", "adv_survivors": 53}
    run_mod._gate(stats, CFG, allow_degraded=True, today="2026-09-04")
    assert stats["ok"] is False and stats["degraded"] is True


def test_gate_ok_leaves_no_files(tmp_path, monkeypatch):
    monkeypatch.setattr(run_mod, "STATS_PATH", tmp_path / "data" / "last_run_stats.json")
    monkeypatch.setattr(run_mod, "OUTPUT_DIR", str(tmp_path / "output"))
    stats = {"date": "2026-09-04", "adv_survivors": 2400}
    run_mod._gate(stats, CFG, allow_degraded=False, today="2026-09-04")
    assert stats["ok"] is True
    assert not (tmp_path / "output").exists()


# ── screen_latest.json ────────────────────────────────────────────────────────

def test_write_latest_json_rows_ranks_and_nan_to_null(tmp_path):
    df = pd.DataFrame({
        "ticker": ["AAA", "BBB"], "name": ["Alpha, Inc.", "Beta"], "sector": ["Tech", "Health"],
        "composite": [1.2, 0.9], "price": [10.0, np.nan], "atr_14": [0.5, 0.7],
        "entry": ["OK", "WAIT"], "conviction": np.array([7, 4], dtype="int64"),
        "stoch_cross": [True, False],
    })
    health = {"ok": True, "adv_survivors": 2400, "reasons": []}
    path = write_latest_json(df, str(tmp_path), "2026-09-04", health=health,
                             regime={"regime": "NORMAL", "scale_factor": 1.0, "reason": "all clear"},
                             ranking_tail=[{"rank": 1, "ticker": "AAA", "composite": 1.2}])
    blob = json.loads(Path(path).read_text())
    assert blob["date"] == "2026-09-04" and blob["health"]["ok"] is True
    assert blob["rows"][0]["rank"] == 1 and blob["rows"][1]["rank"] == 2
    assert blob["rows"][1]["price"] is None            # NaN -> null, never the string "nan"
    assert blob["rows"][0]["conviction"] == 7          # numpy int -> int
    assert blob["rows"][0]["stoch_cross"] is True
    assert blob["rows"][0]["name"] == "Alpha, Inc."    # commas are not a parsing problem in JSON
    assert blob["regime"]["regime"] == "NORMAL"
    assert blob["ranking_tail"][0]["ticker"] == "AAA"
    assert "rank" in blob["columns"]


def test_write_latest_json_empty_frame(tmp_path):
    path = write_latest_json(pd.DataFrame(), str(tmp_path), "2026-09-04",
                             health={"ok": False, "reasons": ["x"]},
                             regime={"regime": "UNKNOWN", "scale_factor": 1.0}, ranking_tail=[])
    blob = json.loads(Path(path).read_text())
    assert blob["rows"] == [] and blob["health"]["ok"] is False


def test_write_latest_json_overwrites_atomically(tmp_path):
    for i in range(2):
        write_latest_json(pd.DataFrame({"ticker": [f"T{i}"], "composite": [1.0]}), str(tmp_path),
                          "2026-09-04", health={"ok": True}, regime={"regime": "NORMAL"}, ranking_tail=[])
    blob = json.loads((tmp_path / "screen_latest.json").read_text())
    assert blob["rows"][0]["ticker"] == "T1"
    assert not (tmp_path / "screen_latest.json.tmp").exists()


# ── run_status carries stats ──────────────────────────────────────────────────

def test_load_stats_only_for_today(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"date": "2026-09-03", "ok": True}))
    assert load_stats("2026-09-04", p) is None
    assert load_stats("2026-09-03", p) == {"date": "2026-09-03", "ok": True}
    assert load_stats("2026-09-03", tmp_path / "missing.json") is None
    p.write_text("{not json")
    assert load_stats("2026-09-03", p) is None


def test_build_status_embeds_stats_and_health_error_line():
    log = ("=== Screener run started: x ===\n"
           "[health] adv_survivors=53 below floor 800\n"
           "src.run.PipelineHealthError: adv_survivors=53 below floor 800\n")
    st = build_status(log, rc=1, started_at="2026-09-04T16:30:00", duration_secs=90,
                      stats={"date": "2026-09-04", "ok": False, "reasons": ["adv_survivors=53 below floor 800"]})
    assert st["result"] == "failed"
    assert st["stats"]["ok"] is False
    assert "PipelineHealthError" in st["error"]


def test_build_status_without_stats_is_none():
    st = build_status("", rc=0, started_at="x", duration_secs=1)
    assert st["stats"] is None
```

- [ ] **Step 2:** Append watchdog tests:

**Apply this unified diff to `tests/test_watchdog.py`** (from the reference patch; `patch -p1 < docs/superpowers/plans/2026-09-04-overhaul-reference.patch` applies all of them at once):

```diff
--- a/tests/test_watchdog.py	2026-08-16 12:32:32
+++ b/tests/test_watchdog.py	2026-09-04 12:27:22
@@ -325,3 +325,36 @@
     from src.watchdog import check_expiring_stops
     orders = [{"symbol": "HUT", "side": "sell", "type": "stop"}]
     assert check_expiring_stops(orders)["status"] == "warn"
+
+
+# ── screen health ─────────────────────────────────────────────────────────────
+from src.watchdog import check_screen_health
+from datetime import datetime as _dt
+from zoneinfo import ZoneInfo as _ZI
+_ET = _ZI("America/New_York")
+
+
+def test_screen_health_fail_on_failed_run():
+    rs = {"date": "2026-09-04", "result": "failed", "error": "src.run.PipelineHealthError: adv_survivors=53 below floor 800"}
+    out = check_screen_health(rs, _dt(2026, 9, 4, 18, 0, tzinfo=_ET))
+    assert out["status"] == "fail" and "PipelineHealthError" in out["detail"]
+
+
+def test_screen_health_fail_on_breached_floor_even_if_rc_zero():
+    rs = {"date": "2026-09-04", "result": "success", "stats": {"ok": False, "reasons": ["ranked=38 below floor 150"]}}
+    out = check_screen_health(rs, _dt(2026, 9, 4, 18, 0, tzinfo=_ET))
+    assert out["status"] == "fail" and "ranked=38" in out["detail"]
+
+
+def test_screen_health_warns_when_no_screen_after_1730_on_weekday():
+    rs = {"date": "2026-09-03", "result": "success", "stats": {"ok": True}}
+    assert check_screen_health(rs, _dt(2026, 9, 4, 17, 45, tzinfo=_ET))["status"] == "warn"
+    assert check_screen_health(rs, _dt(2026, 9, 4, 12, 0, tzinfo=_ET))["status"] == "ok"
+    assert check_screen_health(rs, _dt(2026, 9, 5, 18, 0, tzinfo=_ET))["status"] == "ok"   # Saturday
+
+
+def test_screen_health_ok_and_missing():
+    rs = {"date": "2026-09-04", "result": "success", "stats": {"ok": True, "cap_survivors": 1900}}
+    out = check_screen_health(rs, _dt(2026, 9, 4, 18, 0, tzinfo=_ET))
+    assert out["status"] == "ok" and "1900" in out["detail"]
+    assert check_screen_health(None, _dt(2026, 9, 4, 18, 0, tzinfo=_ET))["status"] == "warn"
```

- [ ] **Step 3:** Run:

```bash
/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m pytest tests/test_pipeline_health.py tests/test_watchdog.py -q
```

Expected: FAIL: `src.run.evaluate_health`, `write_latest_json`, `load_stats`, `check_screen_health` missing.

- [ ] **Step 4:** Apply the output changes (also adds the Phase 1 CSV columns — harmless when absent from the frame):

**Apply this unified diff to `src/output.py`** (from the reference patch; `patch -p1 < docs/superpowers/plans/2026-09-04-overhaul-reference.patch` applies all of them at once):

```diff
--- a/src/output.py	2026-07-30 15:26:22
+++ b/src/output.py	2026-09-04 12:32:33
@@ -1,13 +1,27 @@
+import json
+import math
 import os
 import pandas as pd
 
 CSV_COLUMNS = [
-    "ticker", "name", "sector", "composite", "weight_pct", "conviction", "factor_coverage",
+    "ticker", "alpaca_symbol", "name", "sector", "industry", "composite", "weight_pct", "conviction", "factor_coverage",
     # z-scores for all composite factors
     "z_mom_12_1", "z_residual_mom", "z_rs_6m", "z_rs_accel", "z_rs_slope", "z_pct_from_high",
     "z_sue", "z_rev_breadth", "z_rev_magnitude",
-    "z_gp_assets", "z_insider_z",
+    "z_fund_quality", "z_fund_growth", "z_fund_value", "z_fund_invest", "z_fund_strength",
+    "z_insider_z",
     "z_trend_score", "z_momo_osc_score", "z_volume_score",
+    # finalist re-rank
+    "composite_final", "z_eps_rev_30d",
+    # fundamentals block
+    "fund_score", "fund_quality", "fund_growth", "fund_value", "fund_invest", "fund_strength",
+    "fund_n_subscores",
+    "peTTM", "pfcfShareTTM", "evEbitdaTTM", "psTTM", "roeTTM", "roaTTM", "grossMarginTTM",
+    "revenueGrowthTTMYoy", "epsGrowthTTMYoy", "revenueGrowth3Y",
+    "totalDebt/totalEquityQuarterly", "netInterestCoverageTTM", "currentRatioQuarterly",
+    "accruals", "asset_growth", "net_issuance", "roa", "leverage",
+    # finalist enrichment
+    "eps_rev_30d", "eps_rev_90d", "analyst_target_pct", "days_to_earnings", "days_to_cover",
     # diagnostic z-scores (not in composite)
     "z_streak_z",
     # raw factors
@@ -15,8 +29,9 @@
     "rev_breadth", "sue", "rev_magnitude",
     "gp_assets", "pct_from_high", "short_float", "insider_buys_90d",
     "exec_buys_90d", "insider_buy_value",
-    "price", "market_cap",
+    "price", "market_cap", "cap_source", "avg_dollar_vol_20d", "fractionable",
     # technicals
+    "atr_14", "sma_50", "sma_200",
     "rsi_14", "macd", "vol_surge", "above_sma20", "above_sma50",
     "stoch_k", "stoch_d", "stoch_cross", "bb_pct_b", "bb_width", "adx", "mfi",
     "tech_score", "trend_score", "momo_osc_score", "volume_score",
@@ -51,12 +66,70 @@
         parts.append(f"{int(row['insider_buys_90d'])} insider cluster buy")
     if row.get("z_trend_score", 0) > 1.0:
         parts.append("Strong trend (ADX + MACD)")
+    fs = row.get("fund_score")
+    if fs is not None and fs == fs and float(fs) >= 0.70:
+        parts.append(f"Top-30% fundamentals ({float(fs):.2f})")
+    er = row.get("eps_rev_30d")
+    if er is not None and er == er and float(er) > 0.02:
+        parts.append(f"EPS estimates up {float(er):.1%} in 30d")
     conviction = int(row.get("conviction", 0) or 0)
     if conviction >= 7:
         parts.append(f"High conviction ({conviction}/10)")
     return "; ".join(parts) if parts else "Composite score"
 
 
+def _jsonable(v):
+    """JSON-safe scalar: NaN/NaT -> None, numpy scalars -> Python."""
+    if v is None:
+        return None
+    if isinstance(v, float) and math.isnan(v):
+        return None
+    if hasattr(v, "item"):          # numpy scalar
+        v = v.item()
+        if isinstance(v, float) and math.isnan(v):
+            return None
+        return v
+    if isinstance(v, pd.Timestamp):
+        return v.isoformat()
+    return v
+
+
+def write_latest_json(df: pd.DataFrame, out_dir: str, date_str: str, health: dict,
+                      regime: dict, ranking_tail: list[dict]) -> str:
+    """Machine-readable screen for the trader: output/screen_latest.json.
+
+    Written on EVERY run, including a failed one — then with health.ok False
+    and no rows — so a consumer sees "today's screen failed" rather than
+    yesterday's file with today's date on nothing. `ranking_tail` carries the
+    top ~100 (ticker, rank, composite) so a consumer can tell "fell to #34"
+    from "fell out of the top 100" without the full CSV.
+    """
+    os.makedirs(out_dir, exist_ok=True)
+    path = os.path.join(out_dir, "screen_latest.json")
+    cols = [c for c in CSV_COLUMNS if c in df.columns]
+    rows = []
+    for i, (_, r) in enumerate(df.iterrows(), 1):
+        row = {c: _jsonable(r[c]) for c in cols}
+        row["rank"] = i
+        rows.append(row)
+    payload = {
+        "date": date_str,
+        "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
+        "health": {k: _jsonable(v) if not isinstance(v, (dict, list)) else v
+                   for k, v in (health or {}).items()},
+        "regime": {"regime": regime.get("regime"), "scale_factor": regime.get("scale_factor"),
+                   "reason": regime.get("reason", "")},
+        "columns": cols + ["rank"],
+        "rows": rows,
+        "ranking_tail": ranking_tail or [],
+    }
+    tmp = path + ".tmp"
+    with open(tmp, "w") as f:
+        json.dump(payload, f, indent=1, default=str)
+    os.replace(tmp, path)
+    return path
+
+
 def write_csv(df: pd.DataFrame, out_dir: str, date_str: str) -> str:
     os.makedirs(out_dir, exist_ok=True)
     path = os.path.join(out_dir, f"screen_{date_str}.csv")
@@ -79,8 +152,8 @@
     if has_weight:
         header += " Weight |"
         sep    += "--------|"
-    header += " Conv | Streak | Signal | Entry | Rationale |"
-    sep    += "------|--------|--------|-------|-----------|"
+    header += " Fund | Conv | Streak | Signal | Entry | Rationale |"
+    sep    += "------|------|--------|--------|-------|-----------|"
     lines = [
         f"# Stock Screen — {date_str}",
         "",
@@ -113,7 +186,9 @@
             wt = row.get("weight_pct")
             wt_str = f"{float(wt):.1f}%" if wt is not None and pd.notna(wt) else "—"
             wt_cell = f" {wt_str} |"
-        lines.append(f"| {i} | {row['ticker']} | {name} | {sector} | {comp} |{wt_cell} {conv}/10 | {streak_str} | {es_str} | {entry} | {rationale} |")
+        fs = row.get("fund_score")
+        fund_str = f"{float(fs):.2f}" if fs is not None and pd.notna(fs) else "—"
+        lines.append(f"| {i} | {row['ticker']} | {name} | {sector} | {comp} |{wt_cell} {fund_str} | {conv}/10 | {streak_str} | {es_str} | {entry} | {rationale} |")
 
     lines += [
         "",
```

- [ ] **Step 5:** Apply the run_status changes:

**Apply this unified diff to `src/run_status.py`** (from the reference patch; `patch -p1 < docs/superpowers/plans/2026-09-04-overhaul-reference.patch` applies all of them at once):

```diff
--- a/src/run_status.py	2026-09-03 17:29:15
+++ b/src/run_status.py	2026-09-04 12:24:21
@@ -39,7 +39,23 @@
     }
 
 
-def build_status(log_text: str, rc: int, started_at: str, duration_secs: int) -> dict:
+STATS_PATH = Path("data") / "last_run_stats.json"
+
+
+def load_stats(today: str, path: Path = STATS_PATH) -> dict | None:
+    """Stage counts written by src.run for THIS date, else None. A stats file
+    from an earlier day must not be reported as today's."""
+    if not path.exists():
+        return None
+    try:
+        blob = json.loads(path.read_text())
+    except (OSError, json.JSONDecodeError):
+        return None
+    return blob if blob.get("date") == today else None
+
+
+def build_status(log_text: str, rc: int, started_at: str, duration_secs: int,
+                 stats: dict | None = None) -> dict:
     parsed = parse_log(log_text)
     return {
         "date": datetime.now().date().isoformat(),
@@ -54,6 +70,9 @@
         # A label, not the real hostname: gethostname() leaks the network
         # and machine (e.g. a campus DHCP name) into a public repo.
         "host": os.environ.get("SCREENER_HOST_LABEL", "screener"),
+        # Stage counts + health verdict from src.run (None when the run died
+        # before writing them). The dashboard and the trader read stats.ok.
+        "stats": stats,
     }
 
 
@@ -92,7 +111,9 @@
     log_path = Path(args.log)
     log_text = _last_run_slice(log_path.read_text(errors="replace")) if log_path.exists() else ""
 
-    status = build_status(log_text, args.rc, args.started, args.duration)
+    today = datetime.now().date().isoformat()
+    status = build_status(log_text, args.rc, args.started, args.duration,
+                          stats=load_stats(today))
     STATUS_PATH.write_text(json.dumps(status, indent=2) + "\n")
     print(f"[run_status] {status['result']} → {STATUS_PATH}")
 
```

- [ ] **Step 6:** Apply the watchdog change:

**Apply this unified diff to `src/watchdog.py`** (from the reference patch; `patch -p1 < docs/superpowers/plans/2026-09-04-overhaul-reference.patch` applies all of them at once):

```diff
--- a/src/watchdog.py	2026-08-16 12:32:29
+++ b/src/watchdog.py	2026-09-04 12:24:52
@@ -199,6 +199,30 @@
                   + " — the next post-open stop sync re-places these as whole-share GTC",
                   tickers=sorted((o.get("symbol") or "?") for o in day))
     return _r("expiring_stops", "ok", f"all {len(stops)} stop(s) are GTC")
+
+
+def check_screen_health(run_status: dict | None, now_et: datetime) -> dict:
+    """Did the screener publish a trustworthy list for the latest session?
+
+    FAIL when the last run failed outright or its stats say a health floor was
+    breached. WARN when, after 17:30 ET on a weekday, there is no run dated
+    today — the trader session at 17:15+ would then read yesterday's screen.
+    """
+    if not run_status:
+        return _r("screen", "warn", "no run_status.json")
+    date_ = str(run_status.get("date") or "")
+    if run_status.get("result") == "failed":
+        return _r("screen", "fail",
+                  f"screener run {date_} failed: {str(run_status.get('error') or '?')[:120]}")
+    stats = run_status.get("stats") or {}
+    if stats.get("ok") is False:
+        return _r("screen", "fail",
+                  f"screener {date_} breached a health floor: {'; '.join(stats.get('reasons') or [])[:160]}")
+    today = now_et.date().isoformat()
+    if date_ != today and now_et.weekday() < 5 and (now_et.hour, now_et.minute) >= (17, 30):
+        return _r("screen", "warn", f"no screen for {today} yet (last {date_})")
+    n = stats.get("cap_survivors")
+    return _r("screen", "ok", f"{date_} ok" + (f", {n} names past the cap gate" if n else ""))
 
 
 def check_tests(returncode: int | None, tail: str = "") -> dict:
@@ -253,6 +277,12 @@
 
     snapshot = paper.load_snapshot()
     checks.append(check_snapshot_freshness(snapshot, today))
+    try:
+        rs_path = SCREENER_DIR / "run_status.json"
+        run_status = json.loads(rs_path.read_text()) if rs_path.exists() else None
+    except (OSError, json.JSONDecodeError):
+        run_status = None
+    checks.append(check_screen_health(run_status, now_et))
     checks.append(check_reconciliation(snapshot))
     checks.append(check_cadence((snapshot or {}).get("cadence"), today))
 
```

- [ ] **Step 7:** Apply the compose changes. This diff includes the Phase 1 factor changes; for Phase 0 only the `attrs["ranked_total"]`/`attrs["ranking_tail"]` block is needed, but apply the whole diff now so the file matches the reference:

**Apply this unified diff to `src/compose.py`** (from the reference patch; `patch -p1 < docs/superpowers/plans/2026-09-04-overhaul-reference.patch` applies all of them at once):

```diff
--- a/src/compose.py	2026-06-30 12:53:57
+++ b/src/compose.py	2026-09-04 12:33:28
@@ -11,22 +11,28 @@
 # depended on streak_z), st_reversal was a minor penalty included in the price block.
 # Both remain as diagnostic CSV columns and feed conviction.
 # residual_mom and pct_from_high are new additions.
+# Four blocks (weights in config.yaml): price momentum, earnings momentum,
+# fundamentals (src.fundamentals_block: five percentile sub-scores + insider
+# buying), technical confirmation. gp_assets is no longer a standalone factor —
+# it is an input to fund_quality.
 COMPOSITE_FACTORS = [
     "mom_12_1", "residual_mom", "rs_6m", "rs_accel", "rs_slope", "pct_from_high",
     "sue", "rev_breadth", "rev_magnitude",
-    "gp_assets", "insider_z",
+    "fund_quality", "fund_growth", "fund_value", "fund_invest", "fund_strength", "insider_z",
     "trend_score", "momo_osc_score", "volume_score",
 ]
+FUND_FACTORS = ["fund_quality", "fund_growth", "fund_value", "fund_invest", "fund_strength"]
 
-# Factors that are bounded integers / zero-inflated — use rank normalization
+# Factors that are bounded / percentile / zero-inflated — use rank normalization
 # instead of winsorize+z-score to avoid distributional distortion.
-RANK_NORMALIZE_FACTORS = {"insider_z", "trend_score", "momo_osc_score", "volume_score"}
+RANK_NORMALIZE_FACTORS = {"insider_z", "trend_score", "momo_osc_score", "volume_score",
+                          *FUND_FACTORS}
 
 # Sectors where gp_assets = (revenue-COGS)/assets is meaningless or misleading.
 # These stocks pass the quality gate unconditionally and get NaN for gp_assets factor.
 FINANCIAL_SECTORS = {
     "Financial Services", "Financial", "Financials",
-    "Real Estate", "Banks", "Insurance",
+    "Real Estate", "Banks", "Banking", "Insurance",
     "Asset Management", "Mortgage Finance",
 }
 
@@ -212,7 +218,8 @@
         score += 1
     if max(row.get("z_sue", 0) or 0, row.get("z_rev_breadth", 0) or 0) > 0.5:
         score += 1
-    if (row.get("z_gp_assets", 0) or 0) > 0.3:
+    fs = row.get("fund_score")
+    if fs is not None and not pd.isna(fs) and float(fs) > 0.5:
         score += 1
     if (row.get("trend_score", 0) or 0) >= 3:
         score += 1
@@ -220,7 +227,7 @@
 
 
 def compute_conviction(df: pd.DataFrame) -> pd.DataFrame:
-    rank_comp, streak_comp, tech_comp, agreement_comp = [], [], [], []
+    rank_comp, streak_comp, tech_comp, agreement_comp, fund_comp = [], [], [], [], []
 
     for pos, (_, row) in enumerate(df.iterrows()):
         # Rank (0-3)
@@ -256,11 +263,22 @@
         # Factor agreement (0-4): cross-block consensus — new, non-composite signal
         agreement_comp.append(_compute_factor_agreement(row))
 
+        # Fundamentals (0-2): a business in the top 30% / top half of the
+        # universe on the fundamentals block earns conviction on its own.
+        fs = row.get("fund_score")
+        if fs is not None and not pd.isna(fs) and float(fs) >= 0.70:
+            fund_comp.append(2)
+        elif fs is not None and not pd.isna(fs) and float(fs) >= 0.50:
+            fund_comp.append(1)
+        else:
+            fund_comp.append(0)
+
     raw = (
         pd.Series(rank_comp, dtype=float)
         + pd.Series(streak_comp, dtype=float)
         + pd.Series(tech_comp, dtype=float)
         + pd.Series(agreement_comp, dtype=float)
+        + pd.Series(fund_comp, dtype=float)
     )
     df = df.copy()
     df["conviction"] = raw.clip(1, 10).round().astype(int).values
@@ -271,7 +289,14 @@
     factors_df: pd.DataFrame,
     cfg: dict,
     streak_data: dict | None = None,
+    top_n: int | None = None,
+    with_conviction: bool = True,
 ) -> pd.DataFrame:
+    """Gate, z-score, weight, rank. Returns the top `top_n` (default
+    cfg.output.top_n) with `attrs["ranked_total"]`, `attrs["ranking_tail"]`
+    and `attrs["fund_gate"]`. run.py asks for the finalist pool (top 60),
+    enriches it, re-ranks on composite_final and only then computes
+    conviction on the final top 20 — so `with_conviction=False` there."""
     df = factors_df.copy()
     weights        = cfg["factors"]["weights"]
     winsorize_pct  = cfg["factors"]["winsorize_pct"]
@@ -280,7 +305,15 @@
     min_coverage   = cfg["factors"].get("min_factor_coverage", 0.40)
     rank_norm_set  = set(cfg["factors"].get("rank_normalize_factors", list(RANK_NORMALIZE_FACTORS)))
 
-    df = apply_quality_gate(df, cfg)
+    # The fundamentals floor gate (src.fundamentals_block) replaces the old
+    # gp_assets-only quality gate when fund_score is present. The old gate is
+    # kept for frames without the block (tests, backtests).
+    if "fund_score" in df.columns:
+        from src.fundamentals_block import apply_fundamental_gate
+        df, fund_gate_info = apply_fundamental_gate(df, cfg)
+    else:
+        df = apply_quality_gate(df, cfg)
+        fund_gate_info = {"skipped": "no fund_score column"}
     df = apply_confirmation_gate(df, cfg)
 
     # Attach derived and technical columns
@@ -328,8 +361,17 @@
     df["composite"] = composite
     df = df.sort_values("composite", ascending=False).reset_index(drop=True)
 
-    top_n = cfg["output"]["top_n"]
+    top_n = int(top_n or cfg["output"]["top_n"])
     result = df.head(top_n).reset_index(drop=True)
-    result = compute_conviction(result)
+    if with_conviction:
+        result = compute_conviction(result)
+    result.attrs["fund_gate"] = fund_gate_info
+    # Carry the size of the ranked pool and the top-100 order so run.py's
+    # health gate and screen_latest.json can see past the top_n cut.
+    result.attrs["ranked_total"] = int(len(df))
+    result.attrs["ranking_tail"] = [
+        {"rank": i + 1, "ticker": str(t), "composite": round(float(c), 6)}
+        for i, (t, c) in enumerate(zip(df["ticker"].head(100), df["composite"].head(100)))
+    ]
     print(f"[compose] {len(df)} ranked → top {len(result)} selected")
     return result
```

- [ ] **Step 8:** Then Task 0.7's `run.py` provides `evaluate_health`/`_gate`; run this task's tests after Task 0.7.

**Edge cases this task must handle (each has a test):**
- `evaluate_health` only checks floors for stages present in `stats` (fail fast after stage 2 without a `ranked` count).
- A breach writes stats + failed `screen_latest.json`, writes NO dated CSV, and raises `PipelineHealthError`; `--allow-degraded` records `degraded: true` and continues.
- `write_latest_json`: NaN → null, numpy ints → int, commas in names are not a parsing problem, atomic replace, empty frame OK.
- `load_stats` only returns a stats file dated today.
- `check_screen_health`: FAIL on `result: failed` or `stats.ok: false`; WARN when no screen for today after 17:30 ET on a weekday; OK on weekends.


### Task 0.7: `src/run.py` — new pipeline order with health gates (full rewrite) and `config.yaml`

**Files:**
- Modify: `src/run.py` (full rewrite below)
- Modify: `config.yaml`

Stage order: universe → Alpaca filter → prices → ADV gate → (gate) → Finnhub metrics/profiles → caps → cap gate → (gate) → fundamentals → fundamentals block (Phase 1) → stress → composite → (gate) → finalists (Phase 1) → news → sizing → outputs. The file below is the Phase 0 + Phase 1 end state; the Phase 1 lines (metric columns, `attach_fundamentals_block`, `enrich_finalists`, `composite_final`) are inert until those modules exist, so land the whole file with Task 1.x, or land this file and Tasks 1.2/1.3 together. The config diff carries every new block (universe.alpaca_filter, finnhub, health, fundamentals, trader, new weights).

- [ ] **Step 1:** Apply the config diff (whole file end state):

**Apply this unified diff to `config.yaml`** (from the reference patch; `patch -p1 < docs/superpowers/plans/2026-09-04-overhaul-reference.patch` applies all of them at once):

```diff
--- a/config.yaml	2026-07-30 15:23:50
+++ b/config.yaml	2026-09-04 12:45:16
@@ -2,31 +2,42 @@
   source: sec
   exclude_etfs: true
   min_price: 5.0
+  # Intersect the SEC list with Alpaca's tradable, exchange-listed equities.
+  # Skipped automatically when no Alpaca credentials are present.
+  alpaca_filter: true
+  max_age_days: 7
 
 liquidity_gate:
   min_market_cap: 300000000
   min_avg_dollar_vol_20d: 5000000
 
+# Legacy gate, used only when the fundamentals block is absent (backtests).
 quality_gate:
   gross_profitability_min: q25
 
 factors:
   weights:
-    # Price momentum block (0.50)
-    # residual_mom replaces raw mom_12_1 as primary signal; mom_12_1 kept as secondary
-    mom_12_1: 0.12
-    residual_mom: 0.14
-    rs_6m: 0.10
-    rs_accel: 0.06
-    rs_slope: 0.04
-    pct_from_high: 0.04
-    # Earnings/revision block (0.33)
-    sue: 0.14
-    rev_breadth: 0.10
-    rev_magnitude: 0.09
-    # Quality + insider block (0.09)
-    gp_assets: 0.06
-    insider_z: 0.03
+    # Price momentum block (0.40)
+    mom_12_1: 0.10
+    residual_mom: 0.11
+    rs_6m: 0.08
+    rs_accel: 0.05
+    rs_slope: 0.03
+    pct_from_high: 0.03
+    # Earnings momentum block (0.20). rev_breadth/rev_magnitude are analyst
+    # RATING breadth and its 90-day shift (Finnhub recommendation trends),
+    # not estimate revisions — true revisions enter via eps_rev_30d on the
+    # finalists (see fundamentals.finalist_rev_weight).
+    sue: 0.10
+    rev_breadth: 0.06
+    rev_magnitude: 0.04
+    # Fundamentals block (0.32) — percentile sub-scores from src/fundamentals_block.py
+    fund_quality: 0.09
+    fund_growth: 0.08
+    fund_value: 0.06
+    fund_invest: 0.04
+    fund_strength: 0.03
+    insider_z: 0.02
     # Technical confirmation block (0.08) — rank-normalized, not z-scored
     trend_score: 0.05
     momo_osc_score: 0.02
@@ -39,12 +50,31 @@
     - trend_score
     - momo_osc_score
     - volume_score
+    - fund_quality
+    - fund_growth
+    - fund_value
+    - fund_invest
+    - fund_strength
   # Minimum fraction of factor weight a stock must have data for to be scored
   min_factor_coverage: 0.40
 
 streak:
   lookback_days: 14
 
+fundamentals:
+  # Names below this fund_score percentile are not ranked at all. NaN (no
+  # Finnhub and no EDGAR data) is excluded too.
+  floor_enabled: true
+  floor_percentile: 0.30
+  # If fewer than this share of names have a fund_score the gate is skipped
+  # (cold cache) and the skip is reported in health.reasons.
+  min_coverage: 0.60
+  # Finalists (top N by composite) get yfinance estimate revisions, short
+  # interest and analyst targets; eps_rev_30d re-ranks them at this weight.
+  finalists_n: 60
+  finalist_rev_weight: 0.05
+  enrich_budget_secs: 240
+
 confirmation:
   require_above_sma200: true
   max_pct_below_52w_high: 0.35
@@ -70,7 +100,20 @@
 
 finnhub:
   calls_per_minute: 60
+  # Basic-financials (market cap + ~130 ratios) and profile (industry) cache.
+  metrics_ttl_days: 7
+  # Cap on Finnhub lookups per run so a cold cache degrades the run instead
+  # of stalling it. Warm the cache with: python -m src.cache_maint warm-finnhub
+  max_fetch_per_run: 600
 
+# Pipeline health floors. Breaching one fails the run (exit 1, alert email,
+# run_status.json result=failed) instead of publishing a list ranked from a
+# collapsed universe — which is what happened from 2026-08-31 to 09-03.
+health:
+  min_adv_survivors: 800
+  min_cap_survivors: 500
+  min_ranked: 150
+
 news:
   enabled: true
   analyze_top_n: 20
@@ -83,3 +126,26 @@
   min_market_cap: 50000000      # $50M floor (exitable)
   max_market_cap: 2000000000    # $2B ceiling (below institutional capacity threshold)
   min_avg_dollar_vol_20d: 200000 # $200K tradeable floor (risk control)
+
+# Portfolio engine risk policy (src/portfolio_engine.py). "Aggressive" posture
+# chosen 2026-09-04. Keys mirror DEFAULT_POLICY; omitted keys keep defaults.
+trader:
+  risk_per_trade: 0.0125
+  name_cap: 0.12
+  sector_cap: 0.35
+  max_positions: 8
+  alpha_target:
+    NORMAL: 0.70
+    CAUTION: 0.40
+    STRESS: 0.15
+    UNKNOWN: 0.40
+  core_symbol: SPY
+  cash_buffer: 0.05
+  breaker_drawdown: 0.15
+  breaker_recover: 0.07
+  earnings_no_entry_days: 5
+  earnings_trim_days: 2
+  earnings_max_weight: 0.06
+  entry_band: 20
+  exit_band: 35
+  rotate_out_runs: 3
```

- [ ] **Step 2:** Replace `src/run.py`:

**Create `src/run.py` with exactly this content:**

```python
import json
import logging
import os
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from src.config import load_config
from src.cache import init_db, archive_fundamentals_snapshot, archive_universe_snapshot
from src.universe import load_or_build_universe, load_tradable_assets, filter_tradable
from src.prices import fetch_all_prices, apply_adv_gate, apply_liquidity_gate
from src.finnhub_data import fetch_metrics, fetch_profiles, attach_market_caps
from src.fundamentals import fetch_all_fundamentals
from src.factors import squeeze_flag
from src.compose import build_composite, compute_conviction
from src.fundamentals_block import attach_fundamentals_block
from src.enrich import enrich_finalists, composite_final
from src.finnhub_data import fetch_earnings_calendar
from src.streak import load_streak_history
from src.news import attach_news_overlay
from src.spy_analysis import compute_market_stress_overlay
from src.sizing import attach_weights
from src.output import write_csv, write_markdown, write_latest_json, print_top10

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DB_PATH       = "data/cache.db"
UNIVERSE_PATH = "data/universe.parquet"
OUTPUT_DIR    = "output"
STATS_PATH    = Path("data") / "last_run_stats.json"


class PipelineHealthError(RuntimeError):
    """A stage produced too few names to trust the ranking. The run must fail
    loudly rather than publish a list drawn from a collapsed universe."""


def evaluate_health(stats: dict, cfg: dict) -> tuple[bool, list[str]]:
    """Compare stage counts against config floors. Pure, so it is testable.

    Only floors whose stage has actually run (key present in stats) are
    checked, so this can be called after each stage to fail fast.
    """
    floors = cfg.get("health", {}) or {}
    checks = [
        ("adv_survivors", floors.get("min_adv_survivors")),
        ("cap_survivors", floors.get("min_cap_survivors")),
        ("ranked", floors.get("min_ranked")),
    ]
    reasons = []
    for key, floor in checks:
        if floor is None or key not in stats:
            continue
        if stats[key] < floor:
            reasons.append(f"{key}={stats[key]} below floor {floor}")
    return (not reasons), reasons


def write_stats(stats: dict, path: Path = STATS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(stats, indent=2, default=str))
    os.replace(tmp, path)


def _gate(stats: dict, cfg: dict, allow_degraded: bool, today: str) -> None:
    """Fail the run on a breached floor unless --allow-degraded was passed.
    Records the verdict in stats either way and writes the health-only
    screen_latest.json so the trader sees a failed screen, not a stale one."""
    ok, reasons = evaluate_health(stats, cfg)
    stats["ok"] = ok
    stats["reasons"] = reasons
    if ok:
        return
    for r in reasons:
        print(f"[health] {r}")
    if allow_degraded:
        print("[health] --allow-degraded set: continuing with a degraded universe")
        stats["degraded"] = True
        return
    write_stats(stats)
    write_latest_json(pd.DataFrame(), OUTPUT_DIR, today, health=stats,
                      regime={"regime": "UNKNOWN", "scale_factor": 1.0}, ranking_tail=[])
    raise PipelineHealthError("; ".join(reasons))


def run(force_universe: bool = False, allow_degraded: bool = False):
    cfg = load_config()
    today = date.today().isoformat()

    os.makedirs("data", exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    init_db(DB_PATH)

    stats: dict = {"date": today, "started_at": datetime.now().isoformat(timespec="seconds"),
                   "ok": None, "reasons": []}

    # Stage 1: Universe (SEC list, rebuilt weekly) ∩ Alpaca tradable equities
    universe_df = load_or_build_universe(
        cfg, UNIVERSE_PATH, force=force_universe,
        max_age_days=int(cfg["universe"].get("max_age_days", 7)))
    stats["universe"] = len(universe_df)
    if cfg["universe"].get("alpaca_filter", True):
        assets = load_tradable_assets()
        if assets is None:
            print("[universe] Alpaca asset list unavailable — tradability filter skipped")
        universe_df = filter_tradable(universe_df, assets)
    else:
        universe_df = filter_tradable(universe_df, None)
    stats["tradable"] = len(universe_df)

    # Stage 2: Prices → dollar-volume gate (cheap, before any per-name lookups)
    price_store, factors_df, pstats = fetch_all_prices(universe_df, cfg, DB_PATH)
    stats["prices"] = pstats
    stats["priced"] = int(pstats.get("priced", 0))
    adv_df = apply_adv_gate(factors_df, cfg)
    stats["adv_survivors"] = len(adv_df)
    _gate(stats, cfg, allow_degraded, today)

    # Stage 2.5: Market cap + industry from Finnhub, then the cap gate
    fh_cfg = cfg.get("finnhub", {}) or {}
    tickers = adv_df["ticker"].tolist()
    metrics, mstats = fetch_metrics(
        tickers, DB_PATH, ttl_days=int(fh_cfg.get("metrics_ttl_days", 7)),
        max_fetch=int(fh_cfg.get("max_fetch_per_run", 600)),
        calls_per_minute=int(fh_cfg.get("calls_per_minute", 60)))
    profiles, prstats = fetch_profiles(
        tickers, DB_PATH, cap_ttl_days=int(fh_cfg.get("metrics_ttl_days", 7)),
        max_fetch=int(fh_cfg.get("max_fetch_per_run", 600)),
        calls_per_minute=int(fh_cfg.get("calls_per_minute", 60)))
    stats["finnhub"] = {"metrics": mstats, "profiles": prstats}
    print(f"[finnhub] metrics cached={mstats['cached']} fetched={mstats['fetched']} "
          f"failed={mstats['failed']} unfetched={mstats['unfetched']}; "
          f"profiles cached={prstats['cached']} fetched={prstats['fetched']}")
    adv_df, capstats = attach_market_caps(adv_df, metrics, profiles, DB_PATH)
    stats["cap_sources"] = capstats
    print(f"[caps] sources={capstats}")
    survivors_df = apply_liquidity_gate(adv_df, cfg)
    stats["cap_survivors"] = len(survivors_df)
    print(f"[stage2] {stats['universe']} universe → {stats['tradable']} tradable → "
          f"{stats['priced']} priced → {stats['adv_survivors']} ADV → {stats['cap_survivors']} cap")
    _gate(stats, cfg, allow_degraded, today)

    archive_universe_snapshot(DB_PATH, today, survivors_df)

    # Universe metadata the factor rows do not carry
    extra_cols = [c for c in ["ticker", "alpaca_symbol", "exchange", "fractionable"] if c in universe_df.columns]
    survivors_df = survivors_df.merge(universe_df[extra_cols], on="ticker", how="left")

    # Stage 3: Fundamentals (survivors only). Sector already attached above.
    fund_df = fetch_all_fundamentals(survivors_df, cfg, DB_PATH)
    merged = survivors_df.merge(fund_df, on="ticker", how="left")
    print(f"[stage3] fundamentals fetched for {len(fund_df)} tickers")
    archive_fundamentals_snapshot(DB_PATH, today)

    # Stage 3.2: Finnhub ratios as columns (named as Finnhub names them), then
    # the fundamentals block: five percentile sub-scores + fund_score.
    metric_cols = ["peTTM", "pfcfShareTTM", "evEbitdaTTM", "psTTM", "roeTTM", "roaTTM",
                   "grossMarginTTM", "revenueGrowthTTMYoy", "epsGrowthTTMYoy", "revenueGrowth3Y",
                   "totalDebt/totalEquityQuarterly", "netInterestCoverageTTM", "currentRatioQuarterly"]
    for col in metric_cols:
        merged[col] = merged["ticker"].map(lambda t, c=col: (metrics.get(str(t).upper()) or {}).get(c))
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
    merged = attach_fundamentals_block(merged)
    stats["fund_coverage"] = round(float(merged["fund_score"].notna().mean()), 4) if len(merged) else 0.0
    print(f"[fund_block] fund_score coverage {stats['fund_coverage']:.0%} of {len(merged)} names")

    # Stage 3.5: Market stress overlay — scale top_n down in momentum-crash regimes
    stress = compute_market_stress_overlay()
    scale  = stress["scale_factor"]
    stats["regime"] = stress.get("regime")
    if scale == 0.0:
        print(f"[stress] regime={stress['regime']} — STRESS: outputting empty screen")
        ranked_df = pd.DataFrame(columns=["ticker"])
        csv_path = write_csv(ranked_df, OUTPUT_DIR, today)
        stats["ranked"] = 0
        stats["ok"], stats["reasons"] = True, ["stress regime: screen intentionally empty"]
        write_stats(stats)
        write_latest_json(ranked_df, OUTPUT_DIR, today, health=stats, regime=stress, ranking_tail=[])
        print(f"\n[output] {csv_path} (empty — market stress)")
        return
    elif scale < 1.0:
        original_top_n = cfg["output"]["top_n"]
        new_top_n = max(1, int(original_top_n * scale))
        cfg = dict(cfg)
        cfg["output"] = dict(cfg["output"])
        cfg["output"]["top_n"] = new_top_n
        print(f"[stress] regime={stress['regime']} scale={scale:.1f} "
              f"reason='{stress['reason']}' → top_n {original_top_n}→{new_top_n}")
    else:
        print(f"[stress] regime={stress['regime']} — full screen")

    # Stage 4: Composite score → finalist pool (top finalists_n, no conviction yet)
    fcfg = cfg.get("fundamentals", {}) or {}
    top_n = int(cfg["output"]["top_n"])
    finalists_n = max(top_n, int(fcfg.get("finalists_n", 60)))
    streak_data = load_streak_history(OUTPUT_DIR, lookback_days=cfg.get("streak", {}).get("lookback_days", 14))
    finalists = build_composite(merged, cfg, streak_data=streak_data, top_n=finalists_n, with_conviction=False)
    stats["ranked"] = int(finalists.attrs.get("ranked_total", len(finalists)))
    stats["fund_gate"] = finalists.attrs.get("fund_gate")
    ranking_tail = finalists.attrs.get("ranking_tail", [])
    if (stats["fund_gate"] or {}).get("skipped"):
        stats.setdefault("warnings", []).append(f"fund gate skipped: {stats['fund_gate']['skipped']}")
    _gate(stats, cfg, allow_degraded, today)

    # Stage 4.2: finalist enrichment (estimate revisions, short interest,
    # targets, earnings dates) → composite_final → final top_n → conviction
    calendar = fetch_earnings_calendar()
    finalists, estats = enrich_finalists(finalists, DB_PATH, calendar,
                                         budget_secs=float(fcfg.get("enrich_budget_secs", 240)))
    stats["enrich"] = estats
    finalists = composite_final(finalists, weight=float(fcfg.get("finalist_rev_weight", 0.05)))
    finalists = finalists.sort_values("composite_final", ascending=False).reset_index(drop=True)
    ranked_df = compute_conviction(finalists.head(top_n).reset_index(drop=True))
    ranked_df.attrs.update(finalists.attrs)

    # Stage 4.5: News overlay (entry signal + conviction adjustment)
    if cfg.get("news", {}).get("enabled", True):
        ranked_df = attach_news_overlay(ranked_df, cfg, DB_PATH)

    # Stage 4.6: Position sizing + concentration caps (advisory weights)
    ranked_df = attach_weights(ranked_df, cfg, price_store)

    # Squeeze screen
    squeeze_df = None
    if cfg["output"].get("include_squeeze_screen", False) and "short_float" in merged.columns:
        squeeze_rows = []
        for _, row in merged.iterrows():
            sf  = row.get("short_float", 0) or 0
            dtc = row.get("days_to_cover", 0) or 0
            m1  = row.get("mom_1m", 0) or 0
            if sf == sf and dtc == dtc and squeeze_flag(sf, dtc, m1):
                squeeze_rows.append(row)
        squeeze_df = pd.DataFrame(squeeze_rows) if squeeze_rows else None
        if squeeze_df is not None:
            print(f"[squeeze] {len(squeeze_df)} squeeze candidates")

    # Stage 5: Output
    stats["selected"] = len(ranked_df)
    stats["finished_at"] = datetime.now().isoformat(timespec="seconds")
    write_stats(stats)
    csv_path = write_csv(ranked_df, OUTPUT_DIR, today)
    md_path  = write_markdown(ranked_df, OUTPUT_DIR, today, squeeze_df=squeeze_df)
    json_path = write_latest_json(ranked_df, OUTPUT_DIR, today, health=stats, regime=stress,
                                  ranking_tail=ranking_tail)
    print(f"\n[output] {csv_path}")
    print(f"[output] {md_path}")
    print(f"[output] {json_path}")
    print_top10(ranked_df)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the equity screener")
    parser.add_argument("--force-universe", action="store_true",
                        help="Re-fetch universe from SEC even if parquet exists")
    parser.add_argument("--allow-degraded", action="store_true",
                        help="Publish even when a health floor is breached (manual runs only)")
    args = parser.parse_args()
    run(force_universe=args.force_universe, allow_degraded=args.allow_degraded)
```

- [ ] **Step 3:** Run:

```bash
/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m pytest tests -q
```

Expected: After Tasks 0.1–0.7 (+ Task 1.2 and 1.3 modules present, because run.py imports them): all tests pass. If you are strictly in Phase 0 without Phase 1 modules, temporarily comment out the three Phase 1 imports at the top of run.py and the Stage 3.2/4.2 blocks.

- [ ] **Step 4:** Commit as `feat(run): health floors, screen_latest.json, Finnhub caps and sectors in the pipeline`. Per the user's global CLAUDE.md §6: run `git status` first and **ask the user before committing**. Never add a Co-Authored-By line.

**Edge cases this task must handle (each has a test):**
- `_gate` is called after the ADV gate, after the cap gate and after the composite — each can fail fast.
- Stress regime (`scale_factor == 0`) writes an intentionally empty screen with `health.ok: true`.
- `fetch_all_prices` now returns three values; `apply_adv_gate` precedes Finnhub so lookups are spent on liquid names only.
- `squeeze_flag` skips NaN short interest (it is NaN for non-finalists now).


### Task 0.8: `src/cache_maint.py` — purge quarantine, warm Finnhub, stats

**Files:**
- Create: `src/cache_maint.py`
- Create: `tests/test_cache_maint.py`

Operational CLI. `warm-finnhub` orders tickers by cached 20-day dollar volume so the liquid names are cached first.

- [ ] **Step 1:** **Create `tests/test_cache_maint.py` with exactly this content:**

```python
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
```

- [ ] **Step 2:** Run:

```bash
/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m pytest tests/test_cache_maint.py -q
```

Expected: FAIL at import.

- [ ] **Step 3:** **Create `src/cache_maint.py` with exactly this content:**

```python
"""Cache maintenance CLI.

  python -m src.cache_maint purge-failed [--before ISO]   drop yfinance quarantine rows
  python -m src.cache_maint warm-finnhub [--max N]        pre-fetch Finnhub metrics/profiles
  python -m src.cache_maint stats                          row counts that matter

warm-finnhub exists because a run may only spend `finnhub.max_fetch_per_run`
calls; the first fill of ~2,500 liquid names takes ~85 minutes at 60/min and
should happen outside the 16:30 window. It prioritises tickers that have
prices in the cache and clear the ADV gate, then everything else.
"""
import argparse
import sqlite3

from src.cache import init_db, purge_failed_tickers, count_failed_tickers
from src.config import load_config

DB_PATH = "data/cache.db"


def _adv_ranked_tickers(db_path: str, min_adv: float) -> list[str]:
    """Tickers with cached prices, ordered by 20-day average dollar volume
    (desc), liquid names first, then the rest."""
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        """
        SELECT ticker, AVG(close * volume) AS adv
        FROM (
            SELECT ticker, close, volume,
                   ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) AS rn
            FROM prices
        )
        WHERE rn <= 20
        GROUP BY ticker
        ORDER BY adv DESC
        """
    ).fetchall()
    conn.close()
    liquid = [t for t, adv in rows if adv is not None and adv >= min_adv]
    rest = [t for t, adv in rows if not (adv is not None and adv >= min_adv)]
    return liquid + rest


def cmd_purge_failed(db_path: str, before: str | None) -> dict:
    n_before = count_failed_tickers(db_path)
    deleted = purge_failed_tickers(db_path, before)
    return {"before": n_before, "deleted": deleted, "after": count_failed_tickers(db_path)}


def cmd_warm_finnhub(db_path: str, cfg: dict, max_fetch: int) -> dict:
    from src.finnhub_data import fetch_metrics, fetch_profiles

    tickers = _adv_ranked_tickers(db_path, cfg["liquidity_gate"]["min_avg_dollar_vol_20d"])
    tickers = [t for t in tickers if t != "SPY"]
    cpm = int(cfg.get("finnhub", {}).get("calls_per_minute", 60))
    ttl = int(cfg.get("finnhub", {}).get("metrics_ttl_days", 7))
    _, m_stats = fetch_metrics(tickers, db_path, ttl_days=ttl, max_fetch=max_fetch, calls_per_minute=cpm)
    _, p_stats = fetch_profiles(tickers, db_path, cap_ttl_days=ttl, max_fetch=max_fetch, calls_per_minute=cpm)
    return {"candidates": len(tickers), "metrics": m_stats, "profiles": p_stats}


def cmd_stats(db_path: str) -> dict:
    conn = sqlite3.connect(db_path)
    out = {}
    for table in ("prices", "market_cap", "fundamentals", "edgar", "fh_metrics", "fh_profile", "failed_tickers"):
        try:
            out[table] = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        except sqlite3.OperationalError:
            out[table] = None
    out["distinct_price_tickers"] = conn.execute("SELECT COUNT(DISTINCT ticker) FROM prices").fetchone()[0]
    conn.close()
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="cache_maint", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", default=DB_PATH)
    sub = p.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("purge-failed")
    sp.add_argument("--before", default=None, help="ISO timestamp; only rows older than this")
    sp = sub.add_parser("warm-finnhub")
    sp.add_argument("--max", type=int, default=3000)
    sub.add_parser("stats")
    args = p.parse_args(argv)

    init_db(args.db)
    if args.cmd == "purge-failed":
        out = cmd_purge_failed(args.db, args.before)
    elif args.cmd == "warm-finnhub":
        out = cmd_warm_finnhub(args.db, load_config(), args.max)
    else:
        out = cmd_stats(args.db)
    import json
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4:** Run:

```bash
/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m pytest tests/test_cache_maint.py -q
```

Expected: 3 passed.

- [ ] **Step 5:** Commit as `feat(cache_maint): purge quarantine and warm the Finnhub cache`. Per the user's global CLAUDE.md §6: run `git status` first and **ask the user before committing**. Never add a Co-Authored-By line.


### Task 0.9: Operational steps (no code) — do these before the 16:30 ET run

- [ ] **Step 1:** Purge the 2026-08-24/27 quarantine (everything; dead tickers re-quarantine themselves within one run at negligible cost):

```bash
cd /Users/arhanbarve/Code/screener && /Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m src.cache_maint purge-failed
```
Expected: `{"before": 9620, "deleted": 9620, "after": 0}` (numbers may differ slightly).

- [ ] **Step 2:** Start warming the Finnhub cache in the background (≈85 min for ~2,500 liquid names; the run itself is capped at 600 lookups):

```bash
set -a; source .env; set +a
nohup /Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m src.cache_maint warm-finnhub --max 3000 > logs/warm_finnhub.log 2>&1 &
```
Expected in the log after it finishes: `"metrics": {"fetched": ~2500, ...}, "profiles": {...}`.

- [ ] **Step 3:** Dry-run the pipeline once, timed. This is a real run (writes cache, output, stats) — that is intended; the 16:30 job then skips because a fresh output exists.

```bash
time /Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m src.run 2>&1 | tee logs/manual_run_phase0.log | grep -E "\[universe\]|\[prices\]|\[adv_gate\]|\[finnhub\]|\[caps\]|\[liquidity_gate\]|\[stage2\]|\[fund_block\]|\[fund_gate\]|\[compose\]|\[health\]|\[output\]|Error"
```
Expected: `[universe] tradability filter: 9275 → ~6300 tradable`; `[adv_gate] … → ≥ 800 survivors` (≈2,400 expected); `[liquidity_gate] … → ≥ 500` (≈1,900); `[compose] N ranked → top 20` with N ≥ 150; `[output] output/screen_latest.json`; wall time < 35 min. If a `[health]` line appears the run exits 1 — read the counts; if the universe really is small (holiday), rerun with `--allow-degraded` and lower the floor deliberately.

- [ ] **Step 4:** Verify the artifacts:

```bash
/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -c "import json;d=json.load(open('output/screen_latest.json'));print(d['date'],d['health']['ok'],d['health'].get('adv_survivors'),d['health'].get('cap_survivors'),len(d['rows']),len(d['ranking_tail']))"
/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m src.run_status --log logs/manual_run_phase0.log --rc 0 --started "$(date +%Y-%m-%dT%H:%M:%S)" --duration 0 --no-email && /Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -c "import json;print(json.load(open('run_status.json'))['stats']['ok'])"
/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m src.watchdog --skip-tests | head -5
```
Expected: `2026-09-04 True 2400 1900 20 100`; `True`; watchdog line ` ok   screen …`.

- [ ] **Step 5:** Ask the user to confirm the manual run's top 20 looks like real liquid names (AAPL-class universe, no OTC), then commit Phase 0 (ask first).
