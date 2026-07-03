# Stock Screener Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a nightly US equity screener that ranks ~8,000 tickers on a momentum/revision/surprise/RS composite, gated by liquidity and quality, outputting a top-N CSV and markdown summary.

**Architecture:** Staged pipeline — SEC universe build → yfinance bulk price pull + liquidity gate → Finnhub/EDGAR fundamentals on survivors only → cross-sectional z-score composite → CSV + markdown output. Each stage is a separate module with a well-defined DataFrame interface. SQLite cache prevents redundant network calls on reruns.

**Tech Stack:** Python 3.11+, pandas, numpy, yfinance, finnhub-python, requests, pyyaml, python-dotenv, pandas-ta, sqlite3 (stdlib)

---

## File Map

| File | Responsibility |
|---|---|
| `requirements.txt` | Pinned dependencies |
| `config.yaml` | All thresholds and weights (no magic numbers in code) |
| `.env.template` | API key template |
| `src/config.py` | Load config.yaml + .env; single source of truth |
| `src/cache.py` | SQLite read/write with TTL; three tables: prices, fundamentals, edgar |
| `src/universe.py` | Stage 1: SEC company_tickers.json → universe.parquet |
| `src/factors.py` | Pure factor functions (price series / fundamentals → float) |
| `src/prices.py` | Stage 2: yfinance batch download → price factors + liquidity gate |
| `src/fundamentals.py` | Stage 3: SEC EDGAR + Finnhub pulls on survivors |
| `src/compose.py` | Stage 4: winsorize + z-score + weighted composite + rank |
| `src/output.py` | Stage 5: CSV + markdown writer |
| `src/run.py` | Pipeline orchestrator; calls stages in order |
| `tests/test_factors.py` | Unit tests for every factor formula |
| `tests/test_cache.py` | Unit tests for cache TTL logic |
| `tests/test_compose.py` | Unit tests for z-score and winsorize |
| `backtest/backtest.py` | Optional monthly-rebalance backtest vs SPY |

---

## Task 1: Project Scaffold

**Files:**
- Create: `requirements.txt`
- Create: `config.yaml`
- Create: `.env.template`
- Create: `src/__init__.py`

- [ ] **Step 1: Write requirements.txt**

```
pandas==2.2.2
numpy==1.26.4
yfinance==0.2.40
finnhub-python==2.4.20
requests==2.32.3
pyyaml==6.0.1
python-dotenv==1.0.1
pandas-ta==0.3.14b
scipy==1.13.1
```

- [ ] **Step 2: Write config.yaml**

```yaml
universe:
  source: sec
  exclude_etfs: true
  min_price: 5.0

liquidity_gate:
  min_market_cap: 300000000
  min_avg_dollar_vol_20d: 5000000

quality_gate:
  gross_profitability_min: median

factors:
  weights:
    mom_12_1: 0.35
    rev_breadth: 0.25
    sue: 0.20
    rs_6m: 0.20
  winsorize_pct: 0.01
  missing_factor_treatment: neutral   # neutral | exclude

confirmation:
  require_above_sma200: true
  max_pct_below_52w_high: 0.10

output:
  top_n: 30
  include_squeeze_screen: true

cache:
  price_ttl_hours: 18
  fundamentals_ttl_days: 7
  edgar_ttl_days: 30

finnhub:
  calls_per_minute: 60
```

- [ ] **Step 3: Write .env.template**

```
FINNHUB_API_KEY=your_key_here
SEC_USER_AGENT=YourName your@email.com
```

- [ ] **Step 4: Create src/__init__.py (empty)**

```python
```

- [ ] **Step 5: Install dependencies**

```bash
pip install -r requirements.txt
```

Expected: all packages install without errors.

- [ ] **Step 6: Commit**

```bash
git init
git add requirements.txt config.yaml .env.template src/__init__.py
git commit -m "feat: scaffold project with dependencies and config"
```

---

## Task 2: Config Loader (`src/config.py`)

**Files:**
- Create: `src/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_config.py
import os
import pytest
from src.config import load_config, get_env

def test_load_config_returns_dict():
    cfg = load_config("config.yaml")
    assert isinstance(cfg, dict)
    assert "universe" in cfg
    assert "liquidity_gate" in cfg
    assert "factors" in cfg

def test_load_config_has_weights():
    cfg = load_config("config.yaml")
    weights = cfg["factors"]["weights"]
    assert abs(sum(weights.values()) - 1.0) < 1e-9

def test_get_env_missing_key_raises():
    with pytest.raises(KeyError):
        get_env("NONEXISTENT_KEY_XYZ")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_config.py -v
```

Expected: `ModuleNotFoundError` or `ImportError`.

- [ ] **Step 3: Implement src/config.py**

```python
import os
import yaml
from dotenv import load_dotenv

load_dotenv()

def load_config(path: str = "config.yaml") -> dict:
    f = open(path, "r")
    cfg = yaml.safe_load(f)
    f.close()
    return cfg

def get_env(key: str) -> str:
    val = os.environ.get(key)
    if val is None:
        raise KeyError(f"Missing required env var: {key}")
    return val
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_config.py -v
```

Expected: 3 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/config.py tests/test_config.py
git commit -m "feat: config loader for yaml + dotenv"
```

---

## Task 3: SQLite Cache Layer (`src/cache.py`)

**Files:**
- Create: `src/cache.py`
- Create: `tests/test_cache.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_cache.py
import os
import tempfile
import pandas as pd
from datetime import datetime, timedelta
from src.cache import init_db, put_prices, get_prices, put_fundamentals, get_fundamentals, put_edgar, get_edgar

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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_cache.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement src/cache.py**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_cache.py -v
```

Expected: 5 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/cache.py tests/test_cache.py
git commit -m "feat: sqlite cache layer with TTL for prices, fundamentals, edgar"
```

---

## Task 4: Factor Formulas (`src/factors.py`)

**Files:**
- Create: `src/factors.py`
- Create: `tests/test_factors.py`

The spec requires every factor formula has a unit test against a hand-computed fixture. These tests are the acceptance criteria.

- [ ] **Step 1: Write failing tests with hand-computed fixtures**

```python
# tests/test_factors.py
import numpy as np
import pandas as pd
import pytest
from src.factors import (
    mom_12_1, mom_1m, compute_sue, rev_breadth_score,
    gp_assets_score, rs_vs_spy, pct_from_52w_high,
    breakout_flag, avg_dollar_vol, squeeze_flag,
)

# --- Shared fixture: 252-day synthetic price series ---
# Starts at 100, ends at 110 at t-21, ends at 112 at t=0
# mom_12_1 = (110 / 100) - 1 = 0.10
def make_price_series(n: int = 252) -> pd.Series:
    """Linear ramp: 100 at t-252 → 110 at t-21 → 112 at t=0."""
    prices = np.linspace(100.0, 110.0, n - 21)
    tail = np.linspace(110.0, 112.0, 21 + 1)[1:]
    all_prices = np.concatenate([prices, tail])
    idx = pd.date_range(end="2024-01-31", periods=n, freq="B")
    return pd.Series(all_prices, index=idx)

def test_mom_12_1_known_value():
    close = make_price_series(252)
    result = mom_12_1(close)
    # close[t-252]=100, close[t-21]=110 → (110/100)-1 = 0.10
    assert abs(result - 0.10) < 1e-6

def test_mom_1m_known_value():
    close = make_price_series(252)
    result = mom_1m(close)
    # close[t-21]=110, close[t]=112 → (112/110)-1
    expected = (112.0 / 110.0) - 1.0
    assert abs(result - expected) < 1e-6

def test_mom_12_1_requires_252_bars():
    close = make_price_series(252).iloc[:200]
    with pytest.raises(ValueError, match="252"):
        mom_12_1(close)

def test_sue_with_four_quarters():
    # actuals: [1.0, 1.1, 1.2, 1.3], estimates: [0.9, 1.0, 1.1, 1.0]
    # surprises: [0.1, 0.1, 0.1, 0.3]
    # latest_actual=1.3, latest_estimate=1.0
    # std(surprises)=0.1  →  sue = (1.3 - 1.0) / 0.1 = 3.0
    actuals   = [1.0, 1.1, 1.2, 1.3]
    estimates = [0.9, 1.0, 1.1, 1.0]
    result = compute_sue(actuals, estimates)
    assert abs(result - 3.0) < 1e-6

def test_sue_fallback_fewer_than_four_quarters():
    # Only 2 quarters → fallback: (actual - estimate) / abs(estimate)
    actuals   = [1.3]
    estimates = [1.0]
    result = compute_sue(actuals, estimates)
    # (1.3 - 1.0) / 1.0 = 0.3
    assert abs(result - 0.3) < 1e-6

def test_rev_breadth_score_basic():
    # 3 up, 1 down, 5 total → (3-1)/5 = 0.4
    result = rev_breadth_score(n_up=3, n_down=1, n_total=5)
    assert abs(result - 0.4) < 1e-6

def test_rev_breadth_score_magnitude_fallback():
    # No analyst counts → use magnitude sign
    result = rev_breadth_score(n_up=0, n_down=0, n_total=0,
                               consensus_now=1.1, consensus_90d_ago=1.0)
    # (1.1 - 1.0) / 1.0 = 0.1
    assert abs(result - 0.1) < 1e-6

def test_gp_assets_score():
    result = gp_assets_score(revenue=100.0, cogs=60.0, assets=200.0)
    # (100 - 60) / 200 = 0.20
    assert abs(result - 0.20) < 1e-6

def test_rs_vs_spy():
    stock = make_price_series(252)
    spy   = make_price_series(252) * 0.95  # spy returned less
    result = rs_vs_spy(stock, spy, window=126)
    # stock_return_126d > spy_return_126d → positive RS
    assert result > 0

def test_pct_from_52w_high():
    close = make_price_series(252)
    result = pct_from_52w_high(close)
    # high is at t=0 (112), so pct = (112/112) - 1 = 0
    assert abs(result) < 1e-4

def test_breakout_flag_triggers():
    close  = make_price_series(252)
    # volume spike: last bar 2x average
    volumes = pd.Series(np.ones(252) * 1e6, index=close.index)
    volumes.iloc[-1] = 2.5e6
    result = breakout_flag(close, volumes)
    assert result is True

def test_avg_dollar_vol():
    close   = pd.Series([10.0] * 20)
    volumes = pd.Series([1_000_000] * 20)
    result  = avg_dollar_vol(close, volumes, window=20)
    assert abs(result - 10_000_000.0) < 1.0

def test_squeeze_flag_triggers():
    result = squeeze_flag(short_float=0.20, days_to_cover=6.0, mom_1m_val=0.05)
    assert result is True

def test_squeeze_flag_no_trigger_low_short():
    result = squeeze_flag(short_float=0.05, days_to_cover=6.0, mom_1m_val=0.05)
    assert result is False
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_factors.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement src/factors.py**

```python
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

# Trading-day constants
DAYS_1M  = 21
DAYS_3M  = 63
DAYS_6M  = 126
DAYS_12M = 252

def mom_12_1(close: pd.Series) -> float:
    if len(close) < DAYS_12M:
        raise ValueError(f"Need at least 252 bars; got {len(close)}")
    return float(close.iloc[-DAYS_1M] / close.iloc[-DAYS_12M]) - 1.0

def mom_1m(close: pd.Series) -> float:
    if len(close) < DAYS_1M + 1:
        raise ValueError(f"Need at least {DAYS_1M+1} bars; got {len(close)}")
    return float(close.iloc[-1] / close.iloc[-DAYS_1M]) - 1.0

def compute_sue(actuals: list, estimates: list) -> float:
    """Standardized Unexpected Earnings. actuals/estimates oldest-first."""
    if len(actuals) < 4 or len(estimates) < 4:
        # Fallback
        latest_a = actuals[-1]
        latest_e = estimates[-1]
        if abs(latest_e) < 1e-12:
            return 0.0
        return (latest_a - latest_e) / abs(latest_e)
    surprises = [a - e for a, e in zip(actuals, estimates)]
    std_s = float(np.std(surprises, ddof=1))
    if std_s < 1e-12:
        return 0.0
    return (surprises[-1]) / std_s

def rev_breadth_score(
    n_up: int,
    n_down: int,
    n_total: int,
    consensus_now: float = 0.0,
    consensus_90d_ago: float = 0.0,
) -> float:
    if n_total > 0:
        return (n_up - n_down) / n_total
    # Magnitude fallback
    if abs(consensus_90d_ago) < 1e-12:
        return 0.0
    return (consensus_now - consensus_90d_ago) / abs(consensus_90d_ago)

def gp_assets_score(revenue: float, cogs: float, assets: float) -> float:
    if assets <= 0:
        return float("nan")
    return (revenue - cogs) / assets

def rs_vs_spy(stock_close: pd.Series, spy_close: pd.Series, window: int = DAYS_6M) -> float:
    if len(stock_close) < window + 1 or len(spy_close) < window + 1:
        return float("nan")
    stock_ret = float(stock_close.iloc[-1] / stock_close.iloc[-window]) - 1.0
    spy_ret   = float(spy_close.iloc[-1] / spy_close.iloc[-window]) - 1.0
    return stock_ret - spy_ret

def rs_slope(stock_close: pd.Series, spy_close: pd.Series, window: int = DAYS_3M) -> float:
    """Slope of stock/spy ratio over last `window` days."""
    if len(stock_close) < window or len(spy_close) < window:
        return float("nan")
    ratio = (stock_close / spy_close).iloc[-window:]
    x = np.arange(len(ratio), dtype=float)
    slope, *_ = scipy_stats.linregress(x, ratio.values)
    return float(slope)

def pct_from_52w_high(close: pd.Series) -> float:
    if len(close) < DAYS_12M:
        return float("nan")
    high_52w = close.iloc[-DAYS_12M:].max()
    return float(close.iloc[-1] / high_52w) - 1.0

def breakout_flag(close: pd.Series, volume: pd.Series) -> bool:
    if len(close) < DAYS_12M or len(volume) < 50:
        return False
    high_52w   = close.iloc[-DAYS_12M:].max()
    pct_off    = float(close.iloc[-1] / high_52w) - 1.0
    avg_vol_50 = float(volume.iloc[-50:].mean())
    return pct_off >= -0.05 and float(volume.iloc[-1]) > 1.5 * avg_vol_50

def avg_dollar_vol(close: pd.Series, volume: pd.Series, window: int = 20) -> float:
    n = min(window, len(close), len(volume))
    return float((close.iloc[-n:] * volume.iloc[-n:]).mean())

def squeeze_flag(short_float: float, days_to_cover: float, mom_1m_val: float) -> bool:
    return short_float > 0.15 and days_to_cover > 5.0 and mom_1m_val > 0.0
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_factors.py -v
```

Expected: all 13 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/factors.py tests/test_factors.py
git commit -m "feat: factor formulas with unit tests (mom, sue, rev_breadth, gp_assets, rs, squeeze)"
```

---

## Task 5: Universe Builder (`src/universe.py`)

**Files:**
- Create: `src/universe.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_universe.py
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
import json
from src.universe import parse_sec_tickers, filter_universe

SAMPLE_SEC = {
    "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
    "1": {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft Corp"},
    "2": {"cik_str": 1018724, "ticker": "AMZN-WT", "title": "Amazon Warrant"},
    "3": {"cik_str": 1001085, "ticker": "SPXU", "title": "ProShares UltraPro Short S&P500"},
}

def test_parse_sec_tickers():
    df = parse_sec_tickers(SAMPLE_SEC)
    assert "ticker" in df.columns
    assert "cik" in df.columns
    assert len(df) == 4

def test_filter_removes_warrants():
    df = parse_sec_tickers(SAMPLE_SEC)
    result = filter_universe(df)
    assert "AMZN-WT" not in result["ticker"].values

def test_filter_removes_etfs_by_name():
    df = parse_sec_tickers(SAMPLE_SEC)
    result = filter_universe(df)
    # SPXU contains "proshares" and "short" patterns → excluded
    assert "SPXU" not in result["ticker"].values

def test_cik_formatted_as_10digit_string():
    df = parse_sec_tickers(SAMPLE_SEC)
    assert df["cik"].iloc[0] == "0000320193"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_universe.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement src/universe.py**

```python
import os
import time
import requests
import pandas as pd
from src.config import load_config, get_env

SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

# Terms that identify non-common-stock issues to exclude
_EXCLUDE_TERMS = [
    " warrant", "-wt", " unit", " pfd", " preferred",
    "proshares", "ishares", "invesco", "direxion",
    " etf", "trust", "fund", " lp ", " lp$",
]

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
    return df[~mask_warrant & ~mask_exclude].reset_index(drop=True)

def build_universe(cfg: dict, out_path: str = "data/universe.parquet") -> pd.DataFrame:
    raw = fetch_sec_tickers()
    df = parse_sec_tickers(raw)
    if cfg["universe"].get("exclude_etfs", True):
        df = filter_universe(df)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"[universe] {len(df)} tickers written to {out_path}")
    return df
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_universe.py -v
```

Expected: 4 PASSED.

- [ ] **Step 5: Verify live (optional, needs .env)**

```bash
python -c "
from src.config import load_config
from src.universe import build_universe
cfg = load_config()
df = build_universe(cfg)
print(df.head())
print('rows:', len(df))
"
```

Expected output: ~8,000–10,000 rows with ticker/cik/name columns.

- [ ] **Step 6: Commit**

```bash
git add src/universe.py tests/test_universe.py
git commit -m "feat: universe builder from SEC company_tickers.json with ETF/warrant filtering"
```

---

## Task 6: Price Pipeline (`src/prices.py`)

**Files:**
- Create: `src/prices.py`

This module pulls ~400 days of daily OHLCV for the universe in batches of 200, caches in SQLite, computes all price-based factors, and applies the liquidity gate.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_prices.py
import pandas as pd
import numpy as np
import tempfile, os
from unittest.mock import patch, MagicMock
from src.cache import init_db
from src.prices import compute_price_factors, apply_liquidity_gate

def make_ohlcv(n=252, start_price=50.0):
    prices = np.linspace(start_price, start_price * 1.1, n)
    idx = pd.date_range(end="2024-01-31", periods=n, freq="B")
    return pd.DataFrame({
        "open": prices * 0.99,
        "high": prices * 1.01,
        "low":  prices * 0.98,
        "close": prices,
        "volume": np.ones(n) * 2_000_000,
    }, index=idx)

def test_compute_price_factors_returns_required_columns():
    df = make_ohlcv(252)
    spy = make_ohlcv(252, start_price=450.0)
    result = compute_price_factors("AAPL", df, spy, market_cap=5e11)
    for col in ["mom_12_1", "rs_6m", "pct_from_high", "avg_dollar_vol_20d", "market_cap", "mom_1m"]:
        assert col in result, f"Missing column: {col}"

def test_compute_price_factors_skips_short_series():
    df = make_ohlcv(100)  # < 252
    spy = make_ohlcv(252, start_price=450.0)
    result = compute_price_factors("AAPL", df, spy, market_cap=5e11)
    assert result is None

def test_apply_liquidity_gate():
    rows = [
        {"ticker": "A", "market_cap": 400e6, "avg_dollar_vol_20d": 10e6},
        {"ticker": "B", "market_cap": 200e6, "avg_dollar_vol_20d": 10e6},  # mcap fail
        {"ticker": "C", "market_cap": 400e6, "avg_dollar_vol_20d": 2e6},   # vol fail
    ]
    df = pd.DataFrame(rows)
    cfg = {"liquidity_gate": {"min_market_cap": 300e6, "min_avg_dollar_vol_20d": 5e6}}
    result = apply_liquidity_gate(df, cfg)
    assert list(result["ticker"]) == ["A"]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_prices.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement src/prices.py**

```python
import time
import logging
import pandas as pd
import numpy as np
import yfinance as yf
from typing import Optional
from src.factors import (
    mom_12_1, mom_1m, rs_vs_spy, rs_slope,
    pct_from_52w_high, breakout_flag, avg_dollar_vol,
)
from src.cache import get_prices, put_prices

logger = logging.getLogger(__name__)

BATCH_SIZE = 200
HISTORY_DAYS = 420  # ~400 calendar days → ~280 trading days, enough for 252


def _fetch_batch_yfinance(tickers: list[str]) -> dict[str, pd.DataFrame]:
    joined = " ".join(tickers)
    try:
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
        return {}

    result = {}
    if len(tickers) == 1:
        t = tickers[0]
        raw.columns = [c.lower() for c in raw.columns]
        result[t] = raw.dropna(how="all")
    else:
        for t in tickers:
            if t not in raw.columns.get_level_values(0):
                continue
            df = raw[t].copy()
            df.columns = [c.lower() for c in df.columns]
            df = df.dropna(how="all")
            if len(df) > 0:
                result[t] = df
    return result


def _get_market_cap(ticker: str) -> Optional[float]:
    try:
        info = yf.Ticker(ticker).fast_info
        return float(getattr(info, "market_cap", None) or 0) or None
    except Exception:
        return None


def compute_price_factors(
    ticker: str,
    df: pd.DataFrame,
    spy_df: pd.DataFrame,
    market_cap: Optional[float],
) -> Optional[dict]:
    if len(df) < 252:
        return None
    close  = df["close"]
    volume = df["volume"]
    spy_close = spy_df["close"]

    try:
        return {
            "ticker": ticker,
            "market_cap": market_cap,
            "mom_12_1": mom_12_1(close),
            "mom_1m": mom_1m(close),
            "rs_6m": rs_vs_spy(close, spy_close, window=126),
            "rs_3m": rs_vs_spy(close, spy_close, window=63),
            "rs_slope": rs_slope(close, spy_close),
            "pct_from_high": pct_from_52w_high(close),
            "breakout": breakout_flag(close, volume),
            "avg_dollar_vol_20d": avg_dollar_vol(close, volume, window=20),
            "price": float(close.iloc[-1]),
            "sma_200": float(close.iloc[-252:].mean()),
            "close_series": close,   # kept for SMA200 gate in compose
        }
    except Exception as e:
        logger.warning(f"[prices] factor error for {ticker}: {e}")
        return None


def apply_liquidity_gate(factors_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    gate = cfg["liquidity_gate"]
    min_mcap = gate["min_market_cap"]
    min_vol  = gate["min_avg_dollar_vol_20d"]
    before = len(factors_df)
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
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Returns:
      price_store: ticker → OHLCV DataFrame (all tickers with >=252 bars)
      factors_df:  one row per ticker that passed the liquidity gate
    """
    ttl = cfg["cache"]["price_ttl_hours"]
    tickers = universe_df["ticker"].tolist()

    # Pull SPY first
    spy_data = _fetch_batch_yfinance(["SPY"])
    spy_df   = spy_data.get("SPY", pd.DataFrame())
    if spy_df.empty:
        raise RuntimeError("Failed to fetch SPY — cannot compute relative strength")

    price_store: dict[str, pd.DataFrame] = {"SPY": spy_df}
    factor_rows = []

    batches = [tickers[i:i+BATCH_SIZE] for i in range(0, len(tickers), BATCH_SIZE)]
    for batch_idx, batch in enumerate(batches):
        logger.info(f"[prices] batch {batch_idx+1}/{len(batches)} ({len(batch)} tickers)")

        # Check cache first
        to_fetch = []
        for t in batch:
            cached = get_prices(db_path, t, ttl_hours=ttl)
            if cached is not None and len(cached) >= 252:
                price_store[t] = cached
            else:
                to_fetch.append(t)

        if to_fetch:
            fetched = _fetch_batch_yfinance(to_fetch)
            for t, df in fetched.items():
                put_prices(db_path, t, df)
                price_store[t] = df

        # Compute factors for this batch
        for t in batch:
            df = price_store.get(t)
            if df is None or len(df) < 252:
                continue
            mcap = _get_market_cap(t)
            row = compute_price_factors(t, df, spy_df, mcap)
            if row is not None:
                row["name"] = universe_df.loc[universe_df["ticker"] == t, "name"].values[0] if len(universe_df.loc[universe_df["ticker"] == t]) > 0 else ""
                row["cik"]  = universe_df.loc[universe_df["ticker"] == t, "cik"].values[0]  if len(universe_df.loc[universe_df["ticker"] == t]) > 0 else ""
                factor_rows.append(row)

        # Polite pause between batches
        if batch_idx < len(batches) - 1:
            time.sleep(1.0)

    factors_df = pd.DataFrame([{k: v for k, v in r.items() if k != "close_series"} for r in factor_rows])
    factors_df = apply_liquidity_gate(factors_df, cfg)
    print(f"[prices] {len(tickers)} universe → {len(factors_df)} passed liquidity gate")
    return price_store, factors_df
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_prices.py -v
```

Expected: 3 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/prices.py tests/test_prices.py
git commit -m "feat: price pipeline with yfinance batch download, price factors, liquidity gate"
```

---

## Task 7: Fundamentals Pipeline (`src/fundamentals.py`)

**Files:**
- Create: `src/fundamentals.py`

This module runs only on survivors from Stage 2. It pulls SEC EDGAR for gross profitability and Finnhub for earnings surprise, revisions, short interest, and insider transactions. Rate-limiting is critical here.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_fundamentals.py
import pytest
from unittest.mock import patch, MagicMock
from src.fundamentals import (
    parse_edgar_gp,
    parse_finnhub_surprise,
    parse_finnhub_revisions,
    parse_short_interest,
    parse_insider_buys,
)

SAMPLE_EDGAR = {
    "facts": {
        "us-gaap": {
            "Revenues": {
                "units": {"USD": [
                    {"end": "2023-09-30", "val": 383285000000, "form": "10-K", "filed": "2023-11-03"},
                    {"end": "2022-09-24", "val": 394328000000, "form": "10-K", "filed": "2022-10-28"},
                ]}
            },
            "CostOfGoodsAndServicesSold": {
                "units": {"USD": [
                    {"end": "2023-09-30", "val": 214137000000, "form": "10-K", "filed": "2023-11-03"},
                ]}
            },
            "Assets": {
                "units": {"USD": [
                    {"end": "2023-09-30", "val": 352583000000, "form": "10-K", "filed": "2023-11-03"},
                ]}
            },
        }
    }
}

def test_parse_edgar_gp():
    gp, rev, cogs, assets = parse_edgar_gp(SAMPLE_EDGAR)
    # (383285 - 214137) / 352583 = ~0.4797
    assert abs(gp - (383285e6 - 214137e6) / 352583e6) < 1e-4

def test_parse_edgar_gp_missing_tag_raises():
    with pytest.raises(KeyError):
        parse_edgar_gp({"facts": {"us-gaap": {}}})

SAMPLE_EARNINGS = [
    {"period": "2023-09-30", "actual": 1.46, "estimate": 1.39, "symbol": "AAPL"},
    {"period": "2023-06-30", "actual": 1.26, "estimate": 1.19, "symbol": "AAPL"},
    {"period": "2023-03-31", "actual": 1.52, "estimate": 1.43, "symbol": "AAPL"},
    {"period": "2022-12-31", "actual": 1.88, "estimate": 1.94, "symbol": "AAPL"},
]

def test_parse_finnhub_surprise():
    actuals, estimates = parse_finnhub_surprise(SAMPLE_EARNINGS)
    assert len(actuals) == 4
    assert actuals[-1] == 1.46   # most recent last

def test_parse_finnhub_revisions_empty():
    rev_b, rev_m = parse_finnhub_revisions({})
    assert rev_b == 0.0
    assert rev_m == 0.0

def test_parse_short_interest_from_info():
    info = {"sharesShort": 100_000_000, "floatShares": 500_000_000, "averageVolume": 20_000_000}
    short_float, dtc = parse_short_interest(info)
    assert abs(short_float - 0.20) < 1e-6
    assert abs(dtc - 5.0) < 1e-6

def test_parse_insider_buys():
    transactions = [
        {"transactionCode": "P", "name": "Tim Cook", "transactionDate": "2024-01-10"},
        {"transactionCode": "P", "name": "Luca Maestri", "transactionDate": "2024-01-15"},
        {"transactionCode": "S", "name": "Tim Cook", "transactionDate": "2024-01-20"},
    ]
    count = parse_insider_buys(transactions, days=90)
    assert count == 2  # 2 distinct insiders with purchases
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_fundamentals.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement src/fundamentals.py**

```python
import time
import logging
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
import finnhub
from src.config import load_config, get_env
from src.cache import get_fundamentals, put_fundamentals, get_edgar, put_edgar

logger = logging.getLogger(__name__)

EDGAR_BASE = "https://data.sec.gov/api/xbrl/companyfacts"

# Prioritized tag lists (try first match)
REVENUE_TAGS = [
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "RevenuesNetOfInterestExpense",
    "SalesRevenueNet",
]
COGS_TAGS = [
    "CostOfGoodsAndServicesSold",
    "CostOfRevenue",
    "CostOfGoodsSold",
    "CostOfSales",
]
ASSETS_TAGS = ["Assets"]


def _sec_headers() -> dict:
    return {"User-Agent": get_env("SEC_USER_AGENT")}


def _latest_annual(entries: list) -> Optional[float]:
    annual = [e for e in entries if e.get("form") in ("10-K", "20-F")]
    if not annual:
        return None
    annual.sort(key=lambda e: e.get("end", ""), reverse=True)
    return float(annual[0]["val"])


def _resolve_tag(facts: dict, tag_list: list) -> Optional[float]:
    usgaap = facts.get("us-gaap", {})
    for tag in tag_list:
        if tag in usgaap:
            entries = usgaap[tag].get("units", {}).get("USD", [])
            val = _latest_annual(entries)
            if val is not None:
                return val
    return None


def parse_edgar_gp(data: dict) -> tuple[float, float, float, float]:
    facts = data["facts"]
    revenue = _resolve_tag(facts, REVENUE_TAGS)
    cogs    = _resolve_tag(facts, COGS_TAGS)
    assets  = _resolve_tag(facts, ASSETS_TAGS)
    if revenue is None or cogs is None or assets is None:
        raise KeyError("Could not resolve revenue/cogs/assets XBRL tags")
    gp = (revenue - cogs) / assets
    return gp, revenue, cogs, assets


def fetch_edgar(cik: str, db_path: str, ttl_days: int) -> Optional[dict]:
    cached = get_edgar(db_path, cik, ttl_days=ttl_days)
    if cached is not None:
        return cached
    url = f"{EDGAR_BASE}/CIK{cik}.json"
    try:
        resp = requests.get(url, headers=_sec_headers(), timeout=30)
        resp.raise_for_status()
        data = resp.json()
        gp, rev, cogs, assets = parse_edgar_gp(data)
        put_edgar(db_path, cik, gp_assets=gp, revenue=rev, cogs=cogs, assets=assets)
        return {"gp_assets": gp, "revenue": rev, "cogs": cogs, "assets": assets}
    except Exception as e:
        logger.warning(f"[edgar] failed for CIK {cik}: {e}")
        return None


def parse_finnhub_surprise(earnings: list) -> tuple[list, list]:
    earnings_sorted = sorted(earnings, key=lambda e: e.get("period", ""))
    actuals   = [e["actual"]   for e in earnings_sorted if e.get("actual") is not None]
    estimates = [e["estimate"] for e in earnings_sorted if e.get("estimate") is not None]
    min_len = min(len(actuals), len(estimates))
    return actuals[-min_len:], estimates[-min_len:]


def parse_finnhub_revisions(trend_data: dict) -> tuple[float, float]:
    """Returns (rev_breadth, rev_magnitude) from Finnhub EPS trend."""
    try:
        trend = trend_data.get("trend", [])
        if not trend:
            return 0.0, 0.0
        latest = sorted(trend, key=lambda t: t.get("period", ""), reverse=True)[0]
        eps_up   = latest.get("epsTrendUp", 0) or 0
        eps_down = latest.get("epsTrendDown", 0) or 0
        total    = eps_up + eps_down
        breadth  = (eps_up - eps_down) / total if total > 0 else 0.0
        cur = latest.get("epsTrend", {}).get("current", 0)
        ago = latest.get("epsTrend", {}).get("3month", cur)
        mag = (cur - ago) / abs(ago) if abs(ago) > 1e-12 else 0.0
        return float(breadth), float(mag)
    except Exception:
        return 0.0, 0.0


def parse_short_interest(info: dict) -> tuple[float, float]:
    shares_short = float(info.get("sharesShort") or 0)
    float_shares = float(info.get("floatShares") or 1)
    avg_vol      = float(info.get("averageVolume") or 1)
    short_float  = shares_short / float_shares if float_shares > 0 else 0.0
    dtc          = shares_short / avg_vol if avg_vol > 0 else 0.0
    return short_float, dtc


def parse_insider_buys(transactions: list, days: int = 90) -> int:
    cutoff = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
    buyers = set()
    for tx in transactions:
        if tx.get("transactionCode") == "P" and tx.get("transactionDate", "") >= cutoff:
            buyers.add(tx.get("name", ""))
    return len(buyers)


class _TokenBucket:
    def __init__(self, rate: int):
        self._rate = rate
        self._tokens = rate
        self._last = time.monotonic()

    def consume(self):
        now = time.monotonic()
        self._tokens = min(self._rate, self._tokens + (now - self._last) * self._rate / 60.0)
        self._last = now
        if self._tokens < 1:
            time.sleep((1 - self._tokens) * 60.0 / self._rate)
            self._tokens = 0
        else:
            self._tokens -= 1


def fetch_all_fundamentals(
    survivors_df: pd.DataFrame,
    cfg: dict,
    db_path: str,
) -> pd.DataFrame:
    fh_key = get_env("FINNHUB_API_KEY")
    fh     = finnhub.Client(api_key=fh_key)
    bucket = _TokenBucket(cfg["finnhub"]["calls_per_minute"])

    ttl_fund  = cfg["cache"]["fundamentals_ttl_days"]
    ttl_edgar = cfg["cache"]["edgar_ttl_days"]

    rows = []
    total = len(survivors_df)
    for idx, record in survivors_df.iterrows():
        ticker = record["ticker"]
        cik    = record.get("cik", "")
        logger.info(f"[fundamentals] {idx+1}/{total} {ticker}")

        row = {"ticker": ticker}

        # EDGAR: gross profitability
        if cik:
            edgar = fetch_edgar(cik, db_path, ttl_days=ttl_edgar)
            row["gp_assets"] = edgar["gp_assets"] if edgar else float("nan")
        else:
            row["gp_assets"] = float("nan")

        # Finnhub: earnings surprise
        cached_fund = get_fundamentals(db_path, ticker, ttl_days=ttl_fund)
        if cached_fund:
            row.update(cached_fund)
        else:
            fund = {}
            try:
                bucket.consume()
                earnings = fh.company_earnings(ticker, limit=8)
                actuals, estimates = parse_finnhub_surprise(earnings)
                from src.factors import compute_sue
                fund["sue"] = compute_sue(actuals, estimates) if actuals else 0.0
            except Exception as e:
                logger.warning(f"[fundamentals] earnings for {ticker}: {e}")
                fund["sue"] = float("nan")

            try:
                bucket.consume()
                trend = fh.eps_estimate(ticker, freq="quarterly")
                breadth, mag = parse_finnhub_revisions({"trend": trend.get("data", [])})
                fund["rev_breadth"] = breadth
                fund["rev_magnitude"] = mag
            except Exception as e:
                logger.warning(f"[fundamentals] revisions for {ticker}: {e}")
                fund["rev_breadth"] = float("nan")
                fund["rev_magnitude"] = float("nan")

            try:
                bucket.consume()
                insider_tx = fh.stock_insider_transactions(ticker, _from="", to="")
                buys = parse_insider_buys(insider_tx.get("data", []))
                fund["insider_buys_90d"] = buys
                fund["insider_flag"] = buys >= 2
            except Exception as e:
                logger.warning(f"[fundamentals] insider for {ticker}: {e}")
                fund["insider_buys_90d"] = 0
                fund["insider_flag"] = False

            # Short interest from yfinance .info
            import yfinance as yf
            try:
                info = yf.Ticker(ticker).info
                sf, dtc = parse_short_interest(info)
                fund["short_float"] = sf
                fund["days_to_cover"] = dtc
                fund["sector"] = info.get("sector", "")
            except Exception:
                fund["short_float"] = float("nan")
                fund["days_to_cover"] = float("nan")
                fund["sector"] = ""

            put_fundamentals(db_path, ticker, fund)
            row.update(fund)

        rows.append(row)
        # SEC politeness: max 10 req/sec
        time.sleep(0.12)

    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_fundamentals.py -v
```

Expected: 7 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/fundamentals.py tests/test_fundamentals.py
git commit -m "feat: fundamentals pipeline with EDGAR gp_assets, Finnhub earnings/revisions/insider, rate limiting"
```

---

## Task 8: Compositing and Ranking (`src/compose.py`)

**Files:**
- Create: `src/compose.py`
- Create: `tests/test_compose.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_compose.py
import numpy as np
import pandas as pd
import pytest
from src.compose import winsorize_series, z_score_series, apply_quality_gate, apply_confirmation_gate, build_composite

def make_factors_df(n=50):
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "ticker":          [f"T{i}" for i in range(n)],
        "mom_12_1":        rng.normal(0.10, 0.20, n),
        "rev_breadth":     rng.uniform(-1, 1, n),
        "sue":             rng.normal(0, 2, n),
        "rs_6m":           rng.normal(0, 0.15, n),
        "gp_assets":       rng.uniform(0.0, 0.6, n),
        "price":           rng.uniform(10, 500, n),
        "sma_200":         rng.uniform(10, 500, n),
        "pct_from_high":   rng.uniform(-0.3, 0.0, n),
    })
    df["price"] = df["sma_200"] * 1.1  # all above SMA200 initially
    return df

def test_winsorize_clips_extremes():
    s = pd.Series([1, 2, 3, 4, 100])
    result = winsorize_series(s, pct=0.20)
    assert result.max() < 100

def test_z_score_has_unit_variance():
    s = pd.Series(np.random.default_rng(0).normal(5, 3, 100))
    z = z_score_series(s)
    assert abs(z.std() - 1.0) < 0.01
    assert abs(z.mean()) < 0.01

def test_quality_gate_drops_below_median():
    df = make_factors_df(50)
    cfg = {"quality_gate": {"gross_profitability_min": "median"}}
    result = apply_quality_gate(df, cfg)
    assert len(result) < len(df)
    median = df["gp_assets"].median()
    assert (result["gp_assets"] >= median).all()

def test_quality_gate_float_threshold():
    df = make_factors_df(50)
    cfg = {"quality_gate": {"gross_profitability_min": 0.3}}
    result = apply_quality_gate(df, cfg)
    assert (result["gp_assets"] >= 0.3).all()

def test_confirmation_gate_sma200():
    df = make_factors_df(50)
    df.loc[0, "price"] = df.loc[0, "sma_200"] * 0.9  # first row below SMA200
    cfg = {
        "confirmation": {"require_above_sma200": True, "max_pct_below_52w_high": 0.10},
        "quality_gate": {"gross_profitability_min": 0.0},
    }
    result = apply_confirmation_gate(df, cfg)
    assert "T0" not in result["ticker"].values

def test_build_composite_no_nan():
    df = make_factors_df(50)
    cfg = {
        "factors": {
            "weights": {"mom_12_1": 0.35, "rev_breadth": 0.25, "sue": 0.20, "rs_6m": 0.20},
            "winsorize_pct": 0.01,
            "missing_factor_treatment": "neutral",
        },
        "quality_gate": {"gross_profitability_min": 0.0},
        "confirmation": {"require_above_sma200": False, "max_pct_below_52w_high": 1.0},
        "output": {"top_n": 10},
    }
    result = build_composite(df, cfg)
    assert "composite" in result.columns
    assert not result["composite"].isna().any()
    assert result["composite"].is_monotonic_decreasing

def test_build_composite_sorted_descending():
    df = make_factors_df(50)
    cfg = {
        "factors": {
            "weights": {"mom_12_1": 0.35, "rev_breadth": 0.25, "sue": 0.20, "rs_6m": 0.20},
            "winsorize_pct": 0.01,
            "missing_factor_treatment": "neutral",
        },
        "quality_gate": {"gross_profitability_min": 0.0},
        "confirmation": {"require_above_sma200": False, "max_pct_below_52w_high": 1.0},
        "output": {"top_n": 10},
    }
    result = build_composite(df, cfg)
    assert result["composite"].iloc[0] >= result["composite"].iloc[-1]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_compose.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement src/compose.py**

```python
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

COMPOSITE_FACTORS = ["mom_12_1", "rev_breadth", "sue", "rs_6m"]


def winsorize_series(s: pd.Series, pct: float = 0.01) -> pd.Series:
    lo = s.quantile(pct)
    hi = s.quantile(1 - pct)
    return s.clip(lower=lo, upper=hi)


def z_score_series(s: pd.Series) -> pd.Series:
    mu  = s.mean()
    std = s.std()
    if std < 1e-12:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mu) / std


def apply_quality_gate(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    threshold = cfg["quality_gate"]["gross_profitability_min"]
    if "gp_assets" not in df.columns:
        return df
    gp = df["gp_assets"].fillna(-999)
    if threshold == "median":
        cutoff = gp.median()
    else:
        cutoff = float(threshold)
    before = len(df)
    result = df[gp >= cutoff].reset_index(drop=True)
    logger.info(f"[quality_gate] {before} → {len(result)} survivors")
    print(f"[quality_gate] {before} → {len(result)} survivors (gp_assets >= {cutoff:.4f})")
    return result


def apply_confirmation_gate(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    confirm = cfg["confirmation"]
    mask = pd.Series(True, index=df.index)
    if confirm.get("require_above_sma200", False) and "sma_200" in df.columns:
        mask &= df["price"] >= df["sma_200"]
    max_below = confirm.get("max_pct_below_52w_high", 1.0)
    if "pct_from_high" in df.columns:
        mask &= df["pct_from_high"] >= -(max_below)
    before = len(df)
    result = df[mask].reset_index(drop=True)
    logger.info(f"[confirmation_gate] {before} → {len(result)} survivors")
    print(f"[confirmation_gate] {before} → {len(result)} survivors")
    return result


def build_composite(factors_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = factors_df.copy()
    weights = cfg["factors"]["weights"]
    winsorize_pct = cfg["factors"]["winsorize_pct"]
    missing_treatment = cfg["factors"].get("missing_factor_treatment", "neutral")

    df = apply_quality_gate(df, cfg)
    df = apply_confirmation_gate(df, cfg)

    z_cols = {}
    for factor in COMPOSITE_FACTORS:
        if factor not in df.columns:
            logger.warning(f"[compose] factor {factor} missing entirely — treating all as neutral")
            df[f"z_{factor}"] = 0.0
            continue
        s = df[factor].copy()
        if missing_treatment == "neutral":
            s = s.fillna(s.mean())
        else:
            df = df[s.notna()].reset_index(drop=True)
            s = df[factor]
        s = winsorize_series(s, pct=winsorize_pct)
        z = z_score_series(s)
        df[f"z_{factor.split('_')[0]}_{factor.split('_')[1] if '_' in factor else factor}"] = z
        # Consistent z-column naming
        df[f"z_{factor}"] = z
        z_cols[factor] = f"z_{factor}"

    composite = pd.Series(np.zeros(len(df)), index=df.index)
    for factor, z_col in z_cols.items():
        w = weights.get(factor, 0.0)
        composite += w * df[z_col]

    df["composite"] = composite
    df = df.sort_values("composite", ascending=False).reset_index(drop=True)

    top_n = cfg["output"]["top_n"]
    result = df.head(top_n)
    print(f"[compose] {len(df)} ranked → top {len(result)} selected")
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_compose.py -v
```

Expected: 7 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/compose.py tests/test_compose.py
git commit -m "feat: compositing with winsorize, z-score, quality/confirmation gates, weighted ranking"
```

---

## Task 9: Output Writers (`src/output.py`)

**Files:**
- Create: `src/output.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_output.py
import os, tempfile, pandas as pd, numpy as np
from src.output import write_csv, write_markdown

def make_ranked_df(n=5):
    return pd.DataFrame({
        "ticker":        [f"T{i}" for i in range(n)],
        "name":          [f"Company {i}" for i in range(n)],
        "sector":        ["Tech"] * n,
        "composite":     np.linspace(2.0, 0.5, n),
        "z_mom_12_1":    np.linspace(1.5, 0.0, n),
        "z_rev_breadth": np.linspace(1.0, -0.5, n),
        "z_sue":         np.linspace(0.8, -0.2, n),
        "z_rs_6m":       np.linspace(0.6, 0.0, n),
        "mom_12_1":      np.linspace(0.30, 0.05, n),
        "rev_breadth":   np.linspace(0.6, 0.0, n),
        "sue":           np.linspace(3.0, 0.0, n),
        "rs_6m":         np.linspace(0.15, 0.0, n),
        "gp_assets":     np.linspace(0.40, 0.20, n),
        "pct_from_high": np.linspace(-0.02, -0.10, n),
        "short_float":   np.linspace(0.05, 0.20, n),
        "insider_buys_90d": [2, 1, 0, 1, 3],
        "price":         np.linspace(200.0, 50.0, n),
        "market_cap":    np.linspace(50e9, 1e9, n),
    })

def test_write_csv_creates_file():
    df = make_ranked_df()
    tmpdir = tempfile.mkdtemp()
    path = write_csv(df, tmpdir, "2024-01-31")
    assert os.path.exists(path)
    loaded = pd.read_csv(path)
    assert "composite" in loaded.columns
    assert len(loaded) == 5

def test_write_markdown_creates_file():
    df = make_ranked_df()
    tmpdir = tempfile.mkdtemp()
    path = write_markdown(df, tmpdir, "2024-01-31", squeeze_df=None)
    assert os.path.exists(path)
    content = open(path, "r")
    text = content.read()
    content.close()
    assert "T0" in text
    assert "composite" in text.lower() or "Rank" in text
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_output.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement src/output.py**

```python
import os
import pandas as pd

CSV_COLUMNS = [
    "ticker", "name", "sector", "composite",
    "z_mom_12_1", "z_rev_breadth", "z_sue", "z_rs_6m",
    "mom_12_1", "rev_breadth", "sue", "rs_6m",
    "gp_assets", "pct_from_high", "short_float", "insider_buys_90d",
    "price", "market_cap",
]


def _rationale(row: pd.Series) -> str:
    parts = []
    if row.get("z_mom_12_1", 0) > 1.0:
        parts.append(f"Top-decile 12-1 momentum ({row['mom_12_1']:.1%})")
    if row.get("z_rev_breadth", 0) > 1.0:
        rev = row.get("rev_breadth", 0)
        pct = int(round((rev + 1) / 2 * 100))
        parts.append(f"Analysts revised up (breadth={rev:.2f})")
    if row.get("z_sue", 0) > 1.0:
        parts.append(f"Beat estimate ({row['sue']:.1f} SUE)")
    if row.get("z_rs_6m", 0) > 1.0:
        parts.append(f"Strong 6m RS vs SPY ({row['rs_6m']:.1%})")
    if row.get("insider_buys_90d", 0) >= 2:
        parts.append(f"{int(row['insider_buys_90d'])} insiders bought last 90d")
    return "; ".join(parts) if parts else "Composite score"


def write_csv(df: pd.DataFrame, out_dir: str, date_str: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"screen_{date_str}.csv")
    cols = [c for c in CSV_COLUMNS if c in df.columns]
    out = open(path, "w")
    df[cols].to_csv(out, index=False, float_format="%.6f")
    out.close()
    return path


def write_markdown(
    df: pd.DataFrame,
    out_dir: str,
    date_str: str,
    squeeze_df: pd.DataFrame | None,
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"screen_{date_str}.md")
    lines = [
        f"# Stock Screen — {date_str}",
        "",
        f"**Universe screened:** {len(df)} finalists from US common-stock universe",
        "",
        "## Top Ranked Names",
        "",
        "| Rank | Ticker | Name | Sector | Composite | Rationale |",
        "|------|--------|------|--------|-----------|-----------|",
    ]
    for i, (_, row) in enumerate(df.iterrows(), 1):
        name    = str(row.get("name", ""))[:30]
        sector  = str(row.get("sector", ""))[:20]
        comp    = f"{row.get('composite', 0):.3f}"
        rationale = _rationale(row)
        lines.append(f"| {i} | {row['ticker']} | {name} | {sector} | {comp} | {rationale} |")

    if squeeze_df is not None and len(squeeze_df) > 0:
        lines += [
            "",
            "## Short Squeeze Candidates",
            "",
            "| Ticker | Short Float | Days to Cover | 1M Mom |",
            "|--------|-------------|---------------|--------|",
        ]
        for _, row in squeeze_df.iterrows():
            sf  = f"{row.get('short_float', 0):.1%}"
            dtc = f"{row.get('days_to_cover', 0):.1f}"
            m1  = f"{row.get('mom_1m', 0):.1%}"
            lines.append(f"| {row['ticker']} | {sf} | {dtc} | {m1} |")

    lines += ["", "---", "*Research tool only. Not investment advice.*"]
    f = open(path, "w")
    f.write("\n".join(lines) + "\n")
    f.close()
    return path


def print_top10(df: pd.DataFrame):
    print("\n=== TOP 10 ===")
    for i, (_, row) in enumerate(df.head(10).iterrows(), 1):
        comp = row.get("composite", 0)
        price = row.get("price", 0)
        print(f"  {i:2d}. {row['ticker']:<8} composite={comp:+.3f}  price=${price:.2f}  {_rationale(row)}")
    print()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_output.py -v
```

Expected: 2 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/output.py tests/test_output.py
git commit -m "feat: output writers for dated CSV and markdown with rationale column"
```

---

## Task 10: Pipeline Orchestrator (`src/run.py`)

**Files:**
- Create: `src/run.py`

- [ ] **Step 1: Write src/run.py**

```python
import logging
import os
from datetime import date
from src.config import load_config, get_env
from src.cache import init_db
from src.universe import build_universe
from src.prices import fetch_all_prices
from src.fundamentals import fetch_all_fundamentals
from src.factors import squeeze_flag
from src.compose import build_composite
from src.output import write_csv, write_markdown, print_top10
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DB_PATH       = "data/cache.db"
UNIVERSE_PATH = "data/universe.parquet"
OUTPUT_DIR    = "output"


def run(force_universe: bool = False):
    cfg = load_config()
    today = date.today().isoformat()

    os.makedirs("data", exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    init_db(DB_PATH)

    # Stage 1: Universe
    if force_universe or not os.path.exists(UNIVERSE_PATH):
        universe_df = build_universe(cfg, UNIVERSE_PATH)
    else:
        universe_df = pd.read_parquet(UNIVERSE_PATH)
        print(f"[universe] loaded {len(universe_df)} tickers from cache")

    # Stage 2: Prices + liquidity gate
    price_store, survivors_df = fetch_all_prices(universe_df, cfg, DB_PATH)
    print(f"[stage2] {len(universe_df)} → {len(survivors_df)} after liquidity gate")

    # Attach CIK from universe
    survivors_df = survivors_df.merge(
        universe_df[["ticker", "cik", "name"]],
        on="ticker",
        how="left",
        suffixes=("", "_uni"),
    )
    if "name_uni" in survivors_df.columns:
        survivors_df["name"] = survivors_df["name"].fillna(survivors_df["name_uni"])
        survivors_df = survivors_df.drop(columns=["name_uni"])

    # Stage 3: Fundamentals (survivors only)
    fund_df = fetch_all_fundamentals(survivors_df, cfg, DB_PATH)
    merged = survivors_df.merge(fund_df, on="ticker", how="left")
    print(f"[stage3] fundamentals fetched for {len(fund_df)} tickers")

    # Stage 4: Composite score
    ranked_df = build_composite(merged, cfg)

    # Squeeze screen (separate pass, no quality gate required)
    if cfg["output"].get("include_squeeze_screen", False):
        squeeze_rows = []
        for _, row in merged.iterrows():
            sf  = row.get("short_float", 0) or 0
            dtc = row.get("days_to_cover", 0) or 0
            m1  = row.get("mom_1m", 0) or 0
            if squeeze_flag(sf, dtc, m1):
                squeeze_rows.append(row)
        squeeze_df = pd.DataFrame(squeeze_rows) if squeeze_rows else None
        if squeeze_df is not None:
            print(f"[squeeze] {len(squeeze_df)} squeeze candidates")
    else:
        squeeze_df = None

    # Stage 5: Output
    csv_path = write_csv(ranked_df, OUTPUT_DIR, today)
    md_path  = write_markdown(ranked_df, OUTPUT_DIR, today, squeeze_df=squeeze_df)
    print(f"\n[output] {csv_path}")
    print(f"[output] {md_path}")
    print_top10(ranked_df)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the equity screener")
    parser.add_argument("--force-universe", action="store_true",
                        help="Re-fetch universe from SEC even if parquet exists")
    args = parser.parse_args()
    run(force_universe=args.force_universe)
```

- [ ] **Step 2: Do a dry-run smoke test (mocked, not live)**

```bash
python -c "
from src.config import load_config
from src.cache import init_db
import os; os.makedirs('data', exist_ok=True)
init_db('data/cache.db')
cfg = load_config()
print('config ok:', list(cfg.keys()))
"
```

Expected: `config ok: ['universe', 'liquidity_gate', ...]`

- [ ] **Step 3: Commit**

```bash
git add src/run.py
git commit -m "feat: pipeline orchestrator run.py wiring all stages end-to-end"
```

---

## Task 11: Backtest Module (`backtest/backtest.py`)

**Files:**
- Create: `backtest/__init__.py`
- Create: `backtest/backtest.py`

- [ ] **Step 1: Create backtest/__init__.py (empty)**

```python
```

- [ ] **Step 2: Implement backtest/backtest.py**

```python
"""
Monthly-rebalanced long-only backtest of the composite factor vs SPY.

WARNING: This backtest uses yfinance, which contains only currently-listed tickers.
Any results therefore suffer SURVIVORSHIP BIAS — delisted names (bankruptcies,
mergers, delistings) are absent from the universe, which inflates measured returns.
Treat results as directional sanity checks only, NOT as reliable estimates of
live performance. For a defensible backtest, use a point-in-time universe with
delisted securities (CRSP, Compustat) which are not available for free.

LOOK-AHEAD BIAS WARNING: Factors computed on date t must only use data that was
publicly available on or before t. Earnings estimates and revisions have reporting
lags that are approximated here but not precisely modeled. Treat all results
with appropriate skepticism.
"""

import logging
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date, timedelta
from src.config import load_config
from src.factors import mom_12_1, rs_vs_spy, avg_dollar_vol

logger = logging.getLogger(__name__)

TRANSACTION_COST_BPS = 10  # per side, conservative minimum


def _download_history(tickers: list, start: str, end: str) -> dict:
    joined = " ".join(tickers)
    try:
        raw = yf.download(joined, start=start, end=end, auto_adjust=True, progress=False, group_by="ticker")
    except Exception as e:
        logger.error(f"yfinance download failed: {e}")
        return {}
    result = {}
    if len(tickers) == 1:
        t = tickers[0]
        raw.columns = [c.lower() for c in raw.columns]
        result[t] = raw.dropna(how="all")
    else:
        for t in tickers:
            if t not in raw.columns.get_level_values(0):
                continue
            df = raw[t].copy()
            df.columns = [c.lower() for c in df.columns]
            result[t] = df.dropna(how="all")
    return result


def _monthly_rebalance_dates(start: date, end: date) -> list:
    dates = []
    current = date(start.year, start.month, 1)
    while current <= end:
        dates.append(current)
        if current.month == 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, current.month + 1, 1)
    return dates


def run_backtest(
    tickers: list,
    start: str = "2018-01-01",
    end: str | None = None,
    top_decile_n: int = 50,
    out_path: str = "output/backtest_results.md",
):
    """
    Build monthly-rebalanced long-only portfolios from the top decile of
    12-1 momentum + 6m RS composite, compare to SPY total return.
    """
    if end is None:
        end = date.today().isoformat()

    print("[backtest] Downloading price history — this may take several minutes...")
    all_tickers = tickers + ["SPY"]
    prices = _download_history(all_tickers, start, end)
    spy_df = prices.get("SPY", pd.DataFrame())
    if spy_df.empty:
        raise RuntimeError("Could not fetch SPY")

    rebalance_dates = _monthly_rebalance_dates(
        date.fromisoformat(start), date.fromisoformat(end)
    )

    portfolio_returns = []
    spy_returns = []

    for i, rb_date in enumerate(rebalance_dates[:-1]):
        next_date = rebalance_dates[i + 1]
        rb_str   = rb_date.isoformat()
        next_str = next_date.isoformat()

        scored = []
        for ticker in tickers:
            df = prices.get(ticker, pd.DataFrame())
            if df.empty or len(df) < 252:
                continue
            # Use data up to rb_date only (no look-ahead)
            hist = df[df.index.date <= rb_date]
            if len(hist) < 252:
                continue
            spy_hist = spy_df[spy_df.index.date <= rb_date]
            if len(spy_hist) < 252:
                continue
            try:
                score = (
                    0.5 * mom_12_1(hist["close"]) +
                    0.5 * rs_vs_spy(hist["close"], spy_hist["close"], window=126)
                )
                scored.append((ticker, score))
            except Exception:
                continue

        if not scored:
            continue

        scored.sort(key=lambda x: x[1], reverse=True)
        top = [t for t, _ in scored[:top_decile_n]]

        # Compute equal-weight portfolio return for the month
        monthly_rets = []
        for ticker in top:
            df = prices.get(ticker, pd.DataFrame())
            period = df[(df.index.date >= rb_date) & (df.index.date < next_date)]
            if len(period) < 2:
                continue
            ret = float(period["close"].iloc[-1] / period["close"].iloc[0]) - 1.0
            # Deduct transaction costs (10 bps per side = 20 bps round trip)
            ret -= 2 * TRANSACTION_COST_BPS / 10000
            monthly_rets.append(ret)

        if monthly_rets:
            portfolio_returns.append({"date": rb_str, "return": np.mean(monthly_rets)})

        # SPY return for same period
        spy_period = spy_df[(spy_df.index.date >= rb_date) & (spy_df.index.date < next_date)]
        if len(spy_period) >= 2:
            spy_ret = float(spy_period["close"].iloc[-1] / spy_period["close"].iloc[0]) - 1.0
            spy_returns.append({"date": rb_str, "return": spy_ret})

    if not portfolio_returns:
        print("[backtest] Insufficient data to produce results.")
        return

    port_df = pd.DataFrame(portfolio_returns)
    spy_df2 = pd.DataFrame(spy_returns)

    port_cum = (1 + port_df["return"]).cumprod().iloc[-1] - 1
    spy_cum  = (1 + spy_df2["return"]).cumprod().iloc[-1] - 1 if len(spy_df2) else float("nan")

    port_ann = (1 + port_cum) ** (12 / len(port_df)) - 1
    spy_ann  = (1 + spy_cum)  ** (12 / len(spy_df2)) - 1 if len(spy_df2) else float("nan")

    sharpe_port = port_df["return"].mean() / (port_df["return"].std() + 1e-12) * (12 ** 0.5)

    summary = f"""# Backtest Results

> **SURVIVORSHIP BIAS WARNING:** This backtest uses yfinance, which only contains
> currently-listed tickers. Delisted names (bankruptcies, acquisitions, failures)
> are absent. Results OVERSTATE actual returns. Do NOT use for capital allocation.
>
> **LOOK-AHEAD BIAS:** Earnings estimate data is approximated with reporting lags.
> Treat results as directional sanity checks only.
>
> **Transaction costs:** {TRANSACTION_COST_BPS}bps/side (20bps round-trip) applied.

## Summary ({start} to {end})

| Metric | Portfolio | SPY |
|--------|-----------|-----|
| Cumulative return | {port_cum:.1%} | {spy_cum:.1%} |
| Annualized return | {port_ann:.1%} | {spy_ann:.1%} |
| Monthly Sharpe (ann.) | {sharpe_port:.2f} | — |
| Months tracked | {len(port_df)} | {len(spy_df2)} |

*Model: top-{top_decile_n} by 12-1 momentum + 6m RS, equal-weight, monthly rebalance.*
"""
    f = open(out_path, "w")
    f.write(summary)
    f.close()
    print(summary)
    print(f"[backtest] Results written to {out_path}")


if __name__ == "__main__":
    import sys
    cfg = load_config()
    universe = pd.read_parquet("data/universe.parquet")
    tickers = universe["ticker"].tolist()[:500]  # Limit for speed
    run_backtest(tickers, start="2018-01-01")
```

- [ ] **Step 3: Commit**

```bash
git add backtest/__init__.py backtest/backtest.py
git commit -m "feat: backtest module with survivorship/look-ahead bias warnings, 10bps tx costs"
```

---

## Task 12: End-to-End Smoke Test

- [ ] **Step 1: Run all unit tests**

```bash
pytest tests/ -v
```

Expected: all tests PASS. Zero failures.

- [ ] **Step 2: Run pipeline smoke test with small universe**

```bash
python -c "
import pandas as pd
from src.config import load_config
from src.cache import init_db
from src.universe import build_universe
import os; os.makedirs('data', exist_ok=True)

cfg = load_config()
init_db('data/cache.db')
uni = build_universe(cfg)
print(f'Universe: {len(uni)} tickers')
assert len(uni) > 5000, f'Expected >5000 tickers, got {len(uni)}'
"
```

Expected: `Universe: ~8000+ tickers`

- [ ] **Step 3: Verify full run completes (requires .env with FINNHUB_API_KEY)**

Copy `.env.template` to `.env` and fill in `FINNHUB_API_KEY` and `SEC_USER_AGENT`, then:

```bash
python src/run.py
```

Expected:
- Prints `[universe]`, `[liquidity_gate]`, `[quality_gate]`, `[compose]`, `[output]` stage logs
- Creates `output/screen_YYYY-MM-DD.csv` and `output/screen_YYYY-MM-DD.md`
- Prints top 10 to stdout

- [ ] **Step 4: Verify warm rerun is fast**

```bash
time python src/run.py
```

Expected: under 60 seconds (all data cached).

- [ ] **Step 5: Verify output file correctness**

```bash
python -c "
import pandas as pd
from datetime import date
path = f'output/screen_{date.today().isoformat()}.csv'
df = pd.read_csv(path)
print('Columns:', list(df.columns))
print('Rows:', len(df))
assert not df['composite'].isna().any(), 'NaN in composite column'
assert df['composite'].is_monotonic_decreasing, 'Not sorted by composite'
print('Output validation PASSED')
"
```

Expected: `Output validation PASSED`

- [ ] **Step 6: Final commit**

```bash
git add -A
git commit -m "feat: complete stock screener pipeline — all stages verified"
```

---

## Self-Review Against Spec

**Spec coverage checklist:**

| Spec Section | Covered by Task |
|---|---|
| §1 Architecture: staged pipeline | Tasks 6–10 (each stage separate module) |
| §2 Tech stack | Task 1 (requirements.txt pins all deps) |
| §3 Data sources + API keys | Tasks 1, 5–7 |
| §4 Project structure | Task 1 scaffold + all create paths |
| §5 config.yaml all thresholds | Task 1 |
| §6 Stage 1 universe + CIK | Task 5 |
| §6 Stage 2 batch yfinance + liquidity gate | Task 6 |
| §6 Stage 3 EDGAR + Finnhub, survivors only | Task 7 |
| §6 Stage 4 z-score composite | Task 8 |
| §6 Stage 5 CSV + markdown | Task 9 |
| §7 All factor formulas + unit tests | Tasks 4 + fixture values | 
| §8 Winsorize + z-score + gates | Task 8 |
| §9 SQLite schema 3 tables + TTL | Task 3 |
| §10 CSV columns exact list | Task 9 |
| §10 Markdown rationale column | Task 9 |
| §10 Top 10 to stdout | Task 10 |
| §11 Backtest survivorship warning | Task 11 |
| §11 10bps tx costs | Task 11 |
| §12 Retry/backoff on network calls | Task 7 (`_TokenBucket`, try/except) |
| §12 252-bar validation | Tasks 4 + 6 |
| §12 XBRL tag fallback list | Task 7 (`REVENUE_TAGS`, `COGS_TAGS`) |
| §13 Staged design protects Finnhub quota | Task 7 runs after Stage 2 gate |
| §13 Yahoo isolated in one module | `src/prices.py` |
| §14 `run.py` completes cold run | Task 10 |
| §14 Warm rerun < 60s | Task 12 verification |
| §14 Liquidity + quality gates logged | Tasks 6, 8 |
| §14 No NaN in composite | Task 8 test + Task 12 verification |
| `with` statement ban | All tasks use explicit open/close |

**No placeholders found.** All code steps contain complete implementations.

**Type consistency check:**
- `compute_price_factors` returns `dict | None` → `fetch_all_prices` collects into list → DataFrame ✓
- `fetch_all_fundamentals` returns `pd.DataFrame` → merged in `run.py` on `ticker` ✓
- `build_composite` takes `pd.DataFrame`, returns `pd.DataFrame` ✓
- `write_csv` / `write_markdown` take `pd.DataFrame` → returned path `str` ✓
- `parse_edgar_gp` returns `tuple[float, float, float, float]` → `put_edgar` takes same positional args ✓
