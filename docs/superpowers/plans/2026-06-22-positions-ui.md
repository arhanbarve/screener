# Positions UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Streamlit web app with dark sidebar nav — browse historical screener CSVs and track open positions with live exit signals.

**Architecture:** Single `app.py` Streamlit entrypoint. New `src/positions.py` handles position CRUD (positions.json) and exit signal computation using existing `src/factors.py` functions. Streamlit's `@st.cache_data(ttl=900)` caches yfinance calls per ticker.

**Tech Stack:** Streamlit, pandas, yfinance, existing src/factors.py functions (rsi_14, macd_state, adx_14, mfi_14)

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `requirements.txt` | Modify | Add streamlit |
| `src/positions.py` | Create | Position CRUD + exit signal computation |
| `tests/test_positions.py` | Create | Unit tests for signal computation and CRUD |
| `app.py` | Create | Streamlit entrypoint — sidebar, screener tab, positions tab |
| `positions.json` | Auto-created | Persisted open positions (written by positions.py) |

---

## Task 1: Add Streamlit Dependency

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add streamlit to requirements.txt**

Open `requirements.txt` and add:
```
streamlit>=1.35.0
```

- [ ] **Step 2: Install**

```bash
pip install "streamlit>=1.35.0"
```

Expected: `Successfully installed streamlit-...`

- [ ] **Step 3: Verify import**

```bash
python -c "import streamlit; print(streamlit.__version__)"
```

Expected: version string like `1.35.0` or newer

- [ ] **Step 4: Commit**

```bash
git add requirements.txt
git commit -m "feat: add streamlit dependency for positions UI"
```

---

## Task 2: Create src/positions.py

**Files:**
- Create: `src/positions.py`

- [ ] **Step 1: Create the file**

```python
import json
import math
import os
import tempfile
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import date, datetime
from pathlib import Path

from src.factors import rsi_14, macd_state, adx_14, mfi_14

POSITIONS_FILE = Path("positions.json")


def load_positions() -> list[dict]:
    """Load positions from positions.json. Returns [] if file missing."""
    if not POSITIONS_FILE.exists():
        return []
    with open(POSITIONS_FILE, "r") as f:
        return json.load(f)


def save_positions(positions: list[dict]) -> None:
    """Atomically write positions to positions.json."""
    tmp = POSITIONS_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(positions, f, indent=2)
    os.replace(tmp, POSITIONS_FILE)


def add_position(ticker: str, entry_date: str, entry_price: float) -> None:
    """Append a new position. Raises ValueError if ticker already open."""
    positions = load_positions()
    if any(p["ticker"] == ticker.upper() for p in positions):
        raise ValueError(f"{ticker} already in open positions")
    positions.append({
        "ticker": ticker.upper(),
        "entry_date": entry_date,
        "entry_price": float(entry_price),
    })
    save_positions(positions)


def remove_position(ticker: str) -> None:
    """Remove a position by ticker. No-op if not found."""
    positions = [p for p in load_positions() if p["ticker"] != ticker.upper()]
    save_positions(positions)


def fetch_ohlcv(ticker: str, days: int = 60) -> pd.DataFrame:
    """Fetch OHLCV via yfinance. Returns empty DataFrame on failure."""
    try:
        raw = yf.download(
            ticker,
            period=f"{days}d",
            interval="1d",
            auto_adjust=True,
            progress=False,
        )
        if raw.empty:
            return pd.DataFrame()
        df = raw.copy()
        # Flatten MultiIndex columns if present (single-ticker download)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].lower() for c in df.columns]
        else:
            df.columns = [c.lower() for c in df.columns]
        return df.dropna(how="all")
    except Exception:
        return pd.DataFrame()


def get_current_price(ticker: str) -> float | None:
    """Fetch latest price via yfinance fast_info. Returns None on failure."""
    try:
        info = yf.Ticker(ticker).fast_info
        return float(info.last_price)
    except Exception:
        return None


def compute_exit_signals(df: pd.DataFrame) -> dict:
    """
    Compute 5 exit signals from an OHLCV DataFrame.

    Returns:
        {
            "rsi": bool | None,      # True = exit signal triggered
            "macd": bool | None,
            "stoch": bool | None,
            "adx": bool | None,
            "mfi": bool | None,
            "score": int,            # count of True signals (0-5)
            "rsi_val": float | None,
            "adx_val": float | None,
            "mfi_val": float | None,
            "stoch_k": float | None,
            "stoch_d": float | None,
            "macd_state": str | None,
        }

    Signal definitions:
        RSI   — RSI(14) > 70 AND declining vs 3 bars ago
        MACD  — macd_state in ("bearish", "bearish_cross")
        Stoch — %K was >80 on prev bar AND %K just crossed below %D (bear cross)
        ADX   — ADX(14) now < ADX(10 bars ago) by >5 pts (trend weakening)
        MFI   — MFI(14) < 50 (money flow turned negative)
    """
    base = {
        "rsi": None, "macd": None, "stoch": None, "adx": None, "mfi": None,
        "score": 0,
        "rsi_val": None, "adx_val": None, "mfi_val": None,
        "stoch_k": None, "stoch_d": None, "macd_state": None,
    }

    required_cols = {"close", "high", "low", "volume"}
    if df.empty or not required_cols.issubset(df.columns) or len(df) < 30:
        return base

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # --- RSI: >70 AND declining vs 3 bars ago ---
    try:
        rsi_now = rsi_14(close)
        base["rsi_val"] = rsi_now
        if len(close) >= 17 and not math.isnan(rsi_now):
            rsi_3ago = rsi_14(close.iloc[:-3])
            if not math.isnan(rsi_3ago):
                base["rsi"] = bool(rsi_now > 70 and rsi_now < rsi_3ago)
    except Exception:
        pass

    # --- MACD: bearish or bearish_cross ---
    try:
        m_state = macd_state(close)
        base["macd_state"] = m_state
        base["macd"] = m_state in ("bearish", "bearish_cross")
    except Exception:
        pass

    # --- Stochastic: was overbought (K_prev > 80) AND bear cross (K crossed below D) ---
    try:
        k_period, smooth_k, d_period = 14, 3, 3
        lowest_low = low.rolling(k_period).min()
        highest_high = high.rolling(k_period).max()
        denom = (highest_high - lowest_low).replace(0, np.nan)
        raw_k = 100.0 * (close - lowest_low) / denom
        sk = raw_k.rolling(smooth_k).mean()
        d_ser = sk.rolling(d_period).mean()
        k_now = float(sk.iloc[-1])
        k_prev = float(sk.iloc[-2])
        d_now = float(d_ser.iloc[-1])
        d_prev = float(d_ser.iloc[-2])
        base["stoch_k"] = k_now
        base["stoch_d"] = d_now
        bear_cross = (k_now < d_now) and (k_prev >= d_prev)
        base["stoch"] = bool(k_prev > 80 and bear_cross)
    except Exception:
        pass

    # --- ADX: current < (10-bars-ago value) by more than 5 pts ---
    try:
        adx_now = adx_14(high, low, close)
        base["adx_val"] = adx_now
        if len(close) >= 40 and not math.isnan(adx_now):
            adx_past = adx_14(high.iloc[:-10], low.iloc[:-10], close.iloc[:-10])
            if not math.isnan(adx_past):
                base["adx"] = bool(adx_past > adx_now + 5)
    except Exception:
        pass

    # --- MFI: < 50 ---
    try:
        mfi_now = mfi_14(high, low, close, volume)
        base["mfi_val"] = mfi_now
        if not math.isnan(mfi_now):
            base["mfi"] = bool(mfi_now < 50)
    except Exception:
        pass

    base["score"] = sum(
        1 for k in ("rsi", "macd", "stoch", "adx", "mfi")
        if base[k] is True
    )
    return base
```

- [ ] **Step 2: Verify import works**

```bash
python -c "from src.positions import load_positions, compute_exit_signals; print('OK')"
```

Expected: `OK`

---

## Task 3: Tests for src/positions.py

**Files:**
- Create: `tests/test_positions.py`

- [ ] **Step 1: Create test file**

```python
import json
import math
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch

from src.positions import (
    load_positions,
    save_positions,
    add_position,
    remove_position,
    compute_exit_signals,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_ohlcv(n: int = 60, trend: str = "up") -> pd.DataFrame:
    """Synthetic OHLCV. trend='up' → rising prices; 'down' → falling."""
    idx = pd.date_range("2026-01-01", periods=n, freq="B")
    if trend == "up":
        close = pd.Series([100 + i * 1.5 for i in range(n)], index=idx)
    else:
        close = pd.Series([200 - i * 1.5 for i in range(n)], index=idx)
    high = close * 1.01
    low = close * 0.99
    volume = pd.Series([1_000_000] * n, index=idx)
    return pd.DataFrame({"open": close, "high": high, "low": low, "close": close, "volume": volume})


# ── CRUD tests ────────────────────────────────────────────────────────────────

def test_load_positions_missing_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = load_positions()
    assert result == []


def test_save_and_load_roundtrip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    positions = [{"ticker": "AAPL", "entry_date": "2026-06-01", "entry_price": 150.0}]
    save_positions(positions)
    assert load_positions() == positions


def test_add_position(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    add_position("TSLA", "2026-06-01", 200.0)
    positions = load_positions()
    assert len(positions) == 1
    assert positions[0]["ticker"] == "TSLA"
    assert positions[0]["entry_price"] == 200.0


def test_add_position_duplicate_raises(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    add_position("AAPL", "2026-06-01", 150.0)
    with pytest.raises(ValueError, match="AAPL already in open positions"):
        add_position("aapl", "2026-06-02", 155.0)


def test_remove_position(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    add_position("AAPL", "2026-06-01", 150.0)
    add_position("MSFT", "2026-06-01", 300.0)
    remove_position("AAPL")
    positions = load_positions()
    assert len(positions) == 1
    assert positions[0]["ticker"] == "MSFT"


def test_remove_position_noop_if_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    add_position("AAPL", "2026-06-01", 150.0)
    remove_position("ZZZZ")  # should not raise
    assert len(load_positions()) == 1


# ── compute_exit_signals tests ────────────────────────────────────────────────

def test_exit_signals_empty_df():
    result = compute_exit_signals(pd.DataFrame())
    assert result["score"] == 0
    assert result["rsi"] is None
    assert result["macd"] is None


def test_exit_signals_too_short():
    df = _make_ohlcv(n=10)
    result = compute_exit_signals(df)
    assert result["score"] == 0


def test_exit_signals_returns_expected_keys():
    df = _make_ohlcv(n=60)
    result = compute_exit_signals(df)
    expected_keys = {"rsi", "macd", "stoch", "adx", "mfi", "score",
                     "rsi_val", "adx_val", "mfi_val", "stoch_k", "stoch_d", "macd_state"}
    assert expected_keys == set(result.keys())


def test_exit_signals_score_is_count_of_true():
    df = _make_ohlcv(n=60)
    result = compute_exit_signals(df)
    true_count = sum(1 for k in ("rsi", "macd", "stoch", "adx", "mfi") if result[k] is True)
    assert result["score"] == true_count


def test_exit_signals_score_bounded():
    df = _make_ohlcv(n=60)
    result = compute_exit_signals(df)
    assert 0 <= result["score"] <= 5


def test_exit_signals_rsi_val_is_float_or_none():
    df = _make_ohlcv(n=60)
    result = compute_exit_signals(df)
    if result["rsi_val"] is not None:
        assert isinstance(result["rsi_val"], float)
        assert 0 <= result["rsi_val"] <= 100


def test_exit_signals_macd_state_valid_values():
    df = _make_ohlcv(n=60)
    result = compute_exit_signals(df)
    valid = {"bullish", "bullish_cross", "bearish", "bearish_cross", "unknown", None}
    assert result["macd_state"] in valid


def test_exit_signals_missing_columns():
    df = pd.DataFrame({"close": [100.0] * 50})  # missing high/low/volume
    result = compute_exit_signals(df)
    assert result["score"] == 0
    assert result["rsi"] is None
```

- [ ] **Step 2: Run tests**

```bash
pytest tests/test_positions.py -v
```

Expected: all tests pass. If any signal-specific test fails due to synthetic data not triggering the signal, that's fine — the important tests are CRUD and structural (keys, score=count, bounded).

- [ ] **Step 3: Commit**

```bash
git add src/positions.py tests/test_positions.py
git commit -m "feat: positions CRUD and exit signal computation"
```

---

## Task 4: Create app.py — Screener Results Tab

**Files:**
- Create: `app.py`

- [ ] **Step 1: Create app.py with sidebar + screener tab**

```python
import math
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from src.positions import (
    add_position,
    compute_exit_signals,
    fetch_ohlcv,
    get_current_price,
    load_positions,
    remove_position,
)

st.set_page_config(
    page_title="Stock Screener",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stSidebar"] { background: #1e293b; }
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stRadio label { font-size: 15px; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 📊 Screener")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["📈 Screener Results", "📋 Open Positions"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("*Momentum factor strategy*")


def _render_screener() -> None:
    st.title("📈 Screener Results")

    output_dir = Path("output")
    csv_files = sorted(output_dir.glob("screen_*.csv"), reverse=True)

    if not csv_files:
        st.info("No screen output files found in output/")
        return

    dates = [f.stem.replace("screen_", "") for f in csv_files]
    selected_date = st.selectbox("Screen date", dates)

    df = pd.read_csv(output_dir / f"screen_{selected_date}.csv")
    df.insert(0, "Rank", range(1, len(df) + 1))

    col_map = {
        "ticker": "Ticker", "name": "Name", "sector": "Sector",
        "composite": "Score", "mom_12_1": "Mom 12-1",
        "rs_6m": "RS 6M", "price": "Price", "market_cap": "Mkt Cap",
    }
    display_cols = ["Rank"] + [c for c in col_map if c in df.columns]
    display_df = df[display_cols].rename(columns=col_map)

    if "Score" in display_df.columns:
        display_df["Score"] = display_df["Score"].map(
            lambda x: f"{x:.2f}" if pd.notna(x) else "—"
        )
    if "Mom 12-1" in display_df.columns:
        display_df["Mom 12-1"] = display_df["Mom 12-1"].map(
            lambda x: f"{x:.1%}" if pd.notna(x) else "—"
        )
    if "RS 6M" in display_df.columns:
        display_df["RS 6M"] = display_df["RS 6M"].map(
            lambda x: f"{x:.1%}" if pd.notna(x) else "—"
        )
    if "Price" in display_df.columns:
        display_df["Price"] = display_df["Price"].map(
            lambda x: f"${x:,.2f}" if pd.notna(x) else "—"
        )
    if "Mkt Cap" in display_df.columns:
        display_df["Mkt Cap"] = display_df["Mkt Cap"].map(
            lambda x: f"${x / 1e9:.1f}B" if pd.notna(x) else "—"
        )

    st.metric("Stocks in results", len(df))
    st.dataframe(display_df, use_container_width=True, hide_index=True)


@st.cache_data(ttl=900)
def _cached_position_data(ticker: str) -> tuple[dict, float | None]:
    df = fetch_ohlcv(ticker, days=60)
    signals = compute_exit_signals(df)
    price = get_current_price(ticker)
    return signals, price


def _signal_badge(label: str, triggered: bool | None, val: float | None = None) -> str:
    if triggered is None:
        bg, fg, icon = "#e2e8f0", "#64748b", "—"
    elif triggered:
        bg, fg, icon = "#fee2e2", "#991b1b", "🔴"
    else:
        bg, fg, icon = "#dcfce7", "#166534", "✓"
    val_str = ""
    if val is not None and not (isinstance(val, float) and math.isnan(val)):
        val_str = f" {val:.0f}"
    return (
        f'<span style="background:{bg};color:{fg};padding:3px 10px;'
        f'border-radius:12px;font-size:12px;margin-right:6px">'
        f'{icon} {label}{val_str}</span>'
    )


def _render_position_card(pos: dict) -> None:
    ticker = pos["ticker"]
    entry_price = pos["entry_price"]
    entry_date = pos["entry_date"]
    signals, current_price = _cached_position_data(ticker)
    score = signals.get("score", 0)

    try:
        held_days = (date.today() - datetime.strptime(entry_date, "%Y-%m-%d").date()).days
    except Exception:
        held_days = "?"

    if current_price and entry_price:
        pnl = (current_price - entry_price) / entry_price
        pnl_str = f"+{pnl:.1%}" if pnl >= 0 else f"{pnl:.1%}"
        pnl_color = "#10b981" if pnl >= 0 else "#dc2626"
    else:
        pnl_str, pnl_color = "N/A", "#94a3b8"

    price_str = f"${current_price:,.2f}" if current_price else "N/A"

    if score >= 3:
        border, bg = "#dc2626", "#fff5f5"
    elif score >= 1:
        border, bg = "#f59e0b", "#fffbeb"
    else:
        border, bg = "#10b981", "#f0fdf4"

    badges = (
        _signal_badge("RSI", signals.get("rsi"), signals.get("rsi_val"))
        + _signal_badge("MACD", signals.get("macd"))
        + _signal_badge("Stoch", signals.get("stoch"), signals.get("stoch_k"))
        + _signal_badge("ADX", signals.get("adx"), signals.get("adx_val"))
        + _signal_badge("MFI", signals.get("mfi"), signals.get("mfi_val"))
    )

    exit_banner = ""
    if score >= 3:
        exit_banner = (
            f'<div style="margin-top:10px;color:#dc2626;font-weight:600;font-size:13px">'
            f'🚨 EXIT SIGNAL — {score}/5 momentum signals triggered</div>'
        )

    card_html = f"""
    <div style="border:2px solid {border};border-radius:10px;padding:16px;
                margin-bottom:4px;background:{bg}">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
        <div>
          <span style="font-weight:700;font-size:18px;color:#1e293b">{ticker}</span>
          <span style="color:#64748b;font-size:13px;margin-left:8px">
            · entered {entry_date} @ ${entry_price:,.2f}
            · {held_days}d
            · {price_str} ·
          </span>
          <span style="color:{pnl_color};font-weight:600;font-size:14px">{pnl_str}</span>
        </div>
      </div>
      <div>{badges}</div>
      {exit_banner}
    </div>
    """

    col_card, col_btn = st.columns([11, 1])
    with col_card:
        st.markdown(card_html, unsafe_allow_html=True)
    with col_btn:
        st.markdown("<div style='margin-top:12px'></div>", unsafe_allow_html=True)
        if st.button("✕", key=f"close_{ticker}", help=f"Close {ticker} position"):
            remove_position(ticker)
            st.rerun()


def _render_positions() -> None:
    st.title("📋 Open Positions")

    with st.form("add_position_form", clear_on_submit=True):
        c1, c2, c3, c4 = st.columns([2, 2, 2, 1])
        with c1:
            ticker_input = st.text_input("Ticker", placeholder="AAPL").strip().upper()
        with c2:
            entry_date = st.date_input("Entry date", value=date.today())
        with c3:
            entry_price = st.number_input("Entry price ($)", min_value=0.01, step=0.01, format="%.2f")
        with c4:
            st.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)
            submitted = st.form_submit_button("＋ Add", use_container_width=True)

    if submitted:
        if not ticker_input:
            st.error("Ticker required")
        elif entry_price <= 0:
            st.error("Entry price must be > 0")
        else:
            try:
                add_position(ticker_input, str(entry_date), entry_price)
                st.success(f"Added {ticker_input}")
                st.rerun()
            except ValueError as e:
                st.error(str(e))

    positions = load_positions()
    if not positions:
        st.info("No open positions. Add one above.")
        return

    st.markdown(f"**{len(positions)} open position{'s' if len(positions) != 1 else ''}**")
    st.markdown("---")

    # Sort: most signals first (most urgent at top)
    enriched = []
    for p in positions:
        signals, _ = _cached_position_data(p["ticker"])
        enriched.append({**p, "_score": signals.get("score", 0)})
    enriched.sort(key=lambda x: x["_score"], reverse=True)

    for pos in enriched:
        _render_position_card(pos)


if page == "📈 Screener Results":
    _render_screener()
else:
    _render_positions()
```

- [ ] **Step 2: Run the app**

```bash
streamlit run app.py
```

Expected: browser opens at `http://localhost:8501`. Sidebar shows two nav items.

- [ ] **Step 3: Smoke test — Screener Results**

- Click "📈 Screener Results" in sidebar
- Verify date dropdown shows dates from output/ directory
- Switch between dates, confirm table updates
- Check columns: Rank, Ticker, Name, Sector, Score, Mom 12-1, RS 6M, Price, Mkt Cap

- [ ] **Step 4: Smoke test — Open Positions**

- Click "📋 Open Positions" in sidebar
- Add position: ticker=`AAPL`, entry date=today, price=`150.00`
- Verify card appears with green border (0 signals) and `AAPL` header
- Verify P&L shows (may need a moment for yfinance fetch)
- Verify signal badges appear: RSI, MACD, Stoch, ADX, MFI
- Click ✕ to close position, verify card disappears

- [ ] **Step 5: Commit**

```bash
git add app.py
git commit -m "feat: Streamlit UI for screener results and open positions tracker"
```

---

## Task 5: Add .gitignore Entry

**Files:**
- Modify: `.gitignore` (create if missing)

- [ ] **Step 1: Exclude positions.json and .superpowers from git**

Check if `.gitignore` exists:
```bash
cat .gitignore 2>/dev/null || echo "(no .gitignore)"
```

Add these lines if not already present:
```
positions.json
.superpowers/
```

- [ ] **Step 2: Commit**

```bash
git add .gitignore
git commit -m "chore: gitignore positions.json and .superpowers"
```
