# Positions UI Design — 2026-06-22

## Overview

Streamlit web app (`app.py`) at project root. Two-section sidebar nav:
1. **Screener Results** — browse historical screen output CSVs
2. **Open Positions** — track live holdings, view exit signals per position

Run with: `streamlit run app.py`

---

## Architecture

```
screener/
  app.py                  # Streamlit entrypoint
  positions.json          # Persisted open positions (created on first add)
  src/
    positions.py          # Position CRUD + exit signal computation
  output/                 # Existing CSVs (screen_YYYY-MM-DD.csv)
```

No new dependencies beyond Streamlit and existing `yfinance`/`pandas` stack.

---

## Section 1: Screener Results

**Trigger**: user clicks "Screener Results" in sidebar.

**Behavior:**
- Scan `output/` directory for all `screen_*.csv` files
- Present date dropdown (sorted newest-first, default = latest)
- Load selected CSV with `pandas.read_csv`
- Render as `st.dataframe` with:
  - Rank, Ticker (bold), Name, Sector, Composite score (2 decimal), Rationale
  - Ticker column colored indigo
  - Score column formatted to 2dp
  - Full-width, sortable, searchable via Streamlit's built-in dataframe controls

---

## Section 2: Open Positions

**Trigger**: user clicks "Open Positions" in sidebar.

### Add Position Form

At top of page. Fields inline:
- Ticker (text input, uppercase-forced)
- Entry Date (date input, default today)
- Entry Price (number input, $ prefix)
- "Add Position" button

On submit: append to `positions.json`, recompute signals for new ticker, rerender.

### Position Cards

One expanded card per held position. Cards ordered by exit urgency (most signals first).

**Card layout:**
```
[TICKER]  ·  entered Jun 18 @ $14.20  ·  4 days  ·  +8.3%    [Close ✕]
RSI ✓   MACD ✓   Stoch ⚠   ADX ✓   MFI ✓
```

- Card border: red if ≥3/5 signals, yellow if 1-2/5, green if 0/5
- P&L: live price from yfinance (cached 15 min via existing SQLite cache)
- Signal dots: green ✓ / yellow ⚠ / red 🔴
- "🚨 EXIT SIGNAL — N/5 momentum signals triggered" banner if ≥3/5
- "Close" button removes from positions.json, shows confirmation

### Exit Signal Definitions

Computed in `src/positions.py` from last 30 days of OHLCV:

| Signal | Triggered when |
|--------|---------------|
| RSI | RSI(14) > 70 AND last 3 bars declining |
| MACD | MACD histogram was positive, now declining 2+ consecutive bars |
| Stochastic | %K > 80 AND %K crossed below %D in last 2 bars |
| ADX | ADX(14) peaked in last 10 bars AND now >5 pts below that peak |
| MFI | MFI(14) < 50 (money flow turning negative) |

Score: count of triggered signals (0-5). Exit threshold: ≥3.

---

## Data Storage: positions.json

```json
[
  {
    "ticker": "AEHR",
    "entry_date": "2026-06-18",
    "entry_price": 14.20
  }
]
```

Simple list. Written atomically (write temp file, rename). No DB.

---

## Error Handling

- Ticker not found by yfinance → card shows "⚠ Price unavailable" but still renders entry info
- Empty output/ dir → screener tab shows info message, not crash
- positions.json missing → treat as empty list, create on first add
- Delisted ticker → yfinance returns empty df → signals all show "—" (unknown)

---

## What's Out of Scope

- Authentication (local personal tool)
- Mobile layout
- Historical P&L charting
- Alerts / push notifications
- Multi-user support
