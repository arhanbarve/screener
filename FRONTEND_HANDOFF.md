# Frontend Handoff — Stock Screener

## What this is

A Streamlit-based quantitative stock screener. Single file: `app.py` (1,172 lines). No custom CSS framework, no component library — all styling via `st.markdown(unsafe_allow_html=True)` inline HTML/CSS injection. This is the primary creative constraint and opportunity.

## Chosen design direction

**Dark Financial Terminal** — inspired by Bloomberg Terminal / Linear dark mode.

- Near-black base (current sidebar is `#1e293b` — a start, not a ceiling)
- Single amber/gold accent for primary highlights
- Monospaced tickers, data-dense layouts
- Subtle animations: number countups on load, staggered row reveals, smooth signal-state transitions
- Color reserved strictly for signal meaning: green = bull/confirm, amber = wait/neutral, red = bear/avoid

## Current state — what exists

### Tech stack
- Python 3.12 + Streamlit
- No frontend build step — pure Python
- CSS injected via `st.markdown(..., unsafe_allow_html=True)`
- Custom JS via `st.components.v1.html()`

### Existing color palette (hardcoded inline, no tokens)
| Token (implied) | Value | Used for |
|---|---|---|
| Sidebar bg | `#1e293b` | Sidebar only |
| Text primary | `#1e293b` | Main text |
| Text muted | `#64748b` | Labels, captions |
| Text inverse | `#e2e8f0` | On dark sidebar |
| Surface light | `#f8fafc` | Card backgrounds |
| Border | `#e2e8f0` | Card borders |
| Bull green bg | `#dcfce7` | Positive badges |
| Bull green fg | `#166534` | Positive badge text |
| Bear red bg | `#fee2e2` | Negative badges |
| Bear red fg | `#991b1b` | Negative badge text |
| Caution amber bg | `#fef3c7` | Wait/neutral badges |
| Caution amber fg | `#92400e` | Wait/neutral badge text |
| Blue accent | `#1e40af` / `#dbeafe` | RS badges |
| Purple accent | `#6d28d9` / `#ede9fe` | Score badges |
| Dark card | `#1e293b` | Filing Edge / Confluence top-3 cards |
| Sky blue | `#38bdf8` | Stability values on dark cards |

### Pages and their current problems

#### 1. Screener Results (`_render_screener`)
- Top 3 cards: light `#f8fafc` background, 1px border, emoji medals — feels flat
- Badge soup: 5–6 inline colored spans per card, visually noisy
- `st.dataframe()` for ranked table — Streamlit's built-in, hard to style
- Summary metrics: raw `st.metric()` widgets — look generic
- Entry signal section: expanders per ticker — feels like an afterthought

**Data shown per stock:** ticker, name, sector, composite score, conviction (1–10), streak days, entry signal (confirm/wait/avoid), catalyst, mom 12-1, RS vs SPY 6M, price, market cap

#### 2. Market Regime (`_render_regime`)
- Large emoji + text banner for BULL/BEAR/NEUTRAL — works but unpolished
- Score bar: a raw HTML div with hardcoded `width:%` — no animation
- Signal breakdown: plain HTML `<table>` with alternating row colors — generic
- Sector signal chips: basic rounded spans — OK but uninspired

**Data shown:** regime (BULL/BEAR/NEUTRAL), composite score (0–10), bull/bear signal count, VIX, yield curve, SMA cross, 12+ individual indicator rows, sector headwinds/tailwinds, market news sentiment

#### 3. Open Positions (`_render_positions`)
- Position cards: colored left-border-style via full `border:2px solid` — works
- Exit signal badges per indicator: tiny spans with ✓ or 🔴 — too small
- P&L displayed inline, easy to miss
- No chart/sparkline of position performance
- Exit banner triggers at 3/5 signals — text only, no visual urgency

**Data shown per position:** ticker, entry date, entry price, current price, P&L%, days held, RSI, MACD, Stochastic, ADX, MFI (triggered vs not)

#### 4. Filing Edge (`_render_filing_edge`)
- Top 3 cards: dark `#1e293b` background — best-looking part of the app right now
- Main table: `st.dataframe()` with emoji stability badges
- Watch list table: same treatment
- Academic citation shown — good for credibility, needs better presentation

**Data shown:** ticker, name, sector, composite score, conviction (1–5), text stability (cosine similarity 0–1), doc sim, mkt cap, ADV, change direction, Claude change-characterization

#### 5. Confluence (`_render_confluence`)
- Summary metrics: 4x `st.metric()` — generic
- Strong Buy cards: same dark card style as Filing Edge top-3
- Main table: `st.dataframe()` with combined score progress bar

**Data shown:** ticker, name, signal tier (Strong Buy/Watchlist/Filing Only), combined score, stability, 3M return, RS vs SPY, above 50d SMA, momentum score

### Navigation
- Streamlit `st.radio()` in sidebar — functional, no visual personality
- No page transition animations (Streamlit re-renders full page on navigation)
- Refresh button (R key shortcut implemented via injected JS)

## What needs the most work (priority order)

1. **Global dark theme** — main area is light, sidebar is dark. Inconsistent. Full dark unification is the highest-impact single change.
2. **Typography** — currently Streamlit's default Inter. Needs: a monospace or semi-monospace choice for tickers/numbers, a clear size hierarchy.
3. **Top-3 medal cards** — the hero element on every page. Currently flat. Biggest visual opportunity.
4. **Signal/badge system** — too many small colored chips per row. Needs consolidation and clearer hierarchy.
5. **Loading states** — Streamlit's default spinners. Could be more purposeful.
6. **Animations** — none currently. Number countups on score reveals, staggered row entrances, regime banner state transitions.

## Technical freedoms and constraints

### What Streamlit allows
- Full CSS override via `st.markdown('<style>...</style>', unsafe_allow_html=True)`
- Arbitrary HTML via `st.markdown(..., unsafe_allow_html=True)`
- Custom JS via `st.components.v1.html(js_code, height=0)` (sandboxed iframe)
- Streamlit theme config via `.streamlit/config.toml` (base colors, font, border radius)
- `st.dataframe()` custom column types (progress bars, text, images)

### What Streamlit does NOT allow
- SPA-style routing with animated transitions between pages (full page rerenders)
- React/Vue components without packaging as a custom Streamlit component
- CSS custom properties that target Streamlit's internal component classes reliably (they change between versions)
- Server-side state persistence beyond `st.session_state`

### The CSS injection pattern (how all current styling works)
```python
st.markdown("""
<style>
[data-testid="stSidebar"] { background: #1e293b; }
</style>
""", unsafe_allow_html=True)
```
Streamlit renders into shadow-DOM-like elements with `data-testid` attributes. These are the stable hooks. `stSidebar`, `stMetric`, `stButton`, `stDataFrame` are the main ones.

### Config file
`.streamlit/config.toml` doesn't exist yet — create it to set base theme:
```toml
[theme]
base = "dark"
backgroundColor = "#0a0e13"
secondaryBackgroundColor = "#0f1722"
textColor = "#e2e8f0"
font = "monospace"
```

## Design brief for ui-ux-pro-max

**You have full creative liberty on:**
- Color palette within the Dark Financial Terminal direction (specific hex values, the exact accent color, surface hierarchy)
- Typography choices — what monospace or semi-monospace font to pair with data display
- Card layouts — how the top-3 medal cards look on each page
- Animation choreography — what animates, when, how fast, what easing
- Badge/chip redesign — how signals are displayed (not required to keep the current chip style)
- Navigation redesign — the sidebar can be completely rethought
- Empty states, loading states, error states
- Page-level layout and information hierarchy

**Constraints that must be preserved:**
- All 5 pages must remain (Screener Results, Market Regime, Open Positions, Filing Edge, Confluence)
- All data fields listed above per page must remain visible
- The signal color semantics: green = bull/confirm, amber = wait, red = bear/avoid/exit
- Must work within Streamlit's rendering model (Python file, CSS injection, `unsafe_allow_html`)
- The R keyboard shortcut refresh must still work
- `st.dataframe()` for main ranked tables (or better, if you can inject a custom HTML table)

**Files to edit:**
- `app.py` — the entire frontend lives here
- `.streamlit/config.toml` — create this for base theme

**Run to test:**
```bash
cd /path/to/screener
source venv/bin/activate
streamlit run app.py
```
Open `http://localhost:8501`

## Strategic context

Read `PRODUCT.md` for full brand, user, and principle context before starting. The one-sentence version: this is a tool a quant built for themselves — it should look like that, not like a SaaS product someone sold to a quant.
