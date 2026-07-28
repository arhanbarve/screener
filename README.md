# Equity Screener

A quantitative stock screener that ranks US equities using institutional-grade momentum and fundamental factors, then layers short-term technical oscillators on top so every output name comes with a ready-to-use entry signal — no re-screening required.

---

## What it does

Runs nightly after market close and produces a ranked list of the top 20 stocks from the US common-stock universe, each annotated with:

- **Composite score** — weighted combination of 12-month momentum, analyst revision breadth, earnings surprise (SUE), and 6-month relative strength vs SPY
- **Entry grade** (STRONG / OK / WAIT) — derived from RSI, MACD, MFI, Stochastic, Bollinger %B, ADX, and volume surge
- **Full oscillator detail** — every signal that went into the entry grade, visible per stock

Output is a `.md` file and `.csv` file written to `output/`, plus a summary printed to the terminal.

---

## Factors

### Composite (medium-term, 1–3 month horizon)

| Factor | Weight | What it captures |
|--------|--------|-----------------|
| 12-1 Momentum | 35% | 12-month return skipping last month (Jegadeesh & Titman) |
| Revision Breadth | 25% | Net analyst upgrades vs downgrades over 90 days |
| SUE | 20% | Standardized unexpected earnings — post-earnings drift |
| RS vs SPY (6m) | 20% | Stock's 6-month return minus SPY's return |

### Gates (applied before ranking)

- **Liquidity** — min $300M market cap, min $5M avg daily dollar volume
- **Quality** — gross profit / assets above universe median
- **Confirmation** — price above SMA200 and within 10% of 52-week high

### Short-term entry oscillators (computed from existing OHLCV, no extra API calls)

| Signal | Parameters | Role in entry grade |
|--------|-----------|---------------------|
| RSI | 14-period | Sweet spot 40–65; >75 = WAIT |
| MFI | 14-period | Volume-weighted RSI; >80 = WAIT |
| MACD | 12/26/9 EMA | Bearish = veto; `bullish_cross` = strongest entry signal |
| Stochastic | 14,3,3 | %K > %D and <70 = +1; fresh cross = +2 |
| Bollinger %B | 20-period, 2σ | 0.5–0.85 = healthy trend; >0.92 = WAIT |
| ADX | 14-period | >20 confirms real trend, not chop |
| Vol Surge | 5d / 20d avg | >1.15x = conviction behind the move |

---

## Exiting: the standing exit plan

Screening decides what to buy; a separate engine (`src/exit_plan.py`) decides when to sell. Open positions in `positions.json` are evaluated **once per day on the closing price** — not on page load — and each carries a persisted plan anchored to its own entry, risk unit, and peak.

- **SELL** on a close below the trailing stop (`peak_close` − mult × ATR14), a close below the max-loss floor (entry − 2×ATR14, ratcheting to breakeven after the first trim), or **3+ consecutive** closes below the 50-day SMA.
- **TRIM** on either of two rungs, each firing at most once ever and each meaning sell one third: de-risk at +2R or +20%, and blowoff extension.
- A **weekly** health check (MACD, RS vs SPY, OBV, ADX on weekly bars) can only *tighten* the trailing multiplier — never loosen it, never sell on its own.

Stops only ratchet up, trims are append-only, and a SELL is terminal, so a verdict cannot flip back and forth with day-to-day market moves. `src/exit_alerts.py` emails the instruction and its reason the day it fires, plus a daily digest; positions whose data was stale or failed to fetch are reported as not evaluated rather than shown with a stale verdict. See `STRATEGY.md` §5.1 for the full specification.

---

## Output

### Terminal
```
  1. NVDA     composite=+2.312  $142.50  entry=STRONG  RSI=54  MFI=58  Stoch=61/55  ADX=31  macd=bullish_cross
  2. AEHR     composite=+2.180  $18.20   entry=OK      RSI=62  MFI=55  Stoch=70/68  ADX=24  macd=bullish
```

### Markdown (`output/screen_YYYY-MM-DD.md`)
Two sections:
1. **Top Ranked Names** — composite score, entry grade, fundamental rationale
2. **Entry Timing Details** — full oscillator table; `*` on Stoch %K/%D means a fresh crossover fired today

### CSV (`output/screen_YYYY-MM-DD.csv`)
All raw factor values and signal scores for spreadsheet analysis.

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure `.env`
```
FINNHUB_API_KEY=your_key_here
SEC_USER_AGENT=YourName your@email.com
```

Get a free Finnhub key at [finnhub.io](https://finnhub.io). The SEC user agent just needs your name and email.

### 3. Run
```bash
python3 -m src.run
```

First run: ~60–90 min (fetches full universe cold). Subsequent same-day runs: ~5 min (cached).

---

## Automated daily runs

A cron job runs the screener automatically at 4:15 PM ET on weekdays:

```
15 16 * * 1-5 /path/to/screener/run_screener.sh
```

Logs are written to `logs/run_YYYY-MM-DD.log`. To set it up on a new machine:

```bash
chmod +x run_screener.sh
(crontab -l 2>/dev/null; echo "15 16 * * 1-5 $(pwd)/run_screener.sh") | crontab -
```

---

## Cache

| Data | TTL |
|------|-----|
| Prices (OHLCV) | 18 hours |
| Market cap | 18 hours |
| Fundamentals | 7 days |
| EDGAR filings | 30 days |

Cache lives in `data/cache.db` (SQLite). Delete it to force a full refresh.

---

## Project structure

```
src/
  factors.py      — all factor and oscillator computations
  prices.py       — OHLCV fetching, liquidity gate, entry signals
  fundamentals.py — Finnhub + EDGAR data fetching
  compose.py      — quality/confirmation gates, composite scoring
  universe.py     — SEC EDGAR universe construction
  output.py       — CSV and markdown generation
  cache.py        — SQLite cache layer
  run.py          — main entrypoint
config.yaml       — weights, gates, cache TTLs, output settings
run_screener.sh   — cron wrapper script
```
