# Equity Screener

A quantitative stock screener that ranks US equities using institutional-grade momentum and fundamental factors, then layers short-term technical oscillators on top so every output name comes with a ready-to-use entry signal — no re-screening required.

Around that core sits the rest of a working system: a standing exit engine that
decides when to sell, an Alpaca paper-trading loop with a written decision
journal, a Streamlit dashboard, and event-study backtests used to accept or
reject strategy ideas before they are traded.

~23k lines of Python, 644 tests, running unattended every trading day since
June 2026.

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

Stops only ratchet up, trims are append-only, and a SELL is terminal, so a verdict cannot flip back and forth with day-to-day market moves. `src/exit_alerts.py` emails the instruction and its reason the day it fires, plus a daily digest; positions whose data was stale or failed to fetch are reported as not evaluated rather than shown with a stale verdict.

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

## Dashboard

A Streamlit app renders the whole system. It is deployed privately — it shows
real positions — so the pages are described rather than linked.

| Page | What it shows |
|---|---|
| **Screener** | Ranked results for any run date; top-3 cards, full factor table, entry signals |
| **Regime** | Market regime read used to size risk up or down |
| **Positions** | Open real-money positions with live quotes, and each one's standing exit plan with every trigger and how far it is from firing |
| **Paper** | Paper account equity curve vs SPY, FIFO-matched open lots, realized/unrealized split, resting orders |
| **Monitor** | Last run status, run history, sync health — how a failed overnight run is noticed |

Every page degrades to an empty state rather than an error when its data or
credentials are missing, which is the difference between a dashboard that is
blank and one that is broken. Access is gated by `APP_PASSWORD`.

---

## Paper trading

`src/broker.py` and `src/trader_cli.py` drive an Alpaca paper account: place
orders, sync protective stops, reconcile fills. Each session is written up in
`trading/journal/` — the reasoning before the trade, the fills after it, and a
weekly review that names what went wrong.

The journals are committed deliberately. A record that only survives when it
flatters you is not a record.

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure `.env`
```bash
cp .env.template .env    # then fill it in
```

Only two values are required to screen:

```
FINNHUB_API_KEY=your_key_here
SEC_USER_AGENT=YourName your@email.com
```

Get a free Finnhub key at [finnhub.io](https://finnhub.io). `SEC_USER_AGENT` must be
your own name and email — the SEC requires a real contact string and rate-limits
anonymous traffic. Everything else in `.env.template` is optional and documented
inline.

`OPENAI_API_KEY` powers the Stage 4.5 news overlay (`src/llm.py`). It is optional — without it the overlay is skipped and the run still produces a full ranked list, just with no `entry_signal` column. Models are set in `config.yaml` under `news.model` / `news.prefilter_model`.

### 3. Run
```bash
python3 -m src.run
```

First run: ~60–90 min (fetches full universe cold). Subsequent same-day runs: ~5 min (cached).

---

## Automated daily runs

A launchd job (cron works equally well) runs the screener at 4:15 PM ET on
weekdays. It writes `run_status.json` on failure too — that is how the Monitor
page reports a dead run instead of just going quiet — and then publishes the
day's artifacts to the private data repo:

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

## Tests

```bash
python3 -m pytest tests/ -q      # 644 tests
```

The suite leans on regression cover for bugs that actually cost money. Two
examples: Fidelity renamed its CSV headers from Title Case to Sentence case and
`csv.DictReader`'s exact-key matching silently parsed every money field as `0.0`
for five days, so a zeroed cost basis made the dashboard report the entire
position value as gain; and a zero cost basis had to start rendering as "—"
rather than a confident wrong number. Both are pinned.

---

## Data and privacy

This repo is public. The data it operates on is not, and none of it is in the
git history.

Real holdings, account values, run artifacts and the strategy specification are
gitignored and never committed. The hosted dashboard still renders them by
reading a private companion repo at runtime through `src/datastore.py`, which
resolves local disk first and only falls back to the network when a file is
genuinely absent — so local runs, cron jobs and tests never touch it. A
pre-commit hook blocks staged secrets, account-number patterns and
private-network hostnames, and `scripts/preflight_public.sh` scans the entire
history rather than just `HEAD`. See [SECURITY.md](SECURITY.md).

Consequence for anyone cloning this: it runs, but against your own brokerage
data and API keys, never mine.

---

## Disclaimer

This is a personal research project, not investment advice and not a
recommendation to buy or sell any security. Nothing here is produced by a
registered investment adviser. Backtested and paper-traded results are not
live-money results and do not predict future returns; they exclude slippage,
partial fills, borrow costs, and taxes, and they benefit from hindsight in the
choice of what was tested at all. Anyone running this does so with their own
capital, their own brokerage credentials, and their own losses.

Licensed under the [MIT License](LICENSE).

---

## Project structure

```
src/
  run.py          — screener entrypoint
  universe.py     — SEC EDGAR universe construction
  prices.py       — OHLCV fetching, liquidity gate, entry signals
  fundamentals.py — Finnhub + EDGAR data fetching
  factors.py      — factor and oscillator computations
  compose.py      — quality/confirmation gates, composite scoring
  news.py llm.py  — Stage 4.5 news overlay
  output.py       — CSV and markdown generation
  cache.py        — SQLite cache layer
  datastore.py    — private-data reads (local disk, else private repo)

  exit_plan.py    — standing exit engine: stops, trims, weekly health
  exit_alerts.py  — emails a SELL/TRIM the day it fires
  positions.py    — open-position store
  sizing.py       — position sizing and concentration caps

  broker.py       — Alpaca REST client
  paper.py        — paper portfolio: FIFO lots, equity curve, vs-SPY
  paper_stops.py  — protective stop sync
  trader_cli.py   — trading command line
  fidelity_sync.py— browser-driven Fidelity positions export

  *_backtest.py   — event-study harnesses (PEAD, insider, splits, ASR, …)

app.py            — Streamlit entrypoint
app_shared.py     — shared UI, password gate, rendering
pages/            — Regime, Positions, Paper, Monitor
tests/            — 644 tests
scripts/          — hooks, publication preflight, launchd jobs
config.yaml       — weights, gates, cache TTLs, output settings
```
