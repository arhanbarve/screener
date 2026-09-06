# Screener + Autonomous Trader Overhaul — Design

Date: 2026-09-04. Companion to the vetting report (`docs/strategy-vetting-2026-09-04.md`,
private). Decisions taken with the user: engine + LLM reviewer; SPY core sleeve for
uninvested capital; **aggressive** risk posture; fundamentals block at 32% with a
floor gate.

## Goals

1. The screener ranks the real liquid US universe every day, or fails loudly.
2. Fundamentals (quality, growth, value, investment, strength) decide eligibility and
   carry a third of the composite.
3. The paper account is run by a deterministic portfolio engine with a written risk
   policy; the LLM reviews and journals within bounded powers; risk-reducing legs
   execute even when the LLM session dies.
4. Every artifact the trader consumes is machine-readable and carries a health verdict.

## Non-goals

- Live-money order routing (Fidelity side stays advisory via exit-plan emails).
- A multi-year backtest of the new composite (Phase 3 lays the data groundwork only).
- Dashboard redesign. Pages get new columns and a health strip, nothing more.

## Architecture

```
SEC tickers ∩ Alpaca tradable (NYSE/NASDAQ/AMEX/ARCA)      universe.py + broker.get_assets
  → OHLCV (yfinance, batch-safe, no batch-level quarantine)  prices.py
  → ADV gate → Finnhub metrics/profile (cap, sector, ratios) finnhub_data.py
  → cap gate → EDGAR facts (gp, accruals, asset growth, issuance, ROA, leverage)
  → fundamentals block score + floor gate                     fundamentals_block.py
  → composite (40/20/32/8) → top 60 finalists                 compose.py
  → finalists: eps_trend, earnings date, news overlay          news.py, finnhub_data.py
  → CSV + MD + screen_latest.json (with health)               output.py
  → health gate: fail run if universe/ranked below floors     run.py, run_status.py

Evening trader run (run_trader.sh):
  trader_cli plan   → portfolio_engine.build_plan(screen, book, plans, regime, policy)
  claude -p (reviewer prompt) → may veto/downsize legs, adds news, writes journal
  trader_cli execute-plan <plan.json> [--legs approved]
  fallback: if session crashed and book untouched → execute-plan --risk-only
```

## Components

### universe.py
- `filter_tradable(df, assets)`: keep SEC tickers whose Alpaca-normalised symbol
  (`-` → `.`) is in Alpaca's active, tradable, non-OTC equity list. Drop symbols
  matching `-(P|W|U|R)` suffix patterns. Adds `alpaca_symbol`. If no Alpaca keys,
  returns df unchanged (public clones keep working).
- Universe parquet rebuilt when older than 7 days.

### prices.py
- `_fetch_batch_yfinance` raises `BatchFetchError` on exception instead of returning
  `{}`; `fetch_all_prices` catches it, splits the batch in half and retries down to
  25 tickers, and never quarantines on an exception. Quarantine only when the batch
  returned and the ticker had no rows. Quarantine TTL 7 days.
- Market cap is no longer fetched here. `compute_price_factors` returns rows without
  `market_cap`; the ADV gate runs first, then `finnhub_data.attach_market_caps`.
- `sma_200` becomes a true 200-bar mean.
- `fetch_all_prices` returns `(price_store, factors_df, stats)` where `stats` has
  `universe`, `priced`, `adv_survivors`.

### finnhub_data.py (new)
- Tables: `fh_metrics(ticker, payload, fetched_at)` and `fh_profile(ticker, industry,
  market_cap, shares_out, fetched_at)`.
- `fetch_metrics(tickers, db, ttl_days=7, max_fetch=600)` → `({ticker: metric dict},
  stats)`; fetches at most `max_fetch` missing tickers per call at 60/min so a run
  stays bounded; `{}` is cached for symbols Finnhub does not know.
- `fetch_profiles(tickers, db, cap_ttl_days=7, max_fetch=600)` → industry, cap, shares;
  industry never expires, a stale cap is still returned when the budget runs out.
- `resolve_market_cap` / `attach_market_caps(df, metrics, profiles, db_path)`: cap from
  metrics `marketCapitalization`×1e6 → profile cap → shares × price → legacy
  `market_cap` table → NaN; all bounds-checked; adds `market_cap`, `cap_source`,
  `industry`, `sector`, `is_financial`.
- `sector_for(industry)` maps Finnhub industry to a coarse sector; `is_financial()` for
  {Banking, Financial Services, Insurance, Real Estate}.
- `fetch_earnings_calendar(days_ahead=45)`: one market-wide call, cached 20 h on disk,
  earliest date per symbol; `days_to_earnings(ticker, calendar, today)`.
- Warm-up outside a run: `python -m src.cache_maint warm-finnhub`.

### fundamentals.py
- EDGAR parser extended: from the companyfacts JSON already downloaded, take the two
  latest annual values (10-K/20-F) of Revenues, COGS, NetIncomeLoss, OCF, Assets,
  Liabilities, StockholdersEquity, shares outstanding. Derive `gp_assets`, `roa`,
  `accruals = (NI − OCF)/Assets`, `asset_growth`, `net_issuance`, `leverage`.
  Cached as JSON in a new `edgar_facts(cik, payload, fetched_at)` table (the legacy
  `edgar` table is still refreshed for older readers). Synonym tags are resolved by
  recency of the latest fiscal year; flow items are duration-filtered to ≈12 months.
- Sector comes from `finnhub_data`, not yfinance. yfinance is used only for finalists.

### fundamentals_block.py (new, pure)
Input: frame with Finnhub metric columns + EDGAR columns + `sector`. Output: the
frame with `fund_quality`, `fund_growth`, `fund_value`, `fund_invest`, `fund_strength`
(each the NaN-tolerant mean of its inputs' sign-corrected cross-sectional percentile
ranks, 0–1) and `fund_score` = mean of available sub-scores (≥ 2 required).

| Sub-score | Inputs (sign) |
|---|---|
| quality | gp_assets (+), roeTTM (+), roaTTM (+), accruals (−), grossMarginTTM (+) |
| growth | revenueGrowthTTMYoy (+), epsGrowthTTMYoy (+), revenueGrowth3Y (+) |
| value | 1/peTTM (+, only if EPS>0), 1/pfcfShareTTM (+, if >0), 1/evEbitdaTTM (+, if >0), 1/psTTM (+) |
| invest | asset_growth (−), net_issuance (−) |
| strength | totalDebt/totalEquityQuarterly (−), netInterestCoverageTTM (+), currentRatioQuarterly (+) |

Financials/REITs: quality drops gp_assets and accruals; invest and strength dropped;
their `fund_score` is the mean of what remains.

Floor gate: `fund_score >= 0.30` (config `fundamentals.floor_percentile`). Names with
`fund_score` NaN (no data at all) are excluded — no data is not neutral for a gate.

### compose.py
- `COMPOSITE_FACTORS` adds `fund_quality, fund_growth, fund_value, fund_invest,
  fund_strength` (rank-normalised). Weights in `config.yaml`:
  momentum 0.40, earnings 0.20, fundamentals 0.32, technical 0.08.
- `rev_breadth`/`rev_magnitude` renamed in docs to "analyst rating breadth/shift";
  column names unchanged for compatibility.
- Finalist stage (src/enrich.py): top `finalists_n` (60) get `eps_rev_30d` /
  `eps_rev_90d` (yfinance `eps_trend`, fiscal-year row first, quarter fallback),
  `short_float`, `days_to_cover`, `analyst_target_pct`, `days_to_earnings` (Finnhub);
  `eps_rev_30d` is rank-normalised and added at `finalist_rev_weight` (0.05) *within
  finalists only* → `composite_final`, then the final top_n and conviction.
- Conviction adds a fundamentals component (fund_score ≥ 0.7 → +2, ≥ 0.5 → +1).

### output.py
- Writes `output/screen_latest.json`: `{date, generated_at, health, regime, columns,
  rows, ranking_tail}`; `health` is the run's stats dict (`ok`, `reasons`, `universe`,
  `tradable`, `priced`, `adv_survivors`, `cap_survivors`, `cap_sources`, `fund_gate`,
  `ranked`, `selected`, `warnings`); rows carry `rank`; `ranking_tail` is the top 100.
- CSV gains the fundamentals columns and `alpaca_symbol`.

### run.py / run_status.py
- Health floors in `config.yaml`: `health.min_adv_survivors: 800`,
  `health.min_cap_survivors: 500`, `health.min_ranked: 150`, checked after each stage.
  Breach → `PipelineHealthError` → exit 1 → status failed + alert email; `--allow-degraded`
  continues and records it. Stage counts written to `data/last_run_stats.json`, folded
  into `run_status.json` as `stats` (only when dated today); watchdog `check_screen_health`
  fails on a failed run or breached floor and warns when no screen exists after 17:30.

### portfolio_engine.py (new, pure)
Inputs: `screen` (rows from screen_latest.json), `book` (Alpaca positions +
account), `plans` (exit plans from paper_stops), `regime` (stress overlay), `policy`
(config `trader` block), `today`. Output: `Plan` dict:

```
{
  "as_of": date, "regime": {...}, "equity": float,
  "targets": {"alpha_pct": 0.70, "core_pct": 0.25, "cash_pct": 0.05},
  "breaker": {"active": bool, "drawdown": float, "peak_equity": float},
  "legs": [ {"id": "L1", "kind": "exit|trim|earnings_trim|rotate_out|entry|add|core_buy|core_sell",
             "symbol", "side", "qty"|"notional", "reason", "risk_reducing": bool,
             "source": {...evidence...}} ],
  "candidates_considered": [...], "rejected": [{"symbol","reason"}]
}
```

Rules (policy = aggressive):

| Parameter | Value |
|---|---|
| risk_per_trade | 1.25% of equity at the max-loss floor |
| name_cap | 12% |
| sector_cap | 35% |
| max_positions (alpha) | 8 |
| alpha target by regime | NORMAL 0.70, CAUTION 0.40, STRESS 0.15 |
| core sleeve | SPY; = 1 − alpha − 0.05 cash buffer |
| drawdown breaker | equity < peak × 0.85 → alpha target halved until equity ≥ peak × 0.93 |
| earnings | no entry within 5 sessions of a print; held position > 6% of equity with a print inside 2 sessions → trim to 6% |
| entry filter | rank ≤ entry_band(20), entry ∈ {OK, STRONG}, entry_signal ∉ {avoid}, fund_score ≥ 0.30, tradable, price ≥ 5 |
| exit | exit-plan verdict SELL → sell all; TRIM → sell 1/3; rank > exit_band(35) for 3 consecutive screens → sell all |
| add | position ≥ +1R, still rank ≤ entry_band, entry ∈ {OK, STRONG}, room under name cap → add up to half the original risk |
| ordering | sells → trims → core adjustments → entries |

Leg sizing for an entry: `qty = min(risk_dollars / (price − floor), name_cap × equity / price)`
where `floor` is the exit plan's initial stop (entry − 2×ATR14) computed from the
screen's ATR column (new column `atr_14`). Sector exposure is checked with the
Finnhub sector.

`risk_reducing` = every exit/trim/earnings_trim/rotate_out leg, plus core_sell when
the breaker is active. `execute-plan --risk-only` runs only those.

### trader_cli additions (IO lives in src/trader_plan.py)
- `screen [--top N]`: `screen_latest.json` health + compact top rows.
- `news SYMBOL [--days 7] [--limit 10]`: Finnhub company news; works headless.
- `plan [--target D] [--no-write]`: builds the plan, writes `trading/plans/<target>.json`,
  persists peak/breaker/rank-history state (not `trims_executed`).
- `review LEG DECISION --reason R [--notional N]`: APPROVE / SKIP / DOWNSIZE / DEFER.
- `execute-plan [--legs L1,L2] [--risk-only] [--dry-run]`: places legs in order,
  honouring reviews; idempotent; crash-safe.
- `journal-draft [--force]`: pre-fills `trading/journal/<target>.md`.
- `buy/sell --replace-stop`: cancel the symbol's protective stop first.
- `trading/plans/` and `trading/engine_state.json` are gitignored and published to the
  private data repo; the Paper page reads the latest plan through datastore.

### Trader prompt (trading/PROMPT.md rewritten)
Reviewer role. Steps: `screen` (abort if health.ok false → journal + no trades),
`status`, `plan`, `news` for every leg symbol and every holding, decide per leg
(APPROVE / SKIP / DOWNSIZE with a reason), journal (engine pre-fills snapshot, plan
table, scorecard via `trader_cli journal-draft`), `execute-plan --legs ...`, verify,
`sync-stops --apply`. Powers: may SKIP or DOWNSIZE any leg; may NOT add symbols,
upsize, or override the breaker. `--max-turns 40`.

### run_trader.sh
Builds the plan (`trader_cli plan --target`) before launching the session. After a
nonzero exit with `safe_to_retry: true` on the final attempt (attempt 5 of 5, or
time ≥ 21:30 ET), runs `execute-plan --risk-only` plus `sync-stops --apply`, then
stamps. Tool set drops WebSearch/WebFetch (denied headless anyway); `--max-turns 40`.

### UI (app_shared.py)
- Monitor: health strip from `run_status.json.stats` (stage counts, ok/fail).
- Screener: fundamentals sub-score chips and `fund_score` column.
- Paper: latest plan file rendered as a table with leg status.

## Edge-case register (each has a test in the reference implementation)

Data layer
- yfinance batch exception vs "ticker returned no rows": only the latter quarantines; a
  persistent exception after halving to 25-ticker batches leaves names *unfetched* for
  this run and never quarantines (`fetch_with_split`).
- Quarantine TTL 7 days; the 08-24/08-27 mass entries are purged once with
  `python -m src.cache_maint purge-failed`.
- Market cap precedence: Finnhub metrics → Finnhub profile → shares × price → legacy
  cache → NaN; every source bounds-checked to [$1M, $20T]. `shares_x_price` can
  overstate ADR caps (TSM: ordinary shares × ADR price) — harmless for a ≥$300M gate,
  visible via `cap_source`, never used for ratios.
- Finnhub `{}` for an unknown symbol is cached as "no data" so it is not re-asked daily;
  429 sleeps 60 s and retries once; 403 stops the loop for the run.
- Finnhub/Alpaca write share classes with a dot (BRK.B); SEC and yfinance with a dash.
  Preferreds/warrants/units/rights are excluded by suffix on both spellings.
- The Alpaca asset list caches for 24 h; without credentials or on failure with no cache
  the filter is skipped, so a public clone still screens.
- The universe parquet is rebuilt when older than 7 days; a SEC outage falls back to the
  stale file.
- EDGAR: a 10-K carries the fiscal year and often Q4 under the same `end` date — flow
  items are filtered to ≈12-month durations. Synonym tags are chosen by RECENCY of the
  latest fiscal year, not list order (MSFT's `Revenues` tag stopped in FY2010; tag
  priority produced a negative gross profitability — this bug exists in production
  today). Facts older than 550 days are discarded. Restated values: latest `filed` wins.
  GrossProfit is a fallback when COGS is absent. IFRS-only filers yield "no data".
- Health floors are checked after each stage (fail fast); a breach writes
  `screen_latest.json` with `health.ok=false` and no rows, writes no dated CSV, exits 1,
  and emails. `--allow-degraded` overrides for manual runs and is recorded.
- The stress regime's intentionally empty screen is `health.ok=true` with zero rows.

Fundamentals block
- Negative P/E, P/FCF, EV/EBITDA, P/S are NaN, not "cheap". Negative debt/equity
  (negative equity) ranks as the most levered; NaN stays NaN.
- Financials drop gp_assets and accruals and skip the invest/strength sub-scores.
- fund_score needs ≥ 2 sub-scores; NaN is excluded by the gate. The gate skips itself
  (with a health warning) when fund_score coverage < 60% — a cold Finnhub cache must
  not empty the ranking.
- Finalist enrichment: EPS revisions divide by |then| so negative-EPS names work;
  |then| < 0.02 is NaN; fiscal-year row first, quarter as fallback; per-(ticker, date)
  cache; one retry per ticker; wall-clock budget so the run finishes before 17:15.

Engine
- No entries from a screen with `health.ok=false`, dated before the last completed
  session, or with no rows. Exits, trims, earnings trims and the core sleeve still run.
- Sells first; buys are limited to cash plus 97% of planned sell proceeds. Projected
  cash < 0 → a fixed `core_sell` sized on haircut proceeds.
- Stop distance = 2×ATR14, capped at 15% of price, 8% fallback without ATR; < 2% is
  rejected as too tight.
- Name cap counts the existing position; sector cap counts held plus planned; "Unknown"
  sector is one bucket (conservative). Max positions counts survivors plus entries.
- Non-fractionable names round down to whole shares; below $500 is rejected.
- TRIM rungs execute once: `trims_executed` is committed only when the trim order is
  actually submitted, never at plan time.
- Rotate-out needs 3 consecutive screens below rank 35 (or unranked); rank history is
  keyed by screen date so a rerun cannot double-count, and is not updated from an
  unusable screen.
- Earnings trim: print within 2 sessions and weight > 6% → trim to 6%; entries blocked
  within 5 sessions of a print; unknown dates apply no rule.
- Dust (< $50) is sold only in regular hours (fractional market order).
- A symbol with a resting protective stop gets `cancel_stop_first`; a symbol with a
  resting non-stop order is not re-entered.
- Breaker: −15% from peak halves the alpha target; clears at −7%.
- `cmd_plan` refuses to overwrite a plan that already has executions; `cmd_execute` is
  idempotent and rewrites the plan file after every order; failed legs stay retryable.
- Reviews: non-overridable legs accept APPROVE only; DOWNSIZE needs a smaller notional
  on a buy leg; DEFER only on risk-reducing overridable legs; reasons ≥ 10 characters.
- Runner: the plan is built before the session; a crashed session with an untouched book
  on the final attempt (or after 21:30 ET) executes `--risk-only` and re-arms stops; a
  crash after fills never runs the fallback.

## Testing
- Unit tests for every pure module: `filter_tradable`, batch split/retry logic (mocked
  yfinance), `attach_market_caps` precedence, EDGAR extended parser on a fixture,
  `fundamentals_block` percentiles and financial-sector handling, floor gate,
  composite weights sum, `portfolio_engine` (regime targets, breaker, sizing math,
  sector cap, earnings rules, exit legs from plans, rotate-out streak, ordering),
  `execute-plan` dry-run leg selection, `screen_latest.json` health computation.
- Integration: `python -m src.run` against the live cache after Phase 0 must report
  `adv_survivors ≥ 800` or fail with `PipelineHealthError`.
- Replay harness (Phase 3): `python -m src.replay` over `output/screen_*.csv`.

## Rollout
Phase 0 lands before today's 16:30 ET run. Finnhub cache is warmed in the background
beforehand. Phases 1–2 land the same day if possible; the reviewer prompt switches
over only once `plan` has produced a sane plan against the live book in `--dry-run`.
