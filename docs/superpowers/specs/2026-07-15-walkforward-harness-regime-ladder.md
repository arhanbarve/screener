# Walk-Forward Portfolio Harness + Regime Exposure Ladder — Design

Status: draft for user approval (2026-07-15). Follows the brainstorm/grill
session of 2026-07-15. This spec covers the first two build items only:
the portfolio backtest harness (the proof bar) and the graded regime
exposure ladder with breadth-thrust re-entry. Event sleeve, new XBRL
factors, and sector caps are deliberately out of scope — each gets its own
spec after this ships and passes.

## Decisions locked in the grill session

| Decision | Choice |
|---|---|
| Down-market objective | Preserve capital (long-only, exposure control, no shorts) |
| De-risking mechanism | Graded ladder 100/66/33/0% exposure |
| Universe | Dual sleeve — this spec covers the liquid core only |
| Proof bar | Full walk-forward portfolio backtest; every future change must beat baseline |
| Sizing | Core 10–15 equal-weight positions |
| Cadence | Weekly core rebalance; regime checked daily |
| Data budget | Free (yfinance max-history), survivorship bias documented, wider pass margin demanded |

## Goal

1. Build a portfolio-level backtest harness that simulates the screener's
   actual mechanics (gates → composite → top-20 band portfolio → weekly
   rebalance → transaction costs) over ~10 years of daily data.
2. Add a graded regime exposure ladder + breadth-thrust re-entry, validate
   it in the harness, and only then wire it into the live screener output.

The harness is the deliverable with the longest life: after this project,
no factor/weight/gate change ships without beating the baseline in it.

## Non-goals

- Event sleeve productionization (insider clusters, 13D) — separate spec.
- New composite factors (net issuance, asset growth, accruals, FIP) — after
  harness exists, one at a time.
- Paid data. Point-in-time fundamentals. Shorting. Options.
- Replacing the existing per-event-study harness (`src/event_backtest.py`)
  — it answers a different question and stays as-is.

## Known limitations (accepted, documented in code and report)

1. **Survivorship bias.** yfinance only has currently-listed names. Backtest
   returns are an upper bound; the existing warning block in
   `backtest/backtest.py` carries over verbatim. Mitigation: pass criteria
   demand a wide margin (below), and regime-ladder value is measured as
   *drawdown reduction vs our own unhedged portfolio*, which is much less
   sensitive to survivorship than absolute CAGR.
2. **Fundamental factors not reconstructable.** SUE, revision breadth/
   magnitude, insider counts, and gp_assets cannot be rebuilt point-in-time
   from free sources. **Harness v1 therefore backtests the price-block
   composite only**: `mom_12_1`, `residual_mom`, `rs_6m`, `rs_accel`,
   `rs_slope`, `pct_from_high`, plus the confirmation gate (SMA200,
   52w-high) and liquidity gate — all computable from OHLCV history.
   Weights renormalized to sum to 1 within the price block. This validates
   the momentum engine and the ladder, not the earnings block. Acceptable:
   the ladder decision is portfolio-level and orthogonal to which factors
   pick the names.
3. **Universe reconstruction.** The historical universe at date t is
   approximated as: today's gated universe tickers whose data exists at t
   and which pass the liquidity gate *computed at t* (market cap
   approximated as shares-outstanding-today × price-at-t). Crude but
   consistent across strategy and baseline, so comparisons stay fair.

## Part 1 — Harness

### Data layer

- New module `src/portfolio_backtest.py` (harness) and cache table
  `portfolio_backtest_prices` in `cache.db` (same JSON-payload pattern as
  `event_backtest_prices`, added in `src/cache.py`).
- Fetch `period="max"` daily OHLCV via yfinance, batched like
  `src/prices.py:_fetch_batch_yfinance`, truncated to the last ~11 years
  (10-year sim + 1-year factor warm-up). One-time cost ~3,000 tickers;
  cached forever (prices are immutable history; refresh only appends).
- Benchmarks/instruments fetched the same way: `SPY`, `^VIX`, `^VIX3M`,
  `HYG`, `IEF`.

### Simulation engine

- **Clock:** daily bars. Composite recomputed weekly (last trading day of
  week, close); trades execute at that close (consistent, simple; no
  intraday pretense). Regime evaluated daily (Part 2).
- **Portfolio rule (mirrors production config):** rank by price-block
  composite among gate survivors; enter names ranked in top `entry_band`
  (20) until 15 positions held; hold until rank > `exit_band` (35) or gate
  failure; refill from top of list. Equal weight at entry; no intra-hold
  rebalancing of weights (drift allowed, matches how a human runs it).
- **Costs:** 20 bps per side flat (10 bps commission-equivalent + 10 bps
  spread/slippage; conservative for $300M+ / $5M ADV names). Constant in
  `TRANSACTION_COST_BPS`, printed in every report.
- **Cash:** uninvested capital earns 0% (conservative; ignores T-bill yield,
  which only flatters the ladder — noted in report).

### Baselines (computed in the same run, same universe, same costs)

1. **SPY buy-and-hold** (total return, auto-adjusted closes).
2. **Naive momentum:** top-15 by raw `mom_12_1` only, same band mechanics.
   This isolates whether the multi-factor composite adds anything over the
   one classic factor.

### Metrics (per strategy, full period + per calendar year)

CAGR, annualized vol, Sharpe (rf=0), max drawdown, longest drawdown
(days), average exposure, annual turnover (×), total costs paid, and
**exposure-adjusted CAGR** (CAGR / mean exposure) so the ladder isn't
penalized for holding cash. Output: markdown report to
`output/portfolio_backtest_{date}.md` + per-day equity-curve CSV.

### Anti-overfit protocol (enforced by convention, stated in the report)

- The most recent 2 full calendar years are the **holdout**: iterate on
  2015–2023 only (`--end-date` default), run 2024–2026 once at the end.
- One change per run; every report names the single change it tests.

## Part 2 — Regime exposure ladder

### Signals (each worth 1 point, computed daily from cached data)

| # | Signal | Definition | Rationale |
|---|---|---|---|
| 1 | Trend | SPY close < SPY SMA200 | Classic slow regime anchor (Faber) |
| 2 | Breadth | % of gated universe above own SMA200 < 40% | Internal deterioration leads the index |
| 3 | Volatility | VIX close > 25 **or** VIX > VIX3M (term-structure inversion) | Stress pricing turns before/with tops |
| 4 | Credit | HYG/IEF ratio < its SMA100 | Credit leads equities in real risk-off |

### Exposure map

| Points | Target exposure |
|---|---|
| 0–1 | 100% |
| 2 | 66% |
| 3 | 33% |
| 4 | 0% |

One signal alone is noise; two independent families agreeing is a regime
statement. The mapping is fixed a priori — the report includes a
sensitivity table (all monotone mappings of the 4 signals) as a robustness
check, **not** as a tuning menu: if the chosen mapping is far from the
sensitivity median, that is a red flag to investigate, not a license to
pick the best cell.

**Mechanics in the sim:** exposure applies as a scalar on total equity
allocated to positions (rest cash). De-risking sells pro-rata from lowest-
rank positions first; re-risking refills from the top of the current list.
Exposure changes execute at the daily close the signal fires (regime does
not wait for the weekly rebalance — that was the grill decision).

### Breadth-thrust re-entry override

- Signal: % of gated universe above SMA50 crosses from below 20% to above
  55% within 10 trading sessions (universe-internal proxy for the Zweig
  thrust).
- Effect: force exposure to max(current ladder target, 66%) for 20
  sessions, then hand control back to the ladder.
- Purpose: caps the ladder's known worst case — late re-entry after a
  V-bottom (2020-04, 2023-01).

### Pass criteria (a priori, both vs the unhedged price-block composite over the same period)

1. Max drawdown reduced by **≥ one-third** across the full sim period, and
   specifically in each of the 2020 and 2022 drawdown windows.
2. CAGR give-up **≤ 2 points** annualized (exposure-adjusted CAGR should be
   ~flat or better).
3. Ladder+composite Sharpe ≥ naive-momentum baseline Sharpe.

If pass → Part 3. If fail → report ships anyway with the numbers; no live
wiring; revisit signal set (that is a new spec, not a tuning loop).

## Part 3 — Live wiring (only after Part 2 passes)

- Daily run computes the 4 signals + thrust from already-cached data and
  emits a **regime block** in the markdown output and app: current points,
  which signals fired, target exposure, thrust status.
- Advisory only in v1: the screener tells the user "target exposure 66% —
  new entries half-size / hold cash", it does not resize existing
  positions automatically. Position auto-sizing is a later decision.
- Config additions under a new `regime:` key (thresholds above), all
  defaults matching this spec so the backtest and live logic share one
  code path (`src/regime.py`, imported by both).

## Build order

1. Cache + max-history fetch layer (`src/cache.py`, fetch helper).
2. Point-in-time price-block factor computation + gates at date t.
3. Portfolio sim engine + baselines + report.
4. `src/regime.py` signals + ladder + thrust, integrated into sim.
5. Full run, sensitivity table, holdout run, verdict vs pass criteria.
6. (If pass) live regime block in output + app.

Old `backtest/backtest.py` is superseded by the new harness and deleted in
step 3 (its survivorship warning text moves to the new module).
