# Conclusion: composite stock-picker retired; project pivots to index + regime overlay

Date: 2026-07-16
Branch: `walkforward-harness-regime-ladder`
Follows: `docs/handoffs/HANDOFF-2026-07-16-regime-ladder-FAIL.md` (diagnosis),
`output/portfolio_backtest_2026-07-16.md` (dev-window FAIL report).

## Decision

The momentum/composite stock-selection engine is **retired**. No further
implementation iterations, overlay tuning, or factor changes will be run
against it on the current data infrastructure. The regime exposure ladder is
**kept** and repurposed as an advisory overlay on broad index exposure
(SPY), with a single pre-registered dev-window + holdout evaluation
(criteria below, fixed before the holdout run).

## Evidence (all from our own harness, dev window 2015-2023)

| Strategy | CAGR | Sharpe | MaxDD | Notes |
|---|---|---|---|---|
| SPY buy-and-hold | 11.77% | 0.71 | -33.72% | clean daily data |
| composite (unhedged) | 5.81% | 0.34 | -62.23% | survivorship-inflated |
| naive momentum | 9.09% | 0.42 | -55.53% | survivorship-inflated |
| composite + ladder | 15.53% | 0.61 | -58.84% | survivorship-inflated; -59% MaxDD not survivable live |

Every stock-selection variant loses to SPY buy-and-hold on Sharpe, with the
universe's survivorship bias *helping* the stock-pickers. 15-20 full
implementation iterations produced no exception.

### Why the composite fails structurally

1. **It selects its own crashes.** At the 2021-02-19 equity peak the book's
   top-20 contained six crypto miners (RIOT, MARA, HIVE, BTBT, HUT, CLSK)
   plus NVAX/FCEL/GEVO-class names, with 12-month momentum between +300% and
   +7,100%. The book fell **-32.4% in the first 10 trading days** off that
   peak while SPY was flat. Unbounded momentum ranking + no concentration
   control deterministically assembles a single-theme lottery book at every
   bubble top. No reactive overlay (five signal families tested: market-wide
   ladder, book equity-curve trend, book-vs-SPY relative strength,
   symmetric dwell, asymmetric dwell) can cut a 10-day book-specific crash
   by the required one-third — they all fire around day 10, already -32% in.
   The overall-MaxDD pass criterion is therefore mathematically
   unreachable by exposure overlays on this engine.
2. **The data cannot validate alpha even where it exists.** The universe is
   survivors-only; market caps are reconstructed as shares-today ×
   price-then (RIOT shows ~$50B in Feb-2021 vs ~$6B actual — an ~8x error
   that selectively admits heavy diluters through the liquidity gate); the
   panel contains corrupt rows (e.g. FCEL at $606). The data-quality floor
   sits above the alpha signal being hunted. A "pass" on this foundation
   would not be evidence of live profitability.

### What would reopen stock-picking research

Survivorship-free point-in-time data (e.g. Sharadar/Norgate class,
~$40-90/mo). That is a data purchase decision, not an implementation
decision. Until then, further iterations fit noise.

## What ships instead: regime ladder as index overlay

The 4-signal ladder (`src/regime.py`) is sound at what a market-regime
signal can do (2020 window MaxDD cut ~65% on the composite; behavior
verified not path-luck via return-scaling). Applied to SPY itself — clean
daily data, no survivorship bias, no stock selection — with one addition
(asymmetric dwell: exposure cuts apply immediately, re-risking requires the
higher target to persist 5 consecutive sessions), dev-window results:

| | CAGR | Sharpe | MaxDD |
|---|---|---|---|
| SPY | 11.77% | 0.71 | -33.72% |
| SPY × ladder + asym dwell | 8.58% | 0.78 | -20.46% |

This is drawdown insurance priced at ~3 CAGR points/year in a bull decade —
not an income strategy, and documented as such.

## Pre-registered pass criteria (fixed 2026-07-16, before the holdout run)

Honesty note: the dev window (2015-2023) informed the design (dwell
parameter, the decision to overlay SPY), so dev-window "passes" are
in-sample. The 2024-2026 holdout has never been run and decides the verdict.

For `SPY × ladder + asymmetric dwell(5)` vs `SPY buy-and-hold`, on the
holdout window (2024-01-02 → latest cached date), all three required:

1. **MaxDD cut ≥ 1/3** vs SPY's holdout MaxDD.
2. **CAGR give-up ≤ 4 points** annualized.
3. **Sharpe ≥ SPY's holdout Sharpe.**

Known risk, stated in advance: criterion 1 requires a real drawdown in the
holdout window to cut; criterion 3 will likely fail in a monotone bull
window. The holdout runs once; the numbers ship regardless of verdict.

If PASS → wire the daily regime block (points, signals fired, target
exposure, thrust status) into live screener output as **advisory** guidance
on index exposure, per Part 3 of the original spec.
If FAIL → the numbers are documented and the project stops cleanly.

## Caveats carried forward

- The breadth signal (and thrust override) is computed on the
  survivorship-biased universe; historical breadth reads healthier than
  reality. The other three signals (SPY trend, VIX level/term-structure,
  HYG/IEF credit) are clean. A 3-signal clean-only variant was measured
  (dev: CAGR 5.2-7.3%, Sharpe 0.57-0.70, MaxDD -18 to -29% depending on
  map/dwell) and is not materially different in character; the 4-signal
  spec version is retained unchanged to avoid post-hoc signal selection.
- Overlay simulation uses daily return scaling with 5 bps/side cost on
  exposure changes (SPY spread ~1bp; conservative).
