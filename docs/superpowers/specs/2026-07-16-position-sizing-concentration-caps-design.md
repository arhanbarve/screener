# Position Sizing + Concentration Caps — Design

Date: 2026-07-16
Status: approved, pre-implementation

## Motivation

The screener currently **ranks** stocks and emits a top-N list with no
weights and no cap on how the picks cluster. Diagnosis of the retired
composite backtest (see `docs/CONCLUSION-2026-07-16-composite-stock-picker.md`)
found the engine deterministically assembles a single-theme lottery book at
bubble tops: at the 2021-02-19 peak the top-20 held six crypto miners and fell
-32.4% in 10 trading days. Root cause was unbounded ranking with **no
position sizing and no concentration control** — every name entered at
implied equal conviction, and nothing stopped one theme from dominating.

This feature adds both. It is a **structural risk control**, not an alpha
signal. Its correctness is arithmetic (caps hold, weights sum to 1.0), so it
does not depend on the survivorship-biased backtest to validate — which is
precisely why it can ship where the alpha iterations could not.

## Scope

In scope:
- Inverse-volatility position weights over the selected top-N.
- Per-name weight cap and per-GICS-sector weight cap, enforced by iterative
  redistribution.
- A `weight_pct` column in the live CSV and markdown output (advisory).
- Config-driven parameters under a new `sizing:` block.

Out of scope (explicitly not built):
- Wiring sizing into `portfolio_sim.py` / the walk-forward harness. That
  harness is formally closed, and survivorship bias limits what a backtested
  weight effect would prove.
- Correlation-cluster caps (considered, rejected for now — needs a
  return-correlation matrix; GICS + inverse-vol cover the two main failure
  shapes at lower complexity).
- Conviction- or equal-weighting (rejected: conviction-weighting bets *harder*
  on the crashy high-momentum names; equal-weighting ignores the vol blowup).

## Architecture

New module `src/sizing.py` — three pure functions, no I/O:

```
realized_vol(close_series: pd.Series, window: int = 63) -> float
    Annualized standard deviation of daily log-returns over the trailing
    `window` sessions. Returns NaN if fewer than `window`+1 valid points.
    Uses sqrt(252) annualization.

inverse_vol_weights(vols: dict[str, float]) -> dict[str, float]
    w_i = (1 / vol_i) / Σ_j (1 / vol_j). Names with NaN or non-positive vol
    are dropped before weighting (they cannot be sized). Result sums to 1.0
    (within float tolerance) over the remaining names. Empty input -> {}.

apply_caps(
    weights: dict[str, float],
    sectors: dict[str, str],
    name_cap: float,
    sector_cap: float,
) -> dict[str, float]
    Iterative water-fill:
      1. Clip any name above name_cap to name_cap; redistribute the freed
         weight pro-rata across names still below their cap. Repeat to
         convergence.
      2. For any GICS sector whose total exceeds sector_cap, scale that
         sector's names down proportionally so the sector sums to sector_cap;
         redistribute freed weight pro-rata to names in OTHER sectors that are
         below both their name cap and their sector cap. Repeat to convergence.
    Result sums to 1.0. Convergence is bounded by a max-iteration guard.
```

### Data availability

Each ticker's row already carries `close_series` (full trailing close
history) from `src/prices.py` (`compute_factors_for_ticker`, line ~201) into
the composite DataFrame. Realized vol is therefore computed from data already
in memory — **no new data fetch, no new cache dependency**. ATR was
considered and rejected: it needs high/low, which are not stored past
`prices.py`, whereas `close_series` is; and inverse-realized-vol is the
standard construction.

### Wire point

`src/run.py`, after `build_composite` produces the ranked DataFrame and the
top-N slice is taken. Sizing applies **only to the selected names**, not the
full universe. Everything upstream (gates, factor scoring, composite,
sector-demean) is untouched. The step adds one column, `weight_pct`.

Flow:
```
ranked_df (from build_composite)
  -> select top_n
  -> vols = { ticker: realized_vol(close_series) }
  -> raw = inverse_vol_weights(vols)
  -> capped = apply_caps(raw, sectors, name_cap, sector_cap)
  -> ranked_df["weight_pct"] = capped * 100
  -> write_csv / write_markdown
```

If `sizing.enabled` is false, the column is omitted and behavior is
unchanged from today.

## Configuration

New block in `config.yaml`:

```yaml
sizing:
  enabled: true
  vol_window: 63        # ~3mo realized-vol lookback (trading days)
  name_cap: 0.10        # no single stock above 10% of the book
  sector_cap: 0.25      # no GICS sector above 25% of the book
```

## Known limitations (stated, not hidden)

1. **Sector cap needs somewhere to redistribute.** If the top-N collapses
   into one or two GICS sectors, there is nowhere to move the excess weight.
   The cap then degrades to best-effort (clips as far as the available other
   sectors allow) and logs a warning rather than failing. A single-sector
   top-N is a no-op for the sector cap by construction.
2. **GICS does not capture thematic clusters.** The Feb-2021 crypto miners
   spanned Technology, Financials, and Industrials, so a per-sector cap alone
   would not have contained that specific cluster. What bites it is the
   inverse-vol leg: those names carried 100-200% annualized vol and would be
   auto-shrunk to small weights regardless of sector. The two mechanisms
   cover different failure shapes; both are included deliberately.
3. **Feasibility.** With top_n=20 and name_cap=0.10, per-name capacity is
   200% of the book, so the name cap is always feasible. If a future config
   sets `name_cap * top_n < 1.0`, the weights cannot sum to 1.0 under the
   cap; `apply_caps` clips as far as possible and logs a warning.

## Testing (TDD, red first)

`tests/test_sizing.py`:
- `realized_vol`: known-vol synthetic series returns expected annualized
  value; series shorter than window -> NaN; series with NaNs handled.
- `inverse_vol_weights`: higher-vol name receives strictly less weight;
  output sums to 1.0; NaN/zero-vol names dropped; empty input -> {}.
- `apply_caps`:
  - single name over cap -> clipped to cap, excess redistributed, sum 1.0
  - one sector over cap -> scaled to sector_cap, excess to other sectors
  - already-compliant input -> returned unchanged (sum 1.0)
  - infeasible single-sector input -> best-effort + warning, no crash
  - both caps interacting -> both invariants hold at convergence

`tests/test_run.py` (or existing integration test):
- with `sizing.enabled: true`, output has `weight_pct`; column sums to ~100;
  no name exceeds name_cap; no sector exceeds sector_cap (when feasible)
- with `sizing.enabled: false`, no `weight_pct` column; output unchanged

## Success criteria

- All new unit tests pass; full suite stays green.
- Live screener run emits `weight_pct` in CSV + markdown.
- On real output, invariants verified: max name weight ≤ name_cap, max sector
  weight ≤ sector_cap (or logged best-effort), weights sum to ~100%.
