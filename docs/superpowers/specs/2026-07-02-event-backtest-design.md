# Event-Driven Filing Watcher — Backtest Design (Phase 1 + Phase 2)

Status: approved by user (spec dictated directly, 2026-07-02). Supersedes the
"$100-200 full-corpus Claude backtest" budget in `docs/handoffs/HANDOFF-2026-07-02-1155.md`
and `handoff-thesis-pivot-2026-07-02.md`, which mistakenly scoped Claude
comprehension over the entire historical dataset instead of a bounded
validation sample.

## Goal
Confirm or kill each thesis-A/B event category (see prior handoff) before any
watcher, UI, or Claude spend, using a two-phase backtest:

- **Phase 1**: pure event study, deterministic, $0. This alone must justify
  moving to Phase 2.
- **Phase 2**: materiality validation on a hard-capped sample (≤300 events),
  only for categories that pass Phase 1, ≤$5 if the Claude API path is used.

## Event categories (Phase 1)
Pulled from EDGAR submissions API (per-CIK, reuses `src/filings.py:fetch_submissions`,
already TTL-cached) over the liquidity-gated universe (`lazy_prices` band,
$50M–$2B, ≥$200K ADV — same gate `lazy_run.py` already applies), lookback
default 730 days:

1. `8k_1_01` — 8-K Item 1.01 (material definitive agreement)
2. `8k_3_01` — 8-K Item 3.01 (exchange listing/compliance notice)
3. `delinquent_filer_regains` — an NT 10-K/10-Q followed by the real 10-K/10-Q
   for the same CIK; event date = the real filing's date
4. `odd_lot_tender` — SC TO-I filings; event date = filing date. Body text is
   regex-flagged for odd-lot/no-proration language as metadata, not a Phase-1
   filter (keeps the pull simple, per spec)
5. `going_concern_removed` — consecutive 10-K pairs for the same CIK where a
   plain-string regex for going-concern language matches the prior filing and
   not the current one (reuses `src/filings.py` HTML fetch + `plain_text`)

No Claude calls anywhere in Phase 1.

## Forward returns
- Price history fetched directly via yfinance for the (small) set of tickers
  that actually produced events — not the full universe. New table
  `event_backtest_prices` in `cache.db` (added to `src/cache.py`, same
  JSON-payload pattern as `submissions`/`fundamentals`) caches full history per
  ticker so re-runs don't re-fetch.
- Benchmark: **IWM** (Russell 2000) as a size-matched proxy. This is a known
  simplification vs. a true decile-matched synthetic benchmark; documented in
  code. Reasonable given the universe is already restricted to the $50M–$2B
  band, and building a full cross-sectional matching engine is out of scope
  for a $0 gate.
- Abnormal return = ticker return − IWM return over the same trading-day
  window, computed at 5/20/60-day horizons from the event date.

## Statistics + gate
Per category × horizon: event count, events/month, mean/median abnormal
return, one-sample t-test (scipy) of abnormal returns vs. 0.

Gate (advisory label printed in the summary, not an automatic code-delete):
`PASS` if `|mean_abnormal_return| >= 1.5%` AND `p < 0.10` at the 20-day or
60-day horizon. Categories not marked PASS are excluded from Phase 2 by
default (`--include-category` can override for a manual look).

## Phase 2 — materiality validation
Only runs for categories flagged PASS in Phase 1.

- **Sample**: hard cap `MAX_PHASE2_SAMPLE = 300` total events, stratified
  across Phase-1 return deciles within passing categories (not a training
  set — a check that materiality separates winners from losers). The cap is
  a Python constant with an assertion; no runtime override flag.
- **Zero-cost default**: export the sample to
  `output/phase2_sample_{date}.csv` for hand-labeling (largest/smallest
  apparent-return events first). This is the default path — nothing calls
  the Claude API unless `--claude-batch` is explicitly passed.
- **Claude path (opt-in, `--claude-batch`)**: Haiku only, via the Anthropic
  **Message Batches API** (`client.messages.batches.create`), never
  synchronous per-call loops. Filing text is truncated to the Item 1.01/3.01
  body + named exhibit headers before sending (reuses the `_rough_diff`-style
  section-extraction approach already in `filing_analysis.py`), targeting
  ≤8k input tokens/filing.
  - Cost is estimated **before** submission (input token count × Haiku batch
    rate) and the batch is refused if projected cost exceeds `$5` — Batch API
    is asynchronous and gives no mid-run usage signal, so the $5 abort is
    enforced as a pre-flight gate rather than a mid-loop counter (documented
    deviation from a literal "print every 25 filings" loop, which isn't
    possible against the Batches API).
  - `--dry-run` reports the exact sample size and projected token/cost with
    no network call.
  - A separate retrieval step polls the batch and joins results back onto
    the sample when ready (batches can take up to 24h; this repo won't block
    on it synchronously).
- **Local-model path**: intentionally not implemented. The user's spec
  offered hand-labeling OR a local model (Ollama) as the two zero-cost
  options; Ollama isn't installed/verified in this repo, so hand-labeling via
  CSV export is the zero-cost default actually shipped. Wiring a real local
  model is future work if the CSV path proves too slow.

## Reuse map
- `src/filings.py`: `fetch_submissions`, `fetch_filing_doc`, `plain_text` — reused as-is.
- `src/lazy_run.py`: `_apply_neglect_gate` — reused for the liquidity band.
- `src/cache.py`: new `event_backtest_prices` table + `get_backtest_prices`/`put_backtest_prices`, same pattern as existing cache functions.
- `src/prices.py`: `_fetch_batch_yfinance` extended with optional `start`/`end` params (defaults unchanged — still `period="420d"` when omitted) so the backtest's multi-year price pulls reuse the same batched, rate-limit-conscious fetch the live screener already uses, instead of one HTTP call per ticker.
- `src/news.py`: `_parse_json` reused directly for Phase 2 batch-result parsing (same import path `filing_analysis.py` already uses).

## Out of scope for this spec
- The live daily watcher (Stage 2/3 funnel from the original handoff) — waits
  on Phase 1/2 results.
- Full cross-sectional size/sector-matched benchmark construction.
- Odd-lot tender proration-language filtering as a Phase-1 gate (metadata only).
- Comment-letter cycle (thesis D) — parked per original handoff.

## Files touched
- New: `src/event_backtest.py` (Phase 1 + Phase 2 logic, CLI)
- Edit: `src/cache.py` (new table + 2 functions)
- New: `tests/test_event_backtest.py`
- Edit: `handoff-thesis-pivot-2026-07-02.md` (replace cost-budget section with this two-phase plan)
