# Standing Exit Plan Engine — Design

**Date:** 2026-07-28
**Status:** Approved by user
**Horizon:** Swing (weeks to a few months)

## Problem

The positions page recomputes a 10-signal soft score from the latest daily bars on
every load. All signals are memoryless snapshots; several (MFI, stochastic, 20d
break, RS-decay) flip on a single market-wide red day. Result: every position says
TRIM on a down day and HOLD on the next up day. The recommendation tracks market
mood, not the position, and is useless for deciding actual exits.

## Goal

Replace the daily-opinion grade with a **standing exit plan** per position:
pre-set price levels that only ratchet in one direction, evaluated mechanically at
each close. The system itself decides when a level is breached and issues a
**final verdict** (SELL / TRIM / HOLD) with an email instruction. No conditional
"if it closes below X…" is ever left to the user — conditionals live in the
reasoning, the verdict is the output.

## Non-goals

- No intraday monitoring; closing prices only (daily 4:30pm ET pipeline run).
- No automated order placement — the system instructs, the user executes.
- No changes to entry-side screening or the paper-trading loop.
- Indicators do not gain veto/sell power (weekly tightener only).

## Verdict vocabulary

| Verdict | Meaning | Instruction |
|---|---|---|
| SELL | A stop level breached | Sell entire position at next open |
| TRIM | A profit-taking rung fired (once each) | Sell 1/3 at next open |
| HOLD | Nothing breached | Do nothing |

No WATCH tier. Silence = HOLD.

## Per-position ExitPlan state (persisted in positions.json)

Computed at entry (or backfilled — see Bootstrap):

| Field | Definition | Mutation rule |
|---|---|---|
| `initial_stop` | entry − 2×ATR14(entry date) | Ratchets to breakeven (entry) after de-risk trim; never lowers |
| `risk_R` | entry − initial_stop | Fixed |
| `peak_close` | Highest close since entry | Only rises |
| `trail_mult` | Chandelier multiplier, starts 3.0 | Weekly tightener may lower to 2.5 / 2.0 (floor); never raises |
| `trims_fired` | e.g. `["derisk", "blowoff"]` | Append-only, irreversible |
| `stop_level` | max(effective initial stop, peak_close − trail_mult×ATR14), never lower than yesterday's | One-way ratchet up |
| `verdict` | HOLD / TRIM / SELL | SELL is terminal until position removed |
| `below_50d_streak` | Consecutive closes < 50d SMA | Reset on close above |

## Daily evaluation (inside existing 4:30pm screener run)

Per position, on closing data:

**SELL** if any:
1. close < `stop_level` (trailing/chandelier breach)
2. close < effective `initial_stop` (max-loss floor / breakeven after trim)
3. close < 50d SMA for **3+ consecutive sessions** (confirmed trend break)

**TRIM — de-risk** (fires once): gain ≥ +2R **or** ≥ +20% from entry →
sell 1/3, `initial_stop` ratchets to breakeven.

**TRIM — blowoff** (fires once): close >25% above 50d SMA, **or** weekly
RSI(14) > 80, **or** parabolic burst (price up >3×ATR14 within 5 sessions) →
sell 1/3.

Remaining 1/3 is the runner: rides `stop_level` until SELL.

**Whipsaw invariance property:** every trigger anchors to the position's own
history (entry, R, peak) or requires multi-day confirmation. A market-wide ±3%
day moves no level. Verdicts cannot oscillate: trims are append-only, stops
ratchet one way, SELL is terminal.

## Weekly tightener (Friday close, weekly bars)

Health check per position: weekly MACD state, RS vs SPY, OBV slope, ADX
direction. If ≥2 bearish → tighten `trail_mult` one step (3.0 → 2.5 → 2.0
floor). **Only tightens; never loosens; never issues SELL.** Health shown as a
context badge on the page and in the digest.

Earnings within 5 days remains a non-scoring risk badge (unchanged behavior).

## Email delivery (existing notify.py Gmail SMTP)

1. **Action alert** — separate email, fires only on the day a verdict becomes
   SELL or a new TRIM rung fires. Subject `ACTION: SELL NVDA` /
   `ACTION: TRIM NVDA`. Body: verdict, level breached, exact instruction,
   reasoning (levels, history).
2. **Daily digest** — positions table appended to the existing daily screener
   email: verdict, close vs stop_level distance, trims fired, health badge,
   earnings badge.

## Positions page UI

Per-position card: verdict (large), stop_level and % distance to it, trims
fired/pending, peak_close, gain in R-multiples and %, health badge, earnings
badge. The 10-signal score grid is replaced by a compact health context row.
Old STRONG EXIT / TRIM / WATCH / HOLD grades removed.

## Bootstrap (one-time migration)

For each existing position: fetch daily history since `entry_date`; compute ATR
at entry → `initial_stop`, `risk_R`; walk history to find `peak_close` and
trims already earned (mark fired without emailing per-day alerts). First daily
run after migration may legitimately emit catch-up TRIM/SELL instructions.

## Files touched

- `src/positions.py` — ExitPlan compute/update + verdict engine (replaces grade
  logic in `compute_exit_signals`); state read/write in positions.json
- `src/run.py`, `run_screener.sh` — daily evaluation stage
- `src/notify.py` — action alert email + digest positions section
- `app_shared.py`, `pages/3_Positions.py` — card UI
- `positions.json` — schema extension (additive)
- `tests/test_positions.py` — verdict engine tests

## Test plan

Synthetic OHLCV fixtures, no network:

- Gap-down through stop → SELL, terminal on subsequent recovery
- Whipsaw invariance: alternating ±3% days → verdict stays HOLD, no flips
- De-risk trim at +2R and at +20% (each path); fires exactly once; stop moves
  to breakeven
- Blowoff trim on each of the three triggers; fires exactly once
- 50d break: 2 days below → HOLD; 3rd day → SELL; reset on close above
- Stop ratchet: stop_level never decreases across any input sequence
- Weekly tightener: 2+ bearish → trail_mult steps down; never steps up
- Bootstrap: history with early +25% run → derisk marked fired, breakeven stop

## Risks / rollback

- Schema change is additive; old fields untouched. Rollback = revert commit;
  positions.json extra keys are ignored by old code.
- yfinance outage during eval: skip position, keep yesterday's state, flag in
  digest (never fabricate a verdict from partial data).
