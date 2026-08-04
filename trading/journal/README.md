# Journal coverage and reconstructed entries

The paper-trading account was funded and first traded on **2026-07-24**. Nine
market days elapsed through 2026-08-04, and the trader session completed cleanly
on only three of them. This file records which entries are contemporaneous, which
were reconstructed after the fact, and which days have no entry by design — so
nobody later mistakes a rebuilt entry for a real-time one.

## Coverage

| Date | Session outcome | Entry | Trades |
|------|-----------------|-------|--------|
| 2026-07-23 | gate check failed — session never started | **none, by design** | none |
| 2026-07-24 | completed | contemporaneous | 5 buys |
| 2026-07-27 | crashed | **reconstructed 08-04** | none |
| 2026-07-28 | gate check failed — session never started | **none, by design** | none |
| 2026-07-29 | completed | contemporaneous | 2 sells, 1 buy |
| 2026-07-30 | crashed | **reconstructed 08-04** | none |
| 2026-07-31 | completed | contemporaneous | 3 buys |
| 2026-08-03 | crashed *after executing* | **reconstructed 08-04** | 3 sells, 3 buys |
| 2026-08-04 | crashed | **reconstructed 08-04** | none |

Weekly digests exist for `weekly/2026-W30.md` and `weekly/2026-W31.md`.

## Why some days have no entry

**2026-07-23 and 2026-07-28** are not missing entries — no session ever ran. The
gate check in `run_trader.sh` failed before the agent was invoked (missing keys or
network), so there was no decision-maker present and nothing was owed. Writing an
entry for those days would mean inventing a session that did not happen. The days
are visible instead on the Paper page's run-cadence strip as `GATE FAILED`.

Note that 07-28 was a −2.46% equity day the account rode with no session running.

## What "reconstructed" means here

Reconstructed entries were rebuilt on 2026-08-04 from these sources only:

- **Alpaca order history** (`/v2/orders?status=all`) — fills, quantities, prices,
  order ids, timestamps. Authoritative for everything trade-related.
- **Alpaca portfolio history** (`/v2/account/portfolio/history`) — per-day equity.
  Note these points are stamped at UTC midnight *after* their session, so the
  exchange-local date requires a timezone conversion; see `src/paper.py`.
- **Official closing prices** (yfinance) — position marks.
- **`logs/trader_<date>.log`** — the session's own surviving output, quoted
  verbatim wherever it says anything substantive.
- **`claude-mem` observations** — for **2026-08-03 only**, observations #9420–9431
  captured by the session itself between 09:16 and 11:32 ET.

Cash figures are derived from the fill ledger rather than from `equity − market
value`, because the ledger is exact while marks differ slightly between Alpaca's
consolidated close and yfinance. Where the two disagree the entry says so.

**Rationale was never invented.** Three of the four crashed sessions died before
reaching a decision, and their entries say plainly that no reasoning survives
rather than constructing a plausible one. The 2026-08-03 entry is the exception:
its reasoning is genuinely recovered from the session's own observations, and it
is labelled as recovered-from-summaries rather than as the session's prose. That
entry also states explicitly which section — rejected alternatives — could not be
reconstructed and was left incomplete.

A reconstructed journal is a weaker artefact than a contemporaneous one. It has
the numbers right and the judgement missing.

## Root cause of the gap

All four crashes shared one cause: the job ran while the Mac was asleep. `launchd`
DarkWakes the machine to fire the 09:00 job, the job held no power assertion, and
macOS re-entered sleep within seconds — on 08-03, two seconds after the session
started — killing the API streaming connection mid-response.

The amplifier was `run_trader.sh` writing its once-per-day stamp *before* the
session, so a crash could never retry. Each transient disconnect became a
permanently skipped trading day, with nothing surfacing it.

Both were fixed on 2026-08-04 in commit `824c0ed`: `caffeinate -imsu` wraps the
session, `StartInterval` polling starts one as soon as the Mac is next awake
inside the trading window, the stamp lands after the session, and a crash retries
only when `trader_cli activity-today` confirms the book was never touched.

**Still open:** journaling happens last, so a session that trades and then dies
still loses its record. That is exactly what 2026-08-03 did, and it was only
recoverable because `claude-mem` had incidentally captured the analysis. Luck, not
process.
