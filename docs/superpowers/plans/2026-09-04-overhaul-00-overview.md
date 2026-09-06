# Screener + Autonomous Trader Overhaul — Implementation Plan (00: Overview)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore the screener's data pipeline, add a fundamentals block that decides eligibility and carries 32% of the composite, and replace the discretionary LLM trader with a deterministic portfolio engine reviewed by the LLM — per `docs/superpowers/specs/2026-09-04-screener-trader-overhaul-design.md` and the private vetting report `docs/strategy-vetting-2026-09-04.md`.

**Architecture:** Four phases. Phase 0 makes the pipeline correct and loud (Alpaca-filtered universe, batch-safe yfinance, Finnhub caps/sectors, health floors, `screen_latest.json`). Phase 1 adds Finnhub ratios + extended EDGAR facts → five percentile sub-scores → floor gate + 32% weight, and a finalist enrichment stage (true EPS revisions, earnings dates). Phase 2 adds `portfolio_engine` (pure) + `trader_plan` (IO) + new `trader_cli` subcommands, a reviewer `PROMPT.md`, and a runner that builds the plan first and falls back to risk-only execution. Phases 3–4 add a replay harness, dashboard health/plan views and documentation.

**Tech Stack:** Python 3.14, pandas 3, numpy 2, yfinance 1.4, finnhub-python, requests, pytest; bash for the runner; Streamlit for the dashboard. No new dependencies.

---

## How this plan was vetted, and how to use it

Every line of code in plans 01–04 was built in a throwaway copy of the repository, run against the real test suite (644 → **795 passing**, shell suite 13 → **36 passing**), and probed against the live APIs (Alpaca asset list: 12,634 tradable; Finnhub metrics/profiles for 20 names; EDGAR companyfacts for MSFT/AAPL/JPM/NRIX; yfinance `eps_trend`; a full engine plan against the real paper book, no orders placed). The combined result is the **reference patch**:

`docs/superpowers/plans/2026-09-04-overhaul-reference.patch` (42 files, applies cleanly to commit `00c337d`).

Two ways to execute:

**Mode A — apply, then verify (recommended for a cheaper model).**
```bash
cd /Users/arhanbarve/Code/screener && git status --short        # must be clean; stop and ask if not
patch -p1 --dry-run < docs/superpowers/plans/2026-09-04-overhaul-reference.patch   # expect no "FAILED" lines
patch -p1 < docs/superpowers/plans/2026-09-04-overhaul-reference.patch
/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m pytest tests -q            # expect: 795 passed
bash tests/test_run_trader.sh      # expect: passed 36, failed 0
```
Then walk plans 01–04 as a **review + operational checklist**: read each task's "why" and edge cases, run each task's verification commands, and do the operational steps (purge quarantine, warm the Finnhub cache, dry-run the pipeline, dry-run a plan) in the order given. Commit at the phase boundaries after asking the user.

**Mode B — rebuild task by task.** Each task carries the full content of every new file and the exact unified-diff hunks for every modified file, in TDD order (tests first). Use it if the patch no longer applies because `main` moved; resolve conflicts against the intent stated in each task.

Either way: **never run the trader against the paper account with the new prompt until Task 2.8's dry-run gate has been passed**, and never commit without asking.

## Baseline facts the implementer needs

| Fact | Value |
|---|---|
| Repo | `/Users/arhanbarve/Code/screener`, branch `main`, HEAD `00c337d`, clean tree |
| Python | `/Library/Frameworks/Python.framework/Versions/3.14/bin/python3` (3.14); run everything with it, not `python3` |
| Tests | `/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m pytest tests -q` → 644 passed today; `bash tests/test_run_trader.sh` → 13 cases (its summary line is printed mid-file — fixed in Task 2.6) |
| Secrets | `.env` at repo root (FINNHUB_API_KEY, SEC_USER_AGENT, ALPACA_*); `set -a; source .env; set +a` before live probes |
| Launchd | `com.arhanbarve.screener` 16:30 ET weekdays runs `run_screener.sh`; `com.arhanbarve.trader` polls every 15 min and fires `run_trader.sh` (windows 08:30–15:45 / 16:15–23:59 ET); `com.arhanbarve.stopsync` 09:40/12:30/15:50; `com.arhanbarve.watchdog` every 30 min. All read the working tree — **no reload needed after code changes** |
| Cache | `data/cache.db` (22 GB, SQLite WAL). `failed_tickers` has 9,620 rows (4,344 stamped 2026-08-24); `market_cap` newest row 2026-06-27 |
| Private data | `data/`, `output/`, `positions.json`, `run_status.json`, `STRATEGY.md`, `docs/strategy-*.md` are gitignored; `scripts/publish_data.sh` copies whitelisted files to the private repo `arhanbarve/screener-data` |
| Timing | The screener must finish before the trader's 17:15 ET evening session; today it finishes 17:05–17:15 |

## Phase gates

| Phase | Done when | Plan |
|---|---|---|
| 0 | `/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m src.run` reports `adv_survivors ≥ 800`, `cap_survivors ≥ 500`, writes `output/screen_latest.json` with `health.ok: true`, and finishes in < 35 min; `run_status.json` carries `stats`; the 08-24 quarantine is purged; Finnhub cache warmed for the liquid universe | 01 |
| 1 | The run logs `[fund_block] fund_score coverage ≥ 60%` and `[fund_gate] N → M survivors`; CSV has `fund_score`, `eps_rev_30d`, `days_to_earnings`; `test_config_weights_sum_to_one_and_cover_every_factor` passes | 02 |
| 2 | `trader_cli plan --no-write` against the live book prints a sane plan; `execute-plan --dry-run` sends nothing; one evening session runs with the new PROMPT in **dry-run mode** and its journal is reviewed by the user; then the live switch | 03 |
| 3–4 | `/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m src.replay` prints per-horizon tables; Monitor shows the health strip; Paper shows the latest plan; STRATEGY.md/README describe the new composite and engine | 04 |

## Rollout timing (today is Fri 2026-09-04)

1. Phase 0 code + operational steps (Tasks 0.1–0.9) **before 16:30 ET** if possible; otherwise run the screener manually after the code lands (`run_screener.sh --force`) — the 16:30 job skips when a fresh output exists.
2. Warm the Finnhub cache in the background (`cache_maint warm-finnhub`, ~85 min for ~2,500 liquid names at 60 calls/min). The run tolerates a cold cache (600 lookups per run, gate skips below 60% coverage) but degrades until warm.
3. Phase 1 lands with Phase 0 (same patch); the fundamentals gate self-disables until coverage is real.
4. Phase 2: keep the **old** `trading/PROMPT.md` for tonight's session unless Task 2.8's dry-run gate is passed first. The engine files can land; the runner change (plan-before-session, risk-only fallback) is safe to land because `execute-plan --risk-only` only sells what the exit plan says to sell.
5. Phases 3–4 any time after.

## Risk register and rollback

| Risk | Mitigation | Rollback |
|---|---|---|
| Health floor too high for the real universe (e.g. 780 ADV survivors on a holiday week) | Floors are config (`health:`); the run prints the counts before failing; `--allow-degraded` for a manual run | lower `min_adv_survivors` in `config.yaml` |
| Finnhub 403 on basic financials (plan changes) | `fetch_metrics` stops the loop on 403, run continues with profile caps + legacy cache; gate self-disables | none needed |
| Finnhub cap in wrong units for a symbol | bounds check [$1M, $20T]; `cap_source` column | none needed |
| Alpaca asset list unavailable | 24 h disk cache; without cache the filter is skipped (logged) | none needed |
| yfinance blocks the finalist enrichment | wall-clock budget (240 s), per-ticker retry, neutral composite_final | set `fundamentals.enrich_budget_secs: 0` |
| The engine places an order the user disagrees with | every leg has a reason and a reviewer decision on record; fixed legs are only exit-plan SELL/TRIM/earnings trims | `git checkout trading/PROMPT.md run_trader.sh` restores the discretionary session; engine files are inert without the runner calling `plan` |
| Fractional-share limit orders rejected by Alpaca outside RTH | existing behaviour (the old session placed 964.552-share limit orders and they filled); `execute-plan` records `failed` with the broker message and leaves the leg retryable | reviewer retries with whole shares |
| Patch conflicts with a later `main` | Mode B — each task carries its code | — |

## Consolidated edge-case register

See the spec section "Edge-case register" — every item there maps to a named test in plans 01–04.

## Test count checkpoints

| After | `pytest tests -q` | `bash tests/test_run_trader.sh` |
|---|---|---|
| baseline | 644 passed | passed 13 |
| Phase 0 (Tasks 0.1–0.8) | 709 passed | passed 13 |
| Phase 1 (Tasks 1.1–1.5) | 750 passed | passed 13 |
| Phase 2 (Tasks 2.1–2.7) | 795 passed | passed 36 |
| Phases 3–4 | 795 passed (replay tests are included in the 795) | passed 36 |

Counts are exact for Mode A. In Mode B they are exact if tasks are done in order.
