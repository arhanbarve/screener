# Flow/Liquidity Signal Family Backtests Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and gate-test three structurally new (non-corporate-disclosure) strategy candidates: (A) idiosyncratic no-news drawdown reversal, (B) S&P 500 deletion-overshoot bounce, (C) intra-industry lead-lag spillover.

**Architecture:** Each idea is a standalone `src/<name>_backtest.py` CLI module following the exact pattern of `src/divhike_backtest.py` (detect -> dry-run kill -> abnormal returns -> gate summary -> split-half). Shared new helpers (news split, earnings split, market-cap-at-event proxy) go in `src/backtest_recipe.py`. The return harness (`compute_abnormal_returns`, `summarize`), gate math, and split-half logic are reused unchanged from `src/event_backtest.py` and `src/insider_backtest.py`.

**Tech Stack:** Python 3, pandas, yfinance (free), EDGAR submissions JSON (free), Wikipedia S&P 500 constituent-change table (free), sqlite cache at `data/cache.db`, pytest.

---

## Context primer (read this, you have zero session context)

This repo has killed 8 corporate-event strategies in a row with a hard pre-registered process. Authoritative background: `docs/strategy-registry.md` (rules + test ledger), `docs/handoffs/FIT-2026-07-03.md` (fit constraints), `docs/handoffs/HANDOFF-2026-07-03-1603.md` (session narrative). Read the registry's "PASS confirmation requirements" section before writing any code.

Non-negotiable process rules (from the registry, they apply to every task below):

1. Pre-register gate BEFORE seeing any results (Task 0 does this; do not reorder).
2. Every idea has a control category checked in the same run.
3. Split-half time stability required.
4. Dry-run event count kill BEFORE the full gate run: kill if signal < 10 events/yr OR < 50 total.
5. Ideas run strictly one at a time, in order A -> B -> C. Finish an idea (ledger row committed) before starting the next module's run. (Building code for all three up front is fine; RUNNING them is sequential.)
6. If any idea's gate PASSes: STOP everything, do not build watchers/alerts/UI, report to the user and wait.
7. $0 data cost. No paid APIs, no Claude API calls anywhere in these backtests.

Key existing functions you will reuse (do not reimplement):

- `src.pead_backtest.load_universe()` -> DataFrame `[ticker, market_cap]`, tickers >= $2.5B (2,269 rows). `src.pead_backtest.event_date_from_announcement(iso_str)` -> shifts +1 calendar day.
- `src.event_backtest.get_history_bulk(tickers, db_path, start, end)` -> `dict[ticker, DataFrame]`; frames have a DatetimeIndex and lowercase columns incl. `close` (auto-adjusted). Cached in sqlite (`event_backtest_prices`, 30-day TTL); ~3,316 tickers already cached as of 2026-07-03 so most fetches are instant.
- `src.event_backtest.compute_abnormal_returns(events_df, db_path, horizons, benchmark)` -> adds `ret_{h}d` / `abn_ret_{h}d` columns. Requires `ticker`, `event_date` (iso str), `category` columns.
- `src.event_backtest.summarize(events_df, horizons, gate_horizons)` -> per-category gate table (winsorized mean >= 1.5%, p < 0.10, mean/median sign agreement). Prints PASS/FAIL per category.
- `src.event_backtest.list_filings(cik, db_path, lookback_days)` -> list of dicts with `form`, `filing_date` (iso str), from cached EDGAR submissions JSON.
- `src.insider_backtest.split_half_summary(events)` -> requires a datetime column named `filing_date`.
- `src.backtest_recipe.dedupe_events(events, dedupe_days=20)` -> requires `ticker` + `file_date` columns; earliest event wins.
- `src.backtest_recipe.load_earnings_for_tickers(tickers, db_path)` -> DataFrame `[ticker, announce_date]` (fetches + checkpoints via yfinance).
- `src.cache.init_db(DB_PATH)` must run at the top of every `main()`.

Conventions: run tests with `python3 -m pytest tests/ -q` (236 passing at plan time; keep it green). Run a backtest with `python3 -m src.<name>_backtest --dry-run-only` first, then without the flag. Output CSVs go to `output/` with the `<name>_{events,summary,halves}_<date>_<tag>.csv` naming. Commit messages follow the existing `feat(<idea>): ...` style seen in `git log`. Commit after each green-test milestone. Never commit `output/` files (untracked by convention).

Why these three ideas (one paragraph, so you can write sensible docstrings): all 8 prior kills were "market underreacts to a public disclosure" bets, which die in liquid $2.5B+ names because disclosed information is arbitraged in minutes. These three are liquidity/flow bets instead: they get paid for providing liquidity to non-fundamental selling (forced or panicked sellers), which is a risk premium, not an information edge, and therefore is not necessarily arbitraged away. Idea A: multi-day idiosyncratic price dislocations WITHOUT any news event revert (Da/Liu/Schaumburg: reversal profits come from the liquidity component of returns). Idea B: index deletion forces ~$5T of indexed money to sell regardless of fundamentals; the overshoot reverts over weeks (deletion effect, unlike the inclusion effect, has not decayed in recent literature). Idea C: information diffuses gradually across economically linked names (intra-industry lead-lag), a cross-name effect none of the 8 kills touched.

---

### Task 0: Pre-register all three ideas in the strategy registry

**Files:**
- Modify: `docs/strategy-registry.md` (append new sections at end of file)

This MUST be the first commit, before any backtest code runs. Gates are fixed here and may not be changed after any result is seen.

- [ ] **Step 1: Append the following markdown verbatim to the end of `docs/strategy-registry.md`**

```markdown

## IDEA BANK 2 — 2026-07-03 (flow/liquidity family, post 8-for-8 creative pivot)

The entire first bank drew from one signal family (corporate-disclosure
underreaction) and went 0-for-8. This bank draws from flow/liquidity risk
premia instead: compensation for providing liquidity to non-fundamental
sellers. Same fit constraints, same PASS requirements, same ledger.

Survivorship-bias upgrade, pre-registered for this whole bank: every idea
below filters events on a market-cap-AT-EVENT proxy
(cap_proxy = current market_cap x price_at_event / current_price) >= $2.5B,
NOT current cap alone. This matters far more for reversal-family ideas than
it did for the event bank, because "dropped hard and later recovered enough
to stay in the universe" is exactly the survivorship path that fakes a
reversal edge. Residual caveat (still disclosed): current shares outstanding
are used in the proxy, and names delisted before 2026-07 are absent entirely.

| Rank | Idea | Mechanism | Event source ($0) | Control category | Main kill risk |
|------|------|-----------|-------------------|------------------|----------------|
| A | Idiosyncratic no-news drawdown reversal (Q5) | Liquidity provision: multi-day idio drops with NO news event are flow-driven and revert; news-driven drops do not (Da/Liu/Schaumburg decomposition) | yfinance prices (universe fully cached) + EDGAR submissions for the no-news filter | (a) same-size drops WITH news (expect no bounce); (b) symmetric no-news UP moves (expect nothing) | post-2010 HFT liquidity provision may have compressed even multi-day large-cap reversals |
| B | S&P 500 deletion-overshoot bounce (Q6) | Forced selling by index funds is the purest non-fundamental seller; deletion discount overshoots and reverts over weeks; deletion side has NOT decayed (unlike inclusion) | Wikipedia S&P 500 constituent-change table + yfinance | index ADDITIONS same window (expect null/negative) | event count (~10-15 non-M&A deletions/yr); deleted names delisting before 2026-07 are unpriceable |
| C | Intra-industry lead-lag spillover (Q7) | Information diffuses gradually across economically linked names; peer of a big idiosyncratic winner drifts up over following days-weeks (Cohen-Frazzini economic links; Hou intra-industry lead-lag) | yfinance prices + yfinance sector/industry profile (cached) | same setup with leader DOWN >= 7% (expect null/negative) | published effect concentrates in SMALL laggards; both legs >= $2.5B is the least favorable slice |

### Q5 — Idiosyncratic no-news drawdown reversal (idea A)
- Hypothesis: a 5-trading-day residual return vs SPY that is both <= -7%
  and <= -2.5 residual standard deviations, with NO earnings announcement
  within +/-3 calendar days and NO 8-K/6-K filed in [-3, +1] calendar days
  of the trigger, mean-reverts over the following 5-20 trading days.
- Signal category: `rev_drop_nonews`. Controls: `rev_drop_news` (same
  trigger, but an earnings date or 8-K/6-K IS present — expect materially
  weaker/no bounce; the spread between signal and this control is the
  hypothesis discriminator), `rev_spike_nonews` (z >= +2.5 and resid >=
  +7%, no news — expect nothing).
- Pre-registered parameters (fixed now): RET_WINDOW=5 trading days,
  VOL_WINDOW=60 trading days of daily residual returns ending RET_WINDOW
  days before the trigger (so the move does not inflate its own vol),
  Z_TRIGGER=2.5, RESID_FLOOR=7%, max 5 events per calendar day per
  direction (most-extreme |z| kept), dedupe 20 days per ticker per
  direction (earliest wins), entry = first close on/after trigger+1
  calendar day (harness convention), benchmark SPY, lookback 3 years.
- Gate (pre-registered): winsorized mean abn ret >= +1.5%, p < 0.10,
  mean/median sign agreement, at h5 or h20. Plus mandatory control
  behavior and split-half stability per PASS requirements above.
- Dry-run kill: signal < 10/yr or < 50 total.
- Status: QUEUED (pre-registered 2026-07-03, before any run).

### Q6 — S&P 500 deletion-overshoot bounce (idea B)
- Hypothesis: a stock deleted from the S&P 500 for a non-M&A reason is
  oversold by forced index-fund selling into the effective date and
  rebounds over the following 5-20 trading days.
- Signal category: `sp500_deletion` (non-M&A removals only; removals whose
  stated reason matches acquisition/merger/taken-private/bankruptcy
  regexes are excluded). Control: `sp500_addition` (names added, same
  window — expect null or negative).
- Pre-registered parameters (fixed now): lookback 6 years (longer than the
  3-year default because deletions are rare), entry = first close on/after
  effective-date+1 calendar day, cap_proxy >= $2.5B at event, earnings
  contamination filter +/-3d applies, benchmark SPY.
- Data-integrity kill (pre-registered): if > 20% of non-M&A deletions in
  the window have no recoverable yfinance price history (delisted), KILL
  regardless of gate result — the priceable subsample is survivorship-
  selected in exactly the direction that fakes a bounce.
- Gate: standard (>= +1.5% winsorized, p < 0.10, sign-agree, h5/h20) +
  control + split-half. Dry-run kill: signal < 10/yr or < 50 total.
- Status: QUEUED (pre-registered 2026-07-03, before any run).

### Q7 — Intra-industry lead-lag spillover (idea C)
- Hypothesis: when one stock in a yfinance industry group moves >= +7%
  vs SPY in one day (the leader), the largest-cap same-industry peer that
  did NOT move (|1-day abnormal| <= 2%) and has no own news (+/-3d
  earnings, [-3,+1] 8-K/6-K) drifts up over the following 5-20 days.
- Signal category: `leadlag_up`. Control: `leadlag_down` (leader moved
  <= -7%, same laggard selection — expect null/negative; long-only cannot
  trade it, it exists purely as the directional discriminator).
- Pre-registered parameters (fixed now): LEADER_MOVE=7% 1-day abnormal
  vs SPY, LAGGARD_MAX_MOVE=2% same day, industry = yfinance `industry`
  field (not `sector`), industry must have >= 3 universe members, one
  laggard event max per industry-day (largest-cap qualifying peer), skip
  industry-days that have BOTH a +7% and a -7% leader, dedupe 20d per
  laggard ticker per direction, entry = trigger+1 calendar day, cap_proxy
  filter on the laggard, lookback 3 years, benchmark SPY.
- Gate: standard + control + split-half. Dry-run kill: < 10/yr or < 50 total.
- Status: QUEUED (pre-registered 2026-07-03, before any run).

Run order is A then B then C, strictly sequential, one ledger row each.
Ideas D (unusual-volume accumulation) and E (vol-compression breakout) from
the same pivot analysis were deliberately NOT pre-registered: weaker priors,
and each gate run spends family-wise error. Revisit only if A-C all die.
```

- [ ] **Step 2: Commit**

```bash
git add docs/strategy-registry.md
git commit -m "docs(registry): pre-register flow/liquidity idea bank 2 (Q5-Q7) before any runs"
```

---

### Task 1: Shared helpers in backtest_recipe.py

**Files:**
- Modify: `src/backtest_recipe.py`
- Test: `tests/test_backtest_recipe.py`

Three new helpers used by ideas A and C (and cap proxy by B too): an earnings SPLIT (existing filter only drops; idea A needs the contaminated rows as a control), a filings-news SPLIT, and the cap-at-event proxy.

- [ ] **Step 1: Write the failing tests.** Append to `tests/test_backtest_recipe.py`:

```python
import pandas as pd

from src.backtest_recipe import (
    attach_cap_proxy,
    filter_cap_proxy,
    split_by_earnings,
    split_by_news,
)


def _ts(s):
    return pd.Timestamp(s)


class TestSplitByEarnings:
    def test_splits_contaminated_from_clean(self):
        events = pd.DataFrame({
            "ticker": ["AAA", "BBB"],
            "event_date": ["2025-03-10", "2025-03-10"],
        })
        earnings = pd.DataFrame({
            "ticker": ["AAA"],
            "announce_date": ["2025-03-09"],  # within +/-3d of AAA event
        })
        clean, contaminated = split_by_earnings(events, earnings, window_days=3)
        assert list(clean["ticker"]) == ["BBB"]
        assert list(contaminated["ticker"]) == ["AAA"]

    def test_empty_earnings_means_all_clean(self):
        events = pd.DataFrame({"ticker": ["AAA"], "event_date": ["2025-03-10"]})
        earnings = pd.DataFrame(columns=["ticker", "announce_date"])
        clean, contaminated = split_by_earnings(events, earnings)
        assert len(clean) == 1 and len(contaminated) == 0


class TestSplitByNews:
    def test_filing_inside_window_is_news(self):
        events = pd.DataFrame({
            "ticker": ["AAA", "BBB", "CCC"],
            "trigger_date": [_ts("2025-03-10"), _ts("2025-03-10"), _ts("2025-03-10")],
        })
        filings = {
            "AAA": [_ts("2025-03-08")],   # 2d before -> news
            "BBB": [_ts("2025-03-11")],   # 1d after -> news (days_after=1)
            "CCC": [_ts("2025-03-01")],   # far before -> clean
        }
        clean, news = split_by_news(events, filings, days_before=3, days_after=1,
                                    date_col="trigger_date")
        assert sorted(news["ticker"]) == ["AAA", "BBB"]
        assert list(clean["ticker"]) == ["CCC"]

    def test_ticker_with_no_filings_is_clean(self):
        events = pd.DataFrame({"ticker": ["ZZZ"], "trigger_date": [_ts("2025-03-10")]})
        clean, news = split_by_news(events, {}, date_col="trigger_date")
        assert len(clean) == 1 and len(news) == 0


class TestCapProxy:
    def test_cap_proxy_scales_current_cap_by_price_ratio(self):
        idx = pd.to_datetime(["2025-01-02", "2026-07-01"])
        prices = {"AAA": pd.DataFrame({"close": [50.0, 100.0]}, index=idx)}
        uni = pd.DataFrame({"ticker": ["AAA"], "market_cap": [10e9]})
        events = pd.DataFrame({"ticker": ["AAA"], "event_date": ["2025-01-02"]})
        out = attach_cap_proxy(events, uni, prices)
        # cap_now 10B x (50 / 100) = 5B
        assert abs(out["cap_proxy"].iloc[0] - 5e9) < 1e6

    def test_filter_drops_below_min_and_missing(self):
        events = pd.DataFrame({
            "ticker": ["AAA", "BBB"],
            "event_date": ["2025-01-02", "2025-01-02"],
            "cap_proxy": [5e9, 1e9],
        })
        kept = filter_cap_proxy(events, min_cap=2.5e9)
        assert list(kept["ticker"]) == ["AAA"]

    def test_missing_price_gives_nan_proxy(self):
        events = pd.DataFrame({"ticker": ["NOPE"], "event_date": ["2025-01-02"]})
        uni = pd.DataFrame({"ticker": ["NOPE"], "market_cap": [10e9]})
        out = attach_cap_proxy(events, uni, {})
        assert pd.isna(out["cap_proxy"].iloc[0])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_backtest_recipe.py -q`
Expected: FAIL with `ImportError: cannot import name 'attach_cap_proxy'`

- [ ] **Step 3: Implement.** Append to `src/backtest_recipe.py`:

```python
def split_by_earnings(events: pd.DataFrame, earnings: pd.DataFrame,
                      window_days: int = 3,
                      date_col: str = "event_date") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Like drop_earnings_contamination, but returns BOTH sides:
    (clean, contaminated). Reversal-family ideas use the contaminated rows
    as a control category instead of discarding them."""
    if events.empty:
        return events, events
    if earnings.empty:
        return events.reset_index(drop=True), events.iloc[0:0]
    ann_by_ticker = {}
    for ticker, g in earnings.groupby("ticker"):
        ann_by_ticker[ticker] = pd.to_datetime(g["announce_date"]).tolist()
    contaminated_mask = []
    for _, row in events.iterrows():
        dates = ann_by_ticker.get(row["ticker"], [])
        ev_ts = pd.Timestamp(row[date_col])
        contaminated_mask.append(
            any(abs((ev_ts - d).days) <= window_days for d in dates))
    mask = pd.Series(contaminated_mask, index=events.index)
    return (events[~mask].reset_index(drop=True),
            events[mask].reset_index(drop=True))


def split_by_news(events: pd.DataFrame, filing_dates_by_ticker: dict,
                  days_before: int = 3, days_after: int = 1,
                  date_col: str = "trigger_date") -> tuple[pd.DataFrame, pd.DataFrame]:
    """(no_news, news) split: an event is 'news' if the ticker filed
    anything in filing_dates_by_ticker within [date - days_before,
    date + days_after]. Tickers absent from the dict count as no-news
    (their filings were checked and none matched, or they have no CIK —
    callers must pass an entry per ticker they actually resolved)."""
    if events.empty:
        return events, events
    news_mask = []
    for _, row in events.iterrows():
        dates = filing_dates_by_ticker.get(row["ticker"], [])
        ev_ts = pd.Timestamp(row[date_col])
        news_mask.append(any(
            -days_before <= (d - ev_ts).days <= days_after for d in dates))
    mask = pd.Series(news_mask, index=events.index)
    return (events[~mask].reset_index(drop=True),
            events[mask].reset_index(drop=True))


def attach_cap_proxy(events: pd.DataFrame, uni: pd.DataFrame,
                     price_cache: dict,
                     date_col: str = "event_date") -> pd.DataFrame:
    """cap_proxy = current market_cap x price_at_event / current_price.
    Pre-registered survivorship mitigation (registry IDEA BANK 2 preamble):
    reversal-family signals must not admit events that were sub-$2.5B when
    they happened just because the name later grew into the universe.
    Rows with no usable price history get NaN (caller filters + logs)."""
    if events.empty:
        events = events.copy()
        events["cap_proxy"] = pd.Series(dtype=float)
        return events
    caps_now = dict(zip(uni["ticker"], uni["market_cap"]))
    vals = []
    for _, row in events.iterrows():
        p = price_cache.get(row["ticker"])
        cap_now = caps_now.get(row["ticker"])
        if p is None or p.empty or cap_now is None:
            vals.append(float("nan"))
            continue
        ts = pd.Timestamp(row[date_col])
        upto = p.loc[p.index <= ts, "close"]
        p_now = float(p["close"].iloc[-1])
        if upto.empty or p_now == 0:
            vals.append(float("nan"))
            continue
        vals.append(float(cap_now) * float(upto.iloc[-1]) / p_now)
    events = events.copy()
    events["cap_proxy"] = vals
    return events


def filter_cap_proxy(events: pd.DataFrame, min_cap: float = 2.5e9) -> pd.DataFrame:
    """Keep rows whose cap_proxy is present and >= min_cap."""
    if events.empty:
        return events
    return events[events["cap_proxy"].notna()
                  & (events["cap_proxy"] >= min_cap)].reset_index(drop=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_backtest_recipe.py -q`
Expected: all PASS

- [ ] **Step 5: Full suite green, then commit**

```bash
python3 -m pytest tests/ -q
git add src/backtest_recipe.py tests/test_backtest_recipe.py
git commit -m "feat(recipe): earnings/news split + cap-at-event proxy helpers for idea bank 2"
```

---

### Task 2: Idea A — src/reversal_backtest.py

**Files:**
- Create: `src/reversal_backtest.py`
- Test: `tests/test_reversal_backtest.py`

- [ ] **Step 1: Write the failing tests.** Create `tests/test_reversal_backtest.py`:

```python
import pandas as pd

from src.reversal_backtest import cap_per_day, detect_dislocations


def _mk_prices(closes, start="2024-01-02"):
    idx = pd.bdate_range(start, periods=len(closes))
    return pd.DataFrame({"close": closes}, index=idx)


def _alternating(base, n, step=0.005):
    """Deterministic +/-0.5% alternation — gives nonzero residual vol
    without randomness."""
    out, p = [], base
    for i in range(n):
        p = p * (1 + step if i % 2 == 0 else 1 - step)
        out.append(p)
    return out


class TestDetectDislocations:
    def test_big_idio_drop_triggers_drop_event(self):
        # 150 quiet days, then five straight -2.5% days (~ -11.9% 5d resid)
        closes = _alternating(100.0, 150)
        for _ in range(5):
            closes.append(closes[-1] * 0.975)
        closes += _alternating(closes[-1], 20)
        prices = {"AAA": _mk_prices(closes)}
        bench = _mk_prices([100.0] * len(closes))  # flat benchmark
        start, end = prices["AAA"].index[0], prices["AAA"].index[-1]
        ev = detect_dislocations(prices, bench, start, end)
        drops = ev[ev["direction"] == "drop"]
        assert not drops.empty
        assert set(drops["ticker"]) == {"AAA"}
        assert (drops["resid_5d"] <= -0.07).all()
        assert (drops["z"] <= -2.5).all()

    def test_small_move_does_not_trigger(self):
        closes = _alternating(100.0, 150)
        for _ in range(5):
            closes.append(closes[-1] * 0.994)  # only ~ -3% over 5d
        prices = {"AAA": _mk_prices(closes)}
        bench = _mk_prices([100.0] * len(closes))
        ev = detect_dislocations(prices, bench,
                                 prices["AAA"].index[0], prices["AAA"].index[-1])
        assert ev[ev["direction"] == "drop"].empty

    def test_big_spike_triggers_spike_event(self):
        closes = _alternating(100.0, 150)
        for _ in range(5):
            closes.append(closes[-1] * 1.026)
        closes += _alternating(closes[-1], 20)
        prices = {"AAA": _mk_prices(closes)}
        bench = _mk_prices([100.0] * len(closes))
        ev = detect_dislocations(prices, bench,
                                 prices["AAA"].index[0], prices["AAA"].index[-1])
        assert not ev[ev["direction"] == "spike"].empty


class TestCapPerDay:
    def test_keeps_most_extreme_z_per_day(self):
        d = pd.Timestamp("2025-03-10")
        events = pd.DataFrame({
            "ticker": [f"T{i}" for i in range(7)],
            "trigger_date": [d] * 7,
            "direction": ["drop"] * 7,
            "z": [-2.6, -2.7, -2.8, -2.9, -3.0, -3.1, -3.2],
            "resid_5d": [-0.08] * 7,
        })
        kept = cap_per_day(events, max_per_day=5)
        assert len(kept) == 5
        assert set(kept["z"]) == {-2.8, -2.9, -3.0, -3.1, -3.2}

    def test_directions_capped_independently(self):
        d = pd.Timestamp("2025-03-10")
        events = pd.DataFrame({
            "ticker": [f"T{i}" for i in range(6)],
            "trigger_date": [d] * 6,
            "direction": ["drop"] * 3 + ["spike"] * 3,
            "z": [-3.0, -2.9, -2.8, 3.0, 2.9, 2.8],
            "resid_5d": [-0.08] * 3 + [0.08] * 3,
        })
        kept = cap_per_day(events, max_per_day=5)
        assert len(kept) == 6
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_reversal_backtest.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.reversal_backtest'`

- [ ] **Step 3: Implement.** Create `src/reversal_backtest.py`:

```python
"""Idiosyncratic no-news drawdown reversal — idea A (registry Q5, IDEA BANK 2).

Registry: docs/strategy-registry.md Q5. Fit: docs/handoffs/FIT-2026-07-03.md.

First non-disclosure-event idea after the corporate-event bank went 0-for-8.
Hypothesis is a liquidity risk premium, not an information edge: a 5-day
residual (vs SPY) drop that is both large (<= -7%) and abnormal for the name
(<= -2.5 sigma of its own daily residual vol) WITHOUT any news event
(earnings +/-3d, 8-K/6-K in [-3, +1]) is flow-driven and mean-reverts.
Drops WITH news are the discriminating control (rev_drop_news): if they
bounce just as hard, the "no-news" conditioning is doing nothing and the
signal is not the liquidity mechanism. Symmetric no-news up-moves
(rev_spike_nonews) are the second control (expect nothing).

Simple SPY subtraction (beta=1 assumption) is deliberate: the z-score is
taken against the name's OWN residual vol, so persistently-high-beta names
have persistently wide residual vol and do not trigger spuriously.

Vol window ends RET_WINDOW days before the trigger so the move being tested
does not inflate its own denominator.

Entry follows the harness convention (event_date = trigger + 1 calendar
day; t0 = first close on/after that): the measured window starts at the
close of the day AFTER the trigger, so any same/next-day bounce a scanner
could not have captured is excluded.
"""

import argparse
import logging
import os
from datetime import date, timedelta

import pandas as pd

from src.backtest_recipe import (
    attach_cap_proxy,
    dedupe_events,
    filter_cap_proxy,
    load_earnings_for_tickers,
    split_by_earnings,
    split_by_news,
)
from src.cache import init_db
from src.event_backtest import (
    compute_abnormal_returns,
    get_history_bulk,
    list_filings,
    summarize,
)
from src.insider_backtest import split_half_summary
from src.pead_backtest import event_date_from_announcement, load_universe

logger = logging.getLogger(__name__)

DB_PATH = "data/cache.db"
BENCHMARK = "SPY"
HORIZONS = (5, 10, 20)
GATE_HORIZONS = (5, 20)
LOOKBACK_YEARS = 3

# Pre-registered in docs/strategy-registry.md Q5 — do not tune after runs.
RET_WINDOW = 5          # trading days for the dislocation move
VOL_WINDOW = 60         # trading days of daily residuals for the z denominator
Z_TRIGGER = 2.5
RESID_FLOOR = 0.07
MAX_PER_DAY = 5         # per calendar day per direction, most-extreme |z| kept
NEWS_DAYS_BEFORE = 3
NEWS_DAYS_AFTER = 1
NEWS_FORMS = ("8-K", "6-K")
MIN_CAP_PROXY = 2.5e9

CATEGORY_SIGNAL = "rev_drop_nonews"
CATEGORY_NEWS = "rev_drop_news"
CATEGORY_SPIKE = "rev_spike_nonews"


def detect_dislocations(prices_by_ticker: dict, bench: pd.DataFrame,
                        start, end,
                        ret_window: int = RET_WINDOW,
                        vol_window: int = VOL_WINDOW,
                        z_trigger: float = Z_TRIGGER,
                        resid_floor: float = RESID_FLOOR) -> pd.DataFrame:
    """Pure. Scan every ticker for trading days in [start, end] where the
    ret_window-day residual return vs the benchmark is beyond +/-resid_floor
    AND beyond +/-z_trigger residual sigmas. Returns columns: ticker,
    trigger_date (Timestamp), resid_5d, z, direction ('drop'|'spike')."""
    b_close = bench["close"]
    b_ret1 = b_close.pct_change()
    b_retw = b_close.pct_change(ret_window)
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    min_len = vol_window + 2 * ret_window + 5
    rows = []
    for ticker, df in prices_by_ticker.items():
        if df is None or df.empty:
            continue
        idx = df.index.intersection(b_close.index)
        if len(idx) < min_len:
            continue
        c = df.loc[idx, "close"]
        resid1 = c.pct_change() - b_ret1.loc[idx]
        residw = c.pct_change(ret_window) - b_retw.loc[idx]
        # sigma of daily residuals over vol_window, ending ret_window days
        # BEFORE each date, scaled to the ret_window horizon
        sigma_w = resid1.rolling(vol_window).std().shift(ret_window) * (ret_window ** 0.5)
        z = residw / sigma_w
        in_window = (idx >= start_ts) & (idx <= end_ts)
        valid = z.notna() & residw.notna() & in_window
        drop_mask = valid & (z <= -z_trigger) & (residw <= -resid_floor)
        spike_mask = valid & (z >= z_trigger) & (residw >= resid_floor)
        for mask, direction in ((drop_mask, "drop"), (spike_mask, "spike")):
            for t in idx[mask]:
                rows.append({"ticker": ticker, "trigger_date": t,
                             "resid_5d": float(residw.loc[t]),
                             "z": float(z.loc[t]), "direction": direction})
    cols = ["ticker", "trigger_date", "resid_5d", "z", "direction"]
    return pd.DataFrame(rows, columns=cols)


def cap_per_day(events: pd.DataFrame, max_per_day: int = MAX_PER_DAY) -> pd.DataFrame:
    """Clustered-trigger cap (pre-registered): at most max_per_day events per
    (calendar day, direction), keeping the most extreme |z|. Limits the
    cross-sectional-correlation distortion of the t-test on crash days."""
    if events.empty:
        return events
    return (events.assign(_absz=events["z"].abs())
            .sort_values("_absz", ascending=False)
            .groupby(["trigger_date", "direction"], group_keys=False)
            .head(max_per_day)
            .drop(columns="_absz")
            .sort_values("trigger_date")
            .reset_index(drop=True))


def load_filing_dates(events: pd.DataFrame, uni_cik: pd.DataFrame,
                      lookback_days: int, db_path: str = DB_PATH,
                      forms: tuple = NEWS_FORMS) -> dict:
    """One submissions-JSON scan per distinct CIK among event tickers
    (cached). Returns {ticker: [Timestamp, ...]} of 8-K/6-K filing dates."""
    cik_by_ticker = dict(zip(uni_cik["ticker"], uni_cik["cik"]))
    out = {}
    tickers = sorted(set(events["ticker"]))
    for i, ticker in enumerate(tickers):
        cik = cik_by_ticker.get(ticker)
        if cik is None or pd.isna(cik):
            out[ticker] = []
            continue
        recs = list_filings(str(cik), db_path, lookback_days)
        out[ticker] = [pd.Timestamp(r["filing_date"]) for r in recs
                       if r["form"] in forms and r["filing_date"]]
        if (i + 1) % 100 == 0:
            print(f"[reversal] filings scanned {i + 1}/{len(tickers)}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description="No-news drawdown reversal backtest (registry Q5)")
    ap.add_argument("--years", type=int, default=LOOKBACK_YEARS)
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--tag", default="")
    ap.add_argument("--dry-run-only", action="store_true",
                    help="stop after the signal count kill decision")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    os.makedirs(args.out_dir, exist_ok=True)
    init_db(DB_PATH)

    end = date.today().isoformat()
    start = (date.today() - timedelta(days=365 * args.years)).isoformat()
    # price history must reach back far enough for the vol window
    fetch_start = (date.today() - timedelta(days=365 * args.years + 150)).isoformat()

    uni = load_universe()
    uni_cik = uni.merge(pd.read_parquet("data/universe.parquet")[["ticker", "cik"]],
                        on="ticker", how="left")
    tickers = sorted(uni_cik["ticker"].unique())
    print(f"[reversal] universe: {len(tickers)} tickers >= $2.5B", flush=True)

    print(f"[reversal] loading price history {fetch_start} .. {end} "
          f"(bulk, sqlite-cached)...", flush=True)
    price_cache = get_history_bulk([*tickers, BENCHMARK], DB_PATH, fetch_start, end)
    bench = price_cache.get(BENCHMARK)
    if bench is None or bench.empty:
        raise RuntimeError("no benchmark history")

    candidates = detect_dislocations(
        {t: price_cache.get(t) for t in tickers}, bench, start, end)
    print(f"[reversal] raw dislocations: "
          f"{candidates['direction'].value_counts().to_dict() if not candidates.empty else {}}",
          flush=True)
    if candidates.empty:
        print("[reversal] KILLED at dry-run: no dislocations detected")
        return

    candidates = cap_per_day(candidates)
    # dedupe 20d per ticker per direction, earliest wins
    parts = []
    for direction, g in candidates.groupby("direction"):
        g = g.assign(file_date=g["trigger_date"])
        parts.append(dedupe_events(g))
    candidates = pd.concat(parts, ignore_index=True)
    print(f"[reversal] after per-day cap + dedupe: "
          f"{candidates['direction'].value_counts().to_dict()}", flush=True)

    candidates["event_date"] = candidates["trigger_date"].dt.date.astype(str).map(
        event_date_from_announcement)
    candidates = attach_cap_proxy(candidates, uni, price_cache)
    n_unpriced = int(candidates["cap_proxy"].isna().sum())
    candidates = filter_cap_proxy(candidates, MIN_CAP_PROXY)
    print(f"[reversal] after cap-at-event proxy filter (>= $2.5B): "
          f"{candidates['direction'].value_counts().to_dict() if not candidates.empty else {}} "
          f"({n_unpriced} dropped for missing price data)", flush=True)

    # classify: earnings first, then filings — either one makes a drop "news"
    earnings = load_earnings_for_tickers(set(candidates["ticker"]), DB_PATH)
    lookback_days = 365 * args.years + 30
    filing_dates = load_filing_dates(candidates, uni_cik, lookback_days)

    drops = candidates[candidates["direction"] == "drop"]
    spikes = candidates[candidates["direction"] == "spike"]

    drops_earn_clean, drops_earn = split_by_earnings(drops, earnings)
    drops_clean, drops_filing = split_by_news(
        drops_earn_clean, filing_dates,
        days_before=NEWS_DAYS_BEFORE, days_after=NEWS_DAYS_AFTER)
    drops_news = pd.concat([drops_earn, drops_filing], ignore_index=True)

    spikes_earn_clean, _ = split_by_earnings(spikes, earnings)
    spikes_clean, _ = split_by_news(
        spikes_earn_clean, filing_dates,
        days_before=NEWS_DAYS_BEFORE, days_after=NEWS_DAYS_AFTER)

    drops_clean = drops_clean.assign(category=CATEGORY_SIGNAL)
    drops_news = drops_news.assign(category=CATEGORY_NEWS)
    spikes_clean = spikes_clean.assign(category=CATEGORY_SPIKE)

    years = max(args.years, 1)
    n_signal = len(drops_clean)
    print(f"[reversal] dry-run signal count ({CATEGORY_SIGNAL}): "
          f"{n_signal} ({n_signal / years:.1f}/yr); "
          f"controls: {CATEGORY_NEWS}={len(drops_news)}, "
          f"{CATEGORY_SPIKE}={len(spikes_clean)}", flush=True)

    if n_signal < 10 * years or n_signal < 50:
        print(f"[reversal] KILLED at dry-run: signal count too thin "
              f"(need >=10/yr and >=50 total, got {n_signal} over {years}y)")
        return
    if args.dry_run_only:
        print("[reversal] dry-run-only requested, stopping before price gate")
        return

    events = pd.concat([drops_clean, drops_news, spikes_clean], ignore_index=True)
    events = compute_abnormal_returns(events, DB_PATH, horizons=HORIZONS,
                                      benchmark=BENCHMARK)
    stem = date.today().isoformat() + (f"_{args.tag}" if args.tag else "")
    events.to_csv(os.path.join(args.out_dir, f"reversal_events_{stem}.csv"), index=False)

    summary = summarize(events, horizons=HORIZONS, gate_horizons=GATE_HORIZONS)
    summary.to_csv(os.path.join(args.out_dir, f"reversal_summary_{stem}.csv"), index=False)
    print(summary.to_string(index=False))

    halves = split_half_summary(events.rename(columns={"trigger_date": "filing_date"}))
    halves.to_csv(os.path.join(args.out_dir, f"reversal_halves_{stem}.csv"), index=False)
    print(halves.to_string(index=False))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_reversal_backtest.py -q`
Expected: all PASS. If `detect_dislocations` misses the synthetic drop, print the computed `z`/`residw` around the slide days and check the `shift(ret_window)` alignment; do not loosen the test thresholds.

- [ ] **Step 5: Full suite green, then commit**

```bash
python3 -m pytest tests/ -q
git add src/reversal_backtest.py tests/test_reversal_backtest.py
git commit -m "feat(reversal): no-news drawdown reversal backtest (idea A, registry Q5)"
```

- [ ] **Step 6: Dry run**

Run: `python3 -m src.reversal_backtest --dry-run-only 2>&1 | tee output/reversal_dryrun_$(date +%F).log`

First run fetches any uncached price history (most of the universe is already cached; expect minutes, not hours) and one submissions-JSON per candidate CIK (0.12s sleep each; a few hundred candidates is normal). Expected output ends with either the dry-run signal count line + "stopping before price gate", or a KILLED line. If KILLED: skip Step 7, go to Step 8 and record the kill.

- [ ] **Step 7: Full gate run (only if dry-run passed)**

Run: `python3 -m src.reversal_backtest --tag full 2>&1 | tee output/reversal_run_$(date +%F).log`

Expected: three-category summary table with `gate` column, then split-half table. Sanity checks before recording anything: `rev_drop_nonews` and `rev_drop_news` both have `h5_n >= 50`... if a category has n < 5 its stats are None (known summarize behavior). Hand-read 10 random `rev_drop_nonews` rows from the events CSV: confirm the trigger dates are real (spot-check 2-3 against a chart or yfinance) and that none has an obvious news event the filter missed (PASS requirement #4).

- [ ] **Step 8: Record the result in the registry ledger and Q5 entry**

Append one row to the Test ledger table in `docs/strategy-registry.md` with the next ledger number, date 2026-07-03 (or actual run date), strategy "No-news drawdown reversal (N=<signal> signal, N=<news>+<spike> controls)", categories gated = 3, Result PASS or FAIL, and a Notes cell quoting the actual h5/h20 means and p-values for signal and both controls plus the split-half verdict. Update the Q5 Status line from QUEUED to the outcome. Judge with the pre-registered logic ONLY:

- Gate PASS requires: `rev_drop_nonews` gate=PASS in the summary AND `rev_drop_news` clearly weaker (if the news control passes the same gate in the same direction, the run is a FAIL per PASS requirement #2) AND `rev_spike_nonews` shows nothing AND split-half signs agree for the signal.
- Anything else is FAIL. No horizon shopping, no threshold tweaks.

```bash
git add docs/strategy-registry.md
git commit -m "docs(registry): ledger row for reversal backtest (Q5) — <PASS|FAIL>"
```

**If PASS: STOP the whole plan here. Report to the user and wait for their review (PASS requirements #4 human review and #5 sign-off). Do not start Task 3.** If FAIL, continue.

---

### Task 3: Idea B — src/spdel_backtest.py

**Files:**
- Create: `src/spdel_backtest.py`
- Test: `tests/test_spdel_backtest.py`

- [ ] **Step 1: Write the failing tests.** Create `tests/test_spdel_backtest.py`:

```python
import pandas as pd

from src.spdel_backtest import is_ma_reason, parse_changes


def _raw(rows):
    """Build a frame shaped like Wikipedia's changes table (MultiIndex cols)."""
    cols = pd.MultiIndex.from_tuples([
        ("Date", "Date"), ("Added", "Ticker"), ("Added", "Security"),
        ("Removed", "Ticker"), ("Removed", "Security"), ("Reason", "Reason"),
    ])
    return pd.DataFrame(rows, columns=cols)


class TestIsMaReason:
    def test_acquisition_is_ma(self):
        assert is_ma_reason("Acquired by Broadcom.")
        assert is_ma_reason("Merged with XYZ Corp")
        assert is_ma_reason("Taken private by KKR")
        assert is_ma_reason("Chapter 11 bankruptcy")

    def test_market_cap_change_is_not_ma(self):
        assert not is_ma_reason("Market capitalization change.")
        assert not is_ma_reason("No longer representative of large-cap space")


class TestParseChanges:
    def test_splits_deletions_and_additions(self):
        raw = _raw([
            ["June 10, 2025", "NEWCO", "New Co", "OLDCO", "Old Co",
             "Market capitalization change."],
            ["May 5, 2025", "BUYER", "Buyer Inc", "TARGET", "Target Inc",
             "Acquired by Buyer Inc."],
        ])
        out = parse_changes(raw, start="2025-01-01", end="2025-12-31")
        dels = out[out["category"] == "sp500_deletion"]
        adds = out[out["category"] == "sp500_addition"]
        # TARGET removed for M&A -> excluded from deletions
        assert list(dels["ticker"]) == ["OLDCO"]
        # both additions kept
        assert sorted(adds["ticker"]) == ["BUYER", "NEWCO"]
        # event_date = effective date + 1 calendar day
        assert dels["event_date"].iloc[0] == "2025-06-11"

    def test_window_filter(self):
        raw = _raw([
            ["June 10, 2019", "A", "A Co", "B", "B Co", "Market capitalization change."],
        ])
        out = parse_changes(raw, start="2025-01-01", end="2025-12-31")
        assert out.empty
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_spdel_backtest.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.spdel_backtest'`

- [ ] **Step 3: Implement.** Create `src/spdel_backtest.py`:

```python
"""S&P 500 deletion-overshoot bounce — idea B (registry Q6, IDEA BANK 2).

Registry: docs/strategy-registry.md Q6. Fit: docs/handoffs/FIT-2026-07-03.md.

Hypothesis: index deletion forces indexed money to sell regardless of
fundamentals — the purest identifiable non-fundamental seller — and the
resulting overshoot reverts over the following weeks. Non-M&A deletions
only (an acquired name leaving the index is not a forced-selling event on
a still-trading stock). Control: additions in the same window (expect
null/negative; the inclusion premium is documented as decayed).

Event source is Wikipedia's S&P 500 constituent-change table (free). The
"Date" column is the EFFECTIVE date; entry is effective date + 1 calendar
day, i.e. after the forced selling into the effective close is done.

Pre-registered data-integrity kill (registry Q6): if > 20% of non-M&A
deletions have no usable yfinance price history (delisted since), the
priceable remainder is survivorship-selected in exactly the direction that
fakes a bounce — KILL regardless of gate output.
"""

import argparse
import io
import logging
import os
import re
from datetime import date, timedelta

import pandas as pd
import requests

from src.backtest_recipe import (
    attach_cap_proxy,
    drop_earnings_contamination,
    filter_cap_proxy,
    load_earnings_for_tickers,
)
from src.cache import init_db
from src.event_backtest import compute_abnormal_returns, get_history_bulk, summarize
from src.insider_backtest import split_half_summary
from src.pead_backtest import load_universe

logger = logging.getLogger(__name__)

DB_PATH = "data/cache.db"
BENCHMARK = "SPY"
HORIZONS = (5, 10, 20)
GATE_HORIZONS = (5, 20)
LOOKBACK_YEARS = 6          # pre-registered: deletions are rare, 3y is too thin
MIN_CAP_PROXY = 2.5e9
MAX_UNPRICED_FRAC = 0.20    # pre-registered data-integrity kill threshold

CATEGORY_DELETION = "sp500_deletion"
CATEGORY_ADDITION = "sp500_addition"

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
UA = {"User-Agent": get_env("SEC_USER_AGENT")}

MA_REASON_RE = re.compile(
    r"acquir|merg|taken private|purchas|bought|combin|bankrupt|chapter 11|delist",
    re.IGNORECASE)


def is_ma_reason(reason: str) -> bool:
    """True if the removal reason is M&A/bankruptcy — excluded from the
    signal because the stock either stops trading or the removal is not a
    forced-selling event on a going concern."""
    return bool(MA_REASON_RE.search(reason or ""))


def fetch_changes_table() -> pd.DataFrame:
    """The constituent-changes table from Wikipedia (the one whose columns
    include an 'Added'/'Removed' MultiIndex level)."""
    resp = requests.get(WIKI_URL, headers=UA, timeout=30)
    resp.raise_for_status()
    for t in pd.read_html(io.StringIO(resp.text)):
        if isinstance(t.columns, pd.MultiIndex) and "Added" in t.columns.get_level_values(0):
            return t
    raise RuntimeError("S&P 500 changes table not found on Wikipedia page")


def parse_changes(raw: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Flatten the Wikipedia table into event rows: columns ticker,
    effective_date (Timestamp), event_date (iso str, effective + 1 day),
    reason, category (sp500_deletion for non-M&A removals, sp500_addition
    for all additions)."""
    dates = pd.to_datetime(raw[("Date", "Date")], format="%B %d, %Y", errors="coerce")
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    rows = []
    for i in range(len(raw)):
        d = dates.iloc[i]
        if pd.isna(d) or not (start_ts <= d <= end_ts):
            continue
        reason = str(raw[("Reason", "Reason")].iloc[i])
        removed = raw[("Removed", "Ticker")].iloc[i]
        added = raw[("Added", "Ticker")].iloc[i]
        event_date = (d + pd.Timedelta(days=1)).date().isoformat()
        if isinstance(removed, str) and removed.strip() and not is_ma_reason(reason):
            rows.append({"ticker": removed.strip(), "effective_date": d,
                         "event_date": event_date, "reason": reason,
                         "category": CATEGORY_DELETION})
        if isinstance(added, str) and added.strip():
            rows.append({"ticker": added.strip(), "effective_date": d,
                         "event_date": event_date, "reason": reason,
                         "category": CATEGORY_ADDITION})
    cols = ["ticker", "effective_date", "event_date", "reason", "category"]
    return pd.DataFrame(rows, columns=cols)


def main():
    ap = argparse.ArgumentParser(description="S&P 500 deletion bounce backtest (registry Q6)")
    ap.add_argument("--years", type=int, default=LOOKBACK_YEARS)
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--tag", default="")
    ap.add_argument("--dry-run-only", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    os.makedirs(args.out_dir, exist_ok=True)
    init_db(DB_PATH)

    end = date.today().isoformat()
    start = (date.today() - timedelta(days=365 * args.years)).isoformat()

    raw = fetch_changes_table()
    events = parse_changes(raw, start, end)
    n_del = len(events[events["category"] == CATEGORY_DELETION])
    n_add = len(events[events["category"] == CATEGORY_ADDITION])
    print(f"[spdel] parsed changes in window: {n_del} non-M&A deletions, "
          f"{n_add} additions", flush=True)

    years = max(args.years, 1)
    if n_del < 10 * years or n_del < 50:
        print(f"[spdel] KILLED at dry-run: {n_del} non-M&A deletions over {years}y "
              f"(need >=10/yr and >=50 total)")
        return
    if args.dry_run_only:
        print("[spdel] dry-run-only requested, stopping before price fetch")
        return

    # price data + data-integrity check on the SIGNAL category
    tickers = sorted(set(events["ticker"]))
    fetch_start = (date.today() - timedelta(days=365 * args.years + 30)).isoformat()
    price_cache = get_history_bulk([*tickers, BENCHMARK], DB_PATH, fetch_start, end)

    dels = events[events["category"] == CATEGORY_DELETION]
    unpriced = [t for t in dels["ticker"]
                if price_cache.get(t) is None or price_cache.get(t).empty]
    frac = len(unpriced) / max(len(dels), 1)
    print(f"[spdel] deletions without price history: {len(unpriced)}/{len(dels)} "
          f"({frac:.0%}) {sorted(set(unpriced))}", flush=True)
    if frac > MAX_UNPRICED_FRAC:
        print(f"[spdel] KILLED (pre-registered data-integrity rule): "
              f"{frac:.0%} > {MAX_UNPRICED_FRAC:.0%} of deletions unpriceable — "
              f"remaining sample is survivorship-selected toward bounces")
        return

    uni = load_universe()
    events = attach_cap_proxy(events, uni, price_cache)
    events = filter_cap_proxy(events, MIN_CAP_PROXY)
    print(f"[spdel] after cap-at-event proxy filter: "
          f"{events['category'].value_counts().to_dict() if not events.empty else {}}",
          flush=True)

    earnings = load_earnings_for_tickers(set(events["ticker"]), DB_PATH)
    events = drop_earnings_contamination(events, earnings)
    print(f"[spdel] after earnings-contamination filter: "
          f"{events['category'].value_counts().to_dict() if not events.empty else {}}",
          flush=True)
    if events.empty:
        return

    events = compute_abnormal_returns(events, DB_PATH, horizons=HORIZONS,
                                      benchmark=BENCHMARK)
    stem = date.today().isoformat() + (f"_{args.tag}" if args.tag else "")
    events.to_csv(os.path.join(args.out_dir, f"spdel_events_{stem}.csv"), index=False)

    summary = summarize(events, horizons=HORIZONS, gate_horizons=GATE_HORIZONS)
    summary.to_csv(os.path.join(args.out_dir, f"spdel_summary_{stem}.csv"), index=False)
    print(summary.to_string(index=False))

    halves = split_half_summary(events.rename(columns={"effective_date": "filing_date"}))
    halves.to_csv(os.path.join(args.out_dir, f"spdel_halves_{stem}.csv"), index=False)
    print(halves.to_string(index=False))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_spdel_backtest.py -q`
Expected: all PASS

- [ ] **Step 5: Full suite green, then commit**

```bash
python3 -m pytest tests/ -q
git add src/spdel_backtest.py tests/test_spdel_backtest.py
git commit -m "feat(spdel): S&P 500 deletion bounce backtest (idea B, registry Q6)"
```

- [ ] **Step 6: Dry run**

Run: `python3 -m src.spdel_backtest --dry-run-only 2>&1 | tee output/spdel_dryrun_$(date +%F).log`

Live-data caveat: the real Wikipedia table's column tuples may differ slightly from the synthetic test shape (e.g. a `Notes` level, or the date column named differently). If `parse_changes` raises a KeyError, print `raw.columns.tolist()`, adjust ONLY the column-tuple lookups in `fetch_changes_table`/`parse_changes` to match reality, keep the tests' synthetic shape in sync, and re-run tests. Do NOT adjust thresholds, regexes, or the window.

Honest expectation: this idea plausibly dies right here (~10-15 non-M&A deletions/yr means the >= 50-total leg is tight even at 6 years). A dry-run kill still gets recorded in the Q6 entry (no ledger row, matching the idea-#4 dividend-initiation precedent: dry-run kills do not consume a ledger slot).

- [ ] **Step 7: Full gate run (only if dry-run passed)**

Run: `python3 -m src.spdel_backtest --tag full 2>&1 | tee output/spdel_run_$(date +%F).log`

Watch for the data-integrity kill line. Hand-check (PASS requirement #4): read 10 random deletion rows, confirm each reason string really is non-M&A and the ticker really left the index on that date (spot-check 2-3 via a web search of the S&P press release).

- [ ] **Step 8: Record the result**

Same procedure as Task 2 Step 8: ledger row (if a gate actually ran) with real numbers, Q6 status update, commit as `docs(registry): ledger row for spdel backtest (Q6) — <PASS|FAIL>` or `docs(registry): Q6 killed at dry-run` accordingly.

**If PASS: STOP, report, wait. Otherwise continue to Task 4.**

---

### Task 4: Ticker sector/industry profile cache

**Files:**
- Modify: `src/cache.py` (new table + put/get)
- Test: `tests/test_cache.py`

Idea C needs an industry map for 2,269 tickers. yfinance `.get_info()` is one HTTP call per ticker, so it must be checkpointed in sqlite like every other fetch in this repo.

- [ ] **Step 1: Write the failing tests.** Append to `tests/test_cache.py`:

```python
def test_ticker_profile_roundtrip(tmp_path):
    from src.cache import get_ticker_profile, init_db, put_ticker_profile
    db = str(tmp_path / "t.db")
    init_db(db)
    assert get_ticker_profile(db, "AAA") is None
    put_ticker_profile(db, "AAA", "Technology", "Semiconductors")
    prof = get_ticker_profile(db, "AAA")
    assert prof == {"sector": "Technology", "industry": "Semiconductors"}


def test_ticker_profile_stores_empty_strings(tmp_path):
    from src.cache import get_ticker_profile, init_db, put_ticker_profile
    db = str(tmp_path / "t.db")
    init_db(db)
    put_ticker_profile(db, "BBB", None, None)  # failed lookups cache as empty
    assert get_ticker_profile(db, "BBB") == {"sector": "", "industry": ""}
```

(Check the top of `tests/test_cache.py` first: if it already has a tmp-db fixture pattern, follow that pattern instead of `tmp_path` verbatim, but keep the same assertions.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_cache.py -q`
Expected: the two new tests FAIL with ImportError.

- [ ] **Step 3: Implement.** In `src/cache.py`, add to the `init_db` DDL block (next to the other CREATE TABLE statements):

```python
        CREATE TABLE IF NOT EXISTS ticker_profile_cache (
            ticker TEXT PRIMARY KEY,
            sector TEXT,
            industry TEXT,
            fetched_at TEXT
        );
```

(Match the exact DDL style used by the surrounding statements in that function — some are separate `conn.execute` calls; follow whatever the file actually does.)

And add these functions near `put_market_cap`/`get_market_cap`:

```python
def put_ticker_profile(db_path: str, ticker: str, sector, industry):
    conn = _get_conn(db_path)
    conn.execute(
        "INSERT OR REPLACE INTO ticker_profile_cache VALUES (?,?,?,?)",
        (ticker, _str_or_empty(sector), _str_or_empty(industry), _now_iso()),
    )
    conn.commit()


def get_ticker_profile(db_path: str, ticker: str) -> dict | None:
    """No TTL: sector/industry are effectively static. None = never fetched;
    empty strings = fetched but yfinance had no data (don't refetch)."""
    conn = _get_conn(db_path)
    row = conn.execute(
        "SELECT sector, industry FROM ticker_profile_cache WHERE ticker=?",
        (ticker,)).fetchone()
    if row is None:
        return None
    return {"sector": row[0], "industry": row[1]}
```

- [ ] **Step 4: Run tests, full suite, commit**

```bash
python3 -m pytest tests/test_cache.py -q
python3 -m pytest tests/ -q
git add src/cache.py tests/test_cache.py
git commit -m "feat(cache): ticker sector/industry profile cache table for lead-lag backtest"
```

---

### Task 5: Idea C — src/leadlag_backtest.py

**Files:**
- Create: `src/leadlag_backtest.py`
- Test: `tests/test_leadlag_backtest.py`

- [ ] **Step 1: Write the failing tests.** Create `tests/test_leadlag_backtest.py`:

```python
import pandas as pd

from src.leadlag_backtest import detect_leadlag


def _mk_prices(closes, start="2024-01-02"):
    idx = pd.bdate_range(start, periods=len(closes))
    return pd.DataFrame({"close": closes}, index=idx)


def _flat_then_jump(n_flat, jump, n_after=10, base=100.0):
    closes = [base] * n_flat
    closes.append(closes[-1] * (1 + jump))
    closes += [closes[-1]] * n_after
    return closes


class TestDetectLeadlag:
    def _fixture(self, jump):
        n_flat, n_after = 30, 10
        prices = {
            "LEAD": _mk_prices(_flat_then_jump(n_flat, jump, n_after)),
            "LAG1": _mk_prices([100.0] * (n_flat + 1 + n_after)),  # cap 8B
            "LAG2": _mk_prices([100.0] * (n_flat + 1 + n_after)),  # cap 4B
            "OTHER": _mk_prices([100.0] * (n_flat + 1 + n_after)),  # other industry
        }
        bench = _mk_prices([100.0] * (n_flat + 1 + n_after))
        industry = {"LEAD": "Semis", "LAG1": "Semis", "LAG2": "Semis",
                    "OTHER": "Banks"}
        caps = {"LEAD": 20e9, "LAG1": 8e9, "LAG2": 4e9, "OTHER": 9e9}
        start = prices["LEAD"].index[0]
        end = prices["LEAD"].index[-1]
        return detect_leadlag(prices, bench, industry, caps, start, end)

    def test_up_leader_selects_largest_flat_peer(self):
        ev = self._fixture(jump=0.09)
        up = ev[ev["direction"] == "up"]
        assert len(up) == 1
        assert up["ticker"].iloc[0] == "LAG1"       # largest qualifying peer
        assert up["leader"].iloc[0] == "LEAD"

    def test_down_leader_makes_control_event(self):
        ev = self._fixture(jump=-0.09)
        down = ev[ev["direction"] == "down"]
        assert len(down) == 1
        assert down["ticker"].iloc[0] == "LAG1"

    def test_small_leader_move_no_event(self):
        ev = self._fixture(jump=0.04)
        assert ev.empty
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_leadlag_backtest.py -q`
Expected: FAIL with ModuleNotFoundError.

- [ ] **Step 3: Implement.** Create `src/leadlag_backtest.py`:

```python
"""Intra-industry lead-lag spillover — idea C (registry Q7, IDEA BANK 2).

Registry: docs/strategy-registry.md Q7. Fit: docs/handoffs/FIT-2026-07-03.md.

Hypothesis: information diffuses gradually across economically linked
names (Cohen-Frazzini; Hou). When a leader in a yfinance industry group
moves >= +7% vs SPY in one day, the largest same-industry peer that did
NOT move (|1d abnormal| <= 2%) and has no news of its own drifts up over
the following days-weeks. Control: identical construction off leaders that
moved <= -7% (leadlag_down) — long-only cannot trade it, it exists to show
the effect is directional rather than "any industry excitement".

Honest prior (flagged in the pivot analysis): the published effect
concentrates in SMALL laggards; both legs >= $2.5B is the least favorable
slice. This is idea C, run only after A and B resolve.
"""

import argparse
import logging
import os
import time
from datetime import date, timedelta

import pandas as pd
import yfinance as yf

from src.backtest_recipe import (
    attach_cap_proxy,
    dedupe_events,
    filter_cap_proxy,
    load_earnings_for_tickers,
    split_by_earnings,
    split_by_news,
)
from src.cache import get_ticker_profile, init_db, put_ticker_profile
from src.event_backtest import compute_abnormal_returns, get_history_bulk, summarize
from src.insider_backtest import split_half_summary
from src.pead_backtest import event_date_from_announcement, load_universe
from src.reversal_backtest import cap_per_day, load_filing_dates

logger = logging.getLogger(__name__)

DB_PATH = "data/cache.db"
BENCHMARK = "SPY"
HORIZONS = (5, 10, 20)
GATE_HORIZONS = (5, 20)
LOOKBACK_YEARS = 3

# Pre-registered in docs/strategy-registry.md Q7 — do not tune after runs.
LEADER_MOVE = 0.07
LAGGARD_MAX_MOVE = 0.02
MIN_INDUSTRY_PEERS = 3
NEWS_DAYS_BEFORE = 3
NEWS_DAYS_AFTER = 1
MIN_CAP_PROXY = 2.5e9
PROFILE_SLEEP_S = 0.3

CATEGORY_SIGNAL = "leadlag_up"
CATEGORY_CONTROL = "leadlag_down"


def collect_profiles(tickers: list[str], db_path: str = DB_PATH,
                     sleep_s: float = PROFILE_SLEEP_S) -> dict:
    """{ticker: industry} via yfinance .get_info(), checkpointed in sqlite.
    One-time ~0.3s/ticker cost for the universe; instant afterwards.
    Failed lookups cache as empty so they are not refetched every run."""
    out = {}
    todo = []
    for t in tickers:
        prof = get_ticker_profile(db_path, t)
        if prof is None:
            todo.append(t)
        elif prof["industry"]:
            out[t] = prof["industry"]
    print(f"[leadlag] profiles: {len(out)} cached, {len(todo)} to fetch", flush=True)
    for i, t in enumerate(todo):
        sector, industry = None, None
        try:
            info = yf.Ticker(t).get_info()
            sector = info.get("sector")
            industry = info.get("industry")
        except Exception as e:
            logger.warning(f"[leadlag] profile fetch failed for {t}: {e}")
        put_ticker_profile(db_path, t, sector, industry)
        if industry:
            out[t] = industry
        if (i + 1) % 100 == 0:
            print(f"[leadlag] profile fetch {i + 1}/{len(todo)}", flush=True)
        time.sleep(sleep_s)
    return out


def detect_leadlag(prices_by_ticker: dict, bench: pd.DataFrame,
                   industry_by_ticker: dict, caps_by_ticker: dict,
                   start, end,
                   leader_move: float = LEADER_MOVE,
                   laggard_max_move: float = LAGGARD_MAX_MOVE,
                   min_peers: int = MIN_INDUSTRY_PEERS) -> pd.DataFrame:
    """Pure. One laggard event max per (industry, day): the largest-cap
    same-industry peer with |1d abnormal| <= laggard_max_move on a day when
    some peer moved beyond +/-leader_move. Industry-days with leaders in
    BOTH directions are skipped as ambiguous. Returns columns: ticker,
    trigger_date, leader, leader_ret, direction ('up'|'down'), z (leader
    abnormal ret, kept under the name z so cap_per_day/dedupe reuse works)."""
    b_ret1 = bench["close"].pct_change()
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)

    abn = {}
    for t, df in prices_by_ticker.items():
        if df is None or df.empty or t not in industry_by_ticker:
            continue
        idx = df.index.intersection(b_ret1.index)
        if len(idx) < 10:
            continue
        abn[t] = df.loc[idx, "close"].pct_change() - b_ret1.loc[idx]
    if not abn:
        return pd.DataFrame(columns=["ticker", "trigger_date", "leader",
                                     "leader_ret", "direction", "z"])
    abn_df = pd.DataFrame(abn)

    members_by_industry = {}
    for t in abn_df.columns:
        members_by_industry.setdefault(industry_by_ticker[t], []).append(t)

    rows = []
    for day in abn_df.index:
        if not (start_ts <= day <= end_ts):
            continue
        day_ret = abn_df.loc[day].dropna()
        movers = day_ret[day_ret.abs() >= leader_move]
        if movers.empty:
            continue
        for industry, members in members_by_industry.items():
            if len(members) < min_peers:
                continue
            leaders = [t for t in movers.index if industry_by_ticker[t] == industry]
            if not leaders:
                continue
            has_up = any(day_ret[t] >= leader_move for t in leaders)
            has_down = any(day_ret[t] <= -leader_move for t in leaders)
            if has_up and has_down:
                continue  # ambiguous industry-day, pre-registered skip
            direction = "up" if has_up else "down"
            lead = max(leaders, key=lambda t: abs(day_ret[t]))
            laggards = [t for t in members
                        if t in day_ret.index and t not in leaders
                        and abs(day_ret[t]) <= laggard_max_move]
            if not laggards:
                continue
            lag = max(laggards, key=lambda t: caps_by_ticker.get(t, 0))
            rows.append({"ticker": lag, "trigger_date": day, "leader": lead,
                         "leader_ret": float(day_ret[lead]),
                         "direction": direction,
                         "z": float(day_ret[lead])})
    cols = ["ticker", "trigger_date", "leader", "leader_ret", "direction", "z"]
    return pd.DataFrame(rows, columns=cols)


def main():
    ap = argparse.ArgumentParser(description="Intra-industry lead-lag backtest (registry Q7)")
    ap.add_argument("--years", type=int, default=LOOKBACK_YEARS)
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--tag", default="")
    ap.add_argument("--dry-run-only", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    os.makedirs(args.out_dir, exist_ok=True)
    init_db(DB_PATH)

    end = date.today().isoformat()
    start = (date.today() - timedelta(days=365 * args.years)).isoformat()
    fetch_start = (date.today() - timedelta(days=365 * args.years + 30)).isoformat()

    uni = load_universe()
    uni_cik = uni.merge(pd.read_parquet("data/universe.parquet")[["ticker", "cik"]],
                        on="ticker", how="left")
    tickers = sorted(uni_cik["ticker"].unique())
    print(f"[leadlag] universe: {len(tickers)} tickers >= $2.5B", flush=True)

    industry_by_ticker = collect_profiles(tickers)
    n_ind = len(set(industry_by_ticker.values()))
    print(f"[leadlag] {len(industry_by_ticker)} tickers mapped to {n_ind} industries",
          flush=True)

    price_cache = get_history_bulk([*tickers, BENCHMARK], DB_PATH, fetch_start, end)
    bench = price_cache.get(BENCHMARK)
    if bench is None or bench.empty:
        raise RuntimeError("no benchmark history")

    caps_by_ticker = dict(zip(uni["ticker"], uni["market_cap"]))
    candidates = detect_leadlag({t: price_cache.get(t) for t in tickers}, bench,
                                industry_by_ticker, caps_by_ticker, start, end)
    print(f"[leadlag] raw laggard events: "
          f"{candidates['direction'].value_counts().to_dict() if not candidates.empty else {}}",
          flush=True)
    if candidates.empty:
        print("[leadlag] KILLED at dry-run: no events")
        return

    candidates = cap_per_day(candidates)
    parts = []
    for direction, g in candidates.groupby("direction"):
        g = g.assign(file_date=g["trigger_date"])
        parts.append(dedupe_events(g))
    candidates = pd.concat(parts, ignore_index=True)

    candidates["event_date"] = candidates["trigger_date"].dt.date.astype(str).map(
        event_date_from_announcement)
    candidates = attach_cap_proxy(candidates, uni, price_cache)
    candidates = filter_cap_proxy(candidates, MIN_CAP_PROXY)

    # laggard must have no news of its own — newsy laggards are DISCARDED
    # here (unlike idea A there is no news control; the control is the
    # leader's direction)
    earnings = load_earnings_for_tickers(set(candidates["ticker"]), DB_PATH)
    filing_dates = load_filing_dates(candidates, uni_cik, 365 * args.years + 30)
    clean_earn, _ = split_by_earnings(candidates, earnings)
    clean, _ = split_by_news(clean_earn, filing_dates,
                             days_before=NEWS_DAYS_BEFORE, days_after=NEWS_DAYS_AFTER)

    clean = clean.assign(
        category=clean["direction"].map({"up": CATEGORY_SIGNAL,
                                         "down": CATEGORY_CONTROL}))
    n_signal = len(clean[clean["category"] == CATEGORY_SIGNAL])
    years = max(args.years, 1)
    print(f"[leadlag] dry-run signal count ({CATEGORY_SIGNAL}): {n_signal} "
          f"({n_signal / years:.1f}/yr); control "
          f"{CATEGORY_CONTROL}={len(clean) - n_signal}", flush=True)

    if n_signal < 10 * years or n_signal < 50:
        print(f"[leadlag] KILLED at dry-run: signal count too thin "
              f"(need >=10/yr and >=50 total, got {n_signal} over {years}y)")
        return
    if args.dry_run_only:
        print("[leadlag] dry-run-only requested, stopping before price gate")
        return

    events = compute_abnormal_returns(clean, DB_PATH, horizons=HORIZONS,
                                      benchmark=BENCHMARK)
    stem = date.today().isoformat() + (f"_{args.tag}" if args.tag else "")
    events.to_csv(os.path.join(args.out_dir, f"leadlag_events_{stem}.csv"), index=False)

    summary = summarize(events, horizons=HORIZONS, gate_horizons=GATE_HORIZONS)
    summary.to_csv(os.path.join(args.out_dir, f"leadlag_summary_{stem}.csv"), index=False)
    print(summary.to_string(index=False))

    halves = split_half_summary(events.rename(columns={"trigger_date": "filing_date"}))
    halves.to_csv(os.path.join(args.out_dir, f"leadlag_halves_{stem}.csv"), index=False)
    print(halves.to_string(index=False))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_leadlag_backtest.py -q`
Expected: all PASS

- [ ] **Step 5: Full suite green, then commit**

```bash
python3 -m pytest tests/ -q
git add src/leadlag_backtest.py tests/test_leadlag_backtest.py
git commit -m "feat(leadlag): intra-industry lead-lag backtest (idea C, registry Q7)"
```

- [ ] **Step 6: Dry run**

Run: `python3 -m src.leadlag_backtest --dry-run-only 2>&1 | tee output/leadlag_dryrun_$(date +%F).log`

First run pays the one-time profile fetch (~2,269 tickers x 0.3s = ~12 min plus yfinance latency; checkpointed, safe to interrupt and re-run). Prices are already cached from idea A.

- [ ] **Step 7: Full gate run (only if dry-run passed)**

Run: `python3 -m src.leadlag_backtest --tag full 2>&1 | tee output/leadlag_run_$(date +%F).log`

Hand-check (PASS requirement #4): read 10 random `leadlag_up` rows; verify the leader really moved that day (spot-check against yfinance) and the leader/laggard really share an industry that makes economic sense.

- [ ] **Step 8: Record the result**

Same procedure as Task 2 Step 8: ledger row with real numbers, Q7 status update, commit.

---

## Final wrap-up (after all three resolve)

- [ ] Confirm full suite green: `python3 -m pytest tests/ -q`
- [ ] Confirm all registry updates are committed and each idea's Status line matches its actual outcome.
- [ ] Write a short summary to the user: per idea, where it died (dry-run vs gate vs data-integrity) or PASSed, with the key numbers. If anything PASSed, restate that human review + explicit sign-off (PASS requirements #4/#5) are still pending and NOTHING gets built on top of it (no watcher, no alerts, no UI) until the user says so.

## Self-review notes (already checked)

- Gate values, dry-run bars, and all thresholds appear in Task 0's registry text AND as constants in the modules; they match (1.5%/p<0.10/sign-agree at h5/h20 via `summarize` defaults; 10/yr + 50 total in each `main`).
- `dedupe_events` needs a `file_date` column: both price-triggered modules assign `file_date=trigger_date` before calling it.
- `split_half_summary` needs a datetime `filing_date` column: reversal/leadlag rename `trigger_date`, spdel renames `effective_date`.
- `summarize` returns stats=None for categories with n < 5 at a horizon; expected for thin controls, not a bug.
- yfinance price frames are auto-adjusted with lowercase `close`; `detect_dislocations`/`detect_leadlag` are written against that shape and tested with synthetic frames of the same shape.
- Known residual biases carried over (disclosed in Task 0 registry text): current-shares assumption inside cap_proxy; names delisted before 2026-07 absent from the universe entirely.
