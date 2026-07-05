# Idea Bank 3 Backtests (Q9 run, Q10 insider clusters, Q11 13D) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the three pre-registered idea-bank-3 backtests (PEAD, insider clusters, Schedule 13D — all in the $500M-$2.5B band) against the v2 cost-aware gate, one at a time, writing one ledger row each.

**Architecture:** Each idea is one module following the established bank-2 pattern (detect → dry-run kill → abnormal returns → net-of-cost gate → control + split-half). Q9's module (`src/pead_small_backtest.py`) already exists, is tested, committed, and its dry-run PASSED — Task 1 just runs and interprets it. Q10/Q11 are new thin modules that reuse `src/insider_backtest.py` / `src/sc13d_backtest.py` detection plus the shared band/cost helpers in `src/backtest_recipe.py`.

**Tech Stack:** Python 3, pandas, scipy, yfinance, sqlite cache (`data/cache.db`), pytest. $0 data cost, no Claude API calls anywhere in this plan.

---

## MANDATORY CONTEXT — read before Task 1

### Governing documents (read these files first, in this order)
1. `docs/handoffs/FIT-2026-07-03-v2.md` — the constraint set every decision traces to.
2. `docs/strategy-registry.md` — sections "PASS confirmation requirements", "IDEA BANK 3", Q9/Q10/Q11 entries, and the Test ledger table (rows #1-12 show the exact FAIL-row prose style to imitate).
3. `src/pead_small_backtest.py` — the reference implementation for the v2 gate mechanics.

### The v2 gate, spelled out (identical for all three ideas)
A signal category PASSes only if ALL of the following hold. Anything less is a FAIL.

1. **Net magnitude + significance + sign legs**, read from the NET summary
   table (the one titled "NET of cost model — THE GATE"): at some horizon h
   in that idea's declared gate horizons, `h{h}_mean_abn_ret >= 0.01` AND
   `h{h}_pvalue < 0.10` AND `h{h}_sign_agree == True` AND the mean is
   POSITIVE. **Warning (this has bitten before, see registry gotchas): the
   `gate` column printed by `summarize()` uses `abs(mean)` — a strongly
   NEGATIVE category can print `gate=PASS`. Never copy the printed gate
   column into a verdict without checking the sign of the mean. A negative
   mean is always a FAIL for a long-only book.**
2. **Control behaves**: the control category must NOT show a significant
   positive drift of similar magnitude at the same horizon. If signal and
   control drift together, the effect is a universe artifact → FAIL
   (this exact check killed ledger #2 and #12).
3. **Split-half stability**: in the halves table, the passing horizon must
   show the same SIGN in both halves, and each half's mean must be at
   least roughly half the full-sample mean (registry: "effect sign + rough
   magnitude agree"). A sign flip between halves is an automatic FAIL no
   matter how good the full-sample p-value is (killed ledger #5).
4. **Idea-specific discriminator** (stated per task below), e.g. Q9's
   beat-vs-miss spread.

If a signal PASSes all four: **STOP THE PLAN.** Do not start the next idea,
do not build any watcher/alert/UI. Write the ledger row as PASS, note
"pending hand-read + user sign-off" (PASS requirements #4 and #5 are human
steps), commit, and end the session reporting the pass. Registry rule #6.

### The cost model (already implemented, do not re-derive)
`src/backtest_recipe.py` → `COST_BANDS`, `band_cost()`, `attach_net_returns()`.
Round-trip cost by `cap_proxy`: $500M-$1B → 0.40%; $1B-$2.5B → 0.25%;
≥$2.5B → 0.15%. `abn_ret_net_{h}d = abn_ret_{h}d - band_cost(cap_proxy)`.

### Shared mechanics already in place (reuse, never reimplement)
- `src/backtest_recipe.py`: `attach_cap_proxy` (event-time cap proxy),
  `filter_cap_proxy(min_cap=5e8, max_cap=2.5e9)` (band membership),
  `attach_dollar_vol` / `filter_dollar_vol` ($2M pre-event 20d median
  dollar-volume liquidity leg), `attach_net_returns`, `band_cost`,
  `load_earnings_for_tickers`, `drop_earnings_contamination`.
- `src/event_backtest.py`: `compute_abnormal_returns` (SPY benchmark,
  forward returns from first close on/after event_date), `summarize`
  (winsorized mean + t-test + sign-agreement per category×horizon),
  `get_history_bulk` (batched, cached price fetch).
- `src/pead_small_backtest.py`: `_as_net_frame` (net-basis frame for
  gating), `split_half` (net-basis halves), `beat_vs_miss_spread`.
- Entry-timing convention everywhere: `event_date = disclosure date + 1
  calendar day`, so the harness enters at the first close strictly AFTER
  the signal is public. Never change this.

### Hard process rules (from the registry, non-negotiable)
- One idea at a time, strictly in order Q9 → Q10 → Q11.
- NEVER tune a threshold, horizon, window, phrase, or filter after seeing
  results. If a run surprises you (data shape, column names, API shape),
  you may fix ONLY the mechanical data-access bug, in its own commit, and
  say so in the ledger row (precedent: commit `e8651b4`).
- Every gate run gets exactly one ledger row, PASS or FAIL, in
  `docs/strategy-registry.md`'s Test ledger, written in the same prose
  style as rows #10-12 (numbers for every gated horizon, control behavior,
  split-half verdict, explicit "Qn KILLED" or "Qn PASSED pending review").
- yfinance 401s/"Invalid Crumb"/"Too Many Requests"/"no earnings dates"
  log lines are normal noise, not bugs. The NaN filters tolerate them.
- Do not commit anything in `output/` (gitignored by convention).
- Runtime expectations: Q9 full run ~10-25 min (mostly yfinance price
  backfill for ~1,200 newly-in-universe tickers; run it in the background
  and poll the log). Q10 ~10-15 min (zip parsing + targeted earnings
  fetch). Q11 ~15-40 min (EDGAR FTS pagination; 13G control fetch is the
  slow part).

### State at plan-writing time (verify in Task 0)
- Branch `main`, clean except pre-existing untracked files. Last commits:
  `8741699` (Q9 module), `e58a9d4` (Q8 desk-check kill), `978a218`
  (v2 fit + bank 3 pre-registration).
- Test suite: 267 passed.
- Q9 dry-run already PASSED (2026-07-03 22:52, `output/pead_small_dryrun_2026-07-03.log`):
  30,526 events pre-band; in-band-now approximation `pead_beat_large`
  N=3849, `pead_beat_mid` N=1705, `pead_beat_small` N=985, `pead_miss`
  N=3735 — pooled signal ~2,180/yr vs kill bar of 10/yr-and-50-total.
  82/1178 earnings fetch errors (foreign ADR/CEF noise, expected).
- Earnings checkpoint table `pead_earnings` has 60,890 rows covering the
  full $500M+ universe (fetched 2026-07-03, 7-day TTL in `pead_fetch_log`).
  **If more than 7 days have passed since 2026-07-03, re-run the fetch
  (drop `--skip-fetch` in Task 1) — otherwise `--skip-fetch` is correct
  and saves ~25 minutes.**
- Insider Form 4 quarterly zips on disk: `data/insider/` has 2024q3-2026q1;
  `data/insider/confirm/` has 2020q3-2024q2. No download needed for Q10.

---

### Task 0: Preflight verification

**Files:** none created or modified.

- [ ] **Step 1: Verify git state and test suite**

Run:
```bash
cd /Users/arhanbarve/Code/screener
git log --oneline -4
python3 -m pytest tests/ -q | tail -1
```
Expected: log shows `8741699 feat(pead-small): ...` at or near HEAD;
pytest reports `267 passed` (or more, never fewer).

- [ ] **Step 2: Verify the Q9 dry-run artifact exists and passed**

Run:
```bash
tail -8 output/pead_small_dryrun_2026-07-03.log
```
Expected: the four-category count table shown in "State at plan-writing
time" above. If this file is missing, re-create it first:
`python3 -m src.pead_small_backtest --dry-run-only --skip-fetch` (~2 min)
and confirm the counts are within ~10% of the table above.

- [ ] **Step 3: Verify earnings checkpoint freshness**

Run:
```bash
python3 -c "
import sqlite3
con = sqlite3.connect('data/cache.db')
print(con.execute('SELECT COUNT(*), MAX(fetched_at) FROM pead_fetch_log').fetchone())"
```
Expected: count ≥ 3400 and a max timestamp within the last 7 days. If
stale, note it — Task 1 must then run WITHOUT `--skip-fetch`.

---

### Task 1: Q9 full gate run + ledger row

Everything is already built. This task runs it, interprets strictly per
the gate definition above, and records the verdict.

**Files:**
- Modify: `docs/strategy-registry.md` (one ledger row + Q9 status line)
- Created by the run (not committed): `output/pead_small_events_<date>.csv`,
  `output/pead_small_summary_<date>.csv`, `output/pead_small_spread_<date>.csv`,
  `output/pead_small_halves_<date>.csv`, `output/pead_small_run_<date>.log`

- [ ] **Step 1: Launch the full run in the background**

Run:
```bash
nohup python3 -m src.pead_small_backtest --skip-fetch \
  > output/pead_small_run_$(date +%F).log 2>&1 &
echo $!
```
(Drop `--skip-fetch` only if Task 0 Step 3 found the checkpoint stale.)

- [ ] **Step 2: Poll until complete (~10-25 min)**

Run every few minutes:
```bash
tail -5 output/pead_small_run_$(date +%F).log
```
Progress markers in order: `[pead_small] universe: 3447 tickers ...` →
`[pead_small] 30526 events in last 1095d (pre-band)` → yfinance batch
noise → `[pead_small] band filter 30526 -> N1; dollar-vol filter -> N2` →
`exact post-filter counts: signal=...` → four printed tables (GROSS, NET,
spread, split-half). The run is done when the split-half table has printed.
Sanity expectations (not gates): N1 roughly 8,000-16,000 (band cut), N2
within ~20% of N1 ($2M ADV cuts little in this band).

- [ ] **Step 3: Confirm the exact dry-run leg on true counts**

From the log line `exact post-filter counts: signal=S (R/yr), miss=M`:
require S ≥ 50 AND R ≥ 10. (Near-certain given the approximate counts; if
it somehow fails, that IS the verdict — write a "KILLED at dry-run
(post-band counts)" ledger row per the ledger #7 style and stop Q9 here.)

- [ ] **Step 4: Apply the gate, category by category**

Work from the NET summary table for the three signal categories
(`pead_beat_large`, `pead_beat_mid`, `pead_beat_small`), gate horizons
(5, 20, 40) — all three declared for Q9. For each category × horizon
apply gate leg 1 (mean ≥ 0.01, p < 0.10, sign_agree, mean POSITIVE —
re-read the sign warning in MANDATORY CONTEXT). Then:
- Leg 2 (control): `pead_miss` must not drift positively/significantly at
  the same horizon.
- Leg 4 (discriminator): in the spread table, the same horizon must show
  `spread > 0` and `p_spread < 0.10`. No significant spread → FAIL even
  if a beat bucket clears leg 1 (that is precisely how ledger #2 died).
- Leg 3 (split-half): same-sign, roughly-half-magnitude in both halves
  for the passing category × horizon, read from the halves table.

- [ ] **Step 5: Write the ledger row and Q9 status line**

Append row #13 to the Test ledger table in `docs/strategy-registry.md`,
imitating rows #10-12: date, "PEAD $500M-$2.5B (N=<S> signal, N=<M>
control)", categories gated = 4, Result PASS/FAIL, and a Notes cell that
reports (a) each gated horizon's net mean/p for the best signal bucket,
(b) miss-bucket behavior, (c) spread + p_spread at each horizon,
(d) split-half verdict, (e) closing verdict sentence "Q9 KILLED." or
"Q9 PASSED pending hand-read + user sign-off." Update the Q9 queue
section's `Status:` line to match.

- [ ] **Step 6: Commit**

```bash
git add docs/strategy-registry.md
git commit -m "docs(registry): ledger row for pead-small backtest (Q9) — <PASS|FAIL>"
```

- [ ] **Step 7: Branch on outcome**

FAIL → proceed to Task 2. PASS → STOP THE PLAN (see MANDATORY CONTEXT);
report the pass, the numbers, and the pending human steps, and end.

---

### Task 2: Q10 module — insider clusters, $500M-$2.5B (TDD)

**Files:**
- Create: `src/insider_small_backtest.py`
- Create: `tests/test_insider_small_backtest.py`
- Modify: `src/backtest_recipe.py` (add shared `as_net_frame`)
- Modify: `src/pead_small_backtest.py` (delegate `_as_net_frame` to it)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_insider_small_backtest.py`:

```python
import pandas as pd

from src.backtest_recipe import as_net_frame
from src.insider_small_backtest import (
    GATE_HORIZONS,
    HORIZONS,
    LOOKBACK_DAYS,
    filter_lookback,
)


def test_gate_horizons_are_preregistered():
    # Registry Q10: h20/h40 declared (front-loaded ~1 month), h5 report-only.
    assert HORIZONS == (5, 20, 40)
    assert GATE_HORIZONS == (20, 40)


def test_filter_lookback_drops_old_events():
    events = pd.DataFrame({
        "filing_date": pd.to_datetime(["2020-01-02", "2026-01-02"]),
        "ticker": ["OLD", "NEW"],
    })
    out = filter_lookback(events, today=pd.Timestamp("2026-07-04"),
                          lookback_days=LOOKBACK_DAYS)
    assert list(out["ticker"]) == ["NEW"]


def test_as_net_frame_swaps_columns_shared_helper():
    row = {"cap_proxy": 6e8}
    for h in (5, 20, 40):
        row[f"abn_ret_{h}d"] = 0.03
        row[f"abn_ret_net_{h}d"] = 0.01
    net = as_net_frame(pd.DataFrame([row]), horizons=(5, 20, 40))
    for h in (5, 20, 40):
        assert net[f"abn_ret_{h}d"].iloc[0] == 0.01
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_insider_small_backtest.py -q`
Expected: FAIL/ERROR with `ModuleNotFoundError` / `ImportError`
(`insider_small_backtest` and `as_net_frame` don't exist yet).

- [ ] **Step 3: Add `as_net_frame` to the shared recipe**

Append to `src/backtest_recipe.py` (after `attach_net_returns`):

```python
def as_net_frame(events: pd.DataFrame, horizons: tuple) -> pd.DataFrame:
    """Frame whose abn_ret_{h}d columns hold NET values, so summarize()
    gates on net-of-cost returns without modification."""
    net = events.copy()
    for h in horizons:
        net[f"abn_ret_{h}d"] = net[f"abn_ret_net_{h}d"]
    return net
```

Then in `src/pead_small_backtest.py`, replace the body of `_as_net_frame`
with a delegation (keep the name — its tests and `split_half` call it):

```python
def _as_net_frame(events: pd.DataFrame, horizons: tuple = HORIZONS) -> pd.DataFrame:
    """Frame whose abn_ret_{h}d columns hold NET values, so summarize()
    gates on net without modification."""
    return as_net_frame(events, horizons)
```

and add `as_net_frame` to the `from src.backtest_recipe import (...)` list
in that file.

- [ ] **Step 4: Write the Q10 module**

Create `src/insider_small_backtest.py`:

```python
"""Insider cluster-buying backtest, $500M-$2.5B band — registry Q10, IDEA BANK 3.

Registry: docs/strategy-registry.md Q10. Fit: docs/handoffs/FIT-2026-07-03-v2.md.

Q1 (same detection, $2.5B+ universe) was killed by an out-of-sample
confirmation FAIL (ledger #3-4). The published cluster-buying effect
concentrates in small caps; the v2 band change is a universe change, not a
same-universe variant. Detection, cluster window, dedupe and 10b5-1
exclusion are IDENTICAL to src/insider_backtest.py — only the universe,
band/liquidity filters, horizons and net-of-cost gate differ, all
pre-registered in the registry Q10 section before this file was written.

Data: SEC Form 345 quarterly zips already on disk (data/insider/ has
2024q3-2026q1, data/insider/confirm/ has 2020q3-2024q2). The Q10 run reads
a dedicated directory (data/insider_q10/) holding copies of the quarters
inside the 3-year lookback, so Q1's directories stay untouched.
"""

import argparse
import logging
import os
from datetime import date

import pandas as pd

from src.backtest_recipe import (
    as_net_frame,
    attach_cap_proxy,
    attach_dollar_vol,
    attach_net_returns,
    drop_earnings_contamination,
    filter_cap_proxy,
    filter_dollar_vol,
    load_earnings_for_tickers,
)
from src.cache import init_db
from src.event_backtest import compute_abnormal_returns, get_history_bulk, summarize
from src.insider_backtest import build_events, load_all_quarters
from src.pead_backtest import load_universe

logger = logging.getLogger(__name__)

DB_PATH = "data/cache.db"
DATA_DIR = "data/insider_q10"
BENCHMARK = "SPY"
HORIZONS = (5, 20, 40)
GATE_HORIZONS = (20, 40)      # pre-declared: cluster effect front-loaded ~1 month
LOOKBACK_DAYS = 1095          # 3 years

# Pre-registered in docs/strategy-registry.md Q10 — do not tune after runs.
MIN_CAP = 5e8
MAX_CAP = 2.5e9
MIN_DOLLAR_VOL = 2e6
NET_GATE_MIN = 0.01

SIGNAL_CATEGORY = "insider_cluster_buy"
CONTROL_CATEGORIES = ("insider_cluster_sell", "insider_single_buy")


def filter_lookback(events: pd.DataFrame, today: pd.Timestamp | None = None,
                    lookback_days: int = LOOKBACK_DAYS) -> pd.DataFrame:
    """Quarterly zips cover whole quarters; enforce the exact 3y window."""
    if events.empty:
        return events
    today = today or pd.Timestamp(date.today())
    cutoff = today - pd.Timedelta(days=lookback_days)
    return events[events["filing_date"] >= cutoff].reset_index(drop=True)


def split_half(events: pd.DataFrame) -> pd.DataFrame:
    """PASS requirement #3, computed on the NET frame (gate basis)."""
    mid = events["filing_date"].quantile(0.5)
    frames = []
    for name, half in (("first_half", events[events["filing_date"] <= mid]),
                       ("second_half", events[events["filing_date"] > mid])):
        s = summarize(as_net_frame(half, HORIZONS), horizons=HORIZONS,
                      gate_horizons=GATE_HORIZONS, min_abs_return=NET_GATE_MIN)
        if not s.empty:
            s.insert(0, "half", name)
            frames.append(s)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser(description="Insider cluster $500M-$2.5B backtest (registry Q10)")
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--dry-run-only", action="store_true",
                    help="stop after post-contamination counts (band approximated by CURRENT cap)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    os.makedirs(args.out_dir, exist_ok=True)
    init_db(DB_PATH)

    # Universe: everything currently >= $500M; cap_proxy decides band per event.
    uni = load_universe(min_cap=MIN_CAP)
    in_band_now = uni[uni["market_cap"] < MAX_CAP]
    print(f"[insider_small] universe: {len(uni)} tickers >= ${MIN_CAP/1e9:.1f}B "
          f"({len(in_band_now)} currently in band)", flush=True)

    records = load_all_quarters(args.data_dir)
    events = build_events(records, set(uni["ticker"]))
    events = filter_lookback(events)
    print(f"[insider_small] {len(events)} events in lookback "
          f"({events['category'].value_counts().to_dict() if not events.empty else {}})", flush=True)
    if events.empty:
        print("[insider_small] no events — nothing to summarize")
        return

    # Earnings contamination folded into the dry-run count (ledger #7 fix).
    earnings = load_earnings_for_tickers(set(events["ticker"]), DB_PATH)
    events = drop_earnings_contamination(events, earnings)
    print(f"[insider_small] post-contamination: {len(events)} "
          f"({events['category'].value_counts().to_dict()})", flush=True)

    if args.dry_run_only:
        approx = events[events["ticker"].isin(set(in_band_now["ticker"]))]
        span_yrs = max((approx["filing_date"].max()
                        - approx["filing_date"].min()).days / 365.25, 1e-9)
        counts = approx.groupby("category").size().rename("n").reset_index()
        counts["per_year"] = (counts["n"] / span_yrs).round(1)
        print(counts.to_string(index=False))
        print("[insider_small] dry-run: band approximated by CURRENT cap (true "
              "gate uses cap_proxy at event; exact counts printed in the full "
              "run). Kill bar: signal < 10/yr or < 50 total.", flush=True)
        return

    start = (events["filing_date"].min() - pd.Timedelta(days=60)).date().isoformat()
    end = (events["filing_date"].max() + pd.Timedelta(days=120)).date().isoformat()
    tickers = list(events["ticker"].unique())
    price_cache = get_history_bulk([*tickers, BENCHMARK], DB_PATH, start, end)

    n0 = len(events)
    events = attach_cap_proxy(events, uni, price_cache)
    events = filter_cap_proxy(events, min_cap=MIN_CAP, max_cap=MAX_CAP)
    n1 = len(events)
    events = attach_dollar_vol(events, price_cache)
    events = filter_dollar_vol(events, min_dollar_vol=MIN_DOLLAR_VOL)
    n2 = len(events)
    print(f"[insider_small] band filter {n0} -> {n1}; dollar-vol filter -> {n2}", flush=True)
    if events.empty:
        print("[insider_small] no in-band events — nothing to summarize")
        return

    span_yrs = max((events["filing_date"].max()
                    - events["filing_date"].min()).days / 365.25, 1e-9)
    n_signal = int((events["category"] == SIGNAL_CATEGORY).sum())
    print(f"[insider_small] exact post-filter counts: signal={n_signal} "
          f"({n_signal/span_yrs:.1f}/yr)", flush=True)

    events = compute_abnormal_returns(events, DB_PATH, horizons=HORIZONS,
                                      benchmark=BENCHMARK)
    events = attach_net_returns(events, HORIZONS)

    today = date.today().isoformat()
    events.to_csv(os.path.join(args.out_dir, f"insider_small_events_{today}.csv"), index=False)

    print("\n=== GROSS abnormal returns (comparability with bank 1-2) ===")
    gross = summarize(events, horizons=HORIZONS, gate_horizons=GATE_HORIZONS)
    print(gross.to_string(index=False))

    print(f"\n=== NET of cost model — THE GATE (>= {NET_GATE_MIN:.1%} net) ===")
    net = summarize(as_net_frame(events, HORIZONS), horizons=HORIZONS,
                    gate_horizons=GATE_HORIZONS, min_abs_return=NET_GATE_MIN)
    print(net.to_string(index=False))
    net.to_csv(os.path.join(args.out_dir, f"insider_small_summary_{today}.csv"), index=False)

    print("\n=== Split-half stability (net basis) ===")
    halves = split_half(events)
    print(halves.to_string(index=False))
    halves.to_csv(os.path.join(args.out_dir, f"insider_small_halves_{today}.csv"), index=False)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_insider_small_backtest.py tests/test_pead_small_backtest.py tests/test_insider_backtest.py -q`
Expected: all PASS (the pead_small tests confirm the `_as_net_frame`
delegation didn't change behavior; the insider tests confirm the reused
detection is untouched).

- [ ] **Step 6: Run the full suite**

Run: `python3 -m pytest tests/ -q | tail -1`
Expected: `270 passed` (267 + 3 new), zero failures.

- [ ] **Step 7: Commit**

```bash
git add src/insider_small_backtest.py tests/test_insider_small_backtest.py \
        src/backtest_recipe.py src/pead_small_backtest.py
git commit -m "feat(insider-small): insider cluster \$500M-\$2.5B backtest (idea bank 3, registry Q10)"
```

---

### Task 3: Q10 data prep, dry-run, full run + ledger row

**Files:**
- Create: `data/insider_q10/` (copies of 12 quarterly zips; data dir, not committed)
- Modify: `docs/strategy-registry.md` (ledger row + Q10 status line)

- [ ] **Step 1: Assemble the Q10 quarter directory (3-year lookback)**

Lookback from 2026-07 reaches back to 2023-07, i.e. quarters 2023q3
through the newest on disk (2026q1 — 2026q2's zip isn't published yet;
the ~1 missing quarter is acceptable and must be disclosed in the ledger
row's Notes).

```bash
mkdir -p data/insider_q10
cp data/insider/confirm/2023q3_form345.zip data/insider/confirm/2023q4_form345.zip \
   data/insider/confirm/2024q1_form345.zip data/insider/confirm/2024q2_form345.zip \
   data/insider_q10/
cp data/insider/2024q3_form345.zip data/insider/2024q4_form345.zip \
   data/insider/2025q1_form345.zip data/insider/2025q2_form345.zip \
   data/insider/2025q3_form345.zip data/insider/2025q4_form345.zip \
   data/insider/2026q1_form345.zip data/insider_q10/
ls data/insider_q10/ | wc -l
```
Expected: `11`.

- [ ] **Step 2: Dry-run**

```bash
python3 -m src.insider_small_backtest --dry-run-only 2>&1 | tee output/insider_small_dryrun_$(date +%F).log
```
Expected shape: universe line (3447 / 1178 as in Q9), an events line with
all three categories present, a post-contamination line, then the
approximate per-category count table. Apply the kill bar to
`insider_cluster_buy`: < 10/yr or < 50 total → write a "KILLED at
dry-run" ledger row (ledger #7/Q3.5 style, no Test-ledger slot consumed
if no gate was run — follow the Q3.5 precedent of recording it in the Q10
queue section instead) and SKIP to Task 4. Otherwise continue.

- [ ] **Step 3: Full run in the background**

```bash
nohup python3 -m src.insider_small_backtest \
  > output/insider_small_run_$(date +%F).log 2>&1 &
echo $!
```
Poll with `tail -5` as in Task 1 Step 2. Done when the split-half table
prints. Confirm the exact post-filter signal count still clears the bar
(≥ 50 total, ≥ 10/yr); if not, that IS the verdict (dry-run-style kill,
ledger per Task 1 Step 3's parenthetical).

- [ ] **Step 4: Apply the gate**

Gate horizons for Q10 are **(20, 40) only** (h5 is report-only). Apply
the four legs from MANDATORY CONTEXT with:
- Signal: `insider_cluster_buy` on the NET table.
- Leg 2 (control): `insider_cluster_sell` must not drift positively in
  step with the signal; `insider_single_buy` should be materially weaker
  than the cluster signal (published mechanism: clusters ≫ singles). If
  singles ≈ clusters, note it as evidence against the cluster mechanism.
- Leg 4 (discriminator): cluster-vs-single separation is Q10's
  discriminator; state the two means side by side in the ledger Notes.
- Leg 3: split-half per the standard rule.

- [ ] **Step 5: Ledger row + status line + commit**

Append row #14 to the Test ledger (style of rows #10-12; include the
missing-2026q2 disclosure), update Q10's `Status:` line, then:

```bash
git add docs/strategy-registry.md
git commit -m "docs(registry): ledger row for insider-small backtest (Q10) — <PASS|FAIL>"
```

- [ ] **Step 6: Branch on outcome**

FAIL → Task 4. PASS → STOP THE PLAN (MANDATORY CONTEXT rules).

---

### Task 4: Q11 module — Schedule 13D, $500M-$2.5B (TDD)

**Files:**
- Create: `src/sc13d_small_backtest.py`
- Create: `tests/test_sc13d_small_backtest.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_sc13d_small_backtest.py`:

```python
import pandas as pd

from src.sc13d_small_backtest import (
    GATE_HORIZONS,
    HORIZONS,
    control_cap_for,
    split_half,
)


def test_gate_horizons_are_preregistered():
    # Registry Q11: h5/h20/h40 all declared (activist drift is weeks-scale).
    assert HORIZONS == (5, 20, 40)
    assert GATE_HORIZONS == (5, 20, 40)


def test_control_cap_is_three_x_signal():
    assert control_cap_for(100) == 300


def test_split_half_handles_empty():
    assert split_half(pd.DataFrame(columns=["file_date"])).empty
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_sc13d_small_backtest.py -q`
Expected: ERROR with `ModuleNotFoundError: No module named 'src.sc13d_small_backtest'`.

- [ ] **Step 3: Write the Q11 module**

Create `src/sc13d_small_backtest.py`:

```python
"""Schedule 13D activist-stake backtest, $500M-$2.5B band — registry Q11, IDEA BANK 3.

Registry: docs/strategy-registry.md Q11. Fit: docs/handoffs/FIT-2026-07-03-v2.md.

Q2 (same detection, $2.5B+ universe) was a clean gate FAIL with a
split-half sign flip (ledger #5). Activists concentrate in $500M-$10B and
the published announcement drift is strongest in small/mid caps — the v2
band is the favorable slice the $2.5B floor excluded. Detection (EDGAR FTS,
initial filings only), 13G control design, control cap (3x signal, sampled
across 12 months), dedupe and contamination handling are IDENTICAL to
src/sc13d_backtest.py — only universe, band/liquidity filters, horizons
and the net-of-cost gate differ, all pre-registered in registry Q11.
"""

import argparse
import logging
import os
from datetime import date, timedelta

import pandas as pd

from src.backtest_recipe import (
    as_net_frame,
    attach_cap_proxy,
    attach_dollar_vol,
    attach_net_returns,
    filter_cap_proxy,
    filter_dollar_vol,
)
from src.cache import init_db
from src.event_backtest import compute_abnormal_returns, get_history_bulk, summarize
from src.pead_backtest import load_universe
from src.sc13d_backtest import (
    CONTROL_CATEGORY,
    CONTROL_FORM_EXACT,
    CONTROL_FORM_FILTER,
    CONTROL_PHRASE,
    CONTROL_SAMPLE_MONTHS,
    SIGNAL_CATEGORY,
    SIGNAL_FORM_EXACT,
    SIGNAL_FORM_FILTER,
    SIGNAL_PHRASE,
    _load_earnings_for_tickers,
    build_events,
    fetch_raw_events,
    filter_to_universe,
)

logger = logging.getLogger(__name__)

DB_PATH = "data/cache.db"
BENCHMARK = "SPY"
HORIZONS = (5, 20, 40)
GATE_HORIZONS = (5, 20, 40)   # pre-declared: activist drift is weeks-scale
LOOKBACK_YEARS = 3
CONTROL_CAP_MULTIPLE = 3

# Pre-registered in docs/strategy-registry.md Q11 — do not tune after runs.
MIN_CAP = 5e8
MAX_CAP = 2.5e9
MIN_DOLLAR_VOL = 2e6
NET_GATE_MIN = 0.01


def control_cap_for(n_signal: int) -> int:
    return n_signal * CONTROL_CAP_MULTIPLE


def split_half(events: pd.DataFrame) -> pd.DataFrame:
    """PASS requirement #3, computed on the NET frame (gate basis)."""
    if events.empty:
        return pd.DataFrame()
    mid = events["file_date"].quantile(0.5)
    frames = []
    for name, half in (("first_half", events[events["file_date"] <= mid]),
                       ("second_half", events[events["file_date"] > mid])):
        s = summarize(as_net_frame(half, HORIZONS), horizons=HORIZONS,
                      gate_horizons=GATE_HORIZONS, min_abs_return=NET_GATE_MIN)
        if not s.empty:
            s.insert(0, "half", name)
            frames.append(s)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser(description="Schedule 13D $500M-$2.5B backtest (registry Q11)")
    ap.add_argument("--years", type=int, default=LOOKBACK_YEARS)
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--dry-run-only", action="store_true",
                    help="only count in-universe signal events, do not fetch control/prices")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    os.makedirs(args.out_dir, exist_ok=True)
    init_db(DB_PATH)

    end = date.today().isoformat()
    start = (date.today() - timedelta(days=365 * args.years)).isoformat()

    # Universe: everything currently >= $500M; cap_proxy decides band per event.
    uni = load_universe(min_cap=MIN_CAP)
    in_band_now = uni[uni["market_cap"] < MAX_CAP]
    uni_cik = uni.merge(pd.read_parquet("data/universe.parquet")[["ticker", "cik"]],
                        on="ticker", how="left")
    print(f"[sc13d_small] universe: {len(uni_cik)} tickers >= ${MIN_CAP/1e9:.1f}B "
          f"({len(in_band_now)} currently in band)", flush=True)

    print("[sc13d_small] fetching Schedule 13D hits from EDGAR FTS...", flush=True)
    raw_signal = fetch_raw_events(SIGNAL_PHRASE, SIGNAL_FORM_FILTER, SIGNAL_FORM_EXACT,
                                  start, end, SIGNAL_CATEGORY)
    sig_uni = filter_to_universe(raw_signal, uni_cik)
    sig_band_now = sig_uni[sig_uni["ticker"].isin(set(in_band_now["ticker"]))]
    years = max(args.years, 1)
    n_signal = len(sig_band_now)
    print(f"[sc13d_small] dry-run signal count (band approximated by CURRENT "
          f"cap): {n_signal} ({n_signal / years:.1f}/yr); "
          f"{len(sig_uni)} across the whole >=$500M scan", flush=True)

    if n_signal < 10 * years or n_signal < 50:
        print(f"[sc13d_small] KILLED at dry-run: signal count too thin "
              f"(need >=10/yr and >=50 total, got {n_signal} over {years}y)")
        return

    if args.dry_run_only:
        print("[sc13d_small] dry-run-only requested, stopping before control/price fetch")
        return

    control_cap = control_cap_for(len(sig_uni))
    print(f"[sc13d_small] fetching Schedule 13G hits (capped ~{control_cap} raw "
          f"hits across {CONTROL_SAMPLE_MONTHS} sampled months)...", flush=True)
    raw_control = fetch_raw_events(CONTROL_PHRASE, CONTROL_FORM_FILTER, CONTROL_FORM_EXACT,
                                   start, end, CONTROL_CATEGORY,
                                   max_raw_hits=control_cap,
                                   sample_months=CONTROL_SAMPLE_MONTHS)
    ctl_uni = filter_to_universe(raw_control, uni_cik)
    print(f"[sc13d_small] control count: {len(ctl_uni)} in >=$500M scan", flush=True)

    all_tickers = set(sig_uni["ticker"]) | set(ctl_uni["ticker"])
    earnings = _load_earnings_for_tickers(all_tickers)

    events_signal = build_events(raw_signal, uni_cik, earnings)
    events_control = build_events(raw_control, uni_cik, earnings)
    events = pd.concat([events_signal, events_control], ignore_index=True)
    print(f"[sc13d_small] {len(events)} events after dedup+contamination "
          f"({events['category'].value_counts().to_dict() if not events.empty else {}})",
          flush=True)
    if events.empty:
        return

    start_px = (events["file_date"].min() - pd.Timedelta(days=60)).date().isoformat()
    end_px = (events["file_date"].max() + pd.Timedelta(days=120)).date().isoformat()
    price_cache = get_history_bulk([*events["ticker"].unique(), BENCHMARK],
                                   DB_PATH, start_px, end_px)

    n0 = len(events)
    events = attach_cap_proxy(events, uni, price_cache)
    events = filter_cap_proxy(events, min_cap=MIN_CAP, max_cap=MAX_CAP)
    n1 = len(events)
    events = attach_dollar_vol(events, price_cache)
    events = filter_dollar_vol(events, min_dollar_vol=MIN_DOLLAR_VOL)
    n2 = len(events)
    print(f"[sc13d_small] band filter {n0} -> {n1}; dollar-vol filter -> {n2}", flush=True)
    if events.empty:
        print("[sc13d_small] no in-band events — nothing to summarize")
        return

    span_yrs = max((events["file_date"].max()
                    - events["file_date"].min()).days / 365.25, 1e-9)
    n_sig_final = int((events["category"] == SIGNAL_CATEGORY).sum())
    print(f"[sc13d_small] exact post-filter counts: signal={n_sig_final} "
          f"({n_sig_final/span_yrs:.1f}/yr)", flush=True)

    events = compute_abnormal_returns(events, DB_PATH, horizons=HORIZONS,
                                      benchmark=BENCHMARK)
    events = attach_net_returns(events, HORIZONS)

    today = date.today().isoformat()
    events.to_csv(os.path.join(args.out_dir, f"sc13d_small_events_{today}.csv"), index=False)

    print("\n=== GROSS abnormal returns (comparability with bank 1-2) ===")
    gross = summarize(events, horizons=HORIZONS, gate_horizons=GATE_HORIZONS)
    print(gross.to_string(index=False))

    print(f"\n=== NET of cost model — THE GATE (>= {NET_GATE_MIN:.1%} net) ===")
    net = summarize(as_net_frame(events, HORIZONS), horizons=HORIZONS,
                    gate_horizons=GATE_HORIZONS, min_abs_return=NET_GATE_MIN)
    print(net.to_string(index=False))
    net.to_csv(os.path.join(args.out_dir, f"sc13d_small_summary_{today}.csv"), index=False)

    print("\n=== Split-half stability (net basis) ===")
    halves = split_half(events)
    print(halves.to_string(index=False))
    halves.to_csv(os.path.join(args.out_dir, f"sc13d_small_halves_{today}.csv"), index=False)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_sc13d_small_backtest.py tests/test_sc13d_backtest.py -q`
Expected: all PASS.

- [ ] **Step 5: Run the full suite**

Run: `python3 -m pytest tests/ -q | tail -1`
Expected: `273 passed` (270 + 3 new), zero failures.

- [ ] **Step 6: Commit**

```bash
git add src/sc13d_small_backtest.py tests/test_sc13d_small_backtest.py
git commit -m "feat(sc13d-small): Schedule 13D \$500M-\$2.5B backtest (idea bank 3, registry Q11)"
```

---

### Task 5: Q11 dry-run, full run + ledger row

**Files:**
- Modify: `docs/strategy-registry.md` (ledger row + Q11 status line)

- [ ] **Step 1: Dry-run**

```bash
python3 -m src.sc13d_small_backtest --dry-run-only 2>&1 | tee output/sc13d_small_dryrun_$(date +%F).log
```
The module auto-kills below the bar and prints `KILLED at dry-run` — if
that happens, record it in the Q11 queue section (Q3.5 precedent, no
Test-ledger slot) and skip to Task 6. Note: the Q2 run found activists
DO file plenty of 13Ds; the open question was always the band split, so
expect this to pass but do not assume it.

- [ ] **Step 2: Full run in the background**

```bash
nohup python3 -m src.sc13d_small_backtest \
  > output/sc13d_small_run_$(date +%F).log 2>&1 &
echo $!
```
Poll with `tail -5`; the 13G control fetch is the slow phase (EDGAR FTS
pagination, tens of minutes). Done when the split-half table prints.
Confirm the exact post-filter signal count still clears the bar.

- [ ] **Step 3: Apply the gate**

Gate horizons (5, 20, 40), four legs from MANDATORY CONTEXT with:
- Signal: `sc13d_new` on the NET table.
- Leg 2 (control): `sc13g_new` (passive 5% stake, same disclosure size)
  must not show the same positive drift — 13D-vs-13G separation is also
  Q11's leg-4 discriminator; state both means side by side in the Notes.
- Leg 3: split-half per the standard rule. Q2's kill included a
  first/second-half sign flip — check for the same pattern explicitly and
  name it in the Notes either way.

- [ ] **Step 4: Ledger row + status line + commit**

Append row #15 (or #14 if Q10 died at dry-run) to the Test ledger, update
Q11's `Status:` line, then:

```bash
git add docs/strategy-registry.md
git commit -m "docs(registry): ledger row for sc13d-small backtest (Q11) — <PASS|FAIL>"
```

---

### Task 6: Wrap-up

**Files:**
- Modify: `docs/strategy-registry.md` (only if a closing note is needed)

- [ ] **Step 1: Full suite one last time**

Run: `python3 -m pytest tests/ -q | tail -1`
Expected: same pass count as after the last module task, zero failures.

- [ ] **Step 2: Verify every gate run got its ledger row**

Open `docs/strategy-registry.md`; confirm one Test-ledger row per full
gate run executed in this plan, and that each Q9/Q10/Q11 queue-section
`Status:` line matches its row.

- [ ] **Step 3: Report**

Summarize to the user: verdict per idea with headline numbers (net mean,
p, split-half behavior at the best horizon), plus a one-line answer to
the standing question — if all three died, the registry's own closing
sentence applies: the relaxed-constraint well is also dry, and the honest
next conversation is whether systematic edge-hunting at this capital/data
level should continue at all. Do NOT propose or pre-register new ideas —
that is the user's decision point, not the executor's.

---

## Self-review notes (done at plan-writing time)

- Spec coverage: Q9 run+ledger (Task 1), Q10 build+run+ledger (Tasks 2-3),
  Q11 build+run+ledger (Tasks 4-5), stop-on-PASS rule (context + Steps),
  dry-run kills (Tasks 3/5 Step 1-2), cost-gate mechanics (shared context).
- Type consistency: `as_net_frame(events, horizons)` defined in Task 2
  Step 3, used identically in Tasks 2 and 4 module code; Q10 splits on
  `filing_date` (insider convention), Q11 on `file_date` (FTS convention) —
  intentional, matches each source module's column names.
- Known judgment call baked in: dry-run band membership approximated by
  CURRENT cap (prices not yet fetched at dry-run time); exact cap_proxy
  counts re-checked in every full run before trusting the gate. Same
  convention as Q9, disclosed in every dry-run's printed output.
