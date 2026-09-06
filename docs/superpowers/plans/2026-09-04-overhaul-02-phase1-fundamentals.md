# Phase 1 — Fundamentals Block and Finalist Enrichment (Plan 02)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fundamentals decide who is eligible and carry 32% of the composite; finalists are re-ranked by true EPS estimate revisions and carry earnings dates.

**Architecture:** Finnhub ratios (13 metrics) become columns on the survivors frame; EDGAR-derived accruals/asset growth/net issuance/ROA/leverage come from the companyfacts JSON already downloaded. `fundamentals_block.attach_fundamentals_block` turns them into five percentile sub-scores and a `fund_score`; `apply_fundamental_gate` (inside `build_composite`) excludes names below the 30th percentile, skipping itself when coverage is cold. Weights: momentum 0.40 / earnings 0.20 / fundamentals 0.32 / technical 0.08. `enrich.enrich_finalists` adds `eps_rev_30d/90d`, short interest, analyst target and days-to-earnings for the top 60; `composite_final` re-ranks; conviction gains a fundamentals component.

**Tech Stack:** pandas rank percentiles; yfinance `eps_trend` (verified shape: index `0q,+1q,0y,+1y`; columns `current, 7daysAgo, 30daysAgo, 60daysAgo, 90daysAgo`).

---

### Task 1.1: EDGAR extended facts (already applied in Task 0.5)

**Files:**
- `src/fundamentals.py` — no further change

The parser, `edgar_facts` cache and the survivors-loop columns (`gp_assets, accruals, asset_growth, net_issuance, roa, leverage`) landed with Task 0.5. Verify only.

- [ ] **Step 1:** Run:

```bash
/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m pytest tests/test_fundamentals.py -q -k 'edgar'
```

Expected: all EDGAR-related tests pass (8 selected).


### Task 1.2: `src/fundamentals_block.py` — five sub-scores, fund_score, floor gate

**Files:**
- Create: `src/fundamentals_block.py`
- Create: `tests/test_fundamentals_block.py`

Pure pandas. Percentile ranks (not z-scores) because these ratios are heavy-tailed. Live probe on 19 names produced sensible orderings (NVDA quality 0.94 / growth 0.88; NRIX quality 0.05 / value 0.08 / strength 0.99 — a cash-rich loss-maker; banks scored on ROE/ROA/margin only).

- [ ] **Step 1:** **Create `tests/test_fundamentals_block.py` with exactly this content:**

```python
"""Fundamentals block: percentile sub-scores, financial handling, floor gate."""
import numpy as np
import pandas as pd
import pytest

from src.fundamentals_block import (
    attach_fundamentals_block, apply_fundamental_gate, derive_value_inputs,
    SUBSCORES, MIN_SUBSCORES,
)


def _frame(n=10, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "ticker": [f"T{i}" for i in range(n)],
        "is_financial": [False] * n,
        "gp_assets": rng.uniform(0.1, 0.8, n),
        "roeTTM": rng.uniform(-10, 40, n),
        "roaTTM": rng.uniform(-5, 20, n),
        "accruals": rng.uniform(-0.1, 0.1, n),
        "grossMarginTTM": rng.uniform(20, 80, n),
        "revenueGrowthTTMYoy": rng.uniform(-20, 60, n),
        "epsGrowthTTMYoy": rng.uniform(-50, 100, n),
        "revenueGrowth3Y": rng.uniform(-5, 30, n),
        "peTTM": rng.uniform(8, 60, n),
        "pfcfShareTTM": rng.uniform(10, 80, n),
        "evEbitdaTTM": rng.uniform(5, 40, n),
        "psTTM": rng.uniform(0.5, 15, n),
        "asset_growth": rng.uniform(-0.1, 0.5, n),
        "net_issuance": rng.uniform(-0.05, 0.2, n),
        "totalDebt/totalEquityQuarterly": rng.uniform(0, 3, n),
        "netInterestCoverageTTM": rng.uniform(1, 50, n),
        "currentRatioQuarterly": rng.uniform(0.5, 4, n),
    })


def test_attach_adds_subscores_in_unit_interval_and_fund_score():
    out = attach_fundamentals_block(_frame(20))
    for c in SUBSCORES + ["fund_score"]:
        assert c in out.columns
        assert out[c].between(0, 1).all()
    assert (out["fund_n_subscores"] == 5).all()


def test_higher_is_better_sign_conventions():
    df = _frame(5)
    # Make T0 the best on every input, T4 the worst.
    df["peTTM"] = [5, 10, 20, 30, 60]
    df["roeTTM"] = [40, 30, 20, 10, -5]
    df["accruals"] = [-0.1, -0.05, 0, 0.05, 0.1]
    df["asset_growth"] = [-0.1, 0, 0.1, 0.3, 0.5]
    df["totalDebt/totalEquityQuarterly"] = [0, 0.5, 1, 2, 3]
    out = attach_fundamentals_block(df)
    assert out.loc[0, "fund_value"] > out.loc[4, "fund_value"]
    assert out.loc[0, "fund_quality"] > out.loc[4, "fund_quality"]
    assert out.loc[0, "fund_invest"] > out.loc[4, "fund_invest"]
    assert out.loc[0, "fund_strength"] > out.loc[4, "fund_strength"]


def test_negative_multiples_are_not_cheap():
    df = _frame(4)
    df["peTTM"] = [-5.0, 0.0, 10.0, 20.0]
    df["pfcfShareTTM"] = [np.nan] * 4
    df["evEbitdaTTM"] = [np.nan] * 4
    df["psTTM"] = [np.nan] * 4
    out = attach_fundamentals_block(df)
    assert pd.isna(out.loc[0, "fund_value"]) and pd.isna(out.loc[1, "fund_value"])
    assert out.loc[2, "fund_value"] > out.loc[3, "fund_value"]


def test_negative_equity_ranks_as_most_levered():
    df = _frame(3)
    df["totalDebt/totalEquityQuarterly"] = [0.5, -2.0, 1.5]   # -2.0 = negative equity
    out = derive_value_inputs(df)
    assert out.loc[1, "de_ratio"] == 99.0
    scored = attach_fundamentals_block(df)
    assert scored.loc[1, "fund_strength"] < scored.loc[0, "fund_strength"]


def test_financials_drop_gp_accruals_invest_strength():
    df = _frame(6)
    df.loc[0, "is_financial"] = True
    df.loc[0, "gp_assets"] = np.nan          # banks have no COGS
    out = attach_fundamentals_block(df)
    assert pd.isna(out.loc[0, "fund_invest"]) and pd.isna(out.loc[0, "fund_strength"])
    assert not pd.isna(out.loc[0, "fund_quality"])   # still scored on ROE/ROA/margin
    assert not pd.isna(out.loc[0, "fund_score"])     # 3 sub-scores available
    assert out.loc[0, "fund_n_subscores"] == 3


def test_missing_inputs_are_nan_tolerant():
    df = _frame(6)
    df.loc[0, ["gp_assets", "roeTTM", "roaTTM", "accruals"]] = np.nan   # only margin left
    out = attach_fundamentals_block(df)
    assert not pd.isna(out.loc[0, "fund_quality"])


def test_fund_score_nan_below_min_subscores():
    df = _frame(6)
    cols = [c for c in df.columns if c not in ("ticker", "is_financial", "revenueGrowthTTMYoy")]
    df.loc[0, cols] = np.nan       # only growth remains → 1 sub-score
    out = attach_fundamentals_block(df)
    assert out.loc[0, "fund_n_subscores"] == 1 < MIN_SUBSCORES
    assert pd.isna(out.loc[0, "fund_score"])


def test_attach_is_idempotent_and_keeps_row_count():
    df = _frame(12)
    once = attach_fundamentals_block(df)
    twice = attach_fundamentals_block(once)
    assert len(twice) == 12
    pd.testing.assert_series_equal(once["fund_score"], twice["fund_score"])


def test_attach_with_no_fundamental_columns_gives_nan_scores():
    df = pd.DataFrame({"ticker": ["A", "B"], "price": [1.0, 2.0]})
    out = attach_fundamentals_block(df)
    assert out["fund_score"].isna().all() and (out["fund_n_subscores"] == 0).all()


def test_string_metrics_are_coerced():
    df = _frame(4)
    df["peTTM"] = ["10", "20", None, "abc"]
    out = attach_fundamentals_block(df)
    assert out["fund_value"].notna().all()   # other value inputs still present


# ── gate ──────────────────────────────────────────────────────────────────────

def _gate_cfg(**kw):
    base = {"floor_enabled": True, "floor_percentile": 0.30, "min_coverage": 0.60}
    base.update(kw)
    return {"fundamentals": base}


def test_gate_drops_below_floor_and_nan():
    df = pd.DataFrame({"ticker": list("ABCDE"), "fund_score": [0.9, 0.31, 0.29, np.nan, 0.5]})
    out, info = apply_fundamental_gate(df, _gate_cfg())
    assert list(out["ticker"]) == ["A", "B", "E"]
    assert info["after"] == 3 and info["skipped"] is None and info["coverage"] == 0.8


def test_gate_skips_on_low_coverage():
    df = pd.DataFrame({"ticker": list("ABCDE"), "fund_score": [0.1, np.nan, np.nan, np.nan, np.nan]})
    out, info = apply_fundamental_gate(df, _gate_cfg())
    assert len(out) == 5 and "coverage" in info["skipped"]


def test_gate_disabled_or_missing_column():
    df = pd.DataFrame({"ticker": list("AB"), "fund_score": [0.1, 0.2]})
    out, info = apply_fundamental_gate(df, _gate_cfg(floor_enabled=False))
    assert len(out) == 2 and info["skipped"] == "disabled"
    out, info = apply_fundamental_gate(df.drop(columns="fund_score"), _gate_cfg())
    assert len(out) == 2 and "no fund_score" in info["skipped"]


def test_gate_empty_frame():
    out, info = apply_fundamental_gate(pd.DataFrame({"ticker": [], "fund_score": []}), _gate_cfg())
    assert len(out) == 0
```

- [ ] **Step 2:** Run:

```bash
/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m pytest tests/test_fundamentals_block.py -q
```

Expected: FAIL at import.

- [ ] **Step 3:** **Create `src/fundamentals_block.py` with exactly this content:**

```python
"""Fundamentals block: five percentile sub-scores and one fund_score per stock.

Pure pandas. Inputs are the Finnhub basic-financials metrics (attached by
run.py as columns named exactly as Finnhub names them) plus the EDGAR-derived
columns from src.fundamentals. Every input is converted to a cross-sectional
percentile rank (0..1, higher = better after sign correction) and each
sub-score is the mean of the ranks it has data for. Ranks, not z-scores,
because these ratios are heavy-tailed and often garbage at the extremes
(a 4,000% revenue growth off a $1M base is a rank-1 observation either way).

| sub-score     | inputs (+ good, − bad)                                             |
|---------------|--------------------------------------------------------------------|
| fund_quality  | gp_assets +, roeTTM +, roaTTM +, accruals −, grossMarginTTM +      |
| fund_growth   | revenueGrowthTTMYoy +, epsGrowthTTMYoy +, revenueGrowth3Y +         |
| fund_value    | 1/peTTM +, 1/pfcfShareTTM +, 1/evEbitdaTTM +, 1/psTTM +  (only >0) |
| fund_invest   | asset_growth −, net_issuance −                                     |
| fund_strength | totalDebt/totalEquityQuarterly −, netInterestCoverageTTM +,        |
|               | currentRatioQuarterly +                                            |

Financials (banks, insurers, REITs, financial services): gp_assets, accruals,
fund_invest and fund_strength are meaningless and are dropped for them;
their fund_score is the mean of what remains.

fund_score is NaN when fewer than MIN_SUBSCORES sub-scores exist. NaN is
excluded by the floor gate — no data is not neutral when the question is
"is this business good enough to rank at all".
"""
import numpy as np
import pandas as pd

MIN_SUBSCORES = 2

# (column, sign) — sign +1 means higher is better.
QUALITY_INPUTS = [("gp_assets", +1), ("roeTTM", +1), ("roaTTM", +1), ("accruals", -1), ("grossMarginTTM", +1)]
GROWTH_INPUTS = [("revenueGrowthTTMYoy", +1), ("epsGrowthTTMYoy", +1), ("revenueGrowth3Y", +1)]
VALUE_INPUTS = [("ey_pe", +1), ("fcf_yield", +1), ("ebitda_yield", +1), ("sales_yield", +1)]
INVEST_INPUTS = [("asset_growth", -1), ("net_issuance", -1)]
STRENGTH_INPUTS = [("de_ratio", -1), ("netInterestCoverageTTM", +1), ("currentRatioQuarterly", +1)]

FINANCIAL_DROP = {"gp_assets", "accruals"}
FINANCIAL_SKIP_SUBSCORES = {"fund_invest", "fund_strength"}

SUBSCORES = ["fund_quality", "fund_growth", "fund_value", "fund_invest", "fund_strength"]


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("float64")


def _inverse_positive(s: pd.Series) -> pd.Series:
    """1/x for x > 0, NaN otherwise. A negative P/E or EV/EBITDA is not 'cheap',
    it is a loss-maker with no meaningful multiple."""
    x = _num(s)
    out = pd.Series(np.nan, index=x.index, dtype="float64")
    pos = x > 0
    out[pos] = 1.0 / x[pos]
    return out


def derive_value_inputs(df: pd.DataFrame) -> pd.DataFrame:
    """Add ey_pe, fcf_yield, ebitda_yield, sales_yield, de_ratio columns."""
    out = df.copy()
    n = len(out)
    nan = pd.Series(np.nan, index=out.index, dtype="float64")
    out["ey_pe"] = _inverse_positive(out["peTTM"]) if "peTTM" in out else nan
    out["fcf_yield"] = _inverse_positive(out["pfcfShareTTM"]) if "pfcfShareTTM" in out else nan
    out["ebitda_yield"] = _inverse_positive(out["evEbitdaTTM"]) if "evEbitdaTTM" in out else nan
    out["sales_yield"] = _inverse_positive(out["psTTM"]) if "psTTM" in out else nan
    if "totalDebt/totalEquityQuarterly" in out:
        de = _num(out["totalDebt/totalEquityQuarterly"])
        # Negative D/E means negative equity — treat as the most levered, not as
        # "less than zero debt". 99 ranks below any real ratio.
        de = de.where((de >= 0) | de.isna(), 99.0)
        out["de_ratio"] = de
    else:
        out["de_ratio"] = nan
    assert len(out) == n
    return out


def _pct_rank(s: pd.Series, sign: int) -> pd.Series:
    x = _num(s) * sign
    return x.rank(pct=True, na_option="keep")


def _subscore(df: pd.DataFrame, inputs: list[tuple[str, int]], mask_drop: pd.Series | None,
              drop_cols: set[str]) -> pd.Series:
    """Mean percentile across the inputs present for each row. `mask_drop`
    marks rows (financials) for which `drop_cols` inputs are set NaN first."""
    ranks = []
    for col, sign in inputs:
        if col not in df.columns:
            continue
        s = _num(df[col])
        if mask_drop is not None and col in drop_cols:
            s = s.where(~mask_drop, np.nan)
        ranks.append(_pct_rank(s, sign))
    if not ranks:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.concat(ranks, axis=1).mean(axis=1, skipna=True)


def attach_fundamentals_block(df: pd.DataFrame) -> pd.DataFrame:
    """Add the five sub-scores and fund_score. Idempotent."""
    out = derive_value_inputs(df)
    fin = out["is_financial"].fillna(False).astype(bool) if "is_financial" in out else pd.Series(False, index=out.index)

    out["fund_quality"] = _subscore(out, QUALITY_INPUTS, fin, FINANCIAL_DROP)
    out["fund_growth"] = _subscore(out, GROWTH_INPUTS, None, set())
    out["fund_value"] = _subscore(out, VALUE_INPUTS, None, set())
    out["fund_invest"] = _subscore(out, INVEST_INPUTS, None, set()).where(~fin, np.nan)
    out["fund_strength"] = _subscore(out, STRENGTH_INPUTS, None, set()).where(~fin, np.nan)

    sub = out[SUBSCORES]
    n_avail = sub.notna().sum(axis=1)
    out["fund_score"] = sub.mean(axis=1, skipna=True).where(n_avail >= MIN_SUBSCORES, np.nan)
    out["fund_n_subscores"] = n_avail.astype(int)
    return out


def apply_fundamental_gate(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Drop names below the fund_score floor. Returns (df, info).

    Coverage guard: if fewer than `min_coverage` of the rows have a fund_score
    (cold Finnhub cache, EDGAR outage), the gate is SKIPPED and info["skipped"]
    says why — excluding 70% of the universe for lack of data would be a worse
    ranking than not gating. run.py surfaces the skip in health.reasons.
    """
    fcfg = (cfg.get("fundamentals") or {})
    floor = float(fcfg.get("floor_percentile", 0.30))
    min_cov = float(fcfg.get("min_coverage", 0.60))
    enabled = bool(fcfg.get("floor_enabled", True))
    info = {"enabled": enabled, "floor": floor, "before": len(df), "after": len(df),
            "coverage": None, "skipped": None}
    if not enabled or "fund_score" not in df.columns or len(df) == 0:
        info["skipped"] = "disabled" if not enabled else "no fund_score column"
        return df, info
    cov = float(df["fund_score"].notna().mean())
    info["coverage"] = round(cov, 4)
    if cov < min_cov:
        info["skipped"] = f"fund_score coverage {cov:.0%} below {min_cov:.0%}"
        print(f"[fund_gate] SKIPPED — {info['skipped']}")
        return df, info
    keep = df["fund_score"] >= floor          # NaN compares False -> excluded
    result = df[keep].reset_index(drop=True)
    info["after"] = len(result)
    print(f"[fund_gate] {len(df)} → {len(result)} survivors (fund_score ≥ {floor:.2f}, coverage {cov:.0%})")
    return result, info
```

- [ ] **Step 4:** Run:

```bash
/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m pytest tests/test_fundamentals_block.py -q
```

Expected: 14 passed.

- [ ] **Step 5:** Commit as `feat(fundamentals): percentile block with quality/growth/value/invest/strength and a floor gate`. Per the user's global CLAUDE.md §6: run `git status` first and **ask the user before committing**. Never add a Co-Authored-By line.

**Edge cases this task must handle (each has a test):**
- Negative P/E, P/FCF, EV/EBITDA, P/S → NaN (a loss-maker is not cheap).
- Negative debt/equity → 99 (most levered); NaN stays NaN (this was a bug caught by the tests).
- Financials drop gp_assets/accruals and skip invest/strength; fund_score from the remaining three.
- fund_score NaN below 2 sub-scores; strings coerced; idempotent; frames without any fundamental column produce NaN scores.
- Gate excludes NaN and below-floor; skips itself with a reason below 60% coverage; disabled/missing column/empty frame handled.


### Task 1.3: `src/enrich.py` — finalist enrichment and composite_final

**Files:**
- Create: `src/enrich.py`
- Create: `tests/test_enrich.py`

yfinance is fine for 60 names, not 2,500. Cached per (ticker, date) in `finalist_enrichment`; one retry per ticker; wall-clock budget so the run finishes before the 17:15 trader session. Live probe: AAPL eps_rev_30d ≈ 0.0%, NRIX 90d +160% (negative-EPS base — rank-normalisation handles it).

- [ ] **Step 1:** **Create `tests/test_enrich.py` with exactly this content:**

```python
"""Finalist enrichment: estimate revisions, caching, budget, composite_final."""
import os
import tempfile
from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.cache import init_db
from src import enrich


def _db():
    f = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    f.close()
    init_db(f.name)
    return f.name


def _eps_trend(cur_y=8.8, ago30_y=8.6, ago90_y=8.5, cur_q=1.97, ago30_q=1.99, ago90_q=2.0):
    return pd.DataFrame(
        {"current": [cur_q, 2.9, cur_y, 9.5], "7daysAgo": [cur_q, 2.9, cur_y, 9.5],
         "30daysAgo": [ago30_q, 2.9, ago30_y, 9.5], "60daysAgo": [ago30_q, 2.9, ago30_y, 9.5],
         "90daysAgo": [ago90_q, 2.9, ago90_y, 9.5]},
        index=pd.Index(["0q", "+1q", "0y", "+1y"], name="period"))


def test_eps_revision_prefers_fiscal_year_row():
    r30, r90 = enrich.eps_revision(_eps_trend())
    assert r30 == pytest.approx(8.8 / 8.6 - 1)
    assert r90 == pytest.approx(8.8 / 8.5 - 1)


def test_eps_revision_falls_back_to_quarter_when_year_missing():
    df = _eps_trend().drop(index="0y")
    r30, _ = enrich.eps_revision(df)
    assert r30 == pytest.approx(1.97 / 1.99 - 1)


def test_eps_revision_negative_eps_uses_abs_denominator():
    df = _eps_trend(cur_y=-0.40, ago30_y=-0.50, ago90_y=-0.60)
    r30, r90 = enrich.eps_revision(df)
    assert r30 == pytest.approx(0.20) and r90 == pytest.approx(1 / 3)


def test_eps_revision_tiny_base_is_nan_and_empty_is_nan():
    df = _eps_trend(cur_y=0.01, ago30_y=0.005, ago90_y=0.001, cur_q=0.01, ago30_q=0.005, ago90_q=0.001)
    r30, r90 = enrich.eps_revision(df)
    assert np.isnan(r30) and np.isnan(r90)
    assert all(np.isnan(v) for v in enrich.eps_revision(None))
    assert all(np.isnan(v) for v in enrich.eps_revision(pd.DataFrame()))


def test_target_pct_and_short_interest_helpers():
    assert enrich.target_pct({"mean": 110.0}, 100.0) == pytest.approx(0.10)
    assert np.isnan(enrich.target_pct({"mean": None}, 100.0))
    assert np.isnan(enrich.target_pct({"mean": 110.0}, 0.0))
    assert np.isnan(enrich.target_pct(None, 100.0))
    sf, dtc = enrich.short_interest({"sharesShort": 10, "floatShares": 100, "averageVolume": 5})
    assert sf == pytest.approx(0.10) and dtc == pytest.approx(2.0)
    assert all(np.isnan(v) for v in enrich.short_interest(None))


def test_composite_final_adds_rank_normalised_revision():
    df = pd.DataFrame({"ticker": list("ABCD"), "composite": [1.0, 1.0, 1.0, 1.0],
                       "eps_rev_30d": [0.10, 0.0, -0.10, np.nan]})
    out = enrich.composite_final(df, weight=0.05)
    assert out.loc[0, "composite_final"] > out.loc[1, "composite_final"] > out.loc[2, "composite_final"]
    assert out.loc[3, "z_eps_rev_30d"] == 0.0 and out.loc[3, "composite_final"] == 1.0


def test_composite_final_neutral_when_too_few_revisions():
    df = pd.DataFrame({"ticker": list("AB"), "composite": [1.0, 0.5], "eps_rev_30d": [0.3, np.nan]})
    out = enrich.composite_final(df)
    assert (out["composite_final"] == out["composite"]).all()


def _fake_fetch(payloads: dict, fail: set = frozenset()):
    calls = []
    def fetch(ticker):
        calls.append(ticker)
        if ticker in fail:
            raise RuntimeError("rate limited")
        return payloads[ticker]
    fetch.calls = calls
    return fetch


def test_enrich_finalists_attaches_columns_and_caches():
    db = _db()
    df = pd.DataFrame({"ticker": ["AAA", "BBB"], "price": [100.0, 50.0], "composite": [1.0, 0.9]})
    payloads = {
        "AAA": {"eps_trend": _eps_trend(), "targets": {"mean": 120.0},
                "info": {"sharesShort": 10, "floatShares": 100, "averageVolume": 5}},
        "BBB": {"eps_trend": None, "targets": None, "info": None},
    }
    cal = {"AAA": "2026-09-20"}
    fetch = _fake_fetch(payloads)
    out, stats = enrich.enrich_finalists(df, db, cal, today=date(2026, 9, 4), fetch=fetch, sleep=lambda s: None)
    assert stats["fetched"] == 2 and stats["cached"] == 0
    assert out.loc[0, "eps_rev_30d"] == pytest.approx(8.8 / 8.6 - 1)
    assert out.loc[0, "analyst_target_pct"] == pytest.approx(0.20)
    assert out.loc[0, "short_float"] == pytest.approx(0.10)
    assert out.loc[0, "days_to_earnings"] == 16
    assert np.isnan(out.loc[1, "eps_rev_30d"]) and np.isnan(out.loc[1, "days_to_earnings"])
    # Second call same day: served from cache, fetch not called again.
    out2, stats2 = enrich.enrich_finalists(df, db, cal, today=date(2026, 9, 4), fetch=fetch, sleep=lambda s: None)
    assert stats2["cached"] == 2 and len(fetch.calls) == 2
    assert out2.loc[0, "eps_rev_30d"] == pytest.approx(out.loc[0, "eps_rev_30d"])
    os.unlink(db)


def test_enrich_finalists_retries_once_then_counts_failure():
    db = _db()
    df = pd.DataFrame({"ticker": ["AAA"], "price": [100.0], "composite": [1.0]})
    slept = []
    fetch = _fake_fetch({}, fail={"AAA"})
    out, stats = enrich.enrich_finalists(df, db, {}, today=date(2026, 9, 4), fetch=fetch, sleep=slept.append)
    assert stats["failed"] == 1 and len(fetch.calls) == 2 and slept == [enrich.RETRY_SLEEP]
    assert np.isnan(out.loc[0, "eps_rev_30d"])
    os.unlink(db)


def test_enrich_finalists_respects_wall_clock_budget():
    db = _db()
    df = pd.DataFrame({"ticker": ["AAA", "BBB", "CCC"], "price": [1.0, 1.0, 1.0], "composite": [1, 1, 1]})
    payloads = {t: {"eps_trend": _eps_trend(), "targets": None, "info": None} for t in ["AAA", "BBB", "CCC"]}
    ticks = iter([0.0, 0.0, 100.0, 100.0, 100.0, 100.0])   # budget blown after the first fetch
    out, stats = enrich.enrich_finalists(df, db, {}, today=date(2026, 9, 4), fetch=_fake_fetch(payloads),
                                         budget_secs=10.0, sleep=lambda s: None, clock=lambda: next(ticks))
    assert stats["fetched"] == 1 and stats["budget_exhausted"] == 2
    # days_to_earnings still filled from the calendar even when yfinance is skipped
    assert "days_to_earnings" in out.columns
    os.unlink(db)
```

- [ ] **Step 2:** Run:

```bash
/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m pytest tests/test_enrich.py -q
```

Expected: FAIL at import.

- [ ] **Step 3:** **Create `src/enrich.py` with exactly this content:**

```python
"""Finalist enrichment: the per-ticker lookups that are too slow or too
rate-limited for the whole universe but cheap for ~60 names.

  eps_rev_30d / eps_rev_90d   true estimate-revision momentum from yfinance
                              `eps_trend` (current vs 30/90 days ago, current
                              fiscal year row, current quarter as fallback)
  short_float, days_to_cover  yfinance get_info (was a per-survivor call)
  analyst_target_pct          mean analyst target / price − 1
  days_to_earnings            Finnhub earnings calendar (no yfinance)

Everything is best-effort with a wall-clock budget: Yahoo rate-limits, and a
run must finish before the trader session at 17:15 ET. Results are cached per
(ticker, date) so a rerun the same day costs nothing.
"""
import json
import logging
import math
import time
from datetime import date, datetime

import numpy as np
import pandas as pd

from src.cache import _get_conn, _now_iso
from src.compose import rank_normalize
from src.finnhub_data import days_to_earnings

logger = logging.getLogger(__name__)

ENRICH_COLUMNS = ["eps_rev_30d", "eps_rev_90d", "short_float", "days_to_cover",
                  "analyst_target_pct", "days_to_earnings"]
MIN_ABS_EPS = 0.02   # below this a percent change is meaningless
DEFAULT_BUDGET_SECS = 240.0
RETRY_SLEEP = 1.5


def init_enrich_table(db_path: str) -> None:
    _get_conn(db_path).execute("""
        CREATE TABLE IF NOT EXISTS finalist_enrichment (
            ticker TEXT, as_of_date TEXT, payload TEXT, fetched_at TEXT,
            PRIMARY KEY(ticker, as_of_date)
        )
    """)
    _get_conn(db_path).commit()


def _get_cached(db_path: str, ticker: str, as_of: str) -> dict | None:
    row = _get_conn(db_path).execute(
        "SELECT payload FROM finalist_enrichment WHERE ticker=? AND as_of_date=?",
        (ticker.upper(), as_of)).fetchone()
    return json.loads(row[0]) if row else None


def _put_cached(db_path: str, ticker: str, as_of: str, payload: dict) -> None:
    conn = _get_conn(db_path)
    conn.execute("INSERT OR REPLACE INTO finalist_enrichment VALUES (?,?,?,?)",
                 (ticker.upper(), as_of, json.dumps(payload), _now_iso()))
    conn.commit()


# ── pure helpers ──────────────────────────────────────────────────────────────

def _pct_change(now, then) -> float:
    try:
        now, then = float(now), float(then)
    except (TypeError, ValueError):
        return float("nan")
    if math.isnan(now) or math.isnan(then) or abs(then) < MIN_ABS_EPS:
        return float("nan")
    return (now - then) / abs(then)


def eps_revision(eps_trend: pd.DataFrame | None) -> tuple[float, float]:
    """(rev_30d, rev_90d) from a yfinance eps_trend frame.

    Index is the period ('0q', '+1q', '0y', '+1y'); columns include 'current',
    '30daysAgo', '90daysAgo'. Current fiscal year ('0y') is used first because
    quarterly consensus is noisier; '0q' is the fallback. Negative EPS is
    handled by dividing by |then|, so −0.50 → −0.40 is a +20% revision.
    """
    if eps_trend is None or getattr(eps_trend, "empty", True):
        return float("nan"), float("nan")
    for period in ("0y", "0q"):
        if period not in eps_trend.index:
            continue
        row = eps_trend.loc[period]
        r30 = _pct_change(row.get("current"), row.get("30daysAgo"))
        r90 = _pct_change(row.get("current"), row.get("90daysAgo"))
        if not (math.isnan(r30) and math.isnan(r90)):
            return r30, r90
    return float("nan"), float("nan")


def target_pct(targets: dict | None, price: float | None) -> float:
    mean = (targets or {}).get("mean")
    try:
        mean, price = float(mean), float(price)
    except (TypeError, ValueError):
        return float("nan")
    if not price or math.isnan(mean) or math.isnan(price):
        return float("nan")
    return mean / price - 1.0


def short_interest(info: dict | None) -> tuple[float, float]:
    from src.fundamentals import parse_short_interest
    if not info:
        return float("nan"), float("nan")
    try:
        return parse_short_interest(info)
    except Exception:  # noqa: BLE001
        return float("nan"), float("nan")


def composite_final(df: pd.DataFrame, weight: float = 0.05) -> pd.DataFrame:
    """composite + weight × rank-normalised eps_rev_30d (NaN → 0, i.e. neutral).
    Applied within finalists only, so a name is never dropped from the pool for
    lacking estimate coverage — it just gets no lift."""
    out = df.copy()
    if "eps_rev_30d" in out.columns and out["eps_rev_30d"].notna().sum() >= 3:
        z = rank_normalize(out["eps_rev_30d"])
        z = z.where(out["eps_rev_30d"].notna(), 0.0)
    else:
        z = pd.Series(0.0, index=out.index)
    out["z_eps_rev_30d"] = z
    out["composite_final"] = out["composite"] + weight * z
    return out


# ── yfinance IO ───────────────────────────────────────────────────────────────

def _yf_fetch(ticker: str) -> dict:
    """One ticker's raw yfinance payloads. Raises on failure; caller retries."""
    import yfinance as yf
    t = yf.Ticker(ticker)
    out = {"eps_trend": None, "targets": None, "info": None}
    out["eps_trend"] = t.eps_trend
    try:
        out["targets"] = t.analyst_price_targets
    except Exception:  # noqa: BLE001 — optional
        out["targets"] = None
    try:
        out["info"] = t.get_info()
    except Exception:  # noqa: BLE001 — optional
        out["info"] = None
    return out


def enrich_finalists(
    df: pd.DataFrame,
    db_path: str,
    calendar: dict[str, str],
    today: date | None = None,
    fetch=_yf_fetch,
    budget_secs: float = DEFAULT_BUDGET_SECS,
    sleep=time.sleep,
    clock=time.monotonic,
) -> tuple[pd.DataFrame, dict]:
    """Attach ENRICH_COLUMNS to the finalist frame. Returns (df, stats)."""
    today = today or date.today()
    as_of = today.isoformat()
    init_enrich_table(db_path)
    out = df.copy()
    for c in ENRICH_COLUMNS:
        if c not in out.columns:
            out[c] = np.nan
    stats = {"finalists": len(out), "cached": 0, "fetched": 0, "failed": 0, "budget_exhausted": 0}
    start = clock()

    for idx, row in out.iterrows():
        ticker = str(row["ticker"]).upper()
        dte = days_to_earnings(ticker, calendar, today)
        out.at[idx, "days_to_earnings"] = np.nan if dte is None else float(dte)

        cached = _get_cached(db_path, ticker, as_of)
        if cached is not None:
            stats["cached"] += 1
            for c in ("eps_rev_30d", "eps_rev_90d", "short_float", "days_to_cover", "analyst_target_pct"):
                v = cached.get(c)
                out.at[idx, c] = np.nan if v is None else float(v)
            continue

        if clock() - start > budget_secs:
            stats["budget_exhausted"] += 1
            continue

        raw = None
        for attempt in range(2):
            try:
                raw = fetch(ticker)
                break
            except Exception as e:  # noqa: BLE001
                if attempt == 0:
                    sleep(RETRY_SLEEP)
                else:
                    logger.warning(f"[enrich] {ticker}: {type(e).__name__}: {e}")
        if raw is None:
            stats["failed"] += 1
            continue

        r30, r90 = eps_revision(raw.get("eps_trend"))
        sf, dtc = short_interest(raw.get("info"))
        tp = target_pct(raw.get("targets"), row.get("price"))
        payload = {"eps_rev_30d": _none_if_nan(r30), "eps_rev_90d": _none_if_nan(r90),
                   "short_float": _none_if_nan(sf), "days_to_cover": _none_if_nan(dtc),
                   "analyst_target_pct": _none_if_nan(tp)}
        _put_cached(db_path, ticker, as_of, payload)
        stats["fetched"] += 1
        out.at[idx, "eps_rev_30d"] = r30
        out.at[idx, "eps_rev_90d"] = r90
        out.at[idx, "short_float"] = sf
        out.at[idx, "days_to_cover"] = dtc
        out.at[idx, "analyst_target_pct"] = tp

    print(f"[enrich] finalists={stats['finalists']} cached={stats['cached']} fetched={stats['fetched']} "
          f"failed={stats['failed']} budget_exhausted={stats['budget_exhausted']}")
    return out, stats


def _none_if_nan(v):
    try:
        return None if v is None or math.isnan(float(v)) else float(v)
    except (TypeError, ValueError):
        return None
```

- [ ] **Step 4:** Run:

```bash
/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m pytest tests/test_enrich.py -q
```

Expected: 10 passed.

- [ ] **Step 5:** Commit as `feat(enrich): finalist EPS revisions, short interest, targets, earnings dates`. Per the user's global CLAUDE.md §6: run `git status` first and **ask the user before committing**. Never add a Co-Authored-By line.

**Edge cases this task must handle (each has a test):**
- Fiscal-year row first, quarter fallback; |then| < 0.02 → NaN; negative EPS uses |then|.
- Cache hit on the same day → no fetch; retry once then `failed`; budget exhaustion still fills `days_to_earnings` from the calendar.
- `composite_final` is neutral (equals composite) when fewer than 3 finalists have revisions; NaN revision → z 0.


### Task 1.4: Composite reweighting, conviction, news labels, output columns, earnings via Finnhub

**Files:**
- Modify: `src/compose.py` (diff in Task 0.6)
- Modify: `config.yaml` (diff in Task 0.7)
- Modify: `src/news.py`
- Modify: `src/output.py` (diff in Task 0.6)
- Modify: `src/positions.py`
- Modify: `tests/test_compose.py`

`COMPOSITE_FACTORS` gains the five fund factors (rank-normalised) and loses standalone `gp_assets` (now an input to fund_quality). `build_composite` gains `top_n` and `with_conviction` so run.py can take a 60-name finalist pool, enrich, re-rank and only then compute conviction on the final 20. The fund gate replaces the gp-only quality gate whenever `fund_score` is present. `FINANCIAL_SECTORS` gains 'Banking' (Finnhub's name). `positions.days_to_next_earnings` asks the Finnhub calendar first.

- [ ] **Step 1:** Update the compose tests:

**Apply this unified diff to `tests/test_compose.py`** (from the reference patch; `patch -p1 < docs/superpowers/plans/2026-09-04-overhaul-reference.patch` applies all of them at once):

```diff
--- a/tests/test_compose.py	2026-06-24 09:24:20
+++ b/tests/test_compose.py	2026-09-04 12:33:28
@@ -172,12 +172,75 @@
     df.loc[0:9, "sector"] = "Financial Services"
     result = build_composite(df, _BASE_CFG)
     fin_rows = result[result["sector"] == "Financial Services"]
+    # gp_assets is an input to fund_quality now, not a standalone factor, but
+    # it is still nulled for financials so the block cannot penalise a bank
+    # for having no COGS.
+    assert "z_gp_assets" not in result.columns
     if len(fin_rows) > 0:
-        # gp_assets should be NaN for financial stocks (set in _derive_new_factors)
-        # After z-scoring NaN is filled to mean=0, so z_gp_assets should be near 0
-        assert "z_gp_assets" in result.columns
+        assert fin_rows["gp_assets"].isna().all()
 
 
+def test_gp_assets_not_a_composite_factor_but_fund_block_is():
+    assert "gp_assets" not in COMPOSITE_FACTORS
+    for f in ["fund_quality", "fund_growth", "fund_value", "fund_invest", "fund_strength"]:
+        assert f in COMPOSITE_FACTORS
+
+
+def test_config_weights_sum_to_one_and_cover_every_factor():
+    import yaml
+    cfg = yaml.safe_load(open("config.yaml"))
+    w = cfg["factors"]["weights"]
+    assert abs(sum(w.values()) - 1.0) < 1e-9
+    assert set(w) == set(COMPOSITE_FACTORS)
+    blocks = {
+        "momentum": ["mom_12_1", "residual_mom", "rs_6m", "rs_accel", "rs_slope", "pct_from_high"],
+        "earnings": ["sue", "rev_breadth", "rev_magnitude"],
+        "fundamentals": ["fund_quality", "fund_growth", "fund_value", "fund_invest", "fund_strength", "insider_z"],
+        "technical": ["trend_score", "momo_osc_score", "volume_score"],
+    }
+    sums = {b: round(sum(w[f] for f in fs), 6) for b, fs in blocks.items()}
+    assert sums == {"momentum": 0.40, "earnings": 0.20, "fundamentals": 0.32, "technical": 0.08}
+
+
+def test_build_composite_uses_fund_gate_when_block_present():
+    df = make_factors_df(50)
+    df["fund_score"] = np.linspace(0.0, 1.0, 50)          # 15 names below 0.30
+    cfg = dict(_BASE_CFG); cfg["fundamentals"] = {"floor_percentile": 0.30, "min_coverage": 0.6}
+    cfg["output"] = {"top_n": 60}
+    result = build_composite(df, cfg)
+    assert result.attrs["fund_gate"]["after"] == 35
+    assert (result["fund_score"] >= 0.30).all()
+    assert result.attrs["ranked_total"] == 35
+
+
+def test_build_composite_fund_gate_skipped_on_low_coverage():
+    df = make_factors_df(50)
+    df["fund_score"] = np.nan
+    df.loc[0:9, "fund_score"] = 0.9                         # 20% coverage
+    cfg = dict(_BASE_CFG); cfg["fundamentals"] = {"floor_percentile": 0.30, "min_coverage": 0.6}
+    result = build_composite(df, cfg)
+    assert "coverage" in result.attrs["fund_gate"]["skipped"]
+    assert result.attrs["ranked_total"] == 50
+
+
+def test_build_composite_top_n_override_and_no_conviction():
+    df = make_factors_df(50)
+    result = build_composite(df, _BASE_CFG, top_n=30, with_conviction=False)
+    assert len(result) == 30 and "conviction" not in result.columns
+    assert len(result.attrs["ranking_tail"]) == 50
+
+
+def test_conviction_fund_component():
+    df = make_factors_df(50)
+    df["fund_score"] = 0.0
+    df.loc[0, "fund_score"] = 0.95
+    ranked = build_composite(df, _BASE_CFG, with_conviction=False)
+    ranked["fund_score"] = 0.0
+    base = compute_conviction(ranked)["conviction"].iloc[0]
+    ranked["fund_score"] = 0.95
+    assert compute_conviction(ranked)["conviction"].iloc[0] >= base
+
+
 def test_conviction_range():
     df = make_factors_df(50)
     result = build_composite(df, _BASE_CFG)
```

- [ ] **Step 2:** Apply the news labels:

**Apply this unified diff to `src/news.py`** (from the reference patch; `patch -p1 < docs/superpowers/plans/2026-09-04-overhaul-reference.patch` applies all of them at once):

```diff
--- a/src/news.py	2026-07-30 16:58:03
+++ b/src/news.py	2026-09-04 12:32:33
@@ -21,7 +21,11 @@
     "z_sue":            "earnings surprise (SUE)",
     "z_rev_breadth":    "analyst revision breadth (upgrades vs downgrades)",
     "z_rev_magnitude":  "analyst estimate revision magnitude",
-    "z_gp_assets":      "gross profitability quality",
+    "z_fund_quality":   "business quality (profitability, ROE/ROA, accruals)",
+    "z_fund_growth":    "revenue and EPS growth",
+    "z_fund_value":     "valuation (earnings/FCF/EBITDA yields)",
+    "z_fund_invest":    "conservative investment (low asset growth, no dilution)",
+    "z_fund_strength":  "balance-sheet strength",
     "z_insider_z":      "insider cluster buying",
     "z_trend_score":    "trend strength (ADX + MACD + SMA50)",
     "z_momo_osc_score": "momentum oscillator alignment",
```

- [ ] **Step 3:** Apply the positions change:

**Apply this unified diff to `src/positions.py`** (from the reference patch; `patch -p1 < docs/superpowers/plans/2026-09-04-overhaul-reference.patch` applies all of them at once):

```diff
--- a/src/positions.py	2026-09-03 17:58:39
+++ b/src/positions.py	2026-09-04 12:32:33
@@ -134,11 +134,18 @@
 def days_to_next_earnings(ticker: str) -> int | None:
     """Calendar days until the next scheduled earnings date. None on failure.
 
-    Reuses the yfinance earnings-dates endpoint (same source as
-    src/pead_backtest.py). Best-effort: returns None if the call fails or no
-    future date is listed.
+    Finnhub's earnings calendar (one cached call for the whole market) first;
+    the yfinance per-ticker endpoint is the fallback, because Yahoo
+    rate-limits it and it logs "No earnings dates found" for SPY every run.
     """
     try:
+        from src.finnhub_data import fetch_earnings_calendar, days_to_earnings
+        d = days_to_earnings(ticker, fetch_earnings_calendar())
+        if d is not None:
+            return d
+    except Exception:
+        pass
+    try:
         df = yf.Ticker(ticker).get_earnings_dates(limit=8)
         if df is None or df.empty:
             return None
```

- [ ] **Step 4:** Run:

```bash
/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m pytest tests/test_compose.py tests/test_news.py tests/test_positions.py -q
```

Expected: all pass; `test_config_weights_sum_to_one_and_cover_every_factor` proves the four blocks sum to 0.40/0.20/0.32/0.08.

- [ ] **Step 5:** Commit as `feat(compose): fundamentals block at 32% with floor gate; finalist pool; conviction fundamentals component`. Per the user's global CLAUDE.md §6: run `git status` first and **ask the user before committing**. Never add a Co-Authored-By line.

**Edge cases this task must handle (each has a test):**
- Weights sum to 1.0 and cover exactly `COMPOSITE_FACTORS`.
- Frames without `fund_score` (backtests, old tests) still use the legacy quality gate.
- Fund gate skip on low coverage leaves `ranked_total` unchanged.
- `top_n` override and `with_conviction=False` return the pool without a conviction column; `ranking_tail` covers up to 100.


### Task 1.5: Verify the fundamentals block on a real run

- [ ] **Step 1:** After the Finnhub cache has warmed (Task 0.9 Step 2), run the pipeline and read the fundamentals lines:

```bash
/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m src.run 2>&1 | grep -E "\[fund_block\]|\[fund_gate\]|\[enrich\]|\[compose\]|\[output\]"
```
Expected: `[fund_block] fund_score coverage ≥ 60% of N names`; `[fund_gate] N → M survivors (fund_score ≥ 0.30, coverage xx%)` (M ≈ 0.6–0.7 × N); `[enrich] finalists=60 cached=… fetched=… failed=… budget_exhausted=0`; `[compose] … → top 60 selected`.

- [ ] **Step 2:** Inspect the CSV's new columns:

```bash
/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -c "import pandas as pd;d=pd.read_csv(sorted(__import__('glob').glob('output/screen_*.csv'))[-1]);print(d[['ticker','sector','composite','composite_final','fund_score','fund_quality','fund_growth','fund_value','eps_rev_30d','days_to_earnings','conviction','entry']].head(20).to_string())"
```
Expected: `sector` populated (no blanks), `fund_score` ≥ 0.30 everywhere, `composite_final` sorted descending, `days_to_earnings` numbers or blank.

- [ ] **Step 3:** If coverage is below 60% the gate logs `SKIPPED` and health.warnings carries it — that is expected on day one; re-check after `warm-finnhub` completes.
