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
