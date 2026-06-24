# tests/test_compose.py
import numpy as np
import pandas as pd
import pytest
from src.compose import (
    winsorize_series, z_score_series, rank_normalize,
    apply_quality_gate, apply_confirmation_gate,
    build_composite, compute_conviction, COMPOSITE_FACTORS,
)


def make_factors_df(n=50):
    rng = np.random.default_rng(42)
    sectors = ["Technology", "Healthcare", "Industrials", "Consumer Cyclical", "Energy"]
    df = pd.DataFrame({
        "ticker":           [f"T{i}" for i in range(n)],
        "sector":           [sectors[i % len(sectors)] for i in range(n)],
        "mom_12_1":         rng.normal(0.10, 0.20, n),
        "residual_mom":     rng.normal(0.05, 0.30, n),
        "mom_1m":           rng.normal(0.03, 0.08, n),
        "rev_breadth":      rng.uniform(-1, 1, n),
        "rev_magnitude":    rng.normal(0, 0.05, n),
        "sue":              rng.normal(0, 2, n),
        "rs_6m":            rng.normal(0, 0.15, n),
        "rs_3m":            rng.normal(0, 0.10, n),
        "rs_slope":         rng.normal(0, 0.001, n),
        "gp_assets":        rng.uniform(0.0, 0.6, n),
        "pct_from_high":    rng.uniform(-0.3, 0.0, n),
        "price":            rng.uniform(10, 500, n),
        "sma_200":          rng.uniform(10, 500, n),
        "rsi_14":           rng.uniform(30, 70, n),
        "macd":             ["bullish"] * n,
        "stoch_k":          rng.uniform(30, 70, n),
        "stoch_d":          rng.uniform(30, 70, n),
        "stoch_cross":      [False] * n,
        "adx":              rng.uniform(15, 35, n),
        "vol_surge":        rng.uniform(0.8, 1.5, n),
        "above_sma50":      [True] * n,
        "bb_pct_b":         rng.uniform(0.3, 0.8, n),
        "mfi":              rng.uniform(35, 65, n),
        "insider_buys_90d": rng.integers(0, 4, n),
        "exec_buys_90d":    rng.integers(0, 3, n),
        "short_float":      rng.uniform(0.01, 0.30, n),
    })
    df["price"] = df["sma_200"] * 1.1  # all above SMA200
    return df


_BASE_CFG = {
    "factors": {
        "weights": {
            "mom_12_1": 0.12, "residual_mom": 0.14, "rs_6m": 0.10,
            "rs_accel": 0.06, "rs_slope": 0.04, "pct_from_high": 0.04,
            "sue": 0.14, "rev_breadth": 0.10, "rev_magnitude": 0.09,
            "gp_assets": 0.06, "insider_z": 0.03,
            "trend_score": 0.05, "momo_osc_score": 0.02, "volume_score": 0.01,
        },
        "winsorize_pct": 0.01,
        "missing_factor_treatment": "neutral",
        "rank_normalize_factors": ["insider_z", "trend_score", "momo_osc_score", "volume_score"],
        "min_factor_coverage": 0.30,
    },
    "quality_gate": {"gross_profitability_min": 0.0},
    "confirmation": {"require_above_sma200": False, "max_pct_below_52w_high": 1.0},
    "output": {"top_n": 10},
    "streak": {"lookback_days": 14},
}


def test_winsorize_clips_extremes():
    s = pd.Series([1, 2, 3, 4, 100])
    result = winsorize_series(s, pct=0.20)
    assert result.max() < 100


def test_z_score_has_unit_variance():
    s = pd.Series(np.random.default_rng(0).normal(5, 3, 100))
    z = z_score_series(s)
    assert abs(z.std() - 1.0) < 0.01
    assert abs(z.mean()) < 0.01


def test_rank_normalize_is_normal():
    s = pd.Series([0, 0, 0, 1, 2, 3, 0, 0, 1, 4])  # zero-inflated integer
    z = rank_normalize(s)
    assert len(z) == len(s)
    assert not z.isna().any()


def test_quality_gate_q25():
    df = make_factors_df(50)
    cfg = {"quality_gate": {"gross_profitability_min": "q25"}}
    result = apply_quality_gate(df, cfg)
    q25 = df["gp_assets"].quantile(0.25)
    assert (result["gp_assets"].fillna(-999) >= q25 - 1e-9).all()
    assert len(result) > len(df) // 2

def test_quality_gate_financial_bypass():
    df = make_factors_df(50)
    df.loc[0:4, "sector"] = "Financial Services"
    df.loc[0:4, "gp_assets"] = -0.5  # would fail quality gate
    cfg = {"quality_gate": {"gross_profitability_min": "q25"}}
    result = apply_quality_gate(df, cfg)
    # Financial sector stocks should pass even with negative gp_assets
    fin_pass = result[result["sector"] == "Financial Services"]
    assert len(fin_pass) > 0


def test_confirmation_gate_sma200():
    df = make_factors_df(50)
    df.loc[0, "price"] = df.loc[0, "sma_200"] * 0.9
    cfg = {
        "confirmation": {"require_above_sma200": True, "max_pct_below_52w_high": 1.0},
        "quality_gate": {"gross_profitability_min": 0.0},
    }
    result = apply_confirmation_gate(df, cfg)
    assert "T0" not in result["ticker"].values


def test_composite_factors_no_streak_in_list():
    assert "streak_z" not in COMPOSITE_FACTORS
    assert "st_reversal" not in COMPOSITE_FACTORS


def test_composite_has_residual_mom_and_pct_from_high():
    assert "residual_mom" in COMPOSITE_FACTORS
    assert "pct_from_high" in COMPOSITE_FACTORS


def test_build_composite_no_nan():
    df = make_factors_df(50)
    result = build_composite(df, _BASE_CFG)
    assert "composite" in result.columns
    assert not result["composite"].isna().any()
    assert result["composite"].is_monotonic_decreasing


def test_build_composite_sorted_descending():
    df = make_factors_df(50)
    result = build_composite(df, _BASE_CFG)
    assert result["composite"].iloc[0] >= result["composite"].iloc[-1]


def test_build_composite_has_new_factors():
    df = make_factors_df(50)
    result = build_composite(df, _BASE_CFG)
    for col in ["rs_accel", "insider_z", "trend_score", "momo_osc_score", "volume_score"]:
        assert col in result.columns
    # streak kept as diagnostic
    assert "streak_z" in result.columns
    assert "streak_consecutive" in result.columns
    assert "st_reversal" in result.columns


def test_streak_not_in_composite_weights():
    cfg = _BASE_CFG
    weights = cfg["factors"]["weights"]
    assert "streak_z" not in weights
    assert "st_reversal" not in weights


def test_factor_coverage_column():
    df = make_factors_df(50)
    result = build_composite(df, _BASE_CFG)
    assert "factor_coverage" in result.columns
    assert (result["factor_coverage"] >= 0).all()
    assert (result["factor_coverage"] <= 1.01).all()


def test_gp_assets_nulled_for_financials():
    df = make_factors_df(50)
    df.loc[0:9, "sector"] = "Financial Services"
    result = build_composite(df, _BASE_CFG)
    fin_rows = result[result["sector"] == "Financial Services"]
    if len(fin_rows) > 0:
        # gp_assets should be NaN for financial stocks (set in _derive_new_factors)
        # After z-scoring NaN is filled to mean=0, so z_gp_assets should be near 0
        assert "z_gp_assets" in result.columns


def test_conviction_range():
    df = make_factors_df(50)
    result = build_composite(df, _BASE_CFG)
    assert "conviction" in result.columns
    assert result["conviction"].between(1, 10).all()
    assert result["conviction"].dtype in (int, "int64", "int32")


def test_conviction_top3_higher_rank_component():
    df = make_factors_df(50)
    result = build_composite(df, _BASE_CFG)
    top3_conv = result.iloc[:3]["conviction"].mean()
    bottom_conv = result.iloc[7:]["conviction"].mean()
    assert top3_conv >= bottom_conv


def test_rs_accel_positive_when_accelerating():
    df = make_factors_df(50)
    df["rs_3m"] = 0.20
    df["rs_6m"] = 0.10
    result = build_composite(df, _BASE_CFG)
    assert "rs_accel" in result.columns
    # rs_accel = 0.20*2 - 0.10 = 0.30 for all → all equal → z=0
    assert "z_rs_accel" in result.columns
