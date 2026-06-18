# tests/test_compose.py
import numpy as np
import pandas as pd
import pytest
from src.compose import winsorize_series, z_score_series, apply_quality_gate, apply_confirmation_gate, build_composite

def make_factors_df(n=50):
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "ticker":          [f"T{i}" for i in range(n)],
        "mom_12_1":        rng.normal(0.10, 0.20, n),
        "rev_breadth":     rng.uniform(-1, 1, n),
        "sue":             rng.normal(0, 2, n),
        "rs_6m":           rng.normal(0, 0.15, n),
        "gp_assets":       rng.uniform(0.0, 0.6, n),
        "price":           rng.uniform(10, 500, n),
        "sma_200":         rng.uniform(10, 500, n),
        "pct_from_high":   rng.uniform(-0.3, 0.0, n),
    })
    df["price"] = df["sma_200"] * 1.1  # all above SMA200 initially
    return df

def test_winsorize_clips_extremes():
    s = pd.Series([1, 2, 3, 4, 100])
    result = winsorize_series(s, pct=0.20)
    assert result.max() < 100

def test_z_score_has_unit_variance():
    s = pd.Series(np.random.default_rng(0).normal(5, 3, 100))
    z = z_score_series(s)
    assert abs(z.std() - 1.0) < 0.01
    assert abs(z.mean()) < 0.01

def test_quality_gate_drops_below_median():
    df = make_factors_df(50)
    cfg = {"quality_gate": {"gross_profitability_min": "median"}}
    result = apply_quality_gate(df, cfg)
    assert len(result) < len(df)
    median = df["gp_assets"].median()
    assert (result["gp_assets"] >= median).all()

def test_quality_gate_float_threshold():
    df = make_factors_df(50)
    cfg = {"quality_gate": {"gross_profitability_min": 0.3}}
    result = apply_quality_gate(df, cfg)
    assert (result["gp_assets"] >= 0.3).all()

def test_confirmation_gate_sma200():
    df = make_factors_df(50)
    df.loc[0, "price"] = df.loc[0, "sma_200"] * 0.9  # first row below SMA200
    cfg = {
        "confirmation": {"require_above_sma200": True, "max_pct_below_52w_high": 0.10},
        "quality_gate": {"gross_profitability_min": 0.0},
    }
    result = apply_confirmation_gate(df, cfg)
    assert "T0" not in result["ticker"].values

def test_build_composite_no_nan():
    df = make_factors_df(50)
    cfg = {
        "factors": {
            "weights": {"mom_12_1": 0.35, "rev_breadth": 0.25, "sue": 0.20, "rs_6m": 0.20},
            "winsorize_pct": 0.01,
            "missing_factor_treatment": "neutral",
        },
        "quality_gate": {"gross_profitability_min": 0.0},
        "confirmation": {"require_above_sma200": False, "max_pct_below_52w_high": 1.0},
        "output": {"top_n": 10},
    }
    result = build_composite(df, cfg)
    assert "composite" in result.columns
    assert not result["composite"].isna().any()
    assert result["composite"].is_monotonic_decreasing

def test_build_composite_sorted_descending():
    df = make_factors_df(50)
    cfg = {
        "factors": {
            "weights": {"mom_12_1": 0.35, "rev_breadth": 0.25, "sue": 0.20, "rs_6m": 0.20},
            "winsorize_pct": 0.01,
            "missing_factor_treatment": "neutral",
        },
        "quality_gate": {"gross_profitability_min": 0.0},
        "confirmation": {"require_above_sma200": False, "max_pct_below_52w_high": 1.0},
        "output": {"top_n": 10},
    }
    result = build_composite(df, cfg)
    assert result["composite"].iloc[0] >= result["composite"].iloc[-1]
