import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

COMPOSITE_FACTORS = ["mom_12_1", "rev_breadth", "sue", "rs_6m"]


def winsorize_series(s: pd.Series, pct: float = 0.01) -> pd.Series:
    lo = s.quantile(pct)
    hi = s.quantile(1 - pct)
    return s.clip(lower=lo, upper=hi)


def z_score_series(s: pd.Series) -> pd.Series:
    mu  = s.mean()
    std = s.std()
    if std < 1e-12:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mu) / std


def apply_quality_gate(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    threshold = cfg["quality_gate"]["gross_profitability_min"]
    if "gp_assets" not in df.columns:
        return df
    gp = df["gp_assets"].fillna(-999)
    if threshold == "median":
        cutoff = gp.median()
    else:
        cutoff = float(threshold)
    before = len(df)
    result = df[gp >= cutoff].reset_index(drop=True)
    logger.info(f"[quality_gate] {before} → {len(result)} survivors")
    print(f"[quality_gate] {before} → {len(result)} survivors (gp_assets >= {cutoff:.4f})")
    return result


def apply_confirmation_gate(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    confirm = cfg["confirmation"]
    mask = pd.Series(True, index=df.index)
    if confirm.get("require_above_sma200", False) and "sma_200" in df.columns:
        mask &= df["price"] >= df["sma_200"]
    max_below = confirm.get("max_pct_below_52w_high", 1.0)
    if "pct_from_high" in df.columns:
        mask &= df["pct_from_high"] >= -(max_below)
    before = len(df)
    result = df[mask].reset_index(drop=True)
    logger.info(f"[confirmation_gate] {before} → {len(result)} survivors")
    print(f"[confirmation_gate] {before} → {len(result)} survivors")
    return result


def build_composite(factors_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = factors_df.copy()
    weights = cfg["factors"]["weights"]
    winsorize_pct = cfg["factors"]["winsorize_pct"]
    missing_treatment = cfg["factors"].get("missing_factor_treatment", "neutral")

    df = apply_quality_gate(df, cfg)
    df = apply_confirmation_gate(df, cfg)

    z_cols = {}
    for factor in COMPOSITE_FACTORS:
        if factor not in df.columns:
            logger.warning(f"[compose] factor {factor} missing entirely — treating all as neutral")
            df[f"z_{factor}"] = 0.0
            continue
        s = df[factor].copy()
        if missing_treatment == "neutral":
            fill_val = s.mean()
            s = s.fillna(0.0 if pd.isna(fill_val) else fill_val)
        else:
            df = df[s.notna()].reset_index(drop=True)
            s = df[factor]
        s = winsorize_series(s, pct=winsorize_pct)
        z = z_score_series(s)
        df[f"z_{factor}"] = z
        z_cols[factor] = f"z_{factor}"

    composite = pd.Series(np.zeros(len(df)), index=df.index)
    for factor, z_col in z_cols.items():
        w = weights.get(factor, 0.0)
        composite += w * df[z_col]

    df["composite"] = composite
    df = df.sort_values("composite", ascending=False).reset_index(drop=True)

    top_n = cfg["output"]["top_n"]
    result = df.head(top_n)
    print(f"[compose] {len(df)} ranked → top {len(result)} selected")
    return result
