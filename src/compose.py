import logging
import numpy as np
import pandas as pd
from src.factors import tech_signal_score

logger = logging.getLogger(__name__)

COMPOSITE_FACTORS = ["mom_12_1", "rev_breadth", "sue", "rs_6m", "rs_slope", "tech_score"]


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


def _attach_tech_score(df: pd.DataFrame) -> pd.DataFrame:
    df["tech_score"] = df.apply(tech_signal_score, axis=1)
    return df


def _attach_streak_bonus(df: pd.DataFrame, streak_data: dict, lookback_days: int) -> pd.DataFrame:
    counts, consecutives = [], []
    for ticker in df["ticker"]:
        info = streak_data.get(str(ticker), {})
        counts.append(info.get("count", 0))
        consecutives.append(info.get("consecutive", 0))
    df["streak_count"] = counts
    df["streak_consecutive"] = consecutives
    df["streak_bonus"] = [c / max(lookback_days, 1) for c in counts]
    return df


def compute_conviction(df: pd.DataFrame) -> pd.DataFrame:
    gp_median = df["gp_assets"].median() if "gp_assets" in df.columns else float("nan")

    rank_comp, streak_comp, tech_comp, fund_comp = [], [], [], []

    for pos, (_, row) in enumerate(df.iterrows()):
        # Rank (0–3): position in already-sorted top-N slice
        if pos < 3:
            rank_comp.append(3)
        elif pos < 5:
            rank_comp.append(2)
        elif pos < 10:
            rank_comp.append(1)
        else:
            rank_comp.append(0)

        # Streak (0–3)
        cons = int(row.get("streak_consecutive", 0) or 0)
        if cons >= 7:
            streak_comp.append(3)
        elif cons >= 4:
            streak_comp.append(2)
        elif cons >= 2:
            streak_comp.append(1)
        else:
            streak_comp.append(0)

        # Technical alignment (0–2)
        ts = int(row.get("tech_score", 0) or 0)
        if ts >= 6:
            tech_comp.append(2)
        elif ts >= 4:
            tech_comp.append(1)
        else:
            tech_comp.append(0)

        # Fundamental quality (0–2)
        fc = 0.0
        gp = row.get("gp_assets")
        if pd.notna(gp) and pd.notna(gp_median) and float(gp) > float(gp_median):
            fc += 1.0
        insider = row.get("insider_buys_90d", 0) or 0
        if insider > 0:
            fc += 0.5
        sf = row.get("short_float")
        if pd.notna(sf) and float(sf) < 0.15:
            fc += 0.5
        fund_comp.append(fc)

    raw = (
        pd.Series(rank_comp, dtype=float)
        + pd.Series(streak_comp, dtype=float)
        + pd.Series(tech_comp, dtype=float)
        + pd.Series(fund_comp, dtype=float)
    )
    df = df.copy()
    df["conviction"] = raw.clip(1, 10).round().astype(int).values
    return df


def build_composite(
    factors_df: pd.DataFrame,
    cfg: dict,
    streak_data: dict | None = None,
) -> pd.DataFrame:
    df = factors_df.copy()
    weights = cfg["factors"]["weights"]
    winsorize_pct = cfg["factors"]["winsorize_pct"]
    missing_treatment = cfg["factors"].get("missing_factor_treatment", "neutral")
    lookback_days = cfg.get("streak", {}).get("lookback_days", 14)

    df = apply_quality_gate(df, cfg)
    df = apply_confirmation_gate(df, cfg)

    df = _attach_tech_score(df)
    df = _attach_streak_bonus(df, streak_data or {}, lookback_days)

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

    # Streak bonus: bounded additive (max = streak_weight), bypasses z-scoring
    streak_weight = weights.get("streak_bonus", 0.05)
    composite += streak_weight * df["streak_bonus"].fillna(0.0)

    df["composite"] = composite
    df = df.sort_values("composite", ascending=False).reset_index(drop=True)

    top_n = cfg["output"]["top_n"]
    result = df.head(top_n).reset_index(drop=True)
    result = compute_conviction(result)
    print(f"[compose] {len(df)} ranked → top {len(result)} selected")
    return result
