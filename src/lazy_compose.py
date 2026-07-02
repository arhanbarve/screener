"""Standalone composite for the filing-edge ("Lazy Prices") screen.

Three-factor model:
  text_stability  (0.55) — cosine similarity vs prior comparable filing;
                            high = language unchanged = bullish
  gp_assets       (0.25) — gross profitability quality gate (avoid value traps)
  neglect_score   (0.20) — inverse rank of dollar volume within the neglect band
                            (low dollar-vol = less covered = more potential edge)

All factors z-scored cross-sectionally. Reuses helpers from compose.py.
"""
import logging
from datetime import date
import numpy as np
import pandas as pd
from src.compose import (
    winsorize_series,
    z_score_series,
    rank_normalize,
    FINANCIAL_SECTORS,
)

logger = logging.getLogger(__name__)

FILING_FACTORS = ["text_stability", "gp_assets", "neglect_score", "risk_growth"]


def _compute_filing_age_days(df: pd.DataFrame) -> pd.Series:
    """Days between filing_date (ISO) and today. Unparseable/missing → 9999."""
    today = date.today()

    def _age(val) -> int:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return 9999
        try:
            return (today - date.fromisoformat(str(val)[:10])).days
        except (ValueError, TypeError):
            return 9999

    if "filing_date" not in df.columns:
        return pd.Series([9999] * len(df), index=df.index)
    return df["filing_date"].apply(_age)


def _apply_neglect_quality_gate(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Exclude bottom-quartile gross profitability (non-financial sectors only).
    Financials / REITs pass unconditionally (gp_assets meaningless for them).
    """
    if "gp_assets" not in df.columns:
        return df
    gp = df["gp_assets"].fillna(-999)
    sector = df.get("sector", pd.Series([""] * len(df), index=df.index)).fillna("")
    fin_mask = sector.isin(FINANCIAL_SECTORS)
    cutoff = gp[~fin_mask].quantile(0.25)
    passes = fin_mask | (gp >= cutoff)
    before = len(df)
    result = df[passes].reset_index(drop=True)
    print(f"[lazy_quality_gate] {before} → {len(result)} survivors (gp_assets >= {cutoff:.4f})")
    return result


def _compute_neglect_score(df: pd.DataFrame) -> pd.DataFrame:
    """Neglect = inverse percentile rank of dollar volume within the neglect band.
    Low dollar-vol → high rank → more neglected → positive tilt.
    """
    if "avg_dollar_vol_20d" not in df.columns:
        df["neglect_score"] = 0.0
        return df
    vol = df["avg_dollar_vol_20d"].fillna(df["avg_dollar_vol_20d"].median())
    # Invert: low-vol stocks get high neglect score
    df["neglect_score"] = 1.0 - vol.rank(pct=True)
    return df


def build_lazy_composite(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute filing-edge composite. Returns (ranked_longs, watch_list).

    ranked_longs  — top_n by composite (stable-filing, neglected, profitable)
    watch_list    — watch_list_n names with lowest text_stability (deteriorating filers)
    """
    lp = cfg["lazy_prices"]
    weights = lp["weights"]
    winsorize_pct = lp.get("winsorize_pct", 0.02)
    min_coverage = lp.get("min_factor_coverage", 0.30)
    top_n = lp.get("top_n", 30)
    watch_n = lp.get("watch_list_n", 20)
    max_filing_age_days = lp.get("max_filing_age_days", 9999)
    recency_halflife_days = lp.get("recency_halflife_days", 30)

    df = df.copy()
    df = _apply_neglect_quality_gate(df, cfg)
    df = _compute_neglect_score(df)

    if len(df) == 0:
        return df, df

    # Coverage filter: stock must have data for factors summing ≥ min_coverage weight
    total_w = sum(weights.get(f, 0.0) for f in FILING_FACTORS)
    coverage = pd.Series(0.0, index=df.index)
    for f in FILING_FACTORS:
        if f in df.columns:
            coverage += df[f].notna() * weights.get(f, 0.0)
    df["factor_coverage"] = coverage / max(total_w, 1e-12)
    before = len(df)
    df = df[df["factor_coverage"] >= min_coverage].reset_index(drop=True)
    if len(df) < before:
        print(f"[lazy_coverage] {before} → {len(df)} (min {min_coverage:.0%})")

    # Z-score each factor
    composite = pd.Series(np.zeros(len(df)), index=df.index)
    for f in FILING_FACTORS:
        w = weights.get(f, 0.0)
        if f not in df.columns:
            logger.warning(f"[lazy_compose] factor {f} missing")
            df[f"z_{f}"] = 0.0
            continue
        s = df[f].astype(float)
        if f == "risk_growth":
            # Negative economic direction: higher risk_growth = worse.
            # Negate before winsorize/z so positive weight penalizes it.
            s = -s.fillna(0.0)
        else:
            fill = s.mean()
            s = s.fillna(0.0 if pd.isna(fill) else fill)
        s = winsorize_series(s, pct=winsorize_pct)
        z = z_score_series(s)
        df[f"z_{f}"] = z
        composite += w * z

    df["composite"] = composite
    df = df.sort_values("composite", ascending=False).reset_index(drop=True)

    # Recency: age-decayed composite for long ranking (raw composite kept intact).
    df["filing_age_days"] = _compute_filing_age_days(df)
    hl = max(recency_halflife_days, 1e-9)
    df["composite_decayed"] = df["composite"] * np.power(0.5, df["filing_age_days"] / hl)

    # Conviction: cross-sectional percentile of text_stability (1-5 scale)
    stab_col = df.get("text_stability", pd.Series(np.zeros(len(df)), index=df.index))
    stab_pct = stab_col.rank(pct=True)

    def _conviction(pos: int, pct: float) -> int:
        rank_pts = 3 if pos < 3 else (2 if pos < 8 else (1 if pos < 15 else 0))
        stab_pts = 2 if pct >= 0.80 else (1 if pct >= 0.50 else 0)
        return max(1, min(5, rank_pts + stab_pts))

    df["conviction"] = [
        _conviction(i, float(stab_pct.iloc[i]) if pd.notna(stab_pct.iloc[i]) else 0.0)
        for i in range(len(df))
    ]

    # Long selection: exclude negative changers and 8-K red flags, gate stale
    # filings, then rank by age-decayed composite.
    change_dir = df.get("change_direction", pd.Series([0] * len(df), index=df.index)).fillna(0)
    eight_k = df.get("eight_k_penalty", pd.Series([0] * len(df), index=df.index)).fillna(0)
    long_mask = (
        (change_dir != -1)
        & (eight_k != -1)
        & (df["filing_age_days"] <= max_filing_age_days)
    )
    ranked_longs = (
        df[long_mask]
        .sort_values("composite_decayed", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    # Watch list (not age-gated): confirmed negative changers first, then lowest
    # text_stability. sort key = (change_direction != -1, text_stability) asc.
    watch = df[df["text_stability"].notna()].copy()
    watch_cd = watch.get("change_direction", pd.Series([0] * len(watch), index=watch.index)).fillna(0)
    watch["_cd_not_neg"] = (watch_cd != -1).astype(int)
    watch_list = (
        watch.sort_values(["_cd_not_neg", "text_stability"], ascending=[True, True])
        .drop(columns=["_cd_not_neg"])
        .head(watch_n)
        .reset_index(drop=True)
    )

    print(f"[lazy_compose] {len(df)} scored → {len(ranked_longs)} longs, {len(watch_list)} watch")
    return ranked_longs, watch_list
