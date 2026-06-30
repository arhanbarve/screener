import logging
import numpy as np
import pandas as pd
from scipy.stats import norm as _norm
from src.factors import tech_signal_score, trend_signal_score, oscillator_signal_score, volume_signal_score

logger = logging.getLogger(__name__)

# Economic factor blocks — all flow through winsorize→z-score (or rank_normalize for bounded).
# streak_z and st_reversal removed: streak_z was circular (read prior top-20 CSVs that
# depended on streak_z), st_reversal was a minor penalty included in the price block.
# Both remain as diagnostic CSV columns and feed conviction.
# residual_mom and pct_from_high are new additions.
COMPOSITE_FACTORS = [
    "mom_12_1", "residual_mom", "rs_6m", "rs_accel", "rs_slope", "pct_from_high",
    "sue", "rev_breadth", "rev_magnitude",
    "gp_assets", "insider_z",
    "trend_score", "momo_osc_score", "volume_score",
]

# Factors that are bounded integers / zero-inflated — use rank normalization
# instead of winsorize+z-score to avoid distributional distortion.
RANK_NORMALIZE_FACTORS = {"insider_z", "trend_score", "momo_osc_score", "volume_score"}

# Sectors where gp_assets = (revenue-COGS)/assets is meaningless or misleading.
# These stocks pass the quality gate unconditionally and get NaN for gp_assets factor.
FINANCIAL_SECTORS = {
    "Financial Services", "Financial", "Financials",
    "Real Estate", "Banks", "Insurance",
    "Asset Management", "Mortgage Finance",
}


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


def rank_normalize(s: pd.Series) -> pd.Series:
    """Cross-sectional rank transform → N(0,1) via normal quantile.
    Correct for bounded integers and zero-inflated distributions
    where z-scoring produces distorted scores.
    """
    ranks = s.rank(pct=True, na_option="keep").fillna(0.5)
    ranks = ranks.clip(0.01, 0.99)
    return pd.Series(_norm.ppf(ranks.values), index=s.index)


def apply_quality_gate(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    threshold = cfg["quality_gate"]["gross_profitability_min"]
    if "gp_assets" not in df.columns:
        return df
    gp = df["gp_assets"].fillna(-999)
    sector = df.get("sector", pd.Series([""] * len(df), index=df.index)).fillna("")

    if threshold == "median":
        non_fin = gp[~sector.isin(FINANCIAL_SECTORS)]
        cutoff = non_fin.median() if len(non_fin) > 0 else 0.0
    elif threshold == "q25":
        non_fin = gp[~sector.isin(FINANCIAL_SECTORS)]
        cutoff = non_fin.quantile(0.25) if len(non_fin) > 0 else 0.0
    else:
        cutoff = float(threshold)

    # Financial/REIT sectors pass unconditionally (gp_assets is meaningless for them)
    financial_mask = sector.isin(FINANCIAL_SECTORS)
    passes = financial_mask | (gp >= cutoff)

    before = len(df)
    result = df[passes].reset_index(drop=True)
    logger.info(f"[quality_gate] {before} → {len(result)} survivors")
    print(f"[quality_gate] {before} → {len(result)} survivors (gp_assets >= {cutoff:.4f}, financials bypass)")
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


def _attach_tech_scores(df: pd.DataFrame) -> pd.DataFrame:
    df["tech_score"]     = df.apply(tech_signal_score, axis=1)
    df["trend_score"]    = df.apply(trend_signal_score, axis=1)
    df["momo_osc_score"] = df.apply(oscillator_signal_score, axis=1)
    df["volume_score"]   = df.apply(volume_signal_score, axis=1)
    return df


def _attach_streak_data(df: pd.DataFrame, streak_data: dict, lookback_days: int) -> pd.DataFrame:
    """Streak columns — diagnostic only, NOT in composite (streak_z was circular)."""
    counts, consecutives = [], []
    for ticker in df["ticker"]:
        info = streak_data.get(str(ticker), {})
        counts.append(info.get("count", 0))
        consecutives.append(info.get("consecutive", 0))
    df["streak_count"]       = counts
    df["streak_consecutive"] = consecutives
    df["streak_z"]           = [cons + 0.3 * cnt for cons, cnt in zip(consecutives, counts)]
    df["streak_bonus"]       = [cnt / max(lookback_days, 1) for cnt in counts]
    return df


def _derive_new_factors(df: pd.DataFrame) -> pd.DataFrame:
    """Derive composite factors from already-computed columns."""
    # RS acceleration: positive = RS improving vs 6m pace
    if "rs_3m" in df.columns and "rs_6m" in df.columns:
        df["rs_accel"] = df["rs_3m"] * 2.0 - df["rs_6m"]
    else:
        df["rs_accel"] = 0.0

    # st_reversal: diagnostic column (spike penalty), NOT in composite
    if "mom_1m" in df.columns:
        df["st_reversal"] = -np.clip(df["mom_1m"].fillna(0) - 0.15, 0.0, None)
    else:
        df["st_reversal"] = 0.0

    # Insider buying: use exec_buys_90d (opportunistic signal) when available
    exec_col = "exec_buys_90d" if "exec_buys_90d" in df.columns else "insider_buys_90d"
    if exec_col in df.columns:
        df["insider_z"] = np.log1p(df[exec_col].fillna(0))
    else:
        df["insider_z"] = 0.0

    # Null out gp_assets for financial sectors (meaningless metric for them)
    if "gp_assets" in df.columns and "sector" in df.columns:
        fin_mask = df["sector"].fillna("").isin(FINANCIAL_SECTORS)
        df.loc[fin_mask, "gp_assets"] = float("nan")

    # residual_mom: already computed in prices.py, just ensure it exists
    if "residual_mom" not in df.columns:
        df["residual_mom"] = float("nan")

    # rev_magnitude: already from fundamentals.py
    if "rev_magnitude" not in df.columns:
        df["rev_magnitude"] = float("nan")

    return df


def _sector_demean(df: pd.DataFrame, factors: list[str]) -> pd.DataFrame:
    """Demean each factor within GICS sector before cross-sectional z-scoring.
    Prevents the composite from becoming a sector-rotation bet. Only demean
    sectors with ≥5 stocks with valid data for the factor.
    """
    MIN_SECTOR_SIZE = 5
    df = df.copy()
    sector_col = df["sector"].fillna("Unknown").replace("", "Unknown") if "sector" in df.columns \
        else pd.Series(["Unknown"] * len(df), index=df.index)

    for factor in factors:
        if factor not in df.columns:
            continue
        # Convert to float to avoid dtype assignment errors on integer columns
        col = df[factor].astype(float)
        valid = col.notna()
        if valid.sum() < MIN_SECTOR_SIZE:
            continue

        sec_series = sector_col.loc[col.index]
        sec_means  = col.groupby(sec_series).transform("mean")
        sec_counts = col.groupby(sec_series).transform("count")
        demean_mask = valid & (sec_counts >= MIN_SECTOR_SIZE)
        col_demeaned = col.copy()
        col_demeaned[demean_mask] = col[demean_mask] - sec_means[demean_mask]
        df[factor] = col_demeaned
    return df


def _apply_coverage_filter(df: pd.DataFrame, weights: dict, min_coverage: float) -> pd.DataFrame:
    """Exclude stocks missing most of their factor signal.
    Missing = NaN before fill. Coverage = sum of weights for factors with data.
    """
    total_weight = sum(weights.get(f, 0.0) for f in COMPOSITE_FACTORS)
    coverage = pd.Series(0.0, index=df.index)
    for factor in COMPOSITE_FACTORS:
        if factor in df.columns:
            has_data = df[factor].notna()
            coverage += has_data * weights.get(factor, 0.0)
    df["factor_coverage"] = coverage / max(total_weight, 1e-12)
    before = len(df)
    result = df[df["factor_coverage"] >= min_coverage].reset_index(drop=True)
    if len(result) < before:
        print(f"[coverage_filter] {before} → {len(result)} (min coverage={min_coverage:.0%})")
    return result


def _compute_factor_agreement(row: pd.Series) -> int:
    """0-4: how many independent factor blocks independently confirm this stock.
    Measures cross-block consensus — more informative than re-using composite inputs.
    """
    score = 0
    if max(row.get("z_mom_12_1", 0) or 0, row.get("z_rs_6m", 0) or 0) > 0.5:
        score += 1
    if max(row.get("z_sue", 0) or 0, row.get("z_rev_breadth", 0) or 0) > 0.5:
        score += 1
    if (row.get("z_gp_assets", 0) or 0) > 0.3:
        score += 1
    if (row.get("trend_score", 0) or 0) >= 3:
        score += 1
    return score


def compute_conviction(df: pd.DataFrame) -> pd.DataFrame:
    rank_comp, streak_comp, tech_comp, agreement_comp = [], [], [], []

    for pos, (_, row) in enumerate(df.iterrows()):
        # Rank (0-3)
        if pos < 3:
            rank_comp.append(3)
        elif pos < 5:
            rank_comp.append(2)
        elif pos < 10:
            rank_comp.append(1)
        else:
            rank_comp.append(0)

        # Streak consecutive (0-3) — kept here as it's exogenous from composite now
        cons = int(row.get("streak_consecutive", 0) or 0)
        if cons >= 7:
            streak_comp.append(3)
        elif cons >= 4:
            streak_comp.append(2)
        elif cons >= 2:
            streak_comp.append(1)
        else:
            streak_comp.append(0)

        # Trend strength (0-2)
        ts = int(row.get("trend_score", 0) or 0)
        if ts >= 4:
            tech_comp.append(2)
        elif ts >= 2:
            tech_comp.append(1)
        else:
            tech_comp.append(0)

        # Factor agreement (0-4): cross-block consensus — new, non-composite signal
        agreement_comp.append(_compute_factor_agreement(row))

    raw = (
        pd.Series(rank_comp, dtype=float)
        + pd.Series(streak_comp, dtype=float)
        + pd.Series(tech_comp, dtype=float)
        + pd.Series(agreement_comp, dtype=float)
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
    weights        = cfg["factors"]["weights"]
    winsorize_pct  = cfg["factors"]["winsorize_pct"]
    missing_treat  = cfg["factors"].get("missing_factor_treatment", "neutral")
    lookback_days  = cfg.get("streak", {}).get("lookback_days", 14)
    min_coverage   = cfg["factors"].get("min_factor_coverage", 0.40)
    rank_norm_set  = set(cfg["factors"].get("rank_normalize_factors", list(RANK_NORMALIZE_FACTORS)))

    df = apply_quality_gate(df, cfg)
    df = apply_confirmation_gate(df, cfg)

    # Attach derived and technical columns
    df = _attach_tech_scores(df)
    df = _attach_streak_data(df, streak_data or {}, lookback_days)
    df = _derive_new_factors(df)

    # Coverage filter: drop stocks missing most of their signal
    df = _apply_coverage_filter(df, weights, min_coverage)

    # Sector demean: remove sector-level mean from each factor before z-scoring
    # so composite captures stock-specific deviation, not sector rotation
    if "sector" in df.columns:
        df = _sector_demean(df, COMPOSITE_FACTORS)

    # Z-score pipeline (winsorize → z-score, or rank_normalize for bounded factors)
    z_cols = {}
    for factor in COMPOSITE_FACTORS:
        if factor not in df.columns:
            logger.warning(f"[compose] factor {factor} missing — treating as neutral")
            df[f"z_{factor}"] = 0.0
            continue
        s = df[factor].copy()
        if missing_treat == "neutral":
            fill_val = s.mean()
            s = s.fillna(0.0 if pd.isna(fill_val) else fill_val)
        else:
            df = df[s.notna()].reset_index(drop=True)
            s = df[factor]

        if factor in rank_norm_set:
            z = rank_normalize(s)
        else:
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
    result = df.head(top_n).reset_index(drop=True)
    result = compute_conviction(result)
    print(f"[compose] {len(df)} ranked → top {len(result)} selected")
    return result
