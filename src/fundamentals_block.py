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
