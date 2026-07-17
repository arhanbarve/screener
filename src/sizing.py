"""Position sizing + concentration caps for the screener's top-N picks.

Pure functions, no I/O. Inverse-volatility weights, then iterative
redistribution to enforce a per-name cap and a per-GICS-sector cap.
Correctness is arithmetic (caps hold, weights sum to 1.0), so this needs no
backtest to validate. See
docs/superpowers/specs/2026-07-16-position-sizing-concentration-caps-design.md
"""
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADING_DAYS = 252
_MAX_ITERS = 100


def realized_vol(close_series: pd.Series, window: int = 63) -> float:
    """Annualized std of daily log-returns over the trailing `window` sessions.

    Returns NaN if fewer than `window`+1 valid closes are available."""
    close = pd.Series(close_series).dropna()
    if len(close) < window + 1:
        return float("nan")
    rets = np.log(close / close.shift(1)).dropna().iloc[-window:]
    if len(rets) < window:
        return float("nan")
    return float(rets.std(ddof=1) * np.sqrt(TRADING_DAYS))


def inverse_vol_weights(vols: dict[str, float]) -> dict[str, float]:
    """w_i = (1/vol_i) / Σ(1/vol_j). NaN / non-positive vols are dropped
    (they cannot be sized). Result sums to 1.0 over the remaining names."""
    inv = {t: 1.0 / v for t, v in vols.items()
           if v is not None and np.isfinite(v) and v > 0.0}
    total = sum(inv.values())
    if total <= 0.0:
        return {}
    return {t: x / total for t, x in inv.items()}


def apply_caps(
    weights: dict[str, float],
    sectors: dict[str, str],
    name_cap: float,
    sector_cap: float,
) -> dict[str, float]:
    """Iterative water-fill: clip over-cap names, then over-cap sectors,
    redistributing freed weight pro-rata to names below both their caps.
    Repeats to convergence. Result sums to 1.0. Best-effort + warning when a
    constraint is infeasible (e.g. a single-sector book under a sector cap)."""
    if not weights:
        return {}
    w = dict(weights)

    for _ in range(_MAX_ITERS):
        moved_n = _enforce_name_cap(w, name_cap)
        moved_s = _enforce_sector_cap(w, sectors, name_cap, sector_cap)
        if not (moved_n or moved_s):
            break

    _warn_if_violated(w, sectors, name_cap, sector_cap)
    return w


def _redistribute(
    w: dict[str, float], donors: dict[str, float], recipients: dict[str, float],
) -> bool:
    """Move `moveable = min(donor-excess, recipient-room)` from donors to
    recipients, each proportional to its own excess / room. Sum-preserving.
    Returns True if any weight moved."""
    excess = sum(donors.values())
    room = sum(recipients.values())
    moveable = min(excess, room)
    if moveable <= 1e-12:
        return False
    for t, ex in donors.items():
        w[t] -= moveable * (ex / excess)
    for t, r in recipients.items():
        w[t] += moveable * (r / room)
    return True


def _enforce_name_cap(w: dict[str, float], name_cap: float) -> bool:
    donors = {t: v - name_cap for t, v in w.items() if v > name_cap + 1e-12}
    if not donors:
        return False
    recipients = {t: name_cap - v for t, v in w.items()
                  if v < name_cap - 1e-12 and t not in donors}
    return _redistribute(w, donors, recipients)


def _enforce_sector_cap(
    w: dict[str, float], sectors: dict[str, str],
    name_cap: float, sector_cap: float,
) -> bool:
    """One pass of sector capping. Only moves weight that non-over sectors have
    room to absorb (best-effort when infeasible, e.g. a single-sector book)."""
    sec_tot: dict[str, float] = {}
    for t, v in w.items():
        s = sectors.get(t, "Unknown")
        sec_tot[s] = sec_tot.get(s, 0.0) + v
    over_secs = {s: tot for s, tot in sec_tot.items() if tot > sector_cap + 1e-12}
    if not over_secs:
        return False

    # donor weight per name = its share of the sector's excess
    donors = {}
    for s, tot in over_secs.items():
        sec_excess = tot - sector_cap
        for t in [x for x in w if sectors.get(x, "Unknown") == s]:
            donors[t] = sec_excess * (w[t] / tot)
    # recipients: names in non-over sectors, below their name cap
    recipients = {t: name_cap - w[t] for t in w
                  if sectors.get(t, "Unknown") not in over_secs
                  and w[t] < name_cap - 1e-12}
    return _redistribute(w, donors, recipients)


def _warn_if_violated(
    w: dict[str, float], sectors: dict[str, str],
    name_cap: float, sector_cap: float,
) -> None:
    if w and max(w.values()) > name_cap + 1e-6:
        logger.warning(f"[sizing] name cap {name_cap:.0%} not fully met "
                       f"(max {max(w.values()):.1%}) — infeasible config")
    sec_tot: dict[str, float] = {}
    for t, v in w.items():
        s = sectors.get(t, "Unknown")
        sec_tot[s] = sec_tot.get(s, 0.0) + v
    if sec_tot and max(sec_tot.values()) > sector_cap + 1e-6:
        worst = max(sec_tot, key=sec_tot.get)
        logger.warning(f"[sizing] sector cap {sector_cap:.0%} not met for "
                       f"'{worst}' ({sec_tot[worst]:.1%}) — too few other "
                       f"sectors to redistribute into (best-effort)")


def attach_weights(ranked_df, cfg):
    """Add a `weight_pct` column (inverse-vol, cap-constrained) to the ranked
    top-N. No-op when sizing is disabled, the frame is empty, or the required
    `close_series`/`sector` columns are absent."""
    sizing = cfg.get("sizing", {})
    if not sizing.get("enabled", False):
        return ranked_df
    if len(ranked_df) == 0 or "close_series" not in ranked_df.columns:
        return ranked_df

    window = sizing.get("vol_window", 63)
    name_cap = sizing.get("name_cap", 0.10)
    sector_cap = sizing.get("sector_cap", 0.25)

    vols = {}
    sectors = {}
    for _, row in ranked_df.iterrows():
        t = row["ticker"]
        vols[t] = realized_vol(row["close_series"], window=window)
        sectors[t] = str(row.get("sector", "") or "Unknown")

    raw = inverse_vol_weights(vols)
    capped = apply_caps(raw, sectors, name_cap, sector_cap)

    df = ranked_df.copy()
    df["weight_pct"] = df["ticker"].map(
        lambda t: round(capped.get(t, 0.0) * 100.0, 4)
    )
    return df
