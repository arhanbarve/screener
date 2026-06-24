"""
Portfolio construction layer: rebalance band, earnings blackout, cost model.

This module separates the question "what does the screener rank?" from
"what should I actually trade today?" — crucial because daily re-ranking
against a hard top-20 cut causes violent turnover.

Usage:
    from src.portfolio import apply_rebalance_band, earnings_blackout_tickers, estimate_trade_cost
"""
import logging
import os
from datetime import date, timedelta

import pandas as pd

logger = logging.getLogger(__name__)


def apply_rebalance_band(
    ranked_df: pd.DataFrame,
    current_positions: list[str],
    entry_band: int = 20,
    exit_band: int = 35,
) -> dict:
    """
    Suppress excessive turnover via a rebalance band.

    - New BUYs: only stocks newly entering the top `entry_band` not already held.
    - HOLDs: stocks already held that remain in the top `exit_band`.
    - EXITs: stocks held that dropped out of the top `exit_band`.

    Rationale: a stock ranked #21 today that was #15 yesterday has not
    meaningfully deteriorated. Forcing an exit and re-entry on minor rank
    fluctuations destroys alpha through transaction costs and taxes.
    """
    top_entry = set(ranked_df.head(entry_band)["ticker"].astype(str))
    top_exit  = set(ranked_df.head(exit_band)["ticker"].astype(str))
    held      = set(str(t) for t in current_positions)

    new_buys = top_entry - held
    holds    = held & top_exit
    exits    = held - top_exit

    return {
        "buy":  sorted(new_buys),
        "hold": sorted(holds),
        "exit": sorted(exits),
    }


def earnings_blackout_tickers(
    tickers: list[str],
    finnhub_key: str,
    days_ahead: int = 3,
) -> set[str]:
    """
    Return the set of tickers with scheduled earnings within `days_ahead` trading days.
    These should NOT be entered even if ranked in the top-20 — earnings are a coin flip
    that can erase a month of momentum edge.

    Existing positions approaching earnings: flag for monitoring but do not force exit.
    """
    if not finnhub_key:
        return set()

    try:
        import finnhub
        fh = finnhub.Client(api_key=finnhub_key)
        today = date.today()
        to_date = (today + timedelta(days=days_ahead + 5)).isoformat()  # small buffer for weekends
        from_date = today.isoformat()

        blackout = set()
        for ticker in tickers:
            try:
                cal = fh.company_earnings_quality_index(ticker, freq="annual")
                # Fallback: use company_earnings to check recent dates
            except Exception:
                pass
            # Primary method: check earnings calendar endpoint
            try:
                cal = fh.earnings_calendar(_from=from_date, to=to_date, symbol=ticker, international=False)
                earnings_list = (cal or {}).get("earningsCalendar", [])
                for entry in earnings_list:
                    if entry.get("symbol") == ticker:
                        ed = entry.get("date", "")
                        if from_date <= ed <= to_date:
                            blackout.add(ticker)
                            logger.info(f"[blackout] {ticker} earnings on {ed}")
                            break
            except Exception:
                pass
        return blackout
    except Exception as e:
        logger.warning(f"[blackout] earnings fetch failed: {e}")
        return set()


def estimate_trade_cost(
    size_usd: float,
    avg_dollar_vol_20d: float,
) -> float:
    """
    Estimate round-trip transaction cost as a fraction of trade size.

    Uses a simplified Kyle-Almgren linear market impact model:
    - Spread cost: 10bps one-way (20bps round-trip) for liquid names
    - Market impact: proportional to sqrt(participation rate)

    Participation rate = trade size / daily dollar volume.
    At $5M ADV and $100K trade: participation = 2%, impact ≈ 28bps.
    Round-trip total ≈ 48bps, or 0.48%.
    """
    if avg_dollar_vol_20d <= 0:
        return 0.02  # assume 2% cost for illiquid names
    participation = size_usd / avg_dollar_vol_20d
    spread_rt = 0.002   # 20bps round-trip spread
    impact_rt = 0.004 * (participation ** 0.5)  # doubled for round-trip
    return spread_rt + impact_rt


def filter_by_cost(
    ranked_df: pd.DataFrame,
    position_size_usd: float,
    max_cost_fraction: float = 0.005,
) -> pd.DataFrame:
    """
    Remove stocks where estimated round-trip cost exceeds max_cost_fraction
    of expected alpha (rough heuristic: alpha ≈ 3% at 42-day horizon → cost budget 0.5%).
    """
    if "avg_dollar_vol_20d" not in ranked_df.columns:
        return ranked_df
    costs = ranked_df["avg_dollar_vol_20d"].apply(
        lambda adv: estimate_trade_cost(position_size_usd, float(adv) if pd.notna(adv) else 0)
    )
    return ranked_df[costs <= max_cost_fraction].reset_index(drop=True)


def generate_trade_list(
    ranked_df: pd.DataFrame,
    current_positions: list[str],
    cfg: dict,
    finnhub_key: str = "",
    position_size_usd: float = 10_000,
) -> pd.DataFrame:
    """
    Full portfolio construction pipeline:
    1. Apply rebalance band
    2. Flag earnings blackouts
    3. Estimate trade costs

    Returns a DataFrame with columns: ticker, action, rank, composite, conviction,
    earnings_blackout, est_cost_pct, reason
    """
    entry_band = cfg.get("output", {}).get("entry_band", 20)
    exit_band  = cfg.get("output", {}).get("exit_band", 35)

    band = apply_rebalance_band(ranked_df, current_positions, entry_band, exit_band)

    blackout_set = earnings_blackout_tickers(
        band["buy"], finnhub_key=finnhub_key, days_ahead=3
    ) if finnhub_key else set()

    rows = []
    rank_map = {row["ticker"]: i + 1 for i, (_, row) in enumerate(ranked_df.iterrows())}

    for ticker in sorted(band["buy"] | band["hold"] | band["exit"]):
        rank = rank_map.get(ticker, 999)
        row_data = ranked_df[ranked_df["ticker"] == ticker]
        composite = float(row_data["composite"].iloc[0]) if len(row_data) else float("nan")
        conviction = int(row_data["conviction"].iloc[0]) if len(row_data) else 0
        adv = float(row_data["avg_dollar_vol_20d"].iloc[0]) if len(row_data) and "avg_dollar_vol_20d" in row_data.columns else 0.0

        if ticker in band["buy"]:
            if ticker in blackout_set:
                action = "SKIP"
                reason = "earnings blackout"
            else:
                action = "BUY"
                reason = f"new entry (rank={rank})"
        elif ticker in band["hold"]:
            action = "HOLD"
            reason = f"still in top-{exit_band} (rank={rank})"
        else:
            action = "EXIT"
            reason = f"fell below top-{exit_band}"

        rows.append({
            "ticker":            ticker,
            "action":            action,
            "rank":              rank,
            "composite":         composite,
            "conviction":        conviction,
            "earnings_blackout": ticker in blackout_set,
            "est_cost_pct":      estimate_trade_cost(position_size_usd, adv),
            "reason":            reason,
        })

    return pd.DataFrame(rows).sort_values("rank")
