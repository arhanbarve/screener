"""Filing-edge screen orchestrator ("Lazy Prices" on a neglected universe).

Standalone pipeline — runs independently from the momentum screen (src/run.py).
Produces output/filing_edge_{date}.csv and output/filing_edge_{date}.md.

Pipeline:
  1. Universe: load existing parquet (rebuilt by main screen or --force-universe)
  2. Prices: reuse yfinance cache; apply neglect gate ($50M–$2B, ≥$200K ADV)
  3. EDGAR: fetch gp_assets for quality gate (reuses existing 30-day cache)
  4. Filings: fetch 10-K text-stability score per ticker (permanent cache)
  5. Compose: build standalone lazy composite; split longs vs watch list
  6. Output: write CSV + markdown

Run as:
    python -m src.lazy_run [--force-universe] [--limit N]
"""
import logging
import os
import time
import argparse
from datetime import date

import pandas as pd

from src.config import load_config
from src.cache import init_db, archive_filing_edge_snapshot
from src.universe import build_universe
from src.prices import fetch_all_prices
from src.fundamentals import fetch_edgar
from src.filings import compute_filing_similarity
from src.lazy_compose import build_lazy_composite
from src.filing_analysis import attach_filing_analysis, attach_8k_analysis
from src.output import write_filing_edge_csv, write_filing_edge_markdown

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DB_PATH       = "data/cache.db"
UNIVERSE_PATH = "data/universe.parquet"
OUTPUT_DIR    = "output"


def _apply_neglect_gate(factors_df: pd.DataFrame, lp_cfg: dict, min_price: float = 0.0) -> pd.DataFrame:
    """Apply the filing-edge universe gate (replaces main screen's liquidity gate).

    `min_price` (config: universe.min_price) was declared in config.yaml but
    never actually enforced anywhere — sub-penny microcaps with high share
    counts can clear the $50M-$2B market-cap band and $200K ADV floor on
    volume alone, and their extreme % price swings (reverse splits, thin-float
    pumps) dominate any downstream return statistics. Enforced here.
    """
    min_cap = lp_cfg["min_market_cap"]
    max_cap = lp_cfg["max_market_cap"]
    min_vol = lp_cfg["min_avg_dollar_vol_20d"]
    before = len(factors_df)
    result = factors_df[
        (factors_df["market_cap"] >= min_cap) &
        (factors_df["market_cap"] <= max_cap) &
        (factors_df["avg_dollar_vol_20d"] >= min_vol) &
        (factors_df["price"] >= min_price)
    ].reset_index(drop=True)
    logger.info(f"[neglect_gate] {before} → {len(result)} in $50M–$2B band (ADV≥$200K, price≥${min_price})")
    return result


def run(force_universe: bool = False, limit: int | None = None):
    cfg = load_config()
    lp  = cfg["lazy_prices"]
    today = date.today().isoformat()

    os.makedirs("data", exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    init_db(DB_PATH)

    # --- Stage 1: Universe ---
    if force_universe or not os.path.exists(UNIVERSE_PATH):
        universe_df = build_universe(cfg, UNIVERSE_PATH)
    else:
        universe_df = pd.read_parquet(UNIVERSE_PATH)
        logger.info(f"[universe] loaded {len(universe_df)} tickers from cache")

    # --- Stage 2: Prices (reuse existing cache; apply neglect gate) ---
    # fetch_all_prices applies the *main* liquidity gate internally, so we pass
    # a modified config that sets the gate to match our lower floor, then
    # re-gate for the upper cap in _apply_neglect_gate.
    price_cfg = {
        **cfg,
        "liquidity_gate": {
            "min_market_cap": lp["min_market_cap"],
            "min_avg_dollar_vol_20d": lp["min_avg_dollar_vol_20d"],
        },
    }
    _, factors_df = fetch_all_prices(universe_df, price_cfg, DB_PATH)
    neglect_df = _apply_neglect_gate(factors_df, lp, min_price=cfg["universe"].get("min_price", 0.0))

    if len(neglect_df) == 0:
        logger.info("[lazy_run] No survivors after neglect gate — aborting")
        return

    # Optional: limit universe size for faster dev/test runs
    if limit is not None:
        neglect_df = neglect_df.head(limit)
        logger.info(f"[lazy_run] --limit {limit}: truncated to {len(neglect_df)} tickers")

    # --- Stage 3: EDGAR gp_assets for quality gate ---
    ttl_edgar = cfg["cache"]["edgar_ttl_days"]
    gp_rows = []
    total = len(neglect_df)
    logger.info(f"[lazy_run] fetching EDGAR gp_assets for {total} tickers")
    for i, (_, row) in enumerate(neglect_df.iterrows()):
        ticker = row["ticker"]
        cik = str(row.get("cik", "")) if pd.notna(row.get("cik")) else ""
        gp = float("nan")
        if cik:
            edgar = fetch_edgar(cik, DB_PATH, ttl_days=ttl_edgar)
            if edgar:
                gp = edgar.get("gp_assets", float("nan"))
        gp_rows.append({"ticker": ticker, "gp_assets": gp})
        if (i + 1) % 50 == 0:
            logger.info(f"[lazy_run] edgar {i+1}/{total}")

    gp_df = pd.DataFrame(gp_rows)
    merged = neglect_df.merge(gp_df, on="ticker", how="left", suffixes=("", "_edgar"))
    if "gp_assets_edgar" in merged.columns:
        merged["gp_assets"] = merged["gp_assets_edgar"].fillna(merged.get("gp_assets", float("nan")))
        merged.drop(columns=["gp_assets_edgar"], inplace=True)

    # --- Stage 4: Filing similarity ---
    # Run every configured form (10-K + 10-Q) and keep the MOST RECENT filing per
    # ticker. compute_filing_similarity caches per accession, so the extra form is
    # cheap on repeat runs and reuses the same (cached) submissions fetch.
    filing_forms = lp.get("filing_forms") or [lp.get("filing_form", "10-K")]
    filing_ttl  = lp.get("filing_ttl_hours", 24)
    filing_rows = []
    logger.info(f"[lazy_run] fetching {filing_forms} similarities for {len(merged)} tickers")
    total = len(merged)
    for i, (_, row) in enumerate(merged.iterrows()):
        ticker = row["ticker"]
        cik = str(row.get("cik", "")) if pd.notna(row.get("cik")) else ""
        best = None
        best_key = ""
        if cik:
            for form in filing_forms:
                try:
                    result = compute_filing_similarity(cik, DB_PATH, form=form,
                                                       ttl_hours=filing_ttl)
                except Exception as e:
                    logger.warning(f"[lazy_run] filing similarity failed for {ticker} {form}: {e}")
                    result = None
                if not result:
                    continue
                # Pick the latest filing; filing_date preferred, report_date fallback.
                key = str(result.get("filing_date") or result.get("report_date") or "")
                if best is None or key > best_key:
                    best = result
                    best_key = key
        if best:
            filing_rows.append({"ticker": ticker, **best})
        else:
            filing_rows.append({"ticker": ticker, "text_stability": float("nan")})
        if (i + 1) % 25 == 0:
            logger.info(f"[lazy_run] filings {i+1}/{total}")

    filing_df = pd.DataFrame(filing_rows)
    # Drop heavy text columns before merge (too large for CSV). filing_date and
    # risk_growth are non-text scalar keys and are deliberately KEPT so they flow
    # through onto full_df for the downstream composite/recency layers.
    text_cols = ["risk_text", "prior_risk_text", "mda_text", "prior_mda_text"]
    filing_df.drop(columns=[c for c in text_cols if c in filing_df.columns], inplace=True)
    full_df = merged.merge(filing_df, on="ticker", how="left")

    # Drop tickers with no filing data at all
    has_filing = full_df["text_stability"].notna()
    logger.info(f"[lazy_run] {has_filing.sum()}/{len(full_df)} tickers have filing similarity")
    full_df = full_df[has_filing].reset_index(drop=True)

    if len(full_df) == 0:
        logger.info("[lazy_run] No tickers with filing data — aborting")
        return

    # Merge sector from fundamentals cache if missing
    if "sector" not in full_df.columns or full_df["sector"].isna().all():
        full_df["sector"] = ""

    # --- Stage 4.5: Claude precision layer (conditional on ANTHROPIC_API_KEY) ---
    if os.environ.get("ANTHROPIC_API_KEY"):
        full_df = attach_filing_analysis(full_df, cfg, DB_PATH)
    else:
        logger.info("[lazy_run] ANTHROPIC_API_KEY not set — skipping Claude change characterization")
        full_df["change_direction"] = 0

    # --- Stage 4.75: 8-K veto layer (recent material negative events) ---
    # Populates eight_k_penalty + red_flags. No-ops (0 / []) when the classifier
    # is disabled or ANTHROPIC_API_KEY is absent.
    full_df = attach_8k_analysis(full_df, cfg, DB_PATH)

    # --- Stage 5: Composite ---
    longs_df, watch_df = build_lazy_composite(full_df, cfg)

    # --- Stage 5.5: Archive point-in-time snapshot for backtest validation ---
    archive_filing_edge_snapshot(DB_PATH, today, longs_df, watch_df)
    logger.info(f"[lazy_run] archived {len(longs_df)} longs + {len(watch_df)} watch to filing_edge_snapshots")

    # --- Stage 6: Output ---
    csv_path = write_filing_edge_csv(longs_df, watch_df, OUTPUT_DIR, today)
    md_path  = write_filing_edge_markdown(longs_df, watch_df, OUTPUT_DIR, today)
    logger.info(f"[output] {csv_path}")
    logger.info(f"[output] {md_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the filing-edge screener")
    parser.add_argument("--force-universe", action="store_true",
                        help="Re-fetch universe from SEC even if parquet exists")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit tickers processed (for testing)")
    args = parser.parse_args()
    run(force_universe=args.force_universe, limit=args.limit)
