import logging
import os
from datetime import date
from src.config import load_config
from src.cache import init_db, archive_fundamentals_snapshot, archive_universe_snapshot
from src.universe import build_universe
from src.prices import fetch_all_prices
from src.fundamentals import fetch_all_fundamentals
from src.factors import squeeze_flag
from src.compose import build_composite
from src.streak import load_streak_history
from src.news import attach_news_overlay
from src.spy_analysis import compute_market_stress_overlay
from src.output import write_csv, write_markdown, print_top10
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DB_PATH       = "data/cache.db"
UNIVERSE_PATH = "data/universe.parquet"
OUTPUT_DIR    = "output"


def run(force_universe: bool = False):
    cfg = load_config()
    today = date.today().isoformat()

    os.makedirs("data", exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    init_db(DB_PATH)

    # Stage 1: Universe
    if force_universe or not os.path.exists(UNIVERSE_PATH):
        universe_df = build_universe(cfg, UNIVERSE_PATH)
    else:
        universe_df = pd.read_parquet(UNIVERSE_PATH)
        print(f"[universe] loaded {len(universe_df)} tickers from cache")

    # Stage 2: Prices + liquidity gate
    _, survivors_df = fetch_all_prices(universe_df, cfg, DB_PATH)
    print(f"[stage2] {len(universe_df)} → {len(survivors_df)} after liquidity gate")

    # Archive today's surviving universe for future point-in-time backtesting
    archive_universe_snapshot(DB_PATH, today, survivors_df)

    # Attach CIK from universe
    survivors_df = survivors_df.merge(
        universe_df[["ticker", "cik", "name"]],
        on="ticker",
        how="left",
        suffixes=("", "_uni"),
    )
    if "name_uni" in survivors_df.columns:
        survivors_df["name"] = survivors_df["name"].fillna(survivors_df["name_uni"])
        survivors_df = survivors_df.drop(columns=["name_uni"])

    # Stage 3: Fundamentals (survivors only)
    fund_df = fetch_all_fundamentals(survivors_df, cfg, DB_PATH)
    merged = survivors_df.merge(fund_df, on="ticker", how="left")
    print(f"[stage3] fundamentals fetched for {len(fund_df)} tickers")

    # Archive fundamentals snapshot for point-in-time backtesting
    archive_fundamentals_snapshot(DB_PATH, today)

    # Stage 3.5: Market stress overlay — scale top_n down in momentum-crash regimes
    stress = compute_market_stress_overlay()
    scale  = stress["scale_factor"]
    if scale < 1.0:
        original_top_n = cfg["output"]["top_n"]
        new_top_n = max(1, int(original_top_n * scale))
        cfg = dict(cfg)
        cfg["output"] = dict(cfg["output"])
        cfg["output"]["top_n"] = new_top_n
        print(f"[stress] regime={stress['regime']} scale={scale:.1f} "
              f"reason='{stress['reason']}' → top_n {original_top_n}→{new_top_n}")
    else:
        print(f"[stress] regime={stress['regime']} — full screen")

    if scale == 0.0:
        print("[stress] STRESS regime: skipping ranking, outputting empty screen")
        ranked_df = pd.DataFrame(columns=["ticker"])
        csv_path = write_csv(ranked_df, OUTPUT_DIR, today)
        print(f"\n[output] {csv_path} (empty — market stress)")
        return

    # Stage 4: Composite score
    streak_data = load_streak_history(OUTPUT_DIR, lookback_days=cfg.get("streak", {}).get("lookback_days", 14))
    ranked_df = build_composite(merged, cfg, streak_data=streak_data)

    # Stage 4.5: News overlay (entry signal + conviction adjustment)
    if cfg.get("news", {}).get("enabled", True):
        ranked_df = attach_news_overlay(ranked_df, cfg, DB_PATH)

    # Squeeze screen
    if cfg["output"].get("include_squeeze_screen", False):
        squeeze_rows = []
        for _, row in merged.iterrows():
            sf  = row.get("short_float", 0) or 0
            dtc = row.get("days_to_cover", 0) or 0
            m1  = row.get("mom_1m", 0) or 0
            if squeeze_flag(sf, dtc, m1):
                squeeze_rows.append(row)
        squeeze_df = pd.DataFrame(squeeze_rows) if squeeze_rows else None
        if squeeze_df is not None:
            print(f"[squeeze] {len(squeeze_df)} squeeze candidates")
    else:
        squeeze_df = None

    # Stage 5: Output
    csv_path = write_csv(ranked_df, OUTPUT_DIR, today)
    md_path  = write_markdown(ranked_df, OUTPUT_DIR, today, squeeze_df=squeeze_df)
    print(f"\n[output] {csv_path}")
    print(f"[output] {md_path}")
    print_top10(ranked_df)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the equity screener")
    parser.add_argument("--force-universe", action="store_true",
                        help="Re-fetch universe from SEC even if parquet exists")
    args = parser.parse_args()
    run(force_universe=args.force_universe)
