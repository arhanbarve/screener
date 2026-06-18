import logging
import os
from datetime import date
from src.config import load_config
from src.cache import init_db
from src.universe import build_universe
from src.prices import fetch_all_prices
from src.fundamentals import fetch_all_fundamentals
from src.factors import squeeze_flag
from src.compose import build_composite
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

    # Stage 4: Composite score
    ranked_df = build_composite(merged, cfg)

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
