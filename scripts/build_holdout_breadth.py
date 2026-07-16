"""Build holdout breadth (2024-01-02 → latest cached) from already-cached
prices only — no network. Delisted/uncached names are skipped, matching the
survivorship caveat documented for the dev-window breadth.

Usage: python3 -m scripts.build_holdout_breadth
"""
import logging

import pandas as pd

from src.cache import get_backtest_prices
from src.event_backtest import _covers_range
from src.factor_panel import breadth_series, candidate_tickers

logger = logging.getLogger(__name__)

DB = "data/cache.db"
START = "2024-01-02"
END = "2026-07-14"
WARMUP_START = "2023-01-01"          # SMA200 needs ~10 months of runway
OUT = "output/breadth_holdout.parquet"


def main() -> None:
    tickers = candidate_tickers(DB)
    prices = {}
    for t in tickers:
        df = get_backtest_prices(DB, t, ttl_days=10_000)   # cache-only, no ttl refetch
        if _covers_range(df, WARMUP_START, END, slack_days=10):
            prices[t] = df.loc[WARMUP_START:END]
    logger.info(f"[holdout-breadth] {len(prices)}/{len(tickers)} tickers with cached coverage")
    if len(prices) < 500:
        raise RuntimeError(f"only {len(prices)} tickers covered — cache unexpectedly thin")
    breadth = breadth_series(prices).loc[START:END]
    breadth.to_parquet(OUT)
    logger.info(f"[holdout-breadth] {len(breadth)} days -> {OUT}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
