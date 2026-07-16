"""Build holdout breadth (2024-01-02 → 2026-07-14) for the index-overlay run.

Universe prices are cached only through 2023-12-29 (dev-window panel build),
so the holdout range is fetched fresh from yfinance in batches. Fetched
frames are deliberately NOT written to cache.db: put_backtest_prices
replaces a ticker's whole payload, and a 2023-2026-only frame would clobber
the full 2013-2023 history in the shared production cache.

Tickers yfinance no longer serves (delisted since the universe was built)
are skipped — same survivorship caveat as the dev-window breadth.

Usage: python3 -m scripts.build_holdout_breadth
"""
import logging

import pandas as pd

from src.event_backtest import _fetch_batch_yfinance
from src.factor_panel import breadth_series, candidate_tickers

logger = logging.getLogger(__name__)

DB = "data/cache.db"
START = "2024-01-02"
END = "2026-07-14"
WARMUP_START = "2023-01-01"          # SMA200 needs ~10 months of runway
OUT = "output/breadth_holdout.parquet"
BATCH = 200


def main() -> None:
    tickers = candidate_tickers(DB)
    prices: dict[str, pd.DataFrame] = {}
    batches = [tickers[i:i + BATCH] for i in range(0, len(tickers), BATCH)]
    for i, batch in enumerate(batches):
        fetched = _fetch_batch_yfinance(batch, start=WARMUP_START, end=END)
        for t, df in fetched.items():
            if df is not None and len(df) >= 260:   # enough for SMA200 to form
                prices[t] = df
        logger.info(f"[holdout-breadth] batch {i + 1}/{len(batches)}: "
                    f"{len(prices)} tickers usable so far")
    if len(prices) < 500:
        raise RuntimeError(f"only {len(prices)} tickers fetched — yfinance problem?")
    breadth = breadth_series(prices).loc[START:END]
    breadth.to_parquet(OUT)
    logger.info(f"[holdout-breadth] {len(breadth)} days x {len(prices)} tickers -> {OUT}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
