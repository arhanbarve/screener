"""Cache maintenance CLI.

  python -m src.cache_maint purge-failed [--before ISO]   drop yfinance quarantine rows
  python -m src.cache_maint warm-finnhub [--max N]        pre-fetch Finnhub metrics/profiles
  python -m src.cache_maint stats                          row counts that matter

warm-finnhub exists because a run may only spend `finnhub.max_fetch_per_run`
calls; the first fill of ~2,500 liquid names takes ~85 minutes at 60/min and
should happen outside the 16:30 window. It prioritises tickers that have
prices in the cache and clear the ADV gate, then everything else.
"""
import argparse
import sqlite3

from src.cache import init_db, purge_failed_tickers, count_failed_tickers
from src.config import load_config

DB_PATH = "data/cache.db"


def _adv_ranked_tickers(db_path: str, min_adv: float) -> list[str]:
    """Tickers with cached prices, ordered by 20-day average dollar volume
    (desc), liquid names first, then the rest."""
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        """
        SELECT ticker, AVG(close * volume) AS adv
        FROM (
            SELECT ticker, close, volume,
                   ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) AS rn
            FROM prices
        )
        WHERE rn <= 20
        GROUP BY ticker
        ORDER BY adv DESC
        """
    ).fetchall()
    conn.close()
    liquid = [t for t, adv in rows if adv is not None and adv >= min_adv]
    rest = [t for t, adv in rows if not (adv is not None and adv >= min_adv)]
    return liquid + rest


def cmd_purge_failed(db_path: str, before: str | None) -> dict:
    n_before = count_failed_tickers(db_path)
    deleted = purge_failed_tickers(db_path, before)
    return {"before": n_before, "deleted": deleted, "after": count_failed_tickers(db_path)}


def cmd_warm_finnhub(db_path: str, cfg: dict, max_fetch: int) -> dict:
    from src.finnhub_data import fetch_metrics, fetch_profiles

    tickers = _adv_ranked_tickers(db_path, cfg["liquidity_gate"]["min_avg_dollar_vol_20d"])
    tickers = [t for t in tickers if t != "SPY"]
    cpm = int(cfg.get("finnhub", {}).get("calls_per_minute", 60))
    ttl = int(cfg.get("finnhub", {}).get("metrics_ttl_days", 7))
    _, m_stats = fetch_metrics(tickers, db_path, ttl_days=ttl, max_fetch=max_fetch, calls_per_minute=cpm)
    _, p_stats = fetch_profiles(tickers, db_path, cap_ttl_days=ttl, max_fetch=max_fetch, calls_per_minute=cpm)
    return {"candidates": len(tickers), "metrics": m_stats, "profiles": p_stats}


def cmd_stats(db_path: str) -> dict:
    conn = sqlite3.connect(db_path)
    out = {}
    for table in ("prices", "market_cap", "fundamentals", "edgar", "fh_metrics", "fh_profile", "failed_tickers"):
        try:
            out[table] = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        except sqlite3.OperationalError:
            out[table] = None
    out["distinct_price_tickers"] = conn.execute("SELECT COUNT(DISTINCT ticker) FROM prices").fetchone()[0]
    conn.close()
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="cache_maint", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", default=DB_PATH)
    sub = p.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("purge-failed")
    sp.add_argument("--before", default=None, help="ISO timestamp; only rows older than this")
    sp = sub.add_parser("warm-finnhub")
    sp.add_argument("--max", type=int, default=3000)
    sub.add_parser("stats")
    args = p.parse_args(argv)

    init_db(args.db)
    if args.cmd == "purge-failed":
        out = cmd_purge_failed(args.db, args.before)
    elif args.cmd == "warm-finnhub":
        out = cmd_warm_finnhub(args.db, load_config(), args.max)
    else:
        out = cmd_stats(args.db)
    import json
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
