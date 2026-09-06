import json
import logging
import os
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from src.config import load_config
from src.cache import init_db, archive_fundamentals_snapshot, archive_universe_snapshot
from src.universe import load_or_build_universe, load_tradable_assets, filter_tradable
from src.prices import fetch_all_prices, apply_adv_gate, apply_liquidity_gate
from src.finnhub_data import fetch_metrics, fetch_profiles, attach_market_caps
from src.fundamentals import fetch_all_fundamentals
from src.factors import squeeze_flag
from src.compose import build_composite, compute_conviction
from src.fundamentals_block import attach_fundamentals_block
from src.enrich import enrich_finalists, composite_final
from src.finnhub_data import fetch_earnings_calendar
from src.streak import load_streak_history
from src.news import attach_news_overlay
from src.spy_analysis import compute_market_stress_overlay
from src.sizing import attach_weights
from src.output import write_csv, write_markdown, write_latest_json, print_top10

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DB_PATH       = "data/cache.db"
UNIVERSE_PATH = "data/universe.parquet"
OUTPUT_DIR    = "output"
STATS_PATH    = Path("data") / "last_run_stats.json"


class PipelineHealthError(RuntimeError):
    """A stage produced too few names to trust the ranking. The run must fail
    loudly rather than publish a list drawn from a collapsed universe."""


def evaluate_health(stats: dict, cfg: dict) -> tuple[bool, list[str]]:
    """Compare stage counts against config floors. Pure, so it is testable.

    Only floors whose stage has actually run (key present in stats) are
    checked, so this can be called after each stage to fail fast.
    """
    floors = cfg.get("health", {}) or {}
    checks = [
        ("adv_survivors", floors.get("min_adv_survivors")),
        ("cap_survivors", floors.get("min_cap_survivors")),
        ("ranked", floors.get("min_ranked")),
    ]
    reasons = []
    for key, floor in checks:
        if floor is None or key not in stats:
            continue
        if stats[key] < floor:
            reasons.append(f"{key}={stats[key]} below floor {floor}")
    return (not reasons), reasons


def write_stats(stats: dict, path: Path = STATS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(stats, indent=2, default=str))
    os.replace(tmp, path)


def _gate(stats: dict, cfg: dict, allow_degraded: bool, today: str) -> None:
    """Fail the run on a breached floor unless --allow-degraded was passed.
    Records the verdict in stats either way and writes the health-only
    screen_latest.json so the trader sees a failed screen, not a stale one."""
    ok, reasons = evaluate_health(stats, cfg)
    stats["ok"] = ok
    stats["reasons"] = reasons
    if ok:
        return
    for r in reasons:
        print(f"[health] {r}")
    if allow_degraded:
        print("[health] --allow-degraded set: continuing with a degraded universe")
        stats["degraded"] = True
        return
    write_stats(stats)
    write_latest_json(pd.DataFrame(), OUTPUT_DIR, today, health=stats,
                      regime={"regime": "UNKNOWN", "scale_factor": 1.0}, ranking_tail=[])
    raise PipelineHealthError("; ".join(reasons))


def run(force_universe: bool = False, allow_degraded: bool = False):
    cfg = load_config()
    today = date.today().isoformat()

    os.makedirs("data", exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    init_db(DB_PATH)

    stats: dict = {"date": today, "started_at": datetime.now().isoformat(timespec="seconds"),
                   "ok": None, "reasons": []}

    # Stage 1: Universe (SEC list, rebuilt weekly) ∩ Alpaca tradable equities
    universe_df = load_or_build_universe(
        cfg, UNIVERSE_PATH, force=force_universe,
        max_age_days=int(cfg["universe"].get("max_age_days", 7)))
    stats["universe"] = len(universe_df)
    if cfg["universe"].get("alpaca_filter", True):
        assets = load_tradable_assets()
        if assets is None:
            print("[universe] Alpaca asset list unavailable — tradability filter skipped")
        universe_df = filter_tradable(universe_df, assets)
    else:
        universe_df = filter_tradable(universe_df, None)
    stats["tradable"] = len(universe_df)

    # Stage 2: Prices → dollar-volume gate (cheap, before any per-name lookups)
    price_store, factors_df, pstats = fetch_all_prices(universe_df, cfg, DB_PATH)
    stats["prices"] = pstats
    stats["priced"] = int(pstats.get("priced", 0))
    adv_df = apply_adv_gate(factors_df, cfg)
    stats["adv_survivors"] = len(adv_df)
    _gate(stats, cfg, allow_degraded, today)

    # Stage 2.5: Market cap + industry from Finnhub, then the cap gate
    fh_cfg = cfg.get("finnhub", {}) or {}
    tickers = adv_df["ticker"].tolist()
    metrics, mstats = fetch_metrics(
        tickers, DB_PATH, ttl_days=int(fh_cfg.get("metrics_ttl_days", 7)),
        max_fetch=int(fh_cfg.get("max_fetch_per_run", 600)),
        calls_per_minute=int(fh_cfg.get("calls_per_minute", 60)))
    profiles, prstats = fetch_profiles(
        tickers, DB_PATH, cap_ttl_days=int(fh_cfg.get("metrics_ttl_days", 7)),
        max_fetch=int(fh_cfg.get("max_fetch_per_run", 600)),
        calls_per_minute=int(fh_cfg.get("calls_per_minute", 60)))
    stats["finnhub"] = {"metrics": mstats, "profiles": prstats}
    print(f"[finnhub] metrics cached={mstats['cached']} fetched={mstats['fetched']} "
          f"failed={mstats['failed']} unfetched={mstats['unfetched']}; "
          f"profiles cached={prstats['cached']} fetched={prstats['fetched']}")
    adv_df, capstats = attach_market_caps(adv_df, metrics, profiles, DB_PATH)
    stats["cap_sources"] = capstats
    print(f"[caps] sources={capstats}")
    survivors_df = apply_liquidity_gate(adv_df, cfg)
    stats["cap_survivors"] = len(survivors_df)
    print(f"[stage2] {stats['universe']} universe → {stats['tradable']} tradable → "
          f"{stats['priced']} priced → {stats['adv_survivors']} ADV → {stats['cap_survivors']} cap")
    _gate(stats, cfg, allow_degraded, today)

    archive_universe_snapshot(DB_PATH, today, survivors_df)

    # Universe metadata the factor rows do not carry
    extra_cols = [c for c in ["ticker", "alpaca_symbol", "exchange", "fractionable"] if c in universe_df.columns]
    survivors_df = survivors_df.merge(universe_df[extra_cols], on="ticker", how="left")

    # Stage 3: Fundamentals (survivors only). Sector already attached above.
    fund_df = fetch_all_fundamentals(survivors_df, cfg, DB_PATH)
    merged = survivors_df.merge(fund_df, on="ticker", how="left")
    print(f"[stage3] fundamentals fetched for {len(fund_df)} tickers")
    archive_fundamentals_snapshot(DB_PATH, today)

    # Stage 3.2: Finnhub ratios as columns (named as Finnhub names them), then
    # the fundamentals block: five percentile sub-scores + fund_score.
    metric_cols = ["peTTM", "pfcfShareTTM", "evEbitdaTTM", "psTTM", "roeTTM", "roaTTM",
                   "grossMarginTTM", "revenueGrowthTTMYoy", "epsGrowthTTMYoy", "revenueGrowth3Y",
                   "totalDebt/totalEquityQuarterly", "netInterestCoverageTTM", "currentRatioQuarterly"]
    for col in metric_cols:
        merged[col] = merged["ticker"].map(lambda t, c=col: (metrics.get(str(t).upper()) or {}).get(c))
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
    merged = attach_fundamentals_block(merged)
    stats["fund_coverage"] = round(float(merged["fund_score"].notna().mean()), 4) if len(merged) else 0.0
    print(f"[fund_block] fund_score coverage {stats['fund_coverage']:.0%} of {len(merged)} names")

    # Stage 3.5: Market stress overlay — scale top_n down in momentum-crash regimes
    stress = compute_market_stress_overlay()
    scale  = stress["scale_factor"]
    stats["regime"] = stress.get("regime")
    if scale == 0.0:
        print(f"[stress] regime={stress['regime']} — STRESS: outputting empty screen")
        ranked_df = pd.DataFrame(columns=["ticker"])
        csv_path = write_csv(ranked_df, OUTPUT_DIR, today)
        stats["ranked"] = 0
        stats["ok"], stats["reasons"] = True, ["stress regime: screen intentionally empty"]
        write_stats(stats)
        write_latest_json(ranked_df, OUTPUT_DIR, today, health=stats, regime=stress, ranking_tail=[])
        print(f"\n[output] {csv_path} (empty — market stress)")
        return
    elif scale < 1.0:
        original_top_n = cfg["output"]["top_n"]
        new_top_n = max(1, int(original_top_n * scale))
        cfg = dict(cfg)
        cfg["output"] = dict(cfg["output"])
        cfg["output"]["top_n"] = new_top_n
        print(f"[stress] regime={stress['regime']} scale={scale:.1f} "
              f"reason='{stress['reason']}' → top_n {original_top_n}→{new_top_n}")
    else:
        print(f"[stress] regime={stress['regime']} — full screen")

    # Stage 4: Composite score → finalist pool (top finalists_n, no conviction yet)
    fcfg = cfg.get("fundamentals", {}) or {}
    top_n = int(cfg["output"]["top_n"])
    finalists_n = max(top_n, int(fcfg.get("finalists_n", 60)))
    streak_data = load_streak_history(OUTPUT_DIR, lookback_days=cfg.get("streak", {}).get("lookback_days", 14))
    finalists = build_composite(merged, cfg, streak_data=streak_data, top_n=finalists_n, with_conviction=False)
    stats["ranked"] = int(finalists.attrs.get("ranked_total", len(finalists)))
    stats["fund_gate"] = finalists.attrs.get("fund_gate")
    ranking_tail = finalists.attrs.get("ranking_tail", [])
    if (stats["fund_gate"] or {}).get("skipped"):
        stats.setdefault("warnings", []).append(f"fund gate skipped: {stats['fund_gate']['skipped']}")
    _gate(stats, cfg, allow_degraded, today)

    # Stage 4.2: finalist enrichment (estimate revisions, short interest,
    # targets, earnings dates) → composite_final → final top_n → conviction
    calendar = fetch_earnings_calendar()
    finalists, estats = enrich_finalists(finalists, DB_PATH, calendar,
                                         budget_secs=float(fcfg.get("enrich_budget_secs", 240)))
    stats["enrich"] = estats
    finalists = composite_final(finalists, weight=float(fcfg.get("finalist_rev_weight", 0.05)))
    finalists = finalists.sort_values("composite_final", ascending=False).reset_index(drop=True)
    ranked_df = compute_conviction(finalists.head(top_n).reset_index(drop=True))
    ranked_df.attrs.update(finalists.attrs)

    # Stage 4.5: News overlay (entry signal + conviction adjustment)
    if cfg.get("news", {}).get("enabled", True):
        ranked_df = attach_news_overlay(ranked_df, cfg, DB_PATH)

    # Stage 4.6: Position sizing + concentration caps (advisory weights)
    ranked_df = attach_weights(ranked_df, cfg, price_store)

    # Squeeze screen
    squeeze_df = None
    if cfg["output"].get("include_squeeze_screen", False) and "short_float" in merged.columns:
        squeeze_rows = []
        for _, row in merged.iterrows():
            sf  = row.get("short_float", 0) or 0
            dtc = row.get("days_to_cover", 0) or 0
            m1  = row.get("mom_1m", 0) or 0
            if sf == sf and dtc == dtc and squeeze_flag(sf, dtc, m1):
                squeeze_rows.append(row)
        squeeze_df = pd.DataFrame(squeeze_rows) if squeeze_rows else None
        if squeeze_df is not None:
            print(f"[squeeze] {len(squeeze_df)} squeeze candidates")

    # Stage 5: Output
    stats["selected"] = len(ranked_df)
    stats["finished_at"] = datetime.now().isoformat(timespec="seconds")
    write_stats(stats)
    csv_path = write_csv(ranked_df, OUTPUT_DIR, today)
    md_path  = write_markdown(ranked_df, OUTPUT_DIR, today, squeeze_df=squeeze_df)
    json_path = write_latest_json(ranked_df, OUTPUT_DIR, today, health=stats, regime=stress,
                                  ranking_tail=ranking_tail)
    print(f"\n[output] {csv_path}")
    print(f"[output] {md_path}")
    print(f"[output] {json_path}")
    print_top10(ranked_df)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the equity screener")
    parser.add_argument("--force-universe", action="store_true",
                        help="Re-fetch universe from SEC even if parquet exists")
    parser.add_argument("--allow-degraded", action="store_true",
                        help="Publish even when a health floor is breached (manual runs only)")
    args = parser.parse_args()
    run(force_universe=args.force_universe, allow_degraded=args.allow_degraded)
