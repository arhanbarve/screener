import json
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path

import requests
import pandas as pd
from src.config import get_env

logger = logging.getLogger(__name__)

SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

_EXCLUDE_TERMS = [
    " warrant", "-wt", " unit", " pfd", " preferred",
    "proshares", "ishares", "invesco", "direxion",
    " etf", "trust", "fund", " lp",
]

# SEC writes non-common lines with a dash suffix: -PA/-PC preferred series,
# -WT/-WS warrants, -U units, -R rights. Share classes (-A, -B) are kept.
_SEC_NONCOMMON_SUFFIX_RE = re.compile(r"-(?:P[A-Z]?|W[ST]?[A-Z]?|U|R)$")
# Alpaca spells the same things with a dot: .PRA preferred, .WS warrants,
# .U units, .RT rights.
_ALPACA_NONCOMMON_SUFFIX_RE = re.compile(r"\.(?:PR[A-Z]?|WS[A-Z]?|U|RT)$")

ALPACA_ASSETS_CACHE = Path("data") / "alpaca_assets.json"
ALPACA_EXCHANGES_OK = {"NYSE", "NASDAQ", "AMEX", "ARCA", "BATS"}
UNIVERSE_MAX_AGE_DAYS = 7


def _user_agent() -> str:
    return get_env("SEC_USER_AGENT")


def fetch_sec_tickers() -> dict:
    headers = {"User-Agent": _user_agent()}
    resp = requests.get(SEC_TICKERS_URL, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()


def parse_sec_tickers(raw: dict) -> pd.DataFrame:
    rows = []
    for _, entry in raw.items():
        cik = f"{int(entry['cik_str']):010d}"
        rows.append({
            "ticker": entry["ticker"].upper().strip(),
            "cik": cik,
            "name": entry["title"],
        })
    return pd.DataFrame(rows)


def filter_universe(df: pd.DataFrame) -> pd.DataFrame:
    name_lower = df["name"].str.lower()
    ticker_lower = df["ticker"].str.lower()
    mask_warrant = ticker_lower.str.contains(r"-wt$|\+$|\.wt$", regex=True)
    mask_exclude = name_lower.apply(
        lambda n: any(term in n for term in _EXCLUDE_TERMS)
    )
    mask_suffix = df["ticker"].str.upper().str.contains(_SEC_NONCOMMON_SUFFIX_RE, regex=True)
    return df[~mask_warrant & ~mask_exclude & ~mask_suffix].reset_index(drop=True)


def build_universe(cfg: dict, out_path: str = "data/universe.parquet") -> pd.DataFrame:
    raw = fetch_sec_tickers()
    df = parse_sec_tickers(raw)
    if cfg["universe"].get("exclude_etfs", True):
        df = filter_universe(df)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"[universe] {len(df)} tickers written to {out_path}")
    return df


def load_or_build_universe(cfg: dict, path: str = "data/universe.parquet",
                           force: bool = False, max_age_days: int = UNIVERSE_MAX_AGE_DAYS) -> pd.DataFrame:
    """Cached SEC universe, rebuilt when missing, forced, or older than
    max_age_days. The old code only rebuilt when the file was missing, so a
    universe from June was still in use in September."""
    if not force and os.path.exists(path):
        age_days = (time.time() - os.path.getmtime(path)) / 86400.0
        if age_days <= max_age_days:
            df = pd.read_parquet(path)
            print(f"[universe] loaded {len(df)} tickers from cache ({age_days:.1f}d old)")
            return df
        print(f"[universe] cache is {age_days:.1f}d old — rebuilding")
    try:
        return build_universe(cfg, path)
    except Exception as e:  # noqa: BLE001 — a SEC outage must not kill the run if a cache exists
        if os.path.exists(path):
            logger.warning(f"[universe] rebuild failed ({e!r}); using stale cache")
            return pd.read_parquet(path)
        raise


# ── Alpaca tradability filter ────────────────────────────────────────────────

def to_alpaca_symbol(ticker: str) -> str:
    """SEC/yfinance BRK-B -> Alpaca BRK.B."""
    return ticker.upper().replace("-", ".")


def load_tradable_assets(fetch=None, cache_path: Path | None = None,
                         ttl_hours: int = 24) -> dict[str, dict] | None:
    """{alpaca_symbol: asset} for active, tradable, exchange-listed US equities.

    Cached on disk for a day. Returns None — meaning "do not filter" — when
    there are no Alpaca credentials or the fetch fails with no cache, so a
    public clone without a brokerage still screens. The universe then keeps
    the untradeable OTC/ADR lines it always had; nothing is worse than before.
    """
    path = cache_path or ALPACA_ASSETS_CACHE
    if path.exists():
        try:
            blob = json.loads(path.read_text())
            fetched = datetime.fromisoformat(blob["fetched_at"])
            if (datetime.utcnow() - fetched).total_seconds() < ttl_hours * 3600:
                return blob["assets"]
        except (OSError, KeyError, ValueError, json.JSONDecodeError):
            blob = None
    else:
        blob = None

    if fetch is None:
        if not (os.environ.get("ALPACA_API_KEY") and os.environ.get("ALPACA_SECRET_KEY")):
            return blob["assets"] if blob else None
        from src import broker
        fetch = broker.get_assets
    try:
        raw = fetch()
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[universe] Alpaca assets unavailable ({e!r})")
        return blob["assets"] if blob else None

    assets = {}
    for a in raw or []:
        sym = str(a.get("symbol") or "").upper()
        if not sym or not a.get("tradable"):
            continue
        if a.get("exchange") not in ALPACA_EXCHANGES_OK:
            continue
        if _ALPACA_NONCOMMON_SUFFIX_RE.search(sym):
            continue
        assets[sym] = {"exchange": a.get("exchange"),
                       "fractionable": bool(a.get("fractionable")),
                       "shortable": bool(a.get("shortable"))}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"fetched_at": datetime.utcnow().isoformat(), "assets": assets}))
    except OSError:
        pass
    return assets


def filter_tradable(df: pd.DataFrame, assets: dict[str, dict] | None) -> pd.DataFrame:
    """Keep only names Alpaca can trade on a listed exchange. Adds
    `alpaca_symbol`, `exchange`, `fractionable`. With assets=None the frame is
    returned with alpaca_symbol filled and nothing dropped."""
    out = df.copy()
    out["alpaca_symbol"] = out["ticker"].map(to_alpaca_symbol)
    if assets is None:
        out["exchange"] = ""
        out["fractionable"] = True
        return out.reset_index(drop=True)
    keep = out["alpaca_symbol"].isin(assets.keys())
    out = out[keep].copy()
    out["exchange"] = out["alpaca_symbol"].map(lambda s: assets[s]["exchange"])
    out["fractionable"] = out["alpaca_symbol"].map(lambda s: assets[s]["fractionable"])
    dropped = int((~keep).sum())
    logger.info(f"[universe] tradability filter: {len(df)} → {len(out)} ({dropped} not tradable on Alpaca)")
    print(f"[universe] tradability filter: {len(df)} → {len(out)} tradable")
    return out.reset_index(drop=True)


def apply_neglect_gate(factors_df: pd.DataFrame, lp_cfg: dict, min_price: float = 0.0) -> pd.DataFrame:
    """Capacity-constrained band gate ($50M-$2B market cap, ADV floor) used by
    event_backtest.py's neglected-universe construction."""
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
