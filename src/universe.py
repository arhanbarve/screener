import os
import logging
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
    return df[~mask_warrant & ~mask_exclude].reset_index(drop=True)

def build_universe(cfg: dict, out_path: str = "data/universe.parquet") -> pd.DataFrame:
    raw = fetch_sec_tickers()
    df = parse_sec_tickers(raw)
    if cfg["universe"].get("exclude_etfs", True):
        df = filter_universe(df)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"[universe] {len(df)} tickers written to {out_path}")
    return df


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
