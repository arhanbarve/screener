import time
import logging
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
import finnhub
from src.config import load_config, get_env
from src.cache import get_fundamentals, put_fundamentals, get_edgar, put_edgar

logger = logging.getLogger(__name__)

EDGAR_BASE = "https://data.sec.gov/api/xbrl/companyfacts"

REVENUE_TAGS = [
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "RevenuesNetOfInterestExpense",
    "SalesRevenueNet",
]
COGS_TAGS = [
    "CostOfGoodsAndServicesSold",
    "CostOfRevenue",
    "CostOfGoodsSold",
    "CostOfSales",
]
ASSETS_TAGS = ["Assets"]


def _sec_headers() -> dict:
    return {"User-Agent": get_env("SEC_USER_AGENT")}


def _latest_annual(entries: list) -> Optional[float]:
    annual = [e for e in entries if e.get("form") in ("10-K", "20-F")]
    if not annual:
        return None
    annual.sort(key=lambda e: e.get("end", ""), reverse=True)
    return float(annual[0]["val"])


def _resolve_tag(facts: dict, tag_list: list) -> Optional[float]:
    usgaap = facts.get("us-gaap", {})
    for tag in tag_list:
        if tag in usgaap:
            entries = usgaap[tag].get("units", {}).get("USD", [])
            val = _latest_annual(entries)
            if val is not None:
                return val
    return None


def parse_edgar_gp(data: dict) -> tuple[float, float, float, float]:
    facts = data["facts"]
    revenue = _resolve_tag(facts, REVENUE_TAGS)
    cogs    = _resolve_tag(facts, COGS_TAGS)
    assets  = _resolve_tag(facts, ASSETS_TAGS)
    if revenue is None or cogs is None or assets is None:
        raise KeyError("Could not resolve revenue/cogs/assets XBRL tags")
    gp = (revenue - cogs) / assets
    return gp, revenue, cogs, assets


def fetch_edgar(cik: str, db_path: str, ttl_days: int) -> Optional[dict]:
    cached = get_edgar(db_path, cik, ttl_days=ttl_days)
    if cached is not None:
        return cached
    url = f"{EDGAR_BASE}/CIK{cik}.json"
    try:
        resp = requests.get(url, headers=_sec_headers(), timeout=30)
        resp.raise_for_status()
        data = resp.json()
        gp, rev, cogs, assets = parse_edgar_gp(data)
        put_edgar(db_path, cik, gp_assets=gp, revenue=rev, cogs=cogs, assets=assets)
        return {"gp_assets": gp, "revenue": rev, "cogs": cogs, "assets": assets}
    except Exception as e:
        logger.warning(f"[edgar] failed for CIK {cik}: {e}")
        return None


def parse_finnhub_surprise(earnings: list) -> tuple[list, list]:
    earnings_sorted = sorted(earnings, key=lambda e: e.get("period", ""))
    actuals   = [e["actual"]   for e in earnings_sorted if e.get("actual") is not None]
    estimates = [e["estimate"] for e in earnings_sorted if e.get("estimate") is not None]
    min_len = min(len(actuals), len(estimates))
    return actuals[-min_len:], estimates[-min_len:]


def parse_finnhub_revisions(trend_data: dict) -> tuple[float, float]:
    """Returns (rev_breadth, rev_magnitude) from Finnhub EPS trend."""
    try:
        trend = trend_data.get("trend", [])
        if not trend:
            return 0.0, 0.0
        latest = sorted(trend, key=lambda t: t.get("period", ""), reverse=True)[0]
        eps_up   = latest.get("epsTrendUp", 0) or 0
        eps_down = latest.get("epsTrendDown", 0) or 0
        total    = eps_up + eps_down
        breadth  = (eps_up - eps_down) / total if total > 0 else 0.0
        cur = latest.get("epsTrend", {}).get("current", 0)
        ago = latest.get("epsTrend", {}).get("3month", cur)
        mag = (cur - ago) / abs(ago) if abs(ago) > 1e-12 else 0.0
        return float(breadth), float(mag)
    except Exception:
        return 0.0, 0.0


def parse_short_interest(info: dict) -> tuple[float, float]:
    shares_short = float(info.get("sharesShort") or 0)
    float_shares = float(info.get("floatShares") or 1)
    avg_vol      = float(info.get("averageVolume") or 1)
    short_float  = shares_short / float_shares if float_shares > 0 else 0.0
    dtc          = shares_short / avg_vol if avg_vol > 0 else 0.0
    return short_float, dtc


def parse_insider_buys(transactions: list, days: int = 90) -> int:
    cutoff = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
    buyers = set()
    for tx in transactions:
        if tx.get("transactionCode") == "P" and tx.get("transactionDate", "") >= cutoff:
            buyers.add(tx.get("name", ""))
    return len(buyers)


class _TokenBucket:
    def __init__(self, rate: int):
        self._rate = rate
        self._tokens = rate
        self._last = time.monotonic()

    def consume(self):
        now = time.monotonic()
        self._tokens = min(self._rate, self._tokens + (now - self._last) * self._rate / 60.0)
        self._last = now
        if self._tokens < 1:
            time.sleep((1 - self._tokens) * 60.0 / self._rate)
            self._tokens = 0
        else:
            self._tokens -= 1


def fetch_all_fundamentals(
    survivors_df: pd.DataFrame,
    cfg: dict,
    db_path: str,
) -> pd.DataFrame:
    fh_key = get_env("FINNHUB_API_KEY")
    fh     = finnhub.Client(api_key=fh_key)
    bucket = _TokenBucket(cfg["finnhub"]["calls_per_minute"])

    ttl_fund  = cfg["cache"]["fundamentals_ttl_days"]
    ttl_edgar = cfg["cache"]["edgar_ttl_days"]

    rows = []
    total = len(survivors_df)
    for idx, record in survivors_df.iterrows():
        ticker = record["ticker"]
        cik    = record.get("cik", "")
        logger.info(f"[fundamentals] {idx+1}/{total} {ticker}")

        row = {"ticker": ticker}

        if cik:
            edgar = fetch_edgar(cik, db_path, ttl_days=ttl_edgar)
            row["gp_assets"] = edgar["gp_assets"] if edgar else float("nan")
        else:
            row["gp_assets"] = float("nan")

        cached_fund = get_fundamentals(db_path, ticker, ttl_days=ttl_fund)
        if cached_fund:
            row.update(cached_fund)
        else:
            fund = {}
            try:
                bucket.consume()
                earnings = fh.company_earnings(ticker, limit=8)
                actuals, estimates = parse_finnhub_surprise(earnings)
                from src.factors import compute_sue
                fund["sue"] = compute_sue(actuals, estimates) if actuals else 0.0
            except Exception as e:
                logger.warning(f"[fundamentals] earnings for {ticker}: {e}")
                fund["sue"] = float("nan")

            try:
                bucket.consume()
                trend = fh.eps_estimate(ticker, freq="quarterly")
                breadth, mag = parse_finnhub_revisions({"trend": trend.get("data", [])})
                fund["rev_breadth"] = breadth
                fund["rev_magnitude"] = mag
            except Exception as e:
                logger.warning(f"[fundamentals] revisions for {ticker}: {e}")
                fund["rev_breadth"] = float("nan")
                fund["rev_magnitude"] = float("nan")

            try:
                bucket.consume()
                insider_tx = fh.stock_insider_transactions(ticker, _from="", to="")
                buys = parse_insider_buys(insider_tx.get("data", []))
                fund["insider_buys_90d"] = buys
                fund["insider_flag"] = buys >= 2
            except Exception as e:
                logger.warning(f"[fundamentals] insider for {ticker}: {e}")
                fund["insider_buys_90d"] = 0
                fund["insider_flag"] = False

            import yfinance as yf
            try:
                info = yf.Ticker(ticker).info
                sf, dtc = parse_short_interest(info)
                fund["short_float"] = sf
                fund["days_to_cover"] = dtc
                fund["sector"] = info.get("sector", "")
            except Exception:
                fund["short_float"] = float("nan")
                fund["days_to_cover"] = float("nan")
                fund["sector"] = ""

            put_fundamentals(db_path, ticker, fund)
            row.update(fund)

        rows.append(row)
        time.sleep(0.12)

    return pd.DataFrame(rows)
