import time
import logging
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
import finnhub
import yfinance as yf
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
    if assets <= 0:
        raise KeyError("assets <= 0, gp/assets undefined")
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


def parse_finnhub_revisions(trends: list) -> tuple[float, float]:
    """Returns (rev_breadth, rev_magnitude) from Finnhub recommendation_trends."""
    try:
        if not trends:
            return 0.0, 0.0
        ranked = sorted(trends, key=lambda t: t.get("period", ""), reverse=True)
        latest = ranked[0]
        sb  = latest.get("strongBuy", 0) or 0
        b   = latest.get("buy", 0) or 0
        s   = latest.get("sell", 0) or 0
        ss  = latest.get("strongSell", 0) or 0
        h   = latest.get("hold", 0) or 0
        tot = sb + b + h + s + ss
        breadth = (sb + b - s - ss) / tot if tot > 0 else 0.0
        if len(ranked) >= 2:
            prior = ranked[1]
            pt = sum(prior.get(k, 0) or 0 for k in ["strongBuy", "buy", "hold", "sell", "strongSell"])
            prior_bull = (prior.get("strongBuy", 0) + prior.get("buy", 0)) / pt if pt > 0 else 0.0
            cur_bull   = (sb + b) / tot if tot > 0 else 0.0
            mag = cur_bull - prior_bull
        else:
            mag = 0.0
        return float(breadth), float(mag)
    except Exception:
        return 0.0, 0.0


def parse_short_interest(info: dict) -> tuple[float, float]:
    shares_short = float(info.get("sharesShort") or 0)
    float_shares = float(info.get("floatShares") or 1)
    avg_vol      = float(info.get("averageVolume") or 1)
    short_float  = shares_short / float_shares if float_shares > 0 else 0.0
    short_float  = min(short_float, 1.0)  # cap at 100% — yfinance sometimes returns bad units
    dtc          = shares_short / avg_vol if avg_vol > 0 else 0.0
    dtc          = min(dtc, 365.0)  # cap at 1 year
    return short_float, dtc


_EXEC_ROLES = {"ceo", "cfo", "coo", "president", "chief", "director", "chairman", "general counsel"}


def parse_insider_buys(transactions: list, days: int = 90) -> dict:
    """
    Parse insider purchases. Returns total count AND executive-only count.
    Cohen-Malloy-Pomorski (2012): opportunistic/exec buys have predictive power;
    routine programmatic buys do not. Weight toward executive-level purchasers.
    """
    cutoff = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
    all_buyers: set[str] = set()
    exec_buyers: set[str] = set()
    total_value = 0.0
    for tx in transactions:
        if tx.get("transactionCode") != "P":
            continue
        if tx.get("transactionDate", "") < cutoff:
            continue
        name  = tx.get("name", "")
        role  = (tx.get("officerTitle", "") or "").lower()
        shares = float(tx.get("share", 0) or 0)
        price  = float(tx.get("transactionPrice", 0) or 0)
        all_buyers.add(name)
        total_value += shares * price
        if any(r in role for r in _EXEC_ROLES):
            exec_buyers.add(name)
    return {
        "insider_buys_90d": len(all_buyers),
        "exec_buys_90d":    len(exec_buyers),
        "insider_buy_value": total_value,
    }


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


def _finnhub_key_valid(fh: finnhub.Client) -> bool:
    try:
        fh.company_profile2(symbol="AAPL")
        return True
    except Exception as e:
        if "401" in str(e):
            return False
        return True  # other errors (rate limit etc.) — assume key valid


def fetch_all_fundamentals(
    survivors_df: pd.DataFrame,
    cfg: dict,
    db_path: str,
) -> pd.DataFrame:
    fh_key = get_env("FINNHUB_API_KEY")
    fh     = finnhub.Client(api_key=fh_key)
    bucket = _TokenBucket(cfg["finnhub"]["calls_per_minute"])

    use_finnhub = _finnhub_key_valid(fh)
    if not use_finnhub:
        logger.warning("[fundamentals] Finnhub API key invalid — skipping all Finnhub calls")

    ttl_fund  = cfg["cache"]["fundamentals_ttl_days"]
    ttl_edgar = cfg["cache"]["edgar_ttl_days"]

    rows = []
    total = len(survivors_df)
    # Single shared HTTP session for all yfinance Ticker calls — avoids per-ticker
    # session creation which leaks file descriptors under launchd's low fd limits.
    yf_session = requests.Session()
    try:
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
                # Back-fill empty sector in stale cache entries without full re-fetch
                if not cached_fund.get("sector"):
                    try:
                        info = yf.Ticker(ticker).get_info()
                        sector = (info.get("sector") or info.get("industry") or "").strip()
                        if sector:
                            cached_fund["sector"] = sector
                            put_fundamentals(db_path, ticker, cached_fund)
                    except Exception:
                        pass
                row.update(cached_fund)
            else:
                fund = {}
                if use_finnhub:
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
                        trends = fh.recommendation_trends(ticker)
                        breadth, mag = parse_finnhub_revisions(trends)
                        fund["rev_breadth"] = breadth
                        fund["rev_magnitude"] = mag
                    except Exception as e:
                        logger.warning(f"[fundamentals] revisions for {ticker}: {e}")
                        fund["rev_breadth"] = float("nan")
                        fund["rev_magnitude"] = float("nan")

                    try:
                        bucket.consume()
                        insider_tx = fh.stock_insider_transactions(ticker, _from="", to="")
                        insider_data = parse_insider_buys(insider_tx.get("data", []))
                        fund["insider_buys_90d"]  = insider_data["insider_buys_90d"]
                        fund["exec_buys_90d"]     = insider_data["exec_buys_90d"]
                        fund["insider_buy_value"] = insider_data["insider_buy_value"]
                        fund["insider_flag"]      = insider_data["exec_buys_90d"] >= 2
                    except Exception as e:
                        logger.warning(f"[fundamentals] insider for {ticker}: {e}")
                        fund["insider_buys_90d"]  = 0
                        fund["exec_buys_90d"]     = 0
                        fund["insider_buy_value"] = 0.0
                        fund["insider_flag"]      = False
                else:
                    fund["sue"] = float("nan")
                    fund["rev_breadth"] = float("nan")
                    fund["rev_magnitude"] = float("nan")
                    fund["insider_buys_90d"]  = 0
                    fund["exec_buys_90d"]     = 0
                    fund["insider_buy_value"] = 0.0
                    fund["insider_flag"]      = False

                try:
                    # Use get_info() not .info property — the property bypasses
                    # yfinance's cookie/crumb handling with custom sessions,
                    # causing silent rate-limit failures that blank sector.
                    tk = yf.Ticker(ticker)
                    info = tk.get_info()
                    sf, dtc = parse_short_interest(info)
                    fund["short_float"]   = sf
                    fund["days_to_cover"] = dtc
                    sector = (info.get("sector") or info.get("industry") or "").strip()
                    fund["sector"] = sector
                except Exception:
                    fund["short_float"]   = float("nan")
                    fund["days_to_cover"] = float("nan")
                    fund["sector"]        = ""

                put_fundamentals(db_path, ticker, fund)
                row.update(fund)

            rows.append(row)
            time.sleep(0.12)
    finally:
        yf_session.close()

    return pd.DataFrame(rows)
