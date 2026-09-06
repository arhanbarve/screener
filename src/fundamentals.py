import time
import logging
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from datetime import date, datetime, timedelta
from typing import Optional
import finnhub
from src.config import get_env
from src.cache import (get_fundamentals, put_fundamentals, get_edgar, put_edgar,
                       get_edgar_facts, put_edgar_facts)
from src.factors import compute_sue

logger = logging.getLogger(__name__)

EDGAR_BASE = "https://data.sec.gov/api/xbrl/companyfacts"

# SEC's fair-access guidance caps well-behaved clients around 10 req/sec; this
# stays comfortably under that. Unlike the Finnhub calls in this same loop
# (paced by _TokenBucket below), EDGAR requests had no pacing at all — a
# 2026-09-04 dry run firing them back-to-back triggered real (if transient)
# SEC-side failures under sustained load.
EDGAR_CALLS_PER_MINUTE = 120

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
NET_INCOME_TAGS = ["NetIncomeLoss", "ProfitLoss", "NetIncomeLossAvailableToCommonStockholdersBasic"]
OCF_TAGS = ["NetCashProvidedByUsedInOperatingActivities",
            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"]
LIABILITIES_TAGS = ["Liabilities"]
EQUITY_TAGS = ["StockholdersEquity",
               "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"]
SHARES_DURATION_TAGS = ["WeightedAverageNumberOfDilutedSharesOutstanding",
                        "WeightedAverageNumberOfSharesOutstandingBasic"]
SHARES_INSTANT_TAGS = ["CommonStockSharesOutstanding"]
GROSS_PROFIT_TAGS = ["GrossProfit"]

ANNUAL_FORMS = ("10-K", "20-F", "40-F")
# A 10-K reports the fiscal year (≈365 days) and often the fourth quarter
# (≈90 days) under the SAME end date. Flow items must be filtered to the
# annual duration or a quarterly value can be read as the year's.
ANNUAL_DURATION_DAYS = (340, 390)
# Two consecutive fiscal years are ~365 days apart; allow for 52/53-week years.
YEAR_GAP_DAYS = (300, 430)


def _sec_headers() -> dict:
    return {"User-Agent": get_env("SEC_USER_AGENT")}


def _is_annual_form(form: str) -> bool:
    return any(str(form or "").startswith(f) for f in ANNUAL_FORMS)


def _days_between(a: str, b: str) -> int | None:
    try:
        return (datetime.strptime(b, "%Y-%m-%d") - datetime.strptime(a, "%Y-%m-%d")).days
    except (TypeError, ValueError):
        return None


def _annual_series(entries: list, duration: bool) -> list[tuple[str, float]]:
    """[(end, val)] newest first, one value per fiscal-year end, from annual
    filings only. `duration=True` keeps only ≈12-month periods (flow items);
    False is for balance-sheet instants (no `start`)."""
    best: dict[str, tuple[str, float]] = {}   # end -> (filed, val)
    for e in entries or []:
        if not _is_annual_form(e.get("form")):
            continue
        end = e.get("end")
        if not end or e.get("val") is None:
            continue
        if duration:
            start = e.get("start")
            if start:
                d = _days_between(start, end)
                if d is None or not (ANNUAL_DURATION_DAYS[0] <= d <= ANNUAL_DURATION_DAYS[1]):
                    continue
            # No `start` on a flow item: cannot check the duration, keep it.
        elif e.get("start"):
            continue
        filed = e.get("filed", "")
        if end not in best or filed >= best[end][0]:
            best[end] = (filed, float(e["val"]))
    return sorted(((end, v) for end, (_, v) in best.items()), key=lambda x: x[0], reverse=True)


def _latest_two(series: list[tuple[str, float]]) -> tuple[float | None, float | None]:
    """(latest, one year earlier) — the earlier value must be a real prior
    fiscal year, not a restated duplicate or a stub period."""
    if not series:
        return None, None
    latest_end, latest_val = series[0]
    prev_val = None
    for end, val in series[1:]:
        gap = _days_between(end, latest_end)
        if gap is not None and YEAR_GAP_DAYS[0] <= gap <= YEAR_GAP_DAYS[1]:
            prev_val = val
            break
    return latest_val, prev_val


# A filer whose newest annual value is older than this is not reporting (or
# has moved to a tag we do not know); its facts are treated as missing rather
# than compared against fresh prices and caps.
MAX_FACT_AGE_DAYS = 550


def _resolve_pair(facts: dict, tag_list: list, duration: bool, units: str = "USD",
                  today: date | None = None) -> tuple[float | None, float | None]:
    """(latest, prior-year) for the concept, choosing among synonym tags by
    RECENCY of the latest fiscal year, not by list order.

    Tag priority was the old rule and it is wrong: Microsoft last used
    `Revenues` in FY2010 and `RevenueFromContractWithCustomer…` since, so
    priority returned 2010 revenue against 2026 assets and a negative gross
    profitability. Ties on end date go to the larger value (a total, not a
    component). Anything older than MAX_FACT_AGE_DAYS is discarded.
    """
    usgaap = facts.get("us-gaap", {})
    today = today or date.today()
    best: tuple[str, float, float | None] | None = None   # (end, latest, prev)
    for tag in tag_list:
        if tag not in usgaap:
            continue
        series = _annual_series(usgaap[tag].get("units", {}).get(units, []), duration)
        if not series:
            continue
        end = series[0][0]
        age = _days_between(end, today.isoformat())
        if age is None or age > MAX_FACT_AGE_DAYS:
            continue
        latest, prev = _latest_two(series)
        if best is None or end > best[0] or (end == best[0] and latest > best[1]):
            best = (end, latest, prev)
    return (best[1], best[2]) if best else (None, None)


def parse_edgar_gp(data: dict) -> tuple[float, float, float, float]:
    facts = data["facts"]
    revenue, _ = _resolve_pair(facts, REVENUE_TAGS, duration=True)
    cogs, _    = _resolve_pair(facts, COGS_TAGS, duration=True)
    assets, _  = _resolve_pair(facts, ASSETS_TAGS, duration=False)
    if revenue is None or cogs is None or assets is None:
        raise KeyError("Could not resolve revenue/cogs/assets XBRL tags")
    if assets <= 0:
        raise KeyError("assets <= 0, gp/assets undefined")
    gp = (revenue - cogs) / assets
    return gp, revenue, cogs, assets


def _safe_div(a, b) -> float:
    if a is None or b is None or b == 0:
        return float("nan")
    return float(a) / float(b)


def parse_edgar_facts(data: dict) -> dict:
    """Everything the fundamentals block wants from one companyfacts JSON.

    Returns raw values plus derived ratios; any item that cannot be resolved
    is NaN (never raises — a filer with no COGS tag still has accruals). A
    filer with no usable us-gaap facts at all returns a dict of NaNs plus
    "n_resolved": 0, which callers treat as "no EDGAR data".
    """
    facts = data.get("facts", {}) or {}
    revenue, _            = _resolve_pair(facts, REVENUE_TAGS, duration=True)
    cogs, _               = _resolve_pair(facts, COGS_TAGS, duration=True)
    assets, assets_prev   = _resolve_pair(facts, ASSETS_TAGS, duration=False)
    net_income, _         = _resolve_pair(facts, NET_INCOME_TAGS, duration=True)
    ocf, _                = _resolve_pair(facts, OCF_TAGS, duration=True)
    liabilities, _        = _resolve_pair(facts, LIABILITIES_TAGS, duration=False)
    equity, _             = _resolve_pair(facts, EQUITY_TAGS, duration=False)
    shares, shares_prev   = _resolve_pair(facts, SHARES_DURATION_TAGS, duration=True, units="shares")
    if shares is None:
        shares, shares_prev = _resolve_pair(facts, SHARES_INSTANT_TAGS, duration=False, units="shares")

    gp_assets = float("nan")
    if revenue is not None and cogs is not None and assets and assets > 0:
        gp_assets = (revenue - cogs) / assets
    elif assets and assets > 0:
        gross, _ = _resolve_pair(facts, GROSS_PROFIT_TAGS, duration=True)
        if gross is not None:
            gp_assets = gross / assets
    accruals = float("nan")
    if net_income is not None and ocf is not None and assets and assets > 0:
        accruals = (net_income - ocf) / assets
    asset_growth = float("nan")
    if assets and assets_prev and assets_prev > 0:
        asset_growth = assets / assets_prev - 1.0
    net_issuance = float("nan")
    if shares and shares_prev and shares_prev > 0:
        net_issuance = shares / shares_prev - 1.0
    roa = _safe_div(net_income, assets) if assets and assets > 0 else float("nan")
    leverage = _safe_div(liabilities, assets) if assets and assets > 0 else float("nan")

    raw = {"revenue": revenue, "cogs": cogs, "assets": assets, "assets_prev": assets_prev,
           "net_income": net_income, "ocf": ocf, "liabilities": liabilities, "equity": equity,
           "shares": shares, "shares_prev": shares_prev}
    n_resolved = sum(1 for v in raw.values() if v is not None)
    return {**{k: (float("nan") if v is None else float(v)) for k, v in raw.items()},
            "gp_assets": gp_assets, "accruals": accruals, "asset_growth": asset_growth,
            "net_issuance": net_issuance, "roa": roa, "leverage": leverage,
            "n_resolved": n_resolved}


EDGAR_FACT_COLUMNS = ["gp_assets", "accruals", "asset_growth", "net_issuance", "roa", "leverage"]


# requests' own `timeout=` covers connect/read but a 2026-09-04 live run still
# saw individual EDGAR calls stall for minutes with the process CPU frozen —
# the same "a library timeout kwarg is not a real deadline" failure mode
# already fixed for yfinance in src/prices.py. Enforced again here with a
# thread + future deadline so a stall becomes a normal, loggable failure
# instead of blocking the whole fundamentals loop.
EDGAR_REQUEST_TIMEOUT_SEC = 30


def fetch_edgar(cik: str, db_path: str, ttl_days: int, bucket: "_TokenBucket | None" = None) -> Optional[dict]:
    """Extended EDGAR facts for one CIK, cached in `edgar_facts` (payload JSON).
    Returns None when the fetch fails or nothing resolves. Also refreshes the
    legacy `edgar` row so older readers keep working.

    `bucket` paces the actual network call (skipped entirely on a cache hit)
    — SEC throttles sustained unpaced requests from one client."""
    cached = get_edgar_facts(db_path, cik, ttl_days=ttl_days)
    if cached is not None:
        return cached
    if bucket is not None:
        bucket.consume()
    url = f"{EDGAR_BASE}/CIK{cik}.json"

    def _do_fetch():
        resp = requests.get(url, headers=_sec_headers(), timeout=EDGAR_REQUEST_TIMEOUT_SEC)
        resp.raise_for_status()
        return resp.json()

    executor = ThreadPoolExecutor(max_workers=1)
    try:
        data = executor.submit(_do_fetch).result(timeout=EDGAR_REQUEST_TIMEOUT_SEC)
    except FutureTimeoutError as e:
        logger.warning(f"[edgar] timed out for CIK {cik} after {EDGAR_REQUEST_TIMEOUT_SEC}s")
        return None
    except Exception as e:
        logger.warning(f"[edgar] failed for CIK {cik}: {e}")
        return None
    finally:
        executor.shutdown(wait=False)
    facts = parse_edgar_facts(data)
    if facts["n_resolved"] == 0:
        # Cache the miss too (IFRS filers, shells) so it is not re-requested daily.
        put_edgar_facts(db_path, cik, facts)
        return None
    put_edgar_facts(db_path, cik, facts)
    if facts["gp_assets"] == facts["gp_assets"]:   # not NaN
        put_edgar(db_path, cik, gp_assets=facts["gp_assets"], revenue=facts["revenue"],
                  cogs=facts["cogs"], assets=facts["assets"])
    return facts


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
    def __init__(self, rate: int, sleep=time.sleep):
        self._rate = rate
        self._tokens = rate
        self._last = time.monotonic()
        self._sleep = sleep

    def consume(self):
        now = time.monotonic()
        self._tokens = min(self._rate, self._tokens + (now - self._last) * self._rate / 60.0)
        self._last = now
        if self._tokens < 1:
            self._sleep((1 - self._tokens) * 60.0 / self._rate)
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
    edgar_bucket = _TokenBucket(EDGAR_CALLS_PER_MINUTE)

    use_finnhub = _finnhub_key_valid(fh)
    if not use_finnhub:
        logger.warning("[fundamentals] Finnhub API key invalid — skipping all Finnhub calls")

    ttl_fund  = cfg["cache"]["fundamentals_ttl_days"]
    ttl_edgar = cfg["cache"]["edgar_ttl_days"]

    rows = []
    total = len(survivors_df)
    for idx, record in survivors_df.iterrows():
        ticker = record["ticker"]
        cik    = record.get("cik", "")
        logger.info(f"[fundamentals] {idx+1}/{total} {ticker}")

        row = {"ticker": ticker}

        edgar = fetch_edgar(cik, db_path, ttl_days=ttl_edgar, bucket=edgar_bucket) if cik else None
        for col in EDGAR_FACT_COLUMNS:
            row[col] = edgar[col] if edgar and edgar.get("n_resolved", 0) > 0 else float("nan")

        cached_fund = get_fundamentals(db_path, ticker, ttl_days=ttl_fund)
        if cached_fund:
            # Sector now comes from Finnhub (src.finnhub_data) and is already
            # on the survivors frame; an old cached payload must not shadow it.
            cached_fund.pop("sector", None)
            row.update(cached_fund)
        else:
            fund = {}
            if use_finnhub:
                try:
                    bucket.consume()
                    earnings = fh.company_earnings(ticker, limit=8)
                    actuals, estimates = parse_finnhub_surprise(earnings)
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

            # Short interest used to come from a per-ticker yfinance get_info()
            # call here — one call per survivor, rate-limited at universe scale,
            # and the same call that silently blanked `sector`. It now happens
            # only for the finalists (src.enrich, Phase 1). Left NaN here.
            fund["short_float"]   = float("nan")
            fund["days_to_cover"] = float("nan")

            put_fundamentals(db_path, ticker, fund)
            row.update(fund)

        rows.append(row)
        time.sleep(0.12)

    return pd.DataFrame(rows)
