"""Finnhub-sourced universe data: market cap, industry, fundamental ratios,
earnings dates.

Why this module exists. Market cap and sector used to come from yfinance
`fast_info` / `get_info`, one call per ticker. yfinance 1.x rejects the custom
requests.Session the price fetcher passes, and Yahoo rate-limits per-ticker
calls at universe scale, so by 2026-09 the market_cap cache had not refreshed
since June and `sector` was empty for every ranked row. Finnhub's free tier
answers both in one call per ticker (`company_basic_financials`, ~130 ratios
including marketCapitalization) at 60 calls/minute, and the result is good
for a week.

Rate discipline. A run may fetch at most `max_fetch` missing tickers, so a
cold cache degrades a run (fewer names have caps) instead of stalling it; the
rest are filled by `python -m src.cache_maint warm-finnhub` or by later runs.
Everything cached is read in one bulk query.

Units. Finnhub reports marketCapitalization and shareOutstanding in MILLIONS.
Percent metrics (roeTTM, revenueGrowthTTMYoy, grossMarginTTM) are in percent
units (33.2 means 33.2%). Ratios (peTTM, pb, currentRatioQuarterly) are raw.
"""
import json
import logging
import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path

from src.cache import (
    get_fh_metrics_bulk, put_fh_metrics,
    get_fh_profiles_bulk, put_fh_profile,
    get_market_cap_stale,
)

logger = logging.getLogger(__name__)

MILLION = 1_000_000.0
# Sanity bounds for a market cap in dollars. Outside these the source is
# reporting in the wrong unit or is garbage, and the name is treated as capless.
CAP_MIN = 1_000_000.0          # $1M
CAP_MAX = 20_000_000_000_000.0 # $20T

# Finnhub `finnhubIndustry` strings for which gross-profitability, accruals and
# leverage ratios are meaningless. Checked with `is_financial()`.
FINANCIAL_INDUSTRIES = {
    "Banking", "Financial Services", "Insurance", "Real Estate",
}

# Coarse sector for the composite's sector demeaning and the sizing sector cap.
# Finnhub industries not listed map to themselves — a specific bucket beats
# "Unknown", which is what made the sector cap unenforceable.
_SECTOR_MAP = {
    "Technology": "Technology", "Semiconductors": "Technology",
    "Communications": "Communication Services", "Media": "Communication Services",
    "Telecommunication": "Communication Services",
    "Health Care": "Healthcare", "Biotechnology": "Healthcare",
    "Pharmaceuticals": "Healthcare", "Life Sciences Tools & Services": "Healthcare",
    "Medical Devices": "Healthcare", "Health Care Providers & Services": "Healthcare",
    "Banking": "Financials", "Financial Services": "Financials", "Insurance": "Financials",
    "Real Estate": "Real Estate",
    "Energy": "Energy", "Oil & Gas": "Energy",
    "Utilities": "Utilities", "Electrical Equipment": "Industrials",
    "Industrials": "Industrials", "Machinery": "Industrials", "Aerospace & Defense": "Industrials",
    "Airlines": "Industrials", "Building": "Industrials", "Construction": "Industrials",
    "Logistics & Transportation": "Industrials", "Road & Rail": "Industrials",
    "Trading Companies & Distributors": "Industrials", "Commercial Services & Supplies": "Industrials",
    "Professional Services": "Industrials",
    "Retail": "Consumer Cyclical", "Consumer products": "Consumer Defensive",
    "Textiles, Apparel & Luxury Goods": "Consumer Cyclical", "Hotels, Restaurants & Leisure": "Consumer Cyclical",
    "Automobiles": "Consumer Cyclical", "Auto Components": "Consumer Cyclical",
    "Leisure Products": "Consumer Cyclical", "Distributors": "Consumer Cyclical",
    "Beverages": "Consumer Defensive", "Food Products": "Consumer Defensive",
    "Tobacco": "Consumer Defensive", "Food & Staples Retailing": "Consumer Defensive",
    "Household Products": "Consumer Defensive", "Personal Products": "Consumer Defensive",
    "Chemicals": "Materials", "Metals & Mining": "Materials", "Packaging": "Materials",
    "Paper & Forest": "Materials", "Construction Materials": "Materials",
}


def to_finnhub_symbol(ticker: str) -> str:
    """SEC/yfinance write share classes with a dash (BRK-B); Finnhub and Alpaca
    use a dot (BRK.B)."""
    return ticker.upper().replace("-", ".")


def sector_for(industry: str | None) -> str:
    ind = (industry or "").strip()
    if not ind:
        return "Unknown"
    return _SECTOR_MAP.get(ind, ind)


def is_financial(industry: str | None) -> bool:
    return (industry or "").strip() in FINANCIAL_INDUSTRIES


class _Bucket:
    """Token bucket at `rate` calls per minute (same shape as fundamentals.py)."""

    def __init__(self, rate: int, sleep=time.sleep):
        self._rate = max(1, int(rate))
        self._tokens = float(self._rate)
        self._last = time.monotonic()
        self._sleep = sleep

    def consume(self):
        now = time.monotonic()
        self._tokens = min(self._rate, self._tokens + (now - self._last) * self._rate / 60.0)
        self._last = now
        if self._tokens < 1:
            self._sleep((1 - self._tokens) * 60.0 / self._rate)
            self._tokens = 0.0
        else:
            self._tokens -= 1


def _client():
    import finnhub
    return finnhub.Client(api_key=os.environ.get("FINNHUB_API_KEY", ""))


def _status_of(exc: Exception) -> int | None:
    """HTTP status carried by a finnhub.FinnhubAPIException, else None."""
    code = getattr(exc, "status_code", None)
    if code is not None:
        try:
            return int(code)
        except (TypeError, ValueError):
            return None
    return None


def _call(fn, *args, sleep=time.sleep, **kwargs):
    """One Finnhub call with a single retry on 429. Returns (payload, error)
    where error is None, "forbidden" (403 — stop asking), or "error"."""
    for attempt in range(2):
        try:
            return fn(*args, **kwargs), None
        except Exception as e:  # noqa: BLE001 — every failure is one ticker, never the run
            status = _status_of(e)
            if status == 429 and attempt == 0:
                sleep(60.0)
                continue
            if status == 403:
                return None, "forbidden"
            return None, "error"
    return None, "error"


def fetch_metrics(
    tickers: list[str],
    db_path: str,
    ttl_days: int = 7,
    max_fetch: int = 600,
    calls_per_minute: int = 60,
    client=None,
    sleep=time.sleep,
) -> tuple[dict[str, dict], dict]:
    """Basic-financials `metric` dict per ticker.

    Returns (metrics, stats). `metrics[t]` is {} when Finnhub has no data for
    the symbol (cached so it is not re-asked daily). Tickers beyond `max_fetch`
    that were not cached are simply absent — counted in stats["unfetched"].
    """
    cached = get_fh_metrics_bulk(db_path, ttl_days)
    wanted = [t.upper() for t in tickers]
    out = {t: cached[t] for t in wanted if t in cached}
    missing = [t for t in wanted if t not in cached]
    stats = {"cached": len(out), "fetched": 0, "failed": 0,
             "unfetched": max(0, len(missing) - max_fetch), "forbidden": False}
    if not missing or max_fetch <= 0:
        return out, stats

    fh = client or _client()
    bucket = _Bucket(calls_per_minute, sleep=sleep)
    for t in missing[:max_fetch]:
        bucket.consume()
        payload, err = _call(fh.company_basic_financials, to_finnhub_symbol(t), "all", sleep=sleep)
        if err == "forbidden":
            stats["forbidden"] = True
            logger.warning("[finnhub] 403 on basic financials — key lacks access, stopping")
            break
        if err:
            stats["failed"] += 1
            continue
        metric = (payload or {}).get("metric") or {}
        put_fh_metrics(db_path, t, metric)
        out[t] = metric
        stats["fetched"] += 1
    return out, stats


def fetch_profiles(
    tickers: list[str],
    db_path: str,
    cap_ttl_days: int = 7,
    max_fetch: int = 600,
    calls_per_minute: int = 60,
    client=None,
    sleep=time.sleep,
) -> tuple[dict[str, dict], dict]:
    """Industry + market cap + shares outstanding per ticker via company_profile2.

    Industry never expires. The cap is refreshed when older than cap_ttl_days,
    but a stale row is still returned (with its fetched_at) if the refresh
    budget runs out — a week-old cap is fine for a $300M gate.
    """
    cached = get_fh_profiles_bulk(db_path)
    wanted = [t.upper() for t in tickers]
    cutoff = (datetime.utcnow() - timedelta(days=cap_ttl_days)).isoformat()
    need = [t for t in wanted if t not in cached or (cached[t].get("fetched_at") or "") < cutoff]
    out = {t: cached[t] for t in wanted if t in cached}
    stats = {"cached": len(out), "fetched": 0, "failed": 0,
             "unfetched": max(0, len(need) - max_fetch), "forbidden": False}
    if not need or max_fetch <= 0:
        return out, stats

    fh = client or _client()
    bucket = _Bucket(calls_per_minute, sleep=sleep)
    for t in need[:max_fetch]:
        bucket.consume()
        payload, err = _call(fh.company_profile2, sleep=sleep, symbol=to_finnhub_symbol(t))
        if err == "forbidden":
            stats["forbidden"] = True
            break
        if err:
            stats["failed"] += 1
            continue
        p = payload or {}
        industry = p.get("finnhubIndustry") or ""
        cap = _dollars(p.get("marketCapitalization"))
        shares = _f(p.get("shareOutstanding"))
        shares = shares * MILLION if shares and shares > 0 else None
        put_fh_profile(db_path, t, industry, cap, shares)
        out[t] = {"industry": industry, "market_cap": cap, "shares_out": shares,
                  "fetched_at": datetime.utcnow().isoformat()}
        stats["fetched"] += 1
    return out, stats


def _f(v) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if x == x else None  # NaN -> None


def _dollars(cap_millions) -> float | None:
    """Finnhub millions -> dollars, or None when absent/implausible."""
    x = _f(cap_millions)
    if x is None or x <= 0:
        return None
    dollars = x * MILLION
    if not (CAP_MIN <= dollars <= CAP_MAX):
        return None
    return dollars


def resolve_market_cap(
    ticker: str,
    price: float | None,
    metrics: dict[str, dict],
    profiles: dict[str, dict],
    db_path: str | None = None,
) -> tuple[float | None, str]:
    """Best available cap in dollars and where it came from.

    Precedence: basic-financials cap -> profile cap -> profile shares × price ->
    legacy `market_cap` table (any age) -> None. Every source is bounds-checked.
    """
    t = ticker.upper()
    m = metrics.get(t) or {}
    cap = _dollars(m.get("marketCapitalization"))
    if cap is not None:
        return cap, "fh_metrics"
    p = profiles.get(t) or {}
    cap = p.get("market_cap")
    if cap is not None and CAP_MIN <= float(cap) <= CAP_MAX:
        return float(cap), "fh_profile"
    shares = p.get("shares_out")
    if shares and price and price > 0:
        est = float(shares) * float(price)
        if CAP_MIN <= est <= CAP_MAX:
            return est, "shares_x_price"
    if db_path:
        legacy = get_market_cap_stale(db_path, t)
        if legacy is not None and CAP_MIN <= float(legacy) <= CAP_MAX:
            return float(legacy), "legacy_cache"
    return None, "none"


def attach_market_caps(df, metrics: dict[str, dict], profiles: dict[str, dict],
                       db_path: str | None = None):
    """Add `market_cap`, `cap_source`, `industry`, `sector`, `is_financial`
    columns to a factors frame (must have `ticker` and `price`). Returns
    (df, stats) where stats counts names per cap source."""
    import pandas as pd

    caps, sources, inds, secs, fins = [], [], [], [], []
    for _, row in df.iterrows():
        t = str(row["ticker"]).upper()
        price = row.get("price")
        cap, src = resolve_market_cap(t, price, metrics, profiles, db_path)
        caps.append(cap if cap is not None else float("nan"))
        sources.append(src)
        industry = (profiles.get(t) or {}).get("industry") or ""
        inds.append(industry)
        secs.append(sector_for(industry))
        fins.append(is_financial(industry))
    out = df.copy()
    out["market_cap"] = pd.Series(caps, index=out.index, dtype="float64")
    out["cap_source"] = sources
    out["industry"] = inds
    out["sector"] = secs
    out["is_financial"] = fins
    stats = {s: sources.count(s) for s in set(sources)}
    stats["dropped_no_cap"] = sources.count("none")
    return out, stats


# ── earnings calendar ────────────────────────────────────────────────────────

EARNINGS_CACHE = Path("data") / "earnings_calendar.json"


def fetch_earnings_calendar(
    days_ahead: int = 45,
    cache_path: Path | None = None,
    ttl_hours: int = 20,
    client=None,
    today: date | None = None,
) -> dict[str, str]:
    """{ticker: 'YYYY-MM-DD'} for every US name reporting within days_ahead.

    One call for the whole market (Finnhub's calendar endpoint is free and
    accepts an empty symbol). Cached on disk because every consumer — the
    screener finalists, the portfolio engine, the exit plan — wants it on the
    same evening. Fail-soft: on any error the stale cache is returned, or {}.
    """
    today = today or date.today()
    path = cache_path or EARNINGS_CACHE
    if path.exists():
        try:
            blob = json.loads(path.read_text())
            fetched = datetime.fromisoformat(blob.get("fetched_at", "1970-01-01"))
            if datetime.utcnow() - fetched < timedelta(hours=ttl_hours):
                return blob.get("dates", {})
        except (OSError, ValueError, json.JSONDecodeError):
            blob = {}
    else:
        blob = {}

    fh = client or _client()
    payload, err = _call(fh.earnings_calendar, _from=today.isoformat(),
                         to=(today + timedelta(days=days_ahead)).isoformat(), symbol="")
    if err:
        logger.warning(f"[finnhub] earnings calendar unavailable ({err}); using stale cache")
        return blob.get("dates", {}) if isinstance(blob, dict) else {}

    dates: dict[str, str] = {}
    for e in (payload or {}).get("earningsCalendar", []) or []:
        sym = str(e.get("symbol") or "").upper().replace(".", "-")
        d = str(e.get("date") or "")
        if not sym or len(d) != 10:
            continue
        # Keep the EARLIEST upcoming date per symbol.
        if sym not in dates or d < dates[sym]:
            dates[sym] = d
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"fetched_at": datetime.utcnow().isoformat(), "dates": dates}))
    except OSError:
        pass
    return dates


def days_to_earnings(ticker: str, calendar: dict[str, str], today: date | None = None) -> int | None:
    """Calendar days until the ticker's next report, or None if not scheduled
    inside the calendar window. Never negative: a date before today is stale."""
    today = today or date.today()
    d = calendar.get(ticker.upper())
    if not d:
        return None
    try:
        delta = (date.fromisoformat(d) - today).days
    except ValueError:
        return None
    return delta if delta >= 0 else None
