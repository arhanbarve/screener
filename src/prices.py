import os
import math
import time
import logging
import requests
import pandas as pd
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from datetime import date, timedelta
from requests.adapters import HTTPAdapter
from typing import Optional
from src.universe import to_alpaca_symbol
from src.factors import (
    mom_12_1, mom_1m, rs_vs_spy, rs_slope, residual_momentum,
    pct_from_52w_high, breakout_flag, avg_dollar_vol,
    rsi_14, macd_state, vol_surge_ratio, entry_grade,
    stochastic, bollinger_pct_b, adx_14, mfi_14, atr_14,
    obv_slope, parabolic_sar,
)
from src.cache import (
    get_prices, put_prices,
    put_failed_ticker, is_failed_ticker,
)


_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
yf.set_tz_cache_location(_DATA_DIR)


class _TimeoutAdapter(HTTPAdapter):
    def send(self, *args, **kwargs):
        kwargs.setdefault("timeout", 8)
        return super().send(*args, **kwargs)

_yf_session = requests.Session()
_yf_session.mount("https://", _TimeoutAdapter())
_yf_session.mount("http://", _TimeoutAdapter())

logger = logging.getLogger(__name__)

BATCH_SIZE = 200
# A batch that raises is split in half and retried, down to this size. Below
# it the tickers are skipped for THIS RUN ONLY — never quarantined — because a
# batch-level exception says nothing about any individual ticker.
MIN_SPLIT_BATCH = 25
BATCH_RETRY_SLEEP = 5.0
HISTORY_DAYS = 420
SPY_FETCH_RETRIES = 3
SPY_FETCH_RETRY_DELAY = 5.0
# How stale a cached SPY series may be before it's useless for relative strength.
SPY_STALE_CACHE_MAX_HOURS = 240
FAILED_TICKER_TTL_DAYS = 7


class BatchFetchError(RuntimeError):
    """yfinance raised for a whole batch. Distinct from 'ticker had no rows'."""


# yf.download() has no wall-clock timeout of its own — a stalled connection
# (server accepts, never responds) blocks the call forever with no exception
# ever raised, bypassing fetch_with_split's halving-retry entirely. Enforced
# here with a thread + future deadline so a hang becomes a BatchFetchError
# like any other batch failure, instead of hanging the whole run indefinitely.
#
# 60s, not longer: a healthy 200-ticker/420-day batch completes in ~4-5s
# (measured directly), so 60s is generous headroom for real slowness while
# still failing a genuinely stalled connection fast. A 2026-09-04 live run
# hit a batch where every level of the halving retry stalled — at the
# previous 180s this took ~45 min to give up; at 60s the same worst case is
# ~15 min. The stall itself (observed as a TCP SYN_SENT that never got a
# reply) is intermittent Yahoo/network flakiness, not something a longer
# timeout would have fixed — it would only have waited longer to fail.
DOWNLOAD_TIMEOUT_SEC = 60


def _fetch_batch_yfinance(tickers: list[str], start: str | None = None, end: str | None = None) -> dict[str, pd.DataFrame]:
    """Batched yfinance download. Defaults to the rolling 420-day window used by
    the live screener; pass start/end for an explicit historical range (e.g.
    multi-year backtests) instead.

    Raises BatchFetchError when yfinance itself fails. Returns {} only when
    yfinance answered and had no rows for any ticker — the two cases used to be
    conflated, and on 2026-08-24 one exception quarantined 4,344 tickers.
    """
    joined = " ".join(tickers)

    def _do_download():
        if start is not None:
            return yf.download(
                joined, start=start, end=end, interval="1d",
                auto_adjust=True, progress=False, group_by="ticker", threads=True,
            )
        return yf.download(
            joined,
            period="420d",
            interval="1d",
            auto_adjust=True,
            progress=False,
            group_by="ticker",
            threads=True,
        )

    executor = ThreadPoolExecutor(max_workers=1)
    try:
        raw = executor.submit(_do_download).result(timeout=DOWNLOAD_TIMEOUT_SEC)
    except FutureTimeoutError as e:
        logger.warning(f"yfinance batch timed out after {DOWNLOAD_TIMEOUT_SEC}s")
        raise BatchFetchError(f"timed out after {DOWNLOAD_TIMEOUT_SEC}s") from e
    except Exception as e:
        logger.warning(f"yfinance batch failed: {e}")
        raise BatchFetchError(str(e)) from e
    finally:
        # Don't block on a possibly-still-hung download thread; let it be
        # abandoned and cleaned up whenever (or if) it finally returns.
        executor.shutdown(wait=False)

    if raw is None or raw.empty:
        return {}

    result = {}
    # yfinance 1.x always returns MultiIndex columns (Price, Ticker)
    if isinstance(raw.columns, pd.MultiIndex):
        l0 = set(raw.columns.get_level_values(0))
        l1 = set(raw.columns.get_level_values(1))
        ticker_set = {t.upper() for t in tickers}
        if ticker_set & l1:
            # (Price, Ticker) format
            for t in tickers:
                tu = t.upper()
                cols = [(p, tu) for p in l0 if (p, tu) in raw.columns]
                if not cols:
                    continue
                df = raw[[c for c in raw.columns if c[1] == tu]].copy()
                df.columns = [c[0].lower() for c in df.columns]
                df = df.dropna(how="all")
                if len(df) > 0:
                    result[t] = df
        else:
            # (Ticker, Price) format
            for t in tickers:
                tu = t.upper()
                if tu not in l0:
                    continue
                df = raw[tu].copy()
                df.columns = [c.lower() for c in df.columns]
                df = df.dropna(how="all")
                if len(df) > 0:
                    result[t] = df
    else:
        # Flat columns — single ticker fallback
        t = tickers[0]
        df = raw.copy()
        df.columns = [c.lower() for c in df.columns]
        df = df.dropna(how="all")
        if len(df) > 0:
            result[t] = df
    return result


ALPACA_DATA_URL = "https://data.alpaca.markets"
ALPACA_BAR_REQUEST_TIMEOUT_SEC = 30
# Alpaca caps rows per response page, not symbols per request — a 200-symbol,
# 420-day batch needs several pages. Looped via next_page_token until it's None.
ALPACA_PAGE_LIMIT = 10000


def _alpaca_headers() -> dict:
    key = os.environ.get("ALPACA_API_KEY")
    secret = os.environ.get("ALPACA_SECRET_KEY")
    if not key or not secret:
        raise BatchFetchError("Missing ALPACA_API_KEY / ALPACA_SECRET_KEY env vars")
    return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}


def _fetch_batch_alpaca(tickers: list[str], start: str | None = None, end: str | None = None) -> dict[str, pd.DataFrame]:
    """Batched historical daily bars from Alpaca — an alternative to yfinance.

    Added 2026-09-05: Yahoo Finance rate-limited this environment mid-overhaul
    (confirmed via clean HTTP 429s on finance.yahoo.com), and yfinance's own
    connection stalls under sustained use could not be reliably told apart
    from a genuine outage. Alpaca's bars endpoint is an authenticated,
    documented REST API (not scraped) already used for the broker/paper
    account, with a published 200 req/min limit and no anti-bot heuristics.

    Same contract as `_fetch_batch_yfinance`: returns {ticker: OHLCV df} with
    lowercase open/high/low/close/volume columns; raises BatchFetchError when
    the request itself fails; a ticker Alpaca has no bars for is simply
    absent from the result (no separate exception for that case — Alpaca
    silently omits unknown/no-data symbols rather than erroring).

    `feed=iex` is required — the default SIP feed 403s on this account's
    data subscription tier ("subscription does not permit querying recent
    SIP data"). `adjustment=all` matches yfinance's `auto_adjust=True`
    (splits and dividends both applied).
    """
    if start is None:
        end_date = date.today()
        start_date = end_date - timedelta(days=HISTORY_DAYS)
        start, end = start_date.isoformat(), end_date.isoformat()

    alpaca_to_orig: dict[str, str] = {}
    for t in tickers:
        alpaca_to_orig[to_alpaca_symbol(t)] = t
    symbols = list(alpaca_to_orig.keys())

    def _do_page(page_token: str | None):
        params = {
            "symbols": ",".join(symbols), "timeframe": "1Day",
            "start": start, "end": end, "adjustment": "all",
            "limit": ALPACA_PAGE_LIMIT, "feed": "iex",
        }
        if page_token:
            params["page_token"] = page_token
        resp = requests.get(f"{ALPACA_DATA_URL}/v2/stocks/bars", headers=_alpaca_headers(),
                            params=params, timeout=ALPACA_BAR_REQUEST_TIMEOUT_SEC)
        if resp.status_code == 429:
            raise BatchFetchError(f"rate limited: {resp.text[:200]}")
        if resp.status_code != 200:
            raise BatchFetchError(f"{resp.status_code}: {resp.text[:200]}")
        return resp.json()

    raw: dict[str, list] = {}
    page_token = None
    while True:
        executor = ThreadPoolExecutor(max_workers=1)
        try:
            data = executor.submit(_do_page, page_token).result(timeout=ALPACA_BAR_REQUEST_TIMEOUT_SEC)
        except FutureTimeoutError as e:
            raise BatchFetchError(f"timed out after {ALPACA_BAR_REQUEST_TIMEOUT_SEC}s") from e
        except BatchFetchError:
            raise
        except Exception as e:
            raise BatchFetchError(str(e)) from e
        finally:
            executor.shutdown(wait=False)
        for sym, bars in (data.get("bars") or {}).items():
            raw.setdefault(sym, []).extend(bars)
        page_token = data.get("next_page_token")
        if not page_token:
            break

    result: dict[str, pd.DataFrame] = {}
    for sym, bars in raw.items():
        orig = alpaca_to_orig.get(sym)
        if not orig or not bars:
            continue
        df = pd.DataFrame(bars)
        df.index = pd.to_datetime(df["t"]).dt.tz_localize(None)
        df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
        df = df[["open", "high", "low", "close", "volume"]]
        if len(df) > 0:
            result[orig] = df
    return result


def fetch_with_split(tickers: list[str], fetch=_fetch_batch_alpaca,
                     min_batch: int = MIN_SPLIT_BATCH, sleep=time.sleep) -> tuple[dict[str, pd.DataFrame], list[str], list[str]]:
    """Fetch a batch, halving and retrying on BatchFetchError.

    Returns (fetched, no_data, unfetched):
      fetched   — ticker -> OHLCV
      no_data   — yfinance answered and had no rows for these (quarantine-worthy)
      unfetched — every retry raised; skip this run, do NOT quarantine
    """
    try:
        got = fetch(tickers)
    except BatchFetchError:
        if len(tickers) <= min_batch:
            return {}, [], list(tickers)
        sleep(BATCH_RETRY_SLEEP)
        mid = len(tickers) // 2
        left = fetch_with_split(tickers[:mid], fetch, min_batch, sleep)
        right = fetch_with_split(tickers[mid:], fetch, min_batch, sleep)
        return ({**left[0], **right[0]}, left[1] + right[1], left[2] + right[2])
    no_data = [t for t in tickers if t not in got]
    return got, no_data, []


def compute_price_factors(
    ticker: str,
    df: pd.DataFrame,
    spy_df: pd.DataFrame,
    market_cap: Optional[float] = None,
) -> Optional[dict]:
    """Price-derived factors for one ticker. `market_cap` is passed through
    when the caller already knows it; the live pipeline leaves it None and
    attaches caps afterwards (src.finnhub_data.attach_market_caps)."""
    if len(df) < 252:
        return None
    close  = df["close"]
    volume = df["volume"]
    spy_close = spy_df["close"]

    try:
        high   = df["high"]
        low    = df["low"]

        price_now   = float(close.iloc[-1])
        sma20       = float(close.iloc[-20:].mean())
        sma50       = float(close.iloc[-50:].mean())
        sma200      = float(close.iloc[-200:].mean())
        rsi_val     = rsi_14(close)
        macd        = macd_state(close)
        vol_ratio   = vol_surge_ratio(volume)
        above_sma20 = price_now >= sma20

        stoch_k, stoch_d, stoch_cross = stochastic(high, low, close)
        bb_pct_b, bb_width            = bollinger_pct_b(close)
        adx_val                       = adx_14(high, low, close)
        mfi_val                       = mfi_14(high, low, close, volume)
        obv_slope_val                 = obv_slope(close, volume)
        _, sar_bullish, _             = parabolic_sar(high, low)
        atr_val                       = atr_14(high, low, close)

        return {
            "ticker": ticker,
            "market_cap": market_cap,
            "mom_12_1": mom_12_1(close),
            "mom_1m": mom_1m(close),
            "residual_mom": residual_momentum(close, spy_close),
            "rs_6m": rs_vs_spy(close, spy_close, window=126),
            "rs_3m": rs_vs_spy(close, spy_close, window=63),
            "rs_slope": rs_slope(close, spy_close),
            "pct_from_high": pct_from_52w_high(close),
            "breakout": breakout_flag(close, volume),
            "avg_dollar_vol_20d": avg_dollar_vol(close, volume, window=20),
            "price": price_now,
            "sma_20": sma20,
            "sma_50": sma50,
            "sma_200": sma200,
            "above_sma20": above_sma20,
            "above_sma50": price_now >= sma50,
            "atr_14": atr_val,
            "rsi_14": rsi_val,
            "macd": macd,
            "vol_surge": vol_ratio,
            "stoch_k": stoch_k,
            "stoch_d": stoch_d,
            "stoch_cross": stoch_cross,
            "bb_pct_b": bb_pct_b,
            "bb_width": bb_width,
            "adx": adx_val,
            "mfi": mfi_val,
            "entry": entry_grade(
                rsi_val, macd, vol_ratio, above_sma20,
                adx_val, stoch_k, stoch_d, stoch_cross, bb_pct_b, mfi_val,
                obv_slope_val, sar_bullish,
            ),
            "close_series": close,
        }
    except Exception as e:
        logger.warning(f"[prices] factor error for {ticker}: {e}")
        return None


def apply_adv_gate(factors_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Dollar-volume gate only. Runs BEFORE market caps are fetched so the cap
    lookups are spent on liquid names."""
    min_vol = cfg["liquidity_gate"]["min_avg_dollar_vol_20d"]
    min_price = float(cfg.get("universe", {}).get("min_price", 0.0) or 0.0)
    before = len(factors_df)
    if before == 0:
        return factors_df
    mask = factors_df["avg_dollar_vol_20d"] >= min_vol
    if min_price > 0:
        mask &= factors_df["price"] >= min_price
    result = factors_df[mask].reset_index(drop=True)
    logger.info(f"[adv_gate] {before} → {len(result)} survivors")
    print(f"[adv_gate] {before} in → {len(result)} survivors (ADV ≥ ${min_vol:,.0f}, price ≥ ${min_price:g})")
    return result


def apply_liquidity_gate(factors_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Market-cap + dollar-volume gate. Names with a NaN cap fail (NaN >= x is
    False) — callers must count them separately (see attach_market_caps)."""
    gate = cfg["liquidity_gate"]
    min_mcap = gate["min_market_cap"]
    min_vol  = gate["min_avg_dollar_vol_20d"]
    before = len(factors_df)
    if before == 0 or "market_cap" not in factors_df.columns:
        return factors_df.iloc[0:0] if before else factors_df
    result = factors_df[
        (factors_df["market_cap"] >= min_mcap) &
        (factors_df["avg_dollar_vol_20d"] >= min_vol)
    ].reset_index(drop=True)
    logger.info(f"[liquidity_gate] {before} → {len(result)} survivors")
    print(f"[liquidity_gate] {before} in → {len(result)} survivors")
    return result


def fetch_all_prices(
    universe_df: pd.DataFrame,
    cfg: dict,
    db_path: str,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict]:
    """OHLCV for the universe plus price factors for every name with ≥252 bars.

    Returns (price_store, factors_df, stats). factors_df has NOT been through
    the cap gate — run.py attaches caps from Finnhub and gates afterwards.
    stats: universe, from_cache, fetched, no_data (quarantined this run),
    unfetched (batch errors; skipped, not quarantined), priced (factor rows).
    """
    ttl = cfg["cache"]["price_ttl_hours"]
    tickers = universe_df["ticker"].tolist()

    spy_cached = get_prices(db_path, "SPY", ttl_hours=ttl)
    if spy_cached is not None and len(spy_cached) >= 252:
        spy_df = spy_cached
    else:
        spy_df = pd.DataFrame()
        for attempt in range(SPY_FETCH_RETRIES):
            try:
                spy_data = _fetch_batch_alpaca(["SPY"])
            except BatchFetchError:
                spy_data = {}
            spy_df = spy_data.get("SPY", pd.DataFrame())
            if not spy_df.empty:
                break
            if attempt < SPY_FETCH_RETRIES - 1:
                logger.warning(f"[prices] SPY fetch attempt {attempt+1}/{SPY_FETCH_RETRIES} failed — retrying")
                time.sleep(SPY_FETCH_RETRY_DELAY)
        if not spy_df.empty:
            put_prices(db_path, "SPY", spy_df)
        else:
            stale = get_prices(db_path, "SPY", ttl_hours=SPY_STALE_CACHE_MAX_HOURS)
            if stale is not None and len(stale) >= 252:
                logger.warning(
                    f"[prices] SPY live fetch failed after retries — using stale cache "
                    f"(last row {stale.index[-1].date()})"
                )
                spy_df = stale
    if spy_df is None or spy_df.empty:
        raise RuntimeError("Failed to fetch SPY — cannot compute relative strength")

    price_store: dict[str, pd.DataFrame] = {"SPY": spy_df}
    factor_rows = []
    stats = {"universe": len(tickers), "from_cache": 0, "fetched": 0,
             "no_data": 0, "unfetched": 0, "skipped_quarantined": 0}
    names = universe_df.set_index("ticker")["name"].to_dict() if "name" in universe_df.columns else {}
    ciks  = universe_df.set_index("ticker")["cik"].to_dict() if "cik" in universe_df.columns else {}

    batches = [tickers[i:i+BATCH_SIZE] for i in range(0, len(tickers), BATCH_SIZE)]
    for batch_idx, batch in enumerate(batches):
        logger.info(f"[prices] batch {batch_idx+1}/{len(batches)} ({len(batch)} tickers)")

        to_fetch = []
        for t in batch:
            if is_failed_ticker(db_path, t, ttl_days=FAILED_TICKER_TTL_DAYS):
                stats["skipped_quarantined"] += 1
                continue
            cached = get_prices(db_path, t, ttl_hours=ttl)
            if cached is not None and len(cached) >= 252:
                price_store[t] = cached
                stats["from_cache"] += 1
            else:
                to_fetch.append(t)

        if to_fetch:
            fetched, no_data, unfetched = fetch_with_split(to_fetch)
            for t, df in fetched.items():
                put_prices(db_path, t, df)
                price_store[t] = df
            stats["fetched"] += len(fetched)
            for t in no_data:
                put_failed_ticker(db_path, t, "no_data")
                logger.info(f"[prices] marked {t} as failed (no data returned)")
            stats["no_data"] += len(no_data)
            if unfetched:
                stats["unfetched"] += len(unfetched)
                logger.warning(f"[prices] batch {batch_idx+1}: {len(unfetched)} tickers unfetched "
                               f"after split retries (not quarantined)")

        for t in batch:
            df = price_store.get(t)
            if df is None or len(df) < 252:
                continue
            row = compute_price_factors(t, df, spy_df, None)
            if row is not None:
                row["name"] = names.get(t, "") or ""
                row["cik"]  = ciks.get(t, "") or ""
                factor_rows.append(row)

        if batch_idx < len(batches) - 1:
            time.sleep(1.0)

    factors_df = pd.DataFrame([{k: v for k, v in r.items() if k != "close_series"} for r in factor_rows])
    stats["priced"] = len(factors_df)
    print(f"[prices] {len(tickers)} universe → {len(factors_df)} priced "
          f"(cache {stats['from_cache']}, fetched {stats['fetched']}, no_data {stats['no_data']}, "
          f"unfetched {stats['unfetched']}, quarantined {stats['skipped_quarantined']})")
    return price_store, factors_df, stats
