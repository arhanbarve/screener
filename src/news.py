"""News analysis pipeline: entry signal overlay for momentum screener."""
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import finnhub
import pandas as pd

from src.cache import get_news_sentiment, put_news_sentiment

# z-score column → human label for context injection
_FACTOR_LABELS = {
    "z_mom_12_1":       "12-month momentum",
    "z_rs_6m":          "6-month relative strength vs SPY",
    "z_rs_accel":       "RS acceleration (3m vs 6m pace)",
    "z_rs_slope":       "RS momentum slope",
    "z_streak_z":       "sustained appearance streak",
    "z_sue":            "earnings surprise (SUE)",
    "z_rev_breadth":    "analyst revision breadth (upgrades vs downgrades)",
    "z_rev_magnitude":  "analyst estimate revision magnitude",
    "z_gp_assets":      "gross profitability quality",
    "z_insider_z":      "insider cluster buying",
    "z_trend_score":    "trend strength (ADX + MACD + SMA50)",
    "z_momo_osc_score": "momentum oscillator alignment",
}

_ENTRY_FALLBACK: dict = {
    "entry_signal": "wait",
    "catalyst": "none",
    "priced_in": False,
    "duration": "noise",
    "thesis_consistency": "neutral",
    "conviction_delta": 0,
    "reasoning": "Analysis unavailable.",
}

_CARD_FALLBACK: dict = {
    "sentiment": "neutral",
    "summary": "Analysis unavailable.",
    "key_risks": [],
}


# ── Thread-safe Finnhub rate limiter ─────────────────────────────────────────

class _ThreadSafeTokenBucket:
    def __init__(self, rate: int):
        self._rate = rate
        self._tokens = float(rate)
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def consume(self):
        with self._lock:
            now = time.monotonic()
            self._tokens = min(self._rate, self._tokens + (now - self._last) * self._rate / 60.0)
            self._last = now
            if self._tokens < 1:
                sleep_time = (1 - self._tokens) * 60.0 / self._rate
                self._tokens = 0.0
            else:
                self._tokens -= 1
                sleep_time = 0.0
        if sleep_time > 0:
            time.sleep(sleep_time)


_bucket: _ThreadSafeTokenBucket | None = None
_bucket_init_lock = threading.Lock()


def _get_bucket(rate: int) -> _ThreadSafeTokenBucket:
    global _bucket
    with _bucket_init_lock:
        if _bucket is None:
            _bucket = _ThreadSafeTokenBucket(rate)
        return _bucket


# ── Finnhub fetchers ─────────────────────────────────────────────────────────

def _fh() -> finnhub.Client:
    return finnhub.Client(api_key=os.environ.get("FINNHUB_API_KEY", ""))


def get_market_news(n: int = 20) -> list[dict]:
    try:
        return (_fh().general_news("general") or [])[:n]
    except Exception:
        return []


def get_stock_news(ticker: str, days: int = 7) -> list[dict]:
    try:
        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        return (_fh().company_news(ticker, _from=start, to=end) or [])[:10]
    except Exception:
        return []


# ── Claude helpers ────────────────────────────────────────────────────────────

def _parse_json(text: str) -> dict | None:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
    return None


def _claude(model: str, prompt: str, max_tokens: int) -> str:
    import anthropic
    msg = anthropic.Anthropic().messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return next((b.text for b in reversed(msg.content) if b.type == "text"), "")


# ── Pipeline: pre-filter → context → analyze ─────────────────────────────────

def _prefilter_articles(ticker: str, articles: list[dict]) -> list[dict]:
    """Haiku: classify each article as material or noise. Returns material subset."""
    if not articles:
        return []
    lines = []
    for i, a in enumerate(articles, 1):
        h = a.get("headline", "")
        s = a.get("summary", "")[:200]
        lines.append(f"{i}. {h} — {s}")
    prompt = (
        f"For stock {ticker}, classify each article as material or noise.\n"
        "Material = affects earnings trajectory, analyst estimates, guidance, "
        "competitive position, or institutional flows.\n"
        "Noise = price-recap ('X rose 4%'), listicles, boilerplate, options-volume bots.\n\n"
        "Articles:\n" + "\n".join(lines) + "\n\n"
        'Return ONLY JSON: {"material":[1,3]}'
    )
    try:
        text = _claude("claude-haiku-4-5", prompt, max_tokens=150)
        parsed = _parse_json(text)
        if parsed and "material" in parsed:
            indices = {int(i) - 1 for i in parsed["material"] if str(i).isdigit()}
            result = [a for i, a in enumerate(articles) if i in indices]
            return result if result else articles
    except Exception:
        pass
    return articles  # fail-open


def _build_context_block(rank: int, total: int, row: pd.Series, sector_signal: dict | None) -> str:
    ticker = str(row.get("ticker", "?"))
    lines = [f"{ticker} ranks #{rank} of {total}. Ranking drivers (z-scores vs peers):"]
    found_any = False
    for col, label in _FACTOR_LABELS.items():
        val = row.get(col)
        if val is not None and not pd.isna(val) and float(val) > 0.8:
            lines.append(f"  - {label}: z={float(val):+.1f}")
            found_any = True
    if not found_any:
        lines.append("  - (moderate across all factors — no single dominant driver)")
    entry = row.get("entry", "?")
    conv = int(row.get("conviction", 5) or 5)
    lines.append(f"Technical entry grade: {entry}   Conviction: {conv}/10")
    if sector_signal:
        sector = str(row.get("sector", ""))
        d = sector_signal.get("direction", "")
        s = sector_signal.get("strength", "")
        r = sector_signal.get("reason", "")
        lines.append(f"SECTOR CONTEXT: {sector} — {d.upper()} ({s}). {r}")
    return "\n".join(lines)


def _analyze_stock(ticker: str, articles: list[dict], context_block: str) -> dict:
    """Sonnet: cross-reference news against screener context. Returns entry overlay."""
    material = _prefilter_articles(ticker, articles)
    if not material:
        return {**_ENTRY_FALLBACK, "reasoning": f"No material news found for {ticker}."}

    now = datetime.now()
    article_lines = []
    for a in material[:6]:
        src = a.get("source", "?")
        try:
            age_h = int((now - datetime.fromtimestamp(a.get("datetime", 0))).total_seconds() / 3600)
            age = f"{age_h}h ago"
        except Exception:
            age = "?"
        h = a.get("headline", "")
        s = a.get("summary", "")[:200]
        article_lines.append(f"[{src} | {age}] {h} — {s}")

    prompt = f"""You are a momentum trader's analyst.

{context_block}

Material news (last 7 days):
{chr(10).join(article_lines)}

Reason through in order:
1. Does any item change the EARNINGS TRAJECTORY (likely to trigger sell-side revisions)?
2. Is this PRICED IN (stock already ran on this) or fresh?
3. Does any item CONTRADICT the ranking drivers above?
4. DURATION: noise (1-day), days, or multi-week catalyst?
5. Does the sector context amplify or offset the company news?

Return ONLY JSON:
{{
  "entry_signal": "confirm_entry|wait|avoid",
  "catalyst": "estimate_up|estimate_down|none",
  "priced_in": false,
  "duration": "noise|days|weeks",
  "thesis_consistency": "confirms|neutral|contradicts",
  "conviction_delta": 0,
  "reasoning": "1 sentence naming a specific screener factor"
}}"""

    try:
        text = _claude("claude-sonnet-4-6", prompt, max_tokens=600)
        parsed = _parse_json(text)
        if parsed:
            parsed["conviction_delta"] = max(-2, min(2, int(parsed.get("conviction_delta", 0))))
            return parsed
    except Exception:
        pass
    return dict(_ENTRY_FALLBACK)


def _analyze_market(articles: list[dict], sectors_in_play: list[str]) -> dict:
    """Sonnet: produce sector_signals map from market headlines."""
    if not articles:
        return {"regime_note": "", "sector_signals": {}}
    headlines = "\n".join(f"- {a['headline']}" for a in articles[:15] if a.get("headline"))
    sectors_str = ", ".join(sectors_in_play) if sectors_in_play else "various"
    prompt = f"""You are a macro strategist for equity momentum funds.

Headlines:
{headlines}

Sectors in today's screener top picks: {sectors_str}

Which of these sectors face near-term headwinds or tailwinds based on today's news?

Return ONLY JSON:
{{
  "regime_note": "1 sentence on broad tape",
  "sector_signals": {{
    "Technology": {{"direction": "headwind|tailwind", "strength": "strong|mild", "reason": "brief"}}
  }}
}}
Only emit entries when news genuinely shifts a sector's outlook. Restrict to: {sectors_str}"""
    try:
        text = _claude("claude-sonnet-4-6", prompt, max_tokens=800)
        parsed = _parse_json(text)
        if parsed:
            return parsed
    except Exception:
        pass
    return {"regime_note": "", "sector_signals": {}}


def _process_one(args: tuple) -> tuple[str, dict]:
    """Thread worker: fetch + analyze one stock."""
    ticker, row, rank, total, sector_signal, finnhub_rate, db_path, ttl_hours = args
    cached = get_news_sentiment(db_path, ticker, ttl_hours)
    if cached:
        return ticker, cached
    try:
        _get_bucket(finnhub_rate).consume()
        fh = _fh()
        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        articles = (fh.company_news(ticker, _from=start, to=end) or [])[:10]
    except Exception:
        articles = []
    context = _build_context_block(rank, total, row, sector_signal)
    result = _analyze_stock(ticker, articles, context)
    put_news_sentiment(db_path, ticker, result)
    return ticker, result


def attach_news_overlay(ranked_df: pd.DataFrame, cfg: dict, db_path: str) -> pd.DataFrame:
    """Stage 4.5: add entry_signal overlay and conviction adjustment. Fail-open."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("[news] ANTHROPIC_API_KEY not set — skipping news overlay")
        return ranked_df

    news_cfg = cfg.get("news", {})
    top_n = int(news_cfg.get("analyze_top_n", 10))
    ttl_hours = int(cfg.get("cache", {}).get("news_ttl_hours", 4))
    finnhub_rate = int(cfg.get("finnhub", {}).get("calls_per_minute", 60))

    df = ranked_df.copy()
    total = len(df)

    # Layer 1: market-level sector signals (run once)
    sectors_in_play = [s for s in df["sector"].dropna().unique() if s]
    market_cached = get_news_sentiment(db_path, "__MARKET__", ttl_hours)
    if market_cached:
        market_result = market_cached
        print("[news] market analysis: cache hit")
    else:
        try:
            market_result = _analyze_market(get_market_news(20), sectors_in_play)
            put_news_sentiment(db_path, "__MARKET__", market_result)
            print(f"[news] market: {str(market_result.get('regime_note', ''))[:80]}")
        except Exception as e:
            print(f"[news] market analysis failed: {e}")
            market_result = {"regime_note": "", "sector_signals": {}}

    sector_signals: dict = market_result.get("sector_signals") or {}

    # Layer 2: per-stock analysis (parallel, top_n only)
    worker_args = []
    for i, (_, row) in enumerate(df.head(top_n).iterrows(), 1):
        ticker = str(row.get("ticker", ""))
        sig = sector_signals.get(str(row.get("sector", "")))
        worker_args.append((ticker, row, i, total, sig, finnhub_rate, db_path, ttl_hours))

    results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(_process_one, args): args[0] for args in worker_args}
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                t, r = future.result()
                results[t] = r
            except Exception as e:
                print(f"[news] {ticker} failed: {e}")
                results[ticker] = dict(_ENTRY_FALLBACK)

    print(f"[news] overlay: {len(results)}/{top_n} stocks processed")

    # Initialise new columns with defaults
    for col, default in [
        ("entry_signal", "wait"), ("catalyst", "none"), ("priced_in", False),
        ("duration", "noise"), ("thesis_consistency", "neutral"),
        ("conviction_delta", 0), ("conviction_news", None), ("news_reasoning", ""),
    ]:
        if col not in df.columns:
            df[col] = default

    # Merge results + apply deterministic conviction adjustment
    for idx, row in df.iterrows():
        ticker = str(row.get("ticker", ""))
        if ticker not in results:
            continue
        r = results[ticker]
        for col in ("entry_signal", "catalyst", "priced_in", "duration", "thesis_consistency"):
            df.at[idx, col] = r.get(col)
        df.at[idx, "news_reasoning"] = r.get("reasoning", "")

        delta = max(-2, min(2, int(r.get("conviction_delta", 0))))
        sig = sector_signals.get(str(row.get("sector", "")), {})
        if sig.get("direction") == "headwind" and sig.get("strength") == "strong":
            delta = min(delta, 0)
        if r.get("thesis_consistency") == "contradicts":
            delta = min(delta, -1)

        current = int(row.get("conviction", 5) or 5)
        df.at[idx, "conviction_news"] = current
        df.at[idx, "conviction_delta"] = delta
        df.at[idx, "conviction"] = max(1, min(10, current + delta))

    df.attrs["sector_signals"] = sector_signals
    df.attrs["market_regime_note"] = market_result.get("regime_note", "")
    return df


# ── Legacy UI helpers (Market Regime tab + on-demand stock cards) ─────────────

def analyze_market_news(articles: list[dict]) -> dict:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return {**_CARD_FALLBACK, "summary": "Add ANTHROPIC_API_KEY to .env to enable AI news analysis."}
    if not articles:
        return {**_CARD_FALLBACK, "summary": "No news articles available."}
    headlines = "\n".join(f"- {a['headline']}" for a in articles[:15] if a.get("headline"))
    prompt = f"""Analyze these market news headlines for equity investors.

Headlines:
{headlines}

Return ONLY valid JSON:
{{"sentiment": "bullish|bearish|neutral|mixed", "summary": "2-3 sentences on key themes", "key_risks": ["risk 1", "risk 2"]}}"""
    try:
        text = _claude("claude-haiku-4-5", prompt, max_tokens=600)
        parsed = _parse_json(text)
        return parsed if parsed else {**_CARD_FALLBACK, "summary": text[:300]}
    except Exception as e:
        return {**_CARD_FALLBACK, "summary": f"Analysis error: {e}"}


def analyze_stock_news(ticker: str, articles: list[dict]) -> dict:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return {**_CARD_FALLBACK, "summary": "Add ANTHROPIC_API_KEY to .env to enable AI news analysis."}
    if not articles:
        return {**_CARD_FALLBACK, "summary": f"No recent news found for {ticker}."}
    headlines = "\n".join(f"- {a['headline']}" for a in articles[:8] if a.get("headline"))
    prompt = f"""Analyze recent news for {ticker} from a momentum trading perspective.

Headlines:
{headlines}

Return ONLY valid JSON:
{{"sentiment": "bullish|bearish|neutral|mixed", "summary": "1-2 sentences on momentum impact", "key_risks": ["risk 1"]}}"""
    try:
        text = _claude("claude-haiku-4-5", prompt, max_tokens=400)
        parsed = _parse_json(text)
        return parsed if parsed else {**_CARD_FALLBACK, "summary": text[:300]}
    except Exception as e:
        return {**_CARD_FALLBACK, "summary": f"Analysis error: {e}"}
