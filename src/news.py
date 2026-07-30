"""News analysis pipeline: entry signal overlay for momentum screener."""
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import finnhub
import pandas as pd

from src.cache import get_news_sentiment, put_news_sentiment
from src.llm import LLMError, available as llm_available, complete_json, object_schema

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

DEFAULT_MODEL = "gpt-5.4-mini"            # per-stock + market analysis
DEFAULT_PREFILTER_MODEL = "gpt-5.4-nano"  # material-vs-noise classification only

# entry_signal vocabulary. The first three are judgments the model actually
# made; the last two are the ABSENCE of a judgment and must never be rendered
# as one. Collapsing all five into "wait" is what made 13 of 20 rows on
# 2026-07-29 read as a deliberate "hold off" when 8 simply had no coverage
# and 5 were outright API failures.
SIGNAL_CONFIRM = "confirm_entry"
SIGNAL_WAIT = "wait"
SIGNAL_AVOID = "avoid"
SIGNAL_NO_NEWS = "no_news"          # nothing material published — not a veto
SIGNAL_UNAVAILABLE = "unavailable"  # the analysis failed — not a veto either

_NEUTRAL_OVERLAY: dict = {
    "catalyst": "none",
    "priced_in": False,
    "duration": "noise",
    "thesis_consistency": "neutral",
    "conviction_delta": 0,
}


def _no_news(reasoning: str) -> dict:
    return {**_NEUTRAL_OVERLAY, "entry_signal": SIGNAL_NO_NEWS, "reasoning": reasoning}


def _unavailable(reasoning: str) -> dict:
    return {**_NEUTRAL_OVERLAY, "entry_signal": SIGNAL_UNAVAILABLE, "reasoning": reasoning}


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


# ── Response schemas (OpenAI strict mode) ─────────────────────────────────────

_PREFILTER_SCHEMA = object_schema({
    "material": {"type": "array", "items": {"type": "integer"},
                 "description": "1-based indices of the material articles"},
})

_ENTRY_SCHEMA = object_schema({
    "entry_signal": {"type": "string", "enum": [SIGNAL_CONFIRM, SIGNAL_WAIT, SIGNAL_AVOID]},
    "catalyst": {"type": "string", "enum": ["estimate_up", "estimate_down", "none"]},
    "priced_in": {"type": "boolean"},
    "duration": {"type": "string", "enum": ["noise", "days", "weeks"]},
    "thesis_consistency": {"type": "string", "enum": ["confirms", "neutral", "contradicts"]},
    "conviction_delta": {"type": "integer", "description": "-1, 0 or +1"},
    "reasoning": {"type": "string"},
})

# sector_signals is a LIST, not a keyed object: strict mode requires
# additionalProperties:false, which cannot express "arbitrary sector names as
# keys". Converted back to a dict by _analyze_market.
_MARKET_SCHEMA = object_schema({
    "regime_note": {"type": "string"},
    "sector_signals": {
        "type": "array",
        "items": object_schema({
            "sector": {"type": "string"},
            "direction": {"type": "string", "enum": ["headwind", "tailwind"]},
            "strength": {"type": "string", "enum": ["strong", "mild"]},
            "reason": {"type": "string"},
        }),
    },
})

_CARD_SCHEMA = object_schema({
    "sentiment": {"type": "string", "enum": ["bullish", "bearish", "neutral", "mixed"]},
    "summary": {"type": "string"},
    "key_risks": {"type": "array", "items": {"type": "string"}},
})


# ── Pipeline: pre-filter → context → analyze ─────────────────────────────────

def _prefilter_articles(ticker: str, articles: list[dict], model: str) -> list[dict]:
    """Classify each article as material or noise. Returns the material subset.

    Fails OPEN (returns everything) — a broken filter must not manufacture a
    "no material news" result, because that reads as an editorial judgment
    about the stock rather than as the tooling failure it is.
    """
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
        "Return the 1-based indices of the material articles."
    )
    try:
        parsed = complete_json(prompt, _PREFILTER_SCHEMA, model,
                              max_tokens=500, name="prefilter")
    except LLMError as e:
        print(f"[news] {ticker} prefilter failed ({e}) — keeping all {len(articles)} articles")
        return articles
    indices = {int(i) - 1 for i in parsed.get("material", [])}
    result = [a for i, a in enumerate(articles) if i in indices]
    return result


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


def _analyze_stock(ticker: str, articles: list[dict], context_block: str,
                   model: str, prefilter_model: str) -> dict:
    """Cross-reference news against screener context. Returns entry overlay."""
    if not articles:
        return _no_news(f"No news published for {ticker} in the last 7 days.")
    material = _prefilter_articles(ticker, articles, prefilter_model)
    if not material:
        return _no_news(
            f"{len(articles)} article{'s' if len(articles) != 1 else ''} found for {ticker}, "
            f"none material (price recaps / boilerplate)."
        )

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

For "reasoning": 2-3 plain-English sentences covering what the news actually
says, why it matters for this stock right now, and what specific risk or
opportunity it creates. Be specific and nuanced but avoid finance jargon —
write for a smart investor who does not speak Wall Street.

"conviction_delta" must be -1, 0 or +1."""

    try:
        parsed = complete_json(prompt, _ENTRY_SCHEMA, model, max_tokens=3000, name="entry_overlay")
    except LLMError as e:
        print(f"[news] {ticker} analysis failed: {e}")
        return _unavailable(f"Analysis failed for {ticker}: {e}")
    parsed["conviction_delta"] = max(-1, min(1, int(parsed.get("conviction_delta", 0))))
    return parsed


def _analyze_market(articles: list[dict], sectors_in_play: list[str], model: str) -> dict:
    """Produce a sector_signals map from market headlines."""
    if not articles:
        return {"regime_note": "", "sector_signals": {}}
    headlines = "\n".join(f"- {a['headline']}" for a in articles[:15] if a.get("headline"))
    sectors_str = ", ".join(sectors_in_play) if sectors_in_play else "various"
    prompt = f"""You are a macro strategist for equity momentum funds.

Headlines:
{headlines}

Sectors in today's screener top picks: {sectors_str}

Which of these sectors face near-term headwinds or tailwinds based on today's news?

"regime_note" is 1 sentence on the broad tape. Emit a sector_signals entry
ONLY when the news genuinely shifts that sector's outlook — an empty list is
the correct answer on a quiet tape. Restrict sectors to: {sectors_str}"""
    try:
        parsed = complete_json(prompt, _MARKET_SCHEMA, model, max_tokens=3000, name="market")
    except LLMError as e:
        print(f"[news] market analysis failed: {e}")
        return {"regime_note": "", "sector_signals": {}}
    signals = {
        str(s["sector"]): {"direction": s["direction"], "strength": s["strength"],
                           "reason": s["reason"]}
        for s in parsed.get("sector_signals", []) if s.get("sector")
    }
    return {"regime_note": parsed.get("regime_note", ""), "sector_signals": signals}


def _process_one(args: tuple) -> tuple[str, dict]:
    """Thread worker: fetch + analyze one stock."""
    (ticker, row, rank, total, sector_signal, finnhub_rate, db_path, ttl_hours,
     model, prefilter_model) = args
    cached = get_news_sentiment(db_path, ticker, ttl_hours)
    if cached:
        return ticker, cached
    fetch_failed = False
    try:
        _get_bucket(finnhub_rate).consume()
        fh = _fh()
        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        articles = (fh.company_news(ticker, _from=start, to=end) or [])[:10]
    except Exception as e:
        print(f"[news] {ticker} article fetch failed: {e!r}")
        articles, fetch_failed = [], True

    # A failed Finnhub fetch is not "no news" — the distinction matters
    # because "no news" is a fact about the stock and this is a fact about
    # our plumbing. Don't cache it either; retry on the next run.
    if fetch_failed:
        return ticker, _unavailable(f"Could not fetch news for {ticker} (Finnhub request failed).")

    context = _build_context_block(rank, total, row, sector_signal)
    result = _analyze_stock(ticker, articles, context, model, prefilter_model)
    # Only cache real outcomes. Caching a transient LLM failure for the TTL
    # would pin a whole row to "unavailable" for hours.
    if result.get("entry_signal") != SIGNAL_UNAVAILABLE:
        put_news_sentiment(db_path, ticker, result)
    return ticker, result


def attach_news_overlay(ranked_df: pd.DataFrame, cfg: dict, db_path: str) -> pd.DataFrame:
    """Stage 4.5: add entry_signal overlay and conviction adjustment. Fail-open."""
    if not llm_available():
        print("[news] OPENAI_API_KEY not set — skipping news overlay")
        return ranked_df

    news_cfg = cfg.get("news", {})
    top_n = int(news_cfg.get("analyze_top_n", 10))
    model = str(news_cfg.get("model", DEFAULT_MODEL))
    prefilter_model = str(news_cfg.get("prefilter_model", DEFAULT_PREFILTER_MODEL))
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
            market_result = _analyze_market(get_market_news(20), sectors_in_play, model)
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
        worker_args.append((ticker, row, i, total, sig, finnhub_rate, db_path, ttl_hours,
                           model, prefilter_model))

    results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(_process_one, args): args[0] for args in worker_args}
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                t, r = future.result()
                results[t] = r
            except Exception as e:
                print(f"[news] {ticker} failed: {e!r}")
                results[ticker] = _unavailable(f"Analysis failed for {ticker}: {e!r}")

    tally: dict[str, int] = {}
    for r in results.values():
        sig_name = str(r.get("entry_signal", "?"))
        tally[sig_name] = tally.get(sig_name, 0) + 1
    breakdown = ", ".join(f"{k}={v}" for k, v in sorted(tally.items()))
    print(f"[news] overlay: {len(results)}/{top_n} stocks processed ({breakdown})")

    # Initialise new columns with defaults. entry_signal defaults to EMPTY,
    # not "wait": a row past analyze_top_n was never looked at, and the UI
    # already treats an empty signal as "nothing to show". Defaulting it to a
    # real verdict invents an opinion nobody formed.
    for col, default in [
        ("entry_signal", ""), ("catalyst", "none"), ("priced_in", False),
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

        delta = max(-1, min(1, int(r.get("conviction_delta", 0))))
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

def analyze_market_news(articles: list[dict], model: str = DEFAULT_MODEL) -> dict:
    if not llm_available():
        return {**_CARD_FALLBACK, "summary": "Add OPENAI_API_KEY to .env to enable AI news analysis."}
    if not articles:
        return {**_CARD_FALLBACK, "summary": "No news articles available."}
    headlines = "\n".join(f"- {a['headline']}" for a in articles[:15] if a.get("headline"))
    prompt = f"""Analyze these market news headlines for equity investors.

Headlines:
{headlines}

Give a 2-3 sentence summary of the key themes, plus the main risks."""
    try:
        return complete_json(prompt, _CARD_SCHEMA, model, max_tokens=3000, name="market_card")
    except LLMError as e:
        return {**_CARD_FALLBACK, "summary": f"Analysis error: {e}"}


def analyze_stock_news(ticker: str, articles: list[dict], model: str = DEFAULT_MODEL) -> dict:
    if not llm_available():
        return {**_CARD_FALLBACK, "summary": "Add OPENAI_API_KEY to .env to enable AI news analysis."}
    if not articles:
        return {**_CARD_FALLBACK, "summary": f"No recent news found for {ticker}."}
    headlines = "\n".join(f"- {a['headline']}" for a in articles[:8] if a.get("headline"))
    prompt = f"""Analyze recent news for {ticker} from a momentum trading perspective.

Headlines:
{headlines}

Give a 1-2 sentence summary of the momentum impact, plus the main risks."""
    try:
        return complete_json(prompt, _CARD_SCHEMA, model, max_tokens=3000, name="stock_card")
    except LLMError as e:
        return {**_CARD_FALLBACK, "summary": f"Analysis error: {e}"}
