"""Claude precision layer for the filing-edge screen.

Two classifiers — both run ONLY on the small subset where the deterministic
text-stability score is below the threshold (actual language changers):

1. change_classifier  — Sonnet reads the diff of Risk Factors + MD&A sections
   and classifies whether the change is +1 (positive/benign), 0 (neutral), or
   -1 (negative: new risks, hedging, going-concern, litigation).
   The Lazy Prices paper shows changes are *on average* bad; Claude separates
   bad from neutral/good to lift precision over the raw similarity signal.

2. eight_k_classifier — Haiku classifies recent 8-K item types and flags
   underreacted-to material events: 4.01 auditor change, 5.02 exec departure,
   2.04 covenant / acceleration, 2.06 impairment. Acts as a veto / penalty.

Cost control: both classifiers are gated by similarity_threshold so Claude
spend scales with the (small) fraction of real changers, not with universe size.
"""
import json
import re
import threading
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from src.cache import get_filing_analysis, put_filing_analysis
from src.news import _ThreadSafeTokenBucket, _parse_json, _claude

logger = logging.getLogger(__name__)

_ANALYSIS_FALLBACK = {
    "change_direction": 0,
    "change_type": "unknown",
    "change_reason": "Analysis unavailable.",
}

_8K_RED_FLAGS = {
    "4.01": "auditor_change",
    "5.02": "executive_departure",
    "2.04": "covenant_breach",
    "2.06": "impairment",
}

# Module-level token bucket (shared with news.py if imported together)
_bucket: _ThreadSafeTokenBucket | None = None
_bucket_lock = threading.Lock()


def _get_bucket(rate: int = 30) -> _ThreadSafeTokenBucket:
    global _bucket
    with _bucket_lock:
        if _bucket is None:
            _bucket = _ThreadSafeTokenBucket(rate)
        return _bucket


# ── Diff helper ───────────────────────────────────────────────────────────────

def _rough_diff(text_cur: str, text_prior: str, max_chars: int = 3000) -> str:
    """Produce a human-readable diff summary (added/removed paragraph-level ideas).

    Full difflib is noisy for 50K-char filings. Instead: extract unique sentences
    from each version, find those present in only one side, and return a compact
    summary. Claude reads this, not the raw 50K text.
    """
    def sentences(t: str) -> list[str]:
        t = re.sub(r"\s+", " ", t)
        parts = re.split(r"(?<=[.!?])\s+", t)
        return [s.strip() for s in parts if len(s.strip()) > 40]

    cur_sents = set(sentences(text_cur))
    prior_sents = set(sentences(text_prior))
    added   = list(cur_sents - prior_sents)[:8]
    removed = list(prior_sents - cur_sents)[:8]

    lines = []
    if removed:
        lines.append("REMOVED (in prior, not in current):")
        lines.extend(f"  - {s[:300]}" for s in removed[:5])
    if added:
        lines.append("ADDED (in current, not in prior):")
        lines.extend(f"  + {s[:300]}" for s in added[:5])
    result = "\n".join(lines)
    return result[:max_chars] if result else "(no sentence-level diff detected)"


# ── Change classifier ─────────────────────────────────────────────────────────

def classify_change(
    ticker: str,
    risk_text: str,
    prior_risk_text: str,
    mda_text: str,
    prior_mda_text: str,
    db_path: str,
    accession: str,
) -> dict:
    """Classify whether the filing language change is positive, neutral, or negative.

    Returns dict with change_direction (+1/0/-1), change_type, change_reason.
    Cached permanently by accession (the filing pair is immutable).
    """
    cached = get_filing_analysis(db_path, accession)
    if cached is not None and "change_direction" in cached:
        return cached

    risk_diff = _rough_diff(risk_text, prior_risk_text)
    mda_diff  = _rough_diff(mda_text, prior_mda_text)

    prompt = f"""You are a senior equity analyst reviewing annual report language changes for {ticker}.

The company changed significant language in its annual filing versus the prior year.

RISK FACTORS DIFF:
{risk_diff}

MD&A DIFF:
{mda_diff}

Classify the OVERALL direction of change:
- +1 = positive/benign (raised guidance, new growth initiatives, expanded market, resolved prior risk)
- 0  = neutral (boilerplate reorganization, new legal boilerplate without substance, name/structure changes)
- -1 = negative (new risk factors, increased uncertainty/hedging language, going-concern language, litigation, guidance cut, impairment)

Return ONLY JSON:
{{
  "change_direction": -1,
  "change_type": "new_risk_factors|guidance_cut|going_concern|litigation|growth_initiative|restructuring|boilerplate|other",
  "change_reason": "1 sentence naming the specific change and why you classified it this way"
}}"""

    try:
        _get_bucket().consume()
        text = _claude("claude-sonnet-4-6", prompt, max_tokens=400)
        parsed = _parse_json(text)
        if parsed and "change_direction" in parsed:
            parsed["change_direction"] = max(-1, min(1, int(parsed["change_direction"])))
            put_filing_analysis(db_path, accession, parsed)
            return parsed
    except Exception as e:
        logger.warning(f"[filing_analysis] change classifier failed for {ticker}: {e}")

    result = dict(_ANALYSIS_FALLBACK)
    put_filing_analysis(db_path, accession, result)
    return result


# ── 8-K classifier ────────────────────────────────────────────────────────────

def classify_8k(ticker: str, eight_k_items: list[dict]) -> dict:
    """Haiku: classify 8-K item types; flag underreacted-to material events.

    eight_k_items: list of {"item": "4.01", "headline": "...", "date": "..."}
    Returns {"red_flags": [...], "eight_k_penalty": -1|0}
    """
    if not eight_k_items:
        return {"red_flags": [], "eight_k_penalty": 0}

    lines = [f"{i+1}. Item {r['item']}: {r.get('headline','')}" for i, r in enumerate(eight_k_items)]
    prompt = (
        f"8-K filings for {ticker} in the past 6 months:\n"
        + "\n".join(lines)
        + "\n\nWhich items represent MATERIAL NEGATIVE events that the market likely underreacted to?\n"
        "Focus on: 4.01 (auditor change), 5.02 (exec departure), 2.04 (covenant/acceleration), "
        "2.06 (impairment). Ignore routine press releases.\n"
        'Return ONLY JSON: {"red_flags": ["4.01 auditor change - BDO replaced by KPMG"], "eight_k_penalty": -1}'
        "\n eight_k_penalty: -1 if red flags found, 0 otherwise."
    )
    try:
        text = _claude("claude-haiku-4-5", prompt, max_tokens=300)
        parsed = _parse_json(text)
        if parsed and "eight_k_penalty" in parsed:
            return parsed
    except Exception as e:
        logger.warning(f"[filing_analysis] 8-K classifier failed for {ticker}: {e}")
    return {"red_flags": [], "eight_k_penalty": 0}


# ── Per-row worker ────────────────────────────────────────────────────────────

def _process_one(args: tuple) -> tuple[str, dict]:
    ticker, row, db_path, threshold = args
    stab = float(row.get("text_stability", 1.0) or 1.0)

    if stab >= threshold:
        return ticker, {"change_direction": 0, "change_type": "stable",
                        "change_reason": f"text_stability={stab:.4f} above threshold — no change characterization needed."}

    accession = str(row.get("accession", ""))
    risk_text  = str(row.get("risk_text", ""))
    prior_risk = str(row.get("prior_risk_text", ""))
    mda_text   = str(row.get("mda_text", ""))
    prior_mda  = str(row.get("prior_mda_text", ""))

    if not accession:
        return ticker, dict(_ANALYSIS_FALLBACK)

    result = classify_change(ticker, risk_text, prior_risk, mda_text, prior_mda,
                             db_path, accession)
    return ticker, result


# ── Public entry point ────────────────────────────────────────────────────────

def attach_filing_analysis(
    df: pd.DataFrame,
    cfg: dict,
    db_path: str,
) -> pd.DataFrame:
    """Attach change_direction to rows in df where text_stability < threshold.

    df must have columns: ticker, text_stability, accession,
    risk_text, prior_risk_text, mda_text, prior_mda_text.
    Columns are added/updated in place (copy returned).
    """
    if not cfg.get("lazy_prices", {}).get("enabled_claude_layer", True):
        df["change_direction"] = 0
        return df

    threshold = cfg["lazy_prices"].get("similarity_threshold", 0.85)
    changers = df[df["text_stability"].fillna(1.0) < threshold]
    print(f"[filing_analysis] {len(changers)}/{len(df)} below threshold {threshold} → Claude analysis")

    if len(changers) == 0:
        df["change_direction"] = df.get("change_direction", pd.Series(0, index=df.index))
        return df

    results: dict[str, dict] = {}
    args_list = [
        (str(row["ticker"]), row, db_path, threshold)
        for _, row in changers.iterrows()
    ]

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(_process_one, a): a[0] for a in args_list}
        for fut in as_completed(futures):
            try:
                ticker, res = fut.result()
                results[ticker] = res
            except Exception as e:
                logger.warning(f"[filing_analysis] worker failed: {e}")

    df = df.copy()
    df["change_direction"] = df.apply(
        lambda row: results.get(str(row["ticker"]), {}).get("change_direction", 0), axis=1
    )
    df["change_type"] = df.apply(
        lambda row: results.get(str(row["ticker"]), {}).get("change_type", ""), axis=1
    )
    df["change_reason"] = df.apply(
        lambda row: results.get(str(row["ticker"]), {}).get("change_reason", ""), axis=1
    )
    return df
