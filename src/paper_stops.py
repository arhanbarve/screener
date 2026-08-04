"""Exit plans and resting protective stops for the Alpaca paper book.

Why this exists separately from src/exit_plan.py's daily job: that job reads and
writes positions.json, which mirrors the *Fidelity* account (CRDO, RSI, ENVA,
SPY). The paper book holds EVC, FIX, GTX, HUT, MYE, SPY, STX, VLO, VSXY — only
SPY overlaps. So paper plans cannot be read from positions.json, and writing them
there is forbidden outright (PROMPT.md: never touch positions.json).

Plans are therefore computed here from Alpaca's own avg_entry_price plus price
history, using exit_plan.bootstrap_position — the same logic, same constants,
different book — and persisted to data/alpaca/plans.json.

Only the max-loss floor is ever rested at the broker. The trailing stop and the
50-day trend break remain close-based in exit_plan: a resting intraday stop fires
on a wick, which is exactly the flip-flopping the standing-verdict rewrite
removed. The floor's job is to survive a session that never runs, which — with
four of nine sessions having crashed — is a job worth having.
"""
import json
import os
from datetime import date, datetime, timedelta
from pathlib import Path

SCREENER_DIR = Path(__file__).parent.parent
PLANS_FILE = SCREENER_DIR / "data" / "alpaca" / "plans.json"

# Enough history for ATR14 plus the 50-day context exit_plan replays over.
HISTORY_LOOKBACK_DAYS = 400


def load_plans(path: Path | None = None) -> dict:
    p = path or PLANS_FILE
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text()).get("plans", {})
    except (OSError, json.JSONDecodeError):
        return {}


def save_plans(plans: dict, path: Path | None = None) -> Path:
    p = path or PLANS_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {"updated_at": datetime.now().isoformat(timespec="minutes"), "plans": plans}
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, p)
    return p


def floors_from_plans(plans: dict) -> dict[str, float]:
    """Extract {ticker: max-loss floor}. Skips anything without a usable floor."""
    out = {}
    for ticker, plan in (plans or {}).items():
        floor = (plan or {}).get("stop_floor")
        try:
            floor = float(floor)
        except (TypeError, ValueError):
            continue
        if floor > 0:
            out[ticker.upper()] = floor
    return out


def _held_since(position: dict, fallback: str) -> str:
    """Entry date for the plan. Prefers the FIFO-derived held_since from the
    paper snapshot, since after a partial sale the shares still held began later
    than the first fill."""
    return position.get("held_since") or fallback


def build_plans(positions: list[dict], history: dict, today: str | None = None) -> dict:
    """Compute an exit plan per position by replaying its history since entry.

    positions: dicts with ticker, avg_cost, held_since (paper snapshot shape).
    history:   {ticker: DataFrame with open/high/low/close indexed by date}.
    """
    from src import exit_plan

    today = today or date.today().isoformat()
    plans: dict = {}
    for p in positions:
        ticker = (p.get("ticker") or p.get("symbol") or "").upper()
        try:
            entry_price = float(p.get("avg_cost") or p.get("avg_entry_price") or 0)
        except (TypeError, ValueError):
            entry_price = 0.0
        df = history.get(ticker)
        if not ticker or entry_price <= 0 or df is None or df.empty:
            continue
        pos = {
            "ticker": ticker,
            "entry_date": _held_since(p, today),
            "entry_price": entry_price,
        }
        try:
            pos = exit_plan.bootstrap_position(pos, df)
        except Exception as e:  # noqa: BLE001 — one bad symbol must not sink the run
            plans[ticker] = {"error": f"{type(e).__name__}: {e}"}
            continue
        plan = pos.get("plan") or {}
        plan.pop("pending_notify", None)
        plans[ticker] = {
            "entry_date": pos["entry_date"],
            "entry_price": entry_price,
            "stop_floor": plan.get("stop_floor"),
            "stop_level": plan.get("stop_level"),
            "initial_stop": plan.get("initial_stop"),
            "risk_R": plan.get("risk_R"),
            "peak_close": plan.get("peak_close"),
            "verdict": plan.get("verdict"),
            "last_close": plan.get("last_close"),
        }
    return plans


def fetch_history(tickers: list[str], start: str) -> dict:
    """Daily OHLC per ticker, shaped for exit_plan (lowercase columns)."""
    import yfinance as yf

    if not tickers:
        return {}
    begin = (date.fromisoformat(start) - timedelta(days=HISTORY_LOOKBACK_DAYS)).isoformat()
    end = (date.today() + timedelta(days=1)).isoformat()
    raw = yf.download(sorted(set(tickers)), start=begin, end=end,
                      progress=False, auto_adjust=True, group_by="ticker")
    if raw is None or raw.empty:
        return {}

    out = {}
    for t in tickers:
        try:
            df = raw[t] if len(set(tickers)) > 1 else raw
        except KeyError:
            continue
        df = df.rename(columns=str.lower)[["open", "high", "low", "close"]].dropna()
        if not df.empty:
            out[t] = df
    return out


def sanity_check_floor(floor: float, last_price: float) -> str | None:
    """Reject a floor that would fire the moment it is placed.

    A stop at or above the market is not protection, it is an instant market
    sell. That means the plan is wrong, and dumping the position on a bad plan is
    strictly worse than resting no stop at all.
    """
    if floor <= 0:
        return "floor is not positive"
    if last_price <= 0:
        return "no usable last price"
    if floor >= last_price:
        return (f"floor {floor:.2f} is at or above last {last_price:.2f} — would "
                f"trigger immediately; position is already below its max-loss level "
                f"and belongs to the close-based sell decision, not a resting stop")
    return None
