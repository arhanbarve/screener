"""Alpaca paper-portfolio snapshot.

The Paper page renders from data/alpaca/portfolio.json, never from a live broker
call, for the same reason the Positions page renders from the Fidelity snapshot:
Streamlit Cloud has no ALPACA_API_KEY, and a page that needs credentials to draw
anything is a page that is blank exactly when you want to look at it.

Writes are all-or-nothing. Every required fetch must succeed or the previous
snapshot stays untouched — a partial snapshot is indistinguishable from a real
one once it is on disk, which is the failure mode that cost five days of wrong
P&L on the Fidelity side.

Realized P&L uses FIFO lot matching, because that is what Alpaca itself does.
Average-cost accounting disagrees by $241 on the current book and, more
usefully, FIFO satisfies a hard invariant worth asserting:

    realized + unrealized == equity - deposits

SECURITY: no account identifiers are captured. This file is committed so the
cloud dashboard can read it, and scripts/secret_scan.sh blocks account-shaped
strings from being committed at all.
"""
import json
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

# Alpaca stamps each portfolio-history point at UTC midnight of the day *after*
# the session it describes, so reading the UTC date shifts every point one day
# forward — landing Jul 27 on Jul 25 and Aug 3 on Aug 1, both weekends, which
# silently dropped them from the market-day intersection. Converting to exchange
# time recovers the real trading date. A fixed -4 offset would break across DST.
EXCHANGE_TZ = ZoneInfo("America/New_York")

SCREENER_DIR  = Path(__file__).parent.parent
SNAPSHOT_FILE = SCREENER_DIR / "data" / "alpaca" / "portfolio.json"

# Sparkline history: enough to show the shape of a holding without bloating the
# tracked file. 9 positions x ~40 closes is a few KB.
HISTORY_DAYS = 40


# ── FIFO lot engine ───────────────────────────────────────────────────────────

def _fills(orders: list[dict]) -> list[dict]:
    """Filled orders only, oldest first, normalised to the fields we use."""
    out = []
    for o in orders:
        if not o.get("filled_at"):
            continue
        qty = float(o.get("filled_qty") or 0)
        price = float(o.get("filled_avg_price") or 0)
        if qty <= 0 or price <= 0:
            continue
        out.append({
            "symbol": (o.get("symbol") or "").upper(),
            "side": o.get("side"),
            "qty": qty,
            "price": price,
            "date": o["filled_at"][:10],
        })
    out.sort(key=lambda f: (f["date"], f["symbol"]))
    return out


def match_fifo(fills: list[dict]) -> dict[str, dict]:
    """Run FIFO lot matching per symbol.

    Returns {symbol: {realized, open_lots, buy_qty, sell_qty, closed}} where
    open_lots is a list of {qty, price, date} still held. Matching FIFO (rather
    than average cost) is what makes the remaining lots' cost basis agree with
    Alpaca's own avg_entry_price.
    """
    books: dict[str, dict] = {}
    for f in fills:
        b = books.setdefault(f["symbol"], {
            "realized": 0.0, "open_lots": [], "buy_qty": 0.0, "sell_qty": 0.0,
        })
        if f["side"] == "buy":
            b["open_lots"].append({"qty": f["qty"], "price": f["price"], "date": f["date"]})
            b["buy_qty"] += f["qty"]
            continue

        # Sell: consume oldest lots first.
        b["sell_qty"] += f["qty"]
        remaining = f["qty"]
        while remaining > 1e-9 and b["open_lots"]:
            lot = b["open_lots"][0]
            take = min(remaining, lot["qty"])
            b["realized"] += take * (f["price"] - lot["price"])
            lot["qty"] -= take
            remaining -= take
            if lot["qty"] <= 1e-9:
                b["open_lots"].pop(0)

    for sym, b in books.items():
        b["open_lots"] = [l for l in b["open_lots"] if l["qty"] > 1e-9]
        b["realized"] = round(b["realized"], 4)
        b["closed"] = not b["open_lots"]
    return books


def open_lot_summary(book: dict) -> dict:
    """Cost basis and held-since date for the lots still open.

    held_since is the oldest *remaining* lot, not the first-ever fill. After a
    partial sale FIFO has retired the early lots, so the position you hold today
    genuinely started later — STX first filled Jul 24 but the shares still held
    were bought Jul 29, which is why Alpaca reports 752.67 not a blended 800.17.
    """
    lots = book["open_lots"]
    if not lots:
        return {"qty": 0.0, "avg_cost": None, "held_since": None}
    qty = sum(l["qty"] for l in lots)
    cost = sum(l["qty"] * l["price"] for l in lots)
    return {
        "qty": qty,
        "avg_cost": (cost / qty) if qty else None,
        "held_since": min(l["date"] for l in lots),
    }


def realized_total(books: dict[str, dict]) -> float:
    return round(sum(b["realized"] for b in books.values()), 2)


def closed_trades(books: dict[str, dict], fills: list[dict]) -> list[dict]:
    """Fully-closed symbols, most recently closed first.

    Partially-trimmed symbols are excluded — they still have open lots, so their
    realized portion belongs in the headline figure, not in a table of finished
    trades. Their realized P&L is still counted by realized_total().
    """
    first_buy: dict[str, str] = {}
    last_sell: dict[str, str] = {}
    for f in fills:
        if f["side"] == "buy":
            first_buy.setdefault(f["symbol"], f["date"])
        else:
            last_sell[f["symbol"]] = f["date"]

    rows = []
    for sym, b in books.items():
        if not b["closed"]:
            continue
        entry, exit_ = first_buy.get(sym), last_sell.get(sym)
        cost = 0.0
        for f in fills:
            if f["symbol"] == sym and f["side"] == "buy":
                cost += f["qty"] * f["price"]
        rows.append({
            "ticker": sym,
            "entry_date": entry,
            "exit_date": exit_,
            "held_days": _days_between(entry, exit_),
            "realized": round(b["realized"], 2),
            "realized_pct": round(b["realized"] / cost, 6) if cost else None,
            "cost_basis": round(cost, 2),
        })
    rows.sort(key=lambda r: (r["exit_date"] or ""), reverse=True)
    return rows


def _days_between(a: str | None, b: str | None) -> int | None:
    if not a or not b:
        return None
    try:
        return (date.fromisoformat(b) - date.fromisoformat(a)).days
    except ValueError:
        return None


# ── series normalisation ──────────────────────────────────────────────────────

def pct_series(values: list[float], base: float | None = None) -> list[float | None]:
    """Convert a level series to percent change from base (default: first value).

    None for any point that cannot be expressed against the base, so the chart
    renders a gap rather than a fabricated 0.
    """
    if base is None:
        base = next((v for v in values if v), 0.0) or 0.0
    if not base:
        return [None] * len(values)
    return [None if not v else round((v - base) / base * 100, 4) for v in values]


def equity_by_date(history: dict) -> dict[str, float]:
    """Map Alpaca portfolio history onto real trading dates.

    Each point is stamped at UTC midnight *after* its session, so the UTC date is
    one day late. Reading it naively pushed Jul 27 onto Jul 25 and Aug 3 onto
    Aug 1 — both weekends, so both were dropped when intersected against the
    exchange calendar, losing 2 of 9 points from the curve. Converting to
    exchange time gives the session's own date.

    Zero-equity points (before the account was funded) are skipped.
    """
    out: dict[str, float] = {}
    for ts, eq in zip(history.get("timestamp") or [], history.get("equity") or []):
        if not eq:
            continue
        session = (
            datetime.fromtimestamp(ts, tz=timezone.utc)
            .astimezone(EXCHANGE_TZ)
            .date()
            .isoformat()
        )
        out[session] = float(eq)
    return out


def inception_baseline(first_fill: str | None, market_days: list[str]) -> str | None:
    """The date both series should be normalised to.

    Alpaca backfills equity for days before the account traded, so the earliest
    non-zero equity point is not the start of the strategy — this account shows
    $100,000 flat from Jul 22 but did not deploy a cent until Jul 24. Measuring
    SPY from Jul 22 hands it two days of market move the account was in cash for
    by construction, which understated the gap by 1.3 points.

    The fair baseline is the last market day the account was entirely in cash:
    the market day immediately before the first fill. Falls back to the first
    fill date when no earlier market day is available.
    """
    if not first_fill:
        return market_days[0] if market_days else None
    earlier = [d for d in sorted(market_days) if d < first_fill]
    return earlier[-1] if earlier else first_fill


def first_fill_date(fills: list[dict]) -> str | None:
    return min((f["date"] for f in fills), default=None)


def align_on_dates(
    account: dict[str, float],
    spy: dict[str, float],
    market_days: list[str],
    baseline: str | None = None,
) -> dict:
    """Align account equity and SPY closes onto shared market days.

    Alpaca labels portfolio history in UTC, which shifts a day at the boundary,
    so both series are keyed by date and intersected against the exchange
    calendar rather than zipped positionally.

    baseline trims the window to start there, so both series are normalised to
    the day the strategy actually began rather than to Alpaca's backfill padding.
    """
    days = [d for d in sorted(market_days) if d in account and d in spy]
    if baseline:
        days = [d for d in days if d >= baseline]
    acct_vals = [account[d] for d in days]
    spy_vals  = [spy[d] for d in days]
    return {
        "dates": days,
        "account_pct": pct_series(acct_vals),
        "spy_pct": pct_series(spy_vals),
        "account_level": acct_vals,
        "spy_level": spy_vals,
    }


def vs_spy(curve: dict) -> float | None:
    """Account return minus SPY return over the aligned window, in points."""
    a, s = curve.get("account_pct") or [], curve.get("spy_pct") or []
    if not a or not s or a[-1] is None or s[-1] is None:
        return None
    return round(a[-1] - s[-1], 4)


# ── snapshot assembly ─────────────────────────────────────────────────────────

class SnapshotError(RuntimeError):
    """A required fetch failed. Nothing is written; last good snapshot survives."""


def build_positions(
    raw_positions: list[dict],
    books: dict[str, dict],
    history: dict[str, list[dict]],
) -> list[dict]:
    """Merge broker positions with FIFO lot info and price history."""
    out = []
    for p in raw_positions:
        sym = (p.get("symbol") or "").upper()
        lots = open_lot_summary(books.get(sym, {"open_lots": []}))
        qty = float(p.get("qty") or 0)
        out.append({
            "ticker": sym,
            "quantity": qty,
            "avg_cost": _f(p.get("avg_entry_price")),
            "last_price": _f(p.get("current_price")),
            "market_value": _f(p.get("market_value")),
            "cost_basis": _f(p.get("cost_basis")),
            "total_gl_dollar": _f(p.get("unrealized_pl")),
            "total_gl_pct": _f(p.get("unrealized_plpc")),
            "today_gl_dollar": _f(p.get("unrealized_intraday_pl")),
            "today_gl_pct": _f(p.get("unrealized_intraday_plpc")),
            "lastday_price": _f(p.get("lastday_price")),
            # FIFO-derived: the lots actually still held.
            "held_since": lots["held_since"],
            "realized_to_date": books.get(sym, {}).get("realized", 0.0),
            "history": history.get(sym, []),
        })
    return out


def _f(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def reconcile(realized: float, unrealized: float, equity: float, deposits: float) -> dict:
    """Check realized + unrealized == equity - deposits.

    Not decoration: this is what proved FIFO was the right lot-matching method
    and would catch a future accounting drift. Reported in the snapshot so the
    page can say so rather than silently showing numbers that don't add up.
    """
    expected = round(equity - deposits, 2)
    actual = round(realized + unrealized, 2)
    drift = round(actual - expected, 2)
    return {
        "expected": expected,
        "actual": actual,
        "drift": drift,
        "ok": abs(drift) < 1.0,
    }


def assemble(
    account: dict,
    raw_positions: list[dict],
    orders: list[dict],
    history: dict[str, list[dict]],
    curve: dict,
    cadence: dict,
    deposits: float,
    synced_at: str,
) -> dict:
    """Build the snapshot dict. Pure — all IO happens in the caller."""
    fills = _fills(orders)
    books = match_fifo(fills)

    positions = build_positions(raw_positions, books, history)
    unrealized = round(sum(p["total_gl_dollar"] for p in positions), 2)
    today = round(sum(p["today_gl_dollar"] for p in positions), 2)
    realized = realized_total(books)

    equity = _f(account.get("equity"))
    cash = _f(account.get("cash"))
    long_mv = _f(account.get("long_market_value"))
    last_eq = _f(account.get("last_equity"))

    # Worst total P&L first — the page opens on what needs a decision.
    positions.sort(key=lambda p: p["total_gl_pct"])

    return {
        "synced_at": synced_at,
        "account": {
            "equity": equity,
            "last_equity": last_eq,
            "cash": cash,
            "long_market_value": long_mv,
            "deposits": deposits,
            "deployed_pct": round(long_mv / equity, 6) if equity else 0.0,
            "today_dollar": round(equity - last_eq, 2) if last_eq else today,
            "today_pct": round((equity - last_eq) / last_eq, 6) if last_eq else 0.0,
            "realized": realized,
            "unrealized": unrealized,
            "total_pl": round(realized + unrealized, 2),
        },
        "reconciliation": reconcile(realized, unrealized, equity, deposits),
        "positions": positions,
        "closed_trades": closed_trades(books, fills),
        "open_orders": [
            {"ticker": (o.get("symbol") or "").upper(), "side": o.get("side"),
             "qty": o.get("qty"), "notional": o.get("notional"),
             "submitted_at": (o.get("submitted_at") or "")[:19]}
            for o in orders if o.get("status") in ("new", "accepted", "partially_filled")
        ],
        "curve": curve,
        "vs_spy": vs_spy(curve),
        "cadence": cadence,
    }


def merge_live(snapshot: dict, account: dict, raw_positions: list[dict],
               open_orders: list[dict], now: str) -> dict:
    """Overlay live broker state onto a stored snapshot.

    Split by how fast things move and how expensive they are to fetch. Account,
    positions and open orders are three quick Alpaca calls and change every
    minute, so they come live. The equity curve, run cadence, closed trades and
    per-position price history need portfolio history, the exchange calendar and
    a yfinance download — several seconds — and barely change intraday, so they
    are carried over from the snapshot.

    That split is what makes a live page fast enough to load on every visit. The
    full refresh() stays behind the button for when the slow parts matter.
    """
    snap_positions = {p["ticker"]: p for p in (snapshot.get("positions") or [])}
    realized = float((snapshot.get("account") or {}).get("realized") or 0.0)
    deposits = float((snapshot.get("account") or {}).get("deposits") or 0.0)

    positions = []
    for p in raw_positions:
        sym = (p.get("symbol") or "").upper()
        prior = snap_positions.get(sym, {})
        positions.append({
            "ticker": sym,
            "quantity": _f(p.get("qty")),
            "avg_cost": _f(p.get("avg_entry_price")),
            "last_price": _f(p.get("current_price")),
            "market_value": _f(p.get("market_value")),
            "cost_basis": _f(p.get("cost_basis")),
            "total_gl_dollar": _f(p.get("unrealized_pl")),
            "total_gl_pct": _f(p.get("unrealized_plpc")),
            "today_gl_dollar": _f(p.get("unrealized_intraday_pl")),
            "today_gl_pct": _f(p.get("unrealized_intraday_plpc")),
            "lastday_price": _f(p.get("lastday_price")),
            # Carried: these come from FIFO matching and a price download.
            "held_since": prior.get("held_since"),
            "realized_to_date": prior.get("realized_to_date", 0.0),
            "history": prior.get("history", []),
        })
    positions.sort(key=lambda p: p["total_gl_pct"])

    unrealized = round(sum(p["total_gl_dollar"] for p in positions), 2)
    equity = _f(account.get("equity"))
    last_eq = _f(account.get("last_equity"))
    long_mv = _f(account.get("long_market_value"))

    out = dict(snapshot)
    out["synced_at"] = now
    out["positions"] = positions
    out["open_orders"] = [
        {"ticker": (o.get("symbol") or "").upper(), "side": o.get("side"),
         "qty": o.get("qty"), "notional": o.get("notional"),
         "type": o.get("type"), "time_in_force": o.get("time_in_force"),
         "submitted_at": (o.get("submitted_at") or "")[:19]}
        for o in open_orders
    ]
    out["account"] = {
        **(snapshot.get("account") or {}),
        "equity": equity,
        "last_equity": last_eq,
        "cash": _f(account.get("cash")),
        "long_market_value": long_mv,
        "deployed_pct": round(long_mv / equity, 6) if equity else 0.0,
        "today_dollar": round(equity - last_eq, 2) if last_eq else 0.0,
        "today_pct": round((equity - last_eq) / last_eq, 6) if last_eq else 0.0,
        "realized": realized,
        "unrealized": unrealized,
        "total_pl": round(realized + unrealized, 2),
    }
    out["reconciliation"] = reconcile(realized, unrealized, equity, deposits)

    # Extend the curve's last point to now, so the chart's endpoint is live too.
    curve = dict(snapshot.get("curve") or {})
    dates = list(curve.get("dates") or [])
    acct_pct = list(curve.get("account_pct") or [])
    levels = list(curve.get("account_level") or [])
    if dates and levels and acct_pct:
        today = now[:10]
        base = next((v for v in levels if v), None)
        if base:
            pct = round((equity - base) / base * 100, 4)
            if dates[-1] == today:
                acct_pct[-1], levels[-1] = pct, equity
            else:
                dates.append(today)
                acct_pct.append(pct)
                levels.append(equity)
                spy = list(curve.get("spy_pct") or [])
                spy.append(spy[-1] if spy else None)
                curve["spy_pct"] = spy
        curve.update(dates=dates, account_pct=acct_pct, account_level=levels)
        out["curve"] = curve
        out["vs_spy"] = vs_spy(curve)
    return out


def live_view(snapshot: dict | None = None) -> tuple[dict, str, str | None]:
    """Snapshot overlaid with live broker state. Never raises.

    Returns (data, source, error). source is "live" when the broker answered,
    "snapshot" when it did not — so the page can say which it is showing instead
    of quietly presenting stale numbers as current.
    """
    snap = snapshot if snapshot is not None else load_snapshot()
    if not snap:
        return {}, "none", "no snapshot on disk"
    try:
        from src import broker
        account = broker.get_account()
        raw_positions = broker.get_positions()
        open_orders = broker.get_orders("open")
    except Exception as e:  # noqa: BLE001 — a stale page beats a broken one
        return snap, "snapshot", f"{type(e).__name__}: {e}"
    now = datetime.now().isoformat(timespec="minutes")
    return merge_live(snap, account, raw_positions, open_orders, now), "live", None


def save_snapshot(snap: dict, path: Path | None = None) -> Path:
    p = path or SNAPSHOT_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(snap, indent=2))
    os.replace(tmp, p)
    return p


def load_snapshot(path: Path | None = None) -> dict | None:
    p = path or SNAPSHOT_FILE
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


# ── live refresh (the button) ─────────────────────────────────────────────────

def refresh() -> dict:
    """Fetch everything, assemble, write. Raises SnapshotError on any failure.

    All-or-nothing by construction: every fetch happens before save_snapshot is
    called, so a failure anywhere leaves the previous snapshot in place.
    """
    from src import broker, trader_runlog

    try:
        account = broker.get_account()
        raw_positions = broker.get_positions()
        orders = broker._get("/v2/orders", {"status": "all", "limit": 500, "direction": "asc"})
    except Exception as e:
        raise SnapshotError(f"Alpaca fetch failed: {e}") from e

    try:
        hist = broker._get(
            "/v2/account/portfolio/history", {"period": "3M", "timeframe": "1D"}
        )
    except Exception as e:
        raise SnapshotError(f"Alpaca portfolio history failed: {e}") from e

    account_by_date = equity_by_date(hist)
    # The history series ends at yesterday's close; today's live equity is the
    # point the user actually cares about, so append it.
    account_by_date[date.today().isoformat()] = _f(account.get("equity"))

    if not account_by_date:
        raise SnapshotError("Alpaca portfolio history returned no equity points")

    deposits = _first_nonzero(hist.get("equity") or []) or _f(account.get("equity"))

    start = min(account_by_date)
    try:
        cal = broker._get("/v2/calendar", {"start": start, "end": date.today().isoformat()})
        market_days = [d["date"] for d in cal]
    except Exception as e:
        raise SnapshotError(f"Alpaca calendar fetch failed: {e}") from e

    tickers = sorted({(p.get("symbol") or "").upper() for p in raw_positions})
    try:
        history, spy_by_date = _fetch_history(tickers, start)
    except Exception as e:
        raise SnapshotError(f"Price history fetch failed: {e}") from e

    baseline = inception_baseline(first_fill_date(_fills(orders)), market_days)
    curve = align_on_dates(account_by_date, spy_by_date, market_days, baseline)
    # Cadence only covers days from the baseline on — days before the account
    # traded aren't days the trader failed to run.
    cadence = trader_runlog.collect([d for d in market_days if not baseline or d >= baseline])

    snap = assemble(
        account=account,
        raw_positions=raw_positions,
        orders=orders,
        history=history,
        curve=curve,
        cadence=cadence,
        deposits=deposits,
        synced_at=datetime.now().isoformat(timespec="minutes"),
    )
    save_snapshot(snap)
    return snap


def _first_nonzero(values: list) -> float:
    for v in values:
        if v:
            return float(v)
    return 0.0


def _fetch_history(tickers: list[str], start: str) -> tuple[dict[str, list[dict]], dict[str, float]]:
    """Daily closes for each held ticker plus SPY, from yfinance."""
    import yfinance as yf

    want = sorted(set(tickers) | {"SPY"})
    begin = (date.fromisoformat(start) - timedelta(days=HISTORY_DAYS)).isoformat()
    end = (date.today() + timedelta(days=1)).isoformat()

    raw = yf.download(want, start=begin, end=end, progress=False, auto_adjust=True)
    if raw is None or raw.empty:
        raise RuntimeError("yfinance returned no data")
    closes = raw["Close"]

    history: dict[str, list[dict]] = {}
    for t in tickers:
        if t not in closes.columns:
            continue
        s = closes[t].dropna()
        history[t] = [
            {"date": idx.date().isoformat(), "close": round(float(v), 4)}
            for idx, v in s.items()
        ]

    spy_by_date: dict[str, float] = {}
    if "SPY" in closes.columns:
        for idx, v in closes["SPY"].dropna().items():
            spy_by_date[idx.date().isoformat()] = float(v)
    return history, spy_by_date
