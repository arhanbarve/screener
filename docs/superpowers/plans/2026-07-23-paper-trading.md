# Autonomous Paper Trading Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A fully autonomous daily paper-trading loop: the existing screener produces its ranked CSV at 4:30pm ET; a headless Claude session runs each trading morning (~9:00am ET, with catch-up if the Mac was asleep or off), reads the screener output plus the live Alpaca paper account, makes discretionary entry/exit/sizing decisions, places orders via a narrow CLI, and writes an auditable journal that is committed to the repo.

**Architecture:** Three layers. (1) `src/broker.py` — a thin `requests`-based wrapper over the Alpaca *paper* REST API (account, clock, positions, orders), with a `TRADER_DRY_RUN=1` mode that logs instead of submitting. (2) `src/trader_cli.py` — the only interface the agent touches: JSON-printing subcommands (`status`, `gate`, `buy`, `sell`, `close`, `orders`). (3) `run_trader.sh` + a launchd plist — idempotent wrapper (lockfile, per-day stamp, market-clock gate) that invokes `claude -p` with `trading/PROMPT.md` and a scoped tool allowlist, then commits the journal.

**Tech Stack:** Python 3.14 (`/Library/Frameworks/Python.framework/Versions/3.14/bin/python3`), `requests` (already in requirements), Alpaca Paper Trading API (`https://paper-api.alpaca.markets`), launchd, Claude Code CLI (`/opt/homebrew/bin/claude`, v2.1.218), pytest.

---

## Design decisions (settled with user)

| Decision | Choice |
|---|---|
| Venue | Alpaca paper API (free, email-only signup) |
| Capital | $100k default paper balance |
| Latitude | **Full discretion, zero rails** — screener bands advisory only |
| Universe | Entries from that day's screener CSV (any rank) + SPY/cash as risk-off parking; exits anytime |
| Trigger | launchd → headless `claude -p`, pre-open ~9:00am ET weekdays |
| Missed runs | launchd fires missed calendar jobs on wake; `RunAtLoad` + per-day stamp + market-clock gate covers power-off; window 08:30–15:45 ET |
| Reporting | Daily journal `trading/journal/YYYY-MM-DD.md` + Friday weekly digest, committed |
| Separation | `positions.json` (real Fidelity book) never touched; paper state lives in Alpaca |

**Why morning, not 5pm:** screener CSV is identical either way (produced 4:30pm prior close); a morning session additionally sees overnight news, and the decision→fill gap is ~30 min instead of ~17 hours. A catch-up run that fires mid-day still fills same-day; a missed evening session would leave stale intent overnight.

**Order mechanics:** market DAY orders with `notional` (dollar) sizing, placed pre-open → queue → fill at 9:30 open. Notional orders require `type=market` + `time_in_force=day`, which is exactly what we use. If a catch-up run fires mid-session, orders fill immediately at market — acceptable.

**At-most-once semantics:** the stamp file is written *before* invoking Claude. If the session crashes mid-run it does NOT retry the same day (a retry after partial order placement is worse than a missed day). Failures are visible in `logs/trader_YYYY-MM-DD.log` and the absent journal entry.

## Non-goals

- No shorting, options, margin, or intraday multi-session trading.
- No changes to screener scoring logic.
- No local fill simulator (Alpaca is the ledger).
- Not fixing the two flagged screener data gaps (empty sector column, "Analysis unavailable" news) — separately tracked; the prompt tells the agent to treat those columns as possibly missing.

## Files

- Create: `src/broker.py` — Alpaca REST wrapper
- Create: `tests/test_broker.py`
- Create: `src/trader_cli.py` — agent-facing CLI + run gate
- Create: `tests/test_trader_cli.py`
- Create: `trading/PROMPT.md` — daily session instructions
- Create: `trading/journal/.gitkeep`, `trading/journal/weekly/.gitkeep`
- Create: `run_trader.sh` — launchd entrypoint
- Create: `~/Library/LaunchAgents/com.arhanbarve.trader.plist` (copy kept at `scripts/com.arhanbarve.trader.plist`)
- Modify: `run_screener.sh` — add freshness gate so `RunAtLoad` catch-up is idempotent
- Modify: `~/Library/LaunchAgents/com.arhanbarve.screener.plist` — add `RunAtLoad`
- Modify: `.env` (user step) — `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`

---

### Task 1: `src/broker.py` — Alpaca REST wrapper

**Files:**
- Create: `src/broker.py`
- Test: `tests/test_broker.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_broker.py
import json
from unittest.mock import patch, MagicMock

import pytest

from src import broker


def _resp(status=200, body=None):
    m = MagicMock()
    m.status_code = status
    m.json.return_value = body if body is not None else {}
    m.text = json.dumps(body) if body is not None else ""
    return m


@pytest.fixture(autouse=True)
def alpaca_env(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "test-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test-secret")
    monkeypatch.delenv("TRADER_DRY_RUN", raising=False)


def test_headers_missing_keys_raises(monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    with pytest.raises(broker.BrokerError, match="Missing ALPACA"):
        broker._headers()


def test_get_account_hits_paper_endpoint():
    with patch("src.broker.requests.get", return_value=_resp(body={"equity": "100000"})) as g:
        out = broker.get_account()
    assert out == {"equity": "100000"}
    url = g.call_args[0][0]
    assert url == "https://paper-api.alpaca.markets/v2/account"
    headers = g.call_args[1]["headers"]
    assert headers["APCA-API-KEY-ID"] == "test-key"
    assert headers["APCA-API-SECRET-KEY"] == "test-secret"


def test_get_non_200_raises():
    with patch("src.broker.requests.get", return_value=_resp(status=403, body={"message": "forbidden"})):
        with pytest.raises(broker.BrokerError, match="403"):
            broker.get_clock()


def test_submit_order_notional_payload():
    with patch("src.broker.requests.post", return_value=_resp(body={"id": "abc"})) as p:
        out = broker.submit_order("stx", "buy", notional=5000.555)
    assert out == {"id": "abc"}
    payload = p.call_args[1]["json"]
    assert payload == {
        "symbol": "STX",
        "side": "buy",
        "type": "market",
        "time_in_force": "day",
        "notional": 5000.56,
    }


def test_submit_order_qty_payload():
    with patch("src.broker.requests.post", return_value=_resp(body={"id": "abc"})) as p:
        broker.submit_order("SPY", "sell", qty=3)
    assert p.call_args[1]["json"]["qty"] == "3"
    assert "notional" not in p.call_args[1]["json"]


def test_submit_order_requires_exactly_one_size():
    with pytest.raises(broker.BrokerError):
        broker.submit_order("SPY", "buy")
    with pytest.raises(broker.BrokerError):
        broker.submit_order("SPY", "buy", notional=100, qty=1)


def test_submit_order_dry_run(monkeypatch):
    monkeypatch.setenv("TRADER_DRY_RUN", "1")
    with patch("src.broker.requests.post") as p:
        out = broker.submit_order("STX", "buy", notional=1000)
    p.assert_not_called()
    assert out["dry_run"] is True
    assert out["would_submit"]["symbol"] == "STX"


def test_close_position_dry_run(monkeypatch):
    monkeypatch.setenv("TRADER_DRY_RUN", "1")
    with patch("src.broker.requests.delete") as d:
        out = broker.close_position("stx")
    d.assert_not_called()
    assert out == {"dry_run": True, "would_close": "STX"}


def test_close_position_deletes():
    with patch("src.broker.requests.delete", return_value=_resp(body={"id": "ord1"})) as d:
        out = broker.close_position("STX")
    assert d.call_args[0][0] == "https://paper-api.alpaca.markets/v2/positions/STX"
    assert out == {"id": "ord1"}


def test_get_orders_passes_status():
    with patch("src.broker.requests.get", return_value=_resp(body=[])) as g:
        broker.get_orders("closed")
    assert g.call_args[1]["params"] == {"status": "closed", "limit": 100}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m pytest tests/test_broker.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.broker'` (collection error).

- [ ] **Step 3: Write the implementation**

```python
# src/broker.py
"""Thin Alpaca paper-trading REST wrapper.

All calls hit the PAPER endpoint only — this module can never touch a
live brokerage account. Credentials come from env vars ALPACA_API_KEY /
ALPACA_SECRET_KEY. Set TRADER_DRY_RUN=1 to have order/close calls return
what they *would* send without sending it.
"""
import os

import requests

BASE_URL = "https://paper-api.alpaca.markets"
TIMEOUT = 30


class BrokerError(Exception):
    pass


def _dry_run() -> bool:
    return os.environ.get("TRADER_DRY_RUN") == "1"


def _headers() -> dict:
    key = os.environ.get("ALPACA_API_KEY")
    secret = os.environ.get("ALPACA_SECRET_KEY")
    if not key or not secret:
        raise BrokerError("Missing ALPACA_API_KEY / ALPACA_SECRET_KEY env vars")
    return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}


def _get(path: str, params: dict | None = None):
    resp = requests.get(f"{BASE_URL}{path}", headers=_headers(), params=params, timeout=TIMEOUT)
    if resp.status_code != 200:
        raise BrokerError(f"GET {path} -> {resp.status_code}: {resp.text}")
    return resp.json()


def _post(path: str, payload: dict):
    resp = requests.post(f"{BASE_URL}{path}", headers=_headers(), json=payload, timeout=TIMEOUT)
    if resp.status_code not in (200, 201):
        raise BrokerError(f"POST {path} -> {resp.status_code}: {resp.text}")
    return resp.json()


def _delete(path: str):
    resp = requests.delete(f"{BASE_URL}{path}", headers=_headers(), timeout=TIMEOUT)
    if resp.status_code not in (200, 204, 207):
        raise BrokerError(f"DELETE {path} -> {resp.status_code}: {resp.text}")
    return resp.json() if resp.text else {}


def get_account() -> dict:
    return _get("/v2/account")


def get_clock() -> dict:
    return _get("/v2/clock")


def get_positions() -> list:
    return _get("/v2/positions")


def get_orders(status: str = "open") -> list:
    return _get("/v2/orders", params={"status": status, "limit": 100})


def submit_order(symbol: str, side: str, notional: float | None = None, qty: float | None = None) -> dict:
    """Market DAY order. Exactly one of notional (dollars) or qty (shares)."""
    if (notional is None) == (qty is None):
        raise BrokerError("Provide exactly one of notional or qty")
    payload = {
        "symbol": symbol.upper(),
        "side": side,
        "type": "market",
        "time_in_force": "day",
    }
    if notional is not None:
        payload["notional"] = round(float(notional), 2)
    else:
        payload["qty"] = f"{qty:g}"
    if _dry_run():
        return {"dry_run": True, "would_submit": payload}
    return _post("/v2/orders", payload)


def close_position(symbol: str) -> dict:
    """Liquidate the entire position in symbol at market."""
    if _dry_run():
        return {"dry_run": True, "would_close": symbol.upper()}
    return _delete(f"/v2/positions/{symbol.upper()}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m pytest tests/test_broker.py -v`
Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
git add src/broker.py tests/test_broker.py
git commit -m "feat(trading): Alpaca paper REST wrapper with dry-run mode"
```

---

### Task 2: `src/trader_cli.py` — agent-facing CLI + run gate

**Files:**
- Create: `src/trader_cli.py`
- Test: `tests/test_trader_cli.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_trader_cli.py
import json
from datetime import datetime
from unittest.mock import patch
from zoneinfo import ZoneInfo

from src import trader_cli

ET = ZoneInfo("America/New_York")


def _clock(is_open, next_open):
    return {"is_open": is_open, "next_open": next_open, "next_close": "2026-07-23T16:00:00-04:00"}


def test_gate_weekend_skips():
    now = datetime(2026, 7, 25, 10, 0, tzinfo=ET)  # Saturday
    with patch("src.trader_cli.broker.get_clock",
               return_value=_clock(False, "2026-07-27T09:30:00-04:00")):
        out = trader_cli.cmd_gate(now=now)
    assert out["run"] is False
    assert "not a trading day" in out["reason"]


def test_gate_trading_morning_runs():
    now = datetime(2026, 7, 23, 9, 0, tzinfo=ET)  # Thursday pre-open
    with patch("src.trader_cli.broker.get_clock",
               return_value=_clock(False, "2026-07-23T09:30:00-04:00")):
        out = trader_cli.cmd_gate(now=now)
    assert out["run"] is True


def test_gate_market_open_midday_runs():
    now = datetime(2026, 7, 23, 13, 0, tzinfo=ET)
    with patch("src.trader_cli.broker.get_clock",
               return_value=_clock(True, "2026-07-24T09:30:00-04:00")):
        out = trader_cli.cmd_gate(now=now)
    assert out["run"] is True


def test_gate_before_window_skips():
    now = datetime(2026, 7, 23, 7, 0, tzinfo=ET)
    with patch("src.trader_cli.broker.get_clock",
               return_value=_clock(False, "2026-07-23T09:30:00-04:00")):
        out = trader_cli.cmd_gate(now=now)
    assert out["run"] is False
    assert "before" in out["reason"]


def test_gate_after_cutoff_skips():
    now = datetime(2026, 7, 23, 17, 30, tzinfo=ET)  # post-close
    with patch("src.trader_cli.broker.get_clock",
               return_value=_clock(False, "2026-07-24T09:30:00-04:00")):
        out = trader_cli.cmd_gate(now=now)
    assert out["run"] is False


def test_buy_prints_order_json(capsys):
    with patch("src.trader_cli.broker.submit_order", return_value={"id": "ord1"}) as s:
        rc = trader_cli.main(["buy", "STX", "--notional", "5000"])
    assert rc == 0
    s.assert_called_once_with("STX", "buy", notional=5000.0, qty=None)
    assert json.loads(capsys.readouterr().out) == {"id": "ord1"}


def test_sell_qty(capsys):
    with patch("src.trader_cli.broker.submit_order", return_value={"id": "ord2"}) as s:
        rc = trader_cli.main(["sell", "SPY", "--qty", "3"])
    assert rc == 0
    s.assert_called_once_with("SPY", "sell", notional=None, qty=3.0)


def test_close_position(capsys):
    with patch("src.trader_cli.broker.close_position", return_value={"ok": 1}) as c:
        rc = trader_cli.main(["close", "STX"])
    assert rc == 0
    c.assert_called_once_with("STX")


def test_status_aggregates(capsys):
    with patch("src.trader_cli.broker.get_account", return_value={"equity": "1"}), \
         patch("src.trader_cli.broker.get_clock", return_value={"is_open": True}), \
         patch("src.trader_cli.broker.get_positions", return_value=[]), \
         patch("src.trader_cli.broker.get_orders", return_value=[]):
        rc = trader_cli.main(["status"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert set(out) == {"account", "clock", "positions", "open_orders"}


def test_broker_error_returns_nonzero(capsys):
    from src.broker import BrokerError
    with patch("src.trader_cli.broker.get_account", side_effect=BrokerError("boom")):
        rc = trader_cli.main(["status"])
    assert rc == 1
    assert "boom" in capsys.readouterr().err
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m pytest tests/test_trader_cli.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.trader_cli'`.

- [ ] **Step 3: Write the implementation**

```python
# src/trader_cli.py
"""CLI used by the daily trading agent. Every command prints JSON to stdout.

Commands:
  status              account + clock + positions + open orders
  gate                {"run": bool, "reason": str} — should today's session run now?
  buy SYM (--notional D | --qty N)
  sell SYM (--notional D | --qty N)
  close SYM           liquidate entire position
  orders [--status open|closed|all]
"""
import argparse
import json
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

from src import broker

ET = ZoneInfo("America/New_York")
GATE_EARLIEST = (8, 30)   # don't trade on a midnight reboot with stale context
GATE_LATEST = (15, 45)    # market orders too close to 16:00 close are pointless


def cmd_status() -> dict:
    return {
        "account": broker.get_account(),
        "clock": broker.get_clock(),
        "positions": broker.get_positions(),
        "open_orders": broker.get_orders("open"),
    }


def cmd_gate(now: datetime | None = None) -> dict:
    now = now or datetime.now(ET)
    clock = broker.get_clock()
    today = now.date().isoformat()
    is_trading_day = clock["is_open"] or clock["next_open"][:10] == today
    if not is_trading_day:
        return {"run": False, "reason": f"not a trading day (next open {clock['next_open']})"}
    hm = (now.hour, now.minute)
    if hm < GATE_EARLIEST:
        return {"run": False, "reason": "before 08:30 ET window"}
    if hm > GATE_LATEST:
        return {"run": False, "reason": "after 15:45 ET cutoff"}
    return {"run": True, "reason": "trading day, within 08:30-15:45 ET window"}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="trader_cli", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    sub.add_parser("gate")
    for name in ("buy", "sell"):
        sp = sub.add_parser(name)
        sp.add_argument("symbol")
        g = sp.add_mutually_exclusive_group(required=True)
        g.add_argument("--notional", type=float)
        g.add_argument("--qty", type=float)
    sp = sub.add_parser("close")
    sp.add_argument("symbol")
    sp = sub.add_parser("orders")
    sp.add_argument("--status", default="open", choices=["open", "closed", "all"])
    args = p.parse_args(argv)

    try:
        if args.cmd == "status":
            out = cmd_status()
        elif args.cmd == "gate":
            out = cmd_gate()
        elif args.cmd in ("buy", "sell"):
            out = broker.submit_order(args.symbol.upper(), args.cmd,
                                      notional=args.notional, qty=args.qty)
        elif args.cmd == "close":
            out = broker.close_position(args.symbol.upper())
        else:  # orders
            out = broker.get_orders(args.status)
    except broker.BrokerError as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        return 1
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m pytest tests/test_trader_cli.py -v`
Expected: 10 passed.

- [ ] **Step 5: Run full suite (regression)**

Run: `/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m pytest tests/ -q`
Expected: all pass (same count as before + 20 new).

- [ ] **Step 6: Commit**

```bash
git add src/trader_cli.py tests/test_trader_cli.py
git commit -m "feat(trading): agent-facing trader CLI with run gate"
```

---

### Task 3: `trading/PROMPT.md` — daily session instructions

**Files:**
- Create: `trading/PROMPT.md`
- Create: `trading/journal/.gitkeep`
- Create: `trading/journal/weekly/.gitkeep`

- [ ] **Step 1: Write the prompt file**

```markdown
# Daily Paper-Trading Session

You are the discretionary portfolio manager of an Alpaca PAPER trading
account (started at $100,000). Your single goal: grow account equity.
You have full discretion — no mechanical rules bind you. You own every
decision and must justify each one in the journal.

## Hard boundaries (the only rules)

- PAPER account only. All orders go through `src.trader_cli` — never any
  other mechanism.
- Long-only US equities. No shorting, no options, no margin: after your
  buys, projected cash must stay >= $0.
- New entries must come from the latest screener CSV
  (`output/screen_*.csv`, any rank) or be SPY (risk-off parking).
  Exits: anything, anytime.
- NEVER touch `positions.json` (that mirrors a real brokerage account),
  and never edit files outside `trading/`.
- If `status` errors or data looks corrupt, make NO trades; write a
  journal entry explaining what failed.

## Procedure

1. `PY -m src.trader_cli status` (PY = /Library/Frameworks/Python.framework/Versions/3.14/bin/python3)
   — equity, cash, positions with unrealized P&L, open orders, market clock.
2. Find the latest `output/screen_YYYY-MM-DD.csv`. If it is older than the
   most recent trading day, note the staleness in the journal and weight
   technicals less. Columns worth reading: ticker, composite, weight_pct,
   conviction, entry_signal, rsi_14, macd, adx, pct_from_high, thesis
   columns. `sector` may be empty and news columns may say
   "Analysis unavailable" — treat both as missing data, not signal.
3. Check overnight/pre-market news (WebSearch) for current holdings and
   any candidate you intend to buy or sell.
4. Decide. Consider: current positions vs their screener ranks today,
   better-ranked replacements, concentration, regime (SPY trend), news.
   Doing nothing is a valid decision — say why.
5. Execute: `PY -m src.trader_cli buy TICKER --notional 8000`,
   `... sell TICKER --notional 4000`, or `... close TICKER`.
   Sells/closes BEFORE buys (frees cash; orders fill at next open in
   sequence). Verify with `PY -m src.trader_cli orders --status all`.
6. Journal to `trading/journal/YYYY-MM-DD.md` (today's date, ET):

   ```
   # Trading Journal — YYYY-MM-DD
   ## Snapshot (pre-decision)
   Equity / cash / positions table with unrealized P&L
   ## Market context
   2-4 sentences: SPY regime, notable overnight news
   ## Decisions
   One block per action AND per considered-but-rejected action: what,
   why, screener evidence (rank/composite/conviction), news evidence
   ## Orders placed
   Table: side, ticker, notional or qty, order id, status
   ## Scorecard
   Account equity vs $100,000 baseline (%); SPY vs its price at
   experiment start (record SPY's current price each day)
   ```

7. Fridays: also write `trading/journal/weekly/YYYY-Www.md` — week's
   equity change vs SPY, best/worst call, one lesson, current book.

Keep total session focused: read, decide, execute, journal, stop.
```

- [ ] **Step 2: Create journal dirs**

```bash
mkdir -p trading/journal/weekly
touch trading/journal/.gitkeep trading/journal/weekly/.gitkeep
```

- [ ] **Step 3: Commit**

```bash
git add trading/
git commit -m "feat(trading): daily session prompt and journal structure"
```

---

### Task 4: `run_trader.sh` — launchd entrypoint

**Files:**
- Create: `run_trader.sh` (repo root, chmod +x)

- [ ] **Step 1: Write the script**

```bash
#!/bin/bash
set -euo pipefail

SCREENER_DIR="/Users/arhanbarve/Code/screener"
LOG_DIR="$SCREENER_DIR/logs"
LOG_FILE="$LOG_DIR/trader_$(date +%Y-%m-%d).log"
LOCKFILE="/tmp/trader_run.lock"
STAMP_FILE="$LOG_DIR/trader_last_run"
PY="/Library/Frameworks/Python.framework/Versions/3.14/bin/python3"
CLAUDE="/opt/homebrew/bin/claude"

mkdir -p "$LOG_DIR"

# Prevent overlapping runs (same pattern as run_screener.sh)
if [ -f "$LOCKFILE" ]; then
    existing_pid=$(cat "$LOCKFILE" 2>/dev/null || echo "")
    if [ -n "$existing_pid" ] && kill -0 "$existing_pid" 2>/dev/null; then
        echo "=== Trader already running (PID $existing_pid), skipping ===" >> "$LOG_FILE"
        exit 0
    fi
    rm -f "$LOCKFILE"
fi
echo $$ > "$LOCKFILE"
trap "rm -f '$LOCKFILE'" EXIT

cd "$SCREENER_DIR"

if [ -f "$SCREENER_DIR/.env" ]; then
    set -a
    source "$SCREENER_DIR/.env"
    set +a
fi

TODAY=$(date +%Y-%m-%d)

# At-most-once per day: stamp is written BEFORE the session so a crashed
# session never retries the same day (partial orders + retry is worse
# than a missed day).
if [ -f "$STAMP_FILE" ] && [ "$(cat "$STAMP_FILE")" = "$TODAY" ]; then
    echo "=== Trader already ran today, skipping ===" >> "$LOG_FILE"
    exit 0
fi

# Market-clock gate: trading day, 08:30-15:45 ET only
GATE=$("$PY" -m src.trader_cli gate 2>>"$LOG_FILE") || {
    echo "=== Gate check failed (missing keys / network?) ===" >> "$LOG_FILE"
    exit 0
}
echo "=== Gate: $GATE ===" >> "$LOG_FILE"
if ! echo "$GATE" | grep -q '"run": true'; then
    exit 0
fi

echo "$TODAY" > "$STAMP_FILE"
echo "=== Trader session started: $(date) ===" >> "$LOG_FILE"

"$CLAUDE" -p "$(cat trading/PROMPT.md)" \
    --allowedTools "Bash($PY -m src.trader_cli:*),Read,Glob,Grep,WebSearch,WebFetch,Write(trading/**),Edit(trading/**)" \
    --max-turns 120 \
    >> "$LOG_FILE" 2>&1 || echo "=== claude session exited nonzero ===" >> "$LOG_FILE"

echo "=== Trader session finished: $(date) ===" >> "$LOG_FILE"

# Commit journal (same auto-commit policy as run_screener.sh)
if git diff --quiet HEAD -- trading/ && [ -z "$(git ls-files --others --exclude-standard trading/)" ]; then
    echo "=== No journal changes to commit ===" >> "$LOG_FILE"
else
    git add trading/
    git commit -m "chore(trading): journal $TODAY" >> "$LOG_FILE" 2>&1
    git push >> "$LOG_FILE" 2>&1 || echo "=== git push failed (non-fatal) ===" >> "$LOG_FILE"
fi
```

- [ ] **Step 2: Make executable and smoke-test the gate path**

```bash
chmod +x run_trader.sh
# Without Alpaca keys in .env this must exit 0 and log "Gate check failed"
./run_trader.sh
cat logs/trader_$(date +%Y-%m-%d).log
```

Expected: exits 0; log shows gate failure (no keys yet), no claude session started, no stamp written for a failed gate.

- [ ] **Step 3: Commit**

```bash
git add run_trader.sh
git commit -m "feat(trading): launchd entrypoint with stamp + market-clock gate"
```

---

### Task 5: launchd plist for the trader

**Files:**
- Create: `scripts/com.arhanbarve.trader.plist` (tracked copy)
- Create: `~/Library/LaunchAgents/com.arhanbarve.trader.plist` (installed copy)

- [ ] **Step 1: Write the plist**

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.arhanbarve.trader</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>/Users/arhanbarve/Code/screener/run_trader.sh</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>StartCalendarInterval</key>
    <array>
        <dict><key>Hour</key><integer>9</integer><key>Minute</key><integer>0</integer><key>Weekday</key><integer>1</integer></dict>
        <dict><key>Hour</key><integer>9</integer><key>Minute</key><integer>0</integer><key>Weekday</key><integer>2</integer></dict>
        <dict><key>Hour</key><integer>9</integer><key>Minute</key><integer>0</integer><key>Weekday</key><integer>3</integer></dict>
        <dict><key>Hour</key><integer>9</integer><key>Minute</key><integer>0</integer><key>Weekday</key><integer>4</integer></dict>
        <dict><key>Hour</key><integer>9</integer><key>Minute</key><integer>0</integer><key>Weekday</key><integer>5</integer></dict>
    </array>
    <key>StandardOutPath</key>
    <string>/Users/arhanbarve/Code/screener/logs/trader_launchd.log</string>
    <key>StandardErrorPath</key>
    <string>/Users/arhanbarve/Code/screener/logs/trader_launchd.log</string>
</dict>
</plist>
```

Missed-run coverage: launchd fires missed `StartCalendarInterval` jobs once on wake-from-sleep; `RunAtLoad` fires at login/boot for the powered-off case; the stamp + gate in `run_trader.sh` make every extra fire a no-op.

- [ ] **Step 2: Install and load**

```bash
mkdir -p scripts
cp scripts/com.arhanbarve.trader.plist ~/Library/LaunchAgents/
plutil -lint ~/Library/LaunchAgents/com.arhanbarve.trader.plist
launchctl load ~/Library/LaunchAgents/com.arhanbarve.trader.plist
launchctl list | grep trader
```

Expected: `plutil` says OK; label appears in `launchctl list`. RunAtLoad fires immediately → gate fails (no keys) → harmless, proves the wiring.

- [ ] **Step 3: Commit**

```bash
git add scripts/com.arhanbarve.trader.plist
git commit -m "feat(trading): launchd schedule 9:00am ET weekdays with catch-up"
```

---

### Task 6: screener catch-up (freshness gate + RunAtLoad)

The morning trader needs a fresh prior-close CSV even if the Mac was off at 4:30pm. launchd wake-coalescing already re-fires the screener job on wake; adding `RunAtLoad` covers boot. The gate below makes any extra fire idempotent.

**Files:**
- Modify: `run_screener.sh` (insert after `.env` sourcing, before the run)
- Modify: `~/Library/LaunchAgents/com.arhanbarve.screener.plist`

- [ ] **Step 1: Add freshness gate to `run_screener.sh`**

Insert after the `.env` block:

```bash
# Skip if output for the most recent completed trading close already exists.
# (Makes RunAtLoad / wake-coalesced re-fires idempotent. --force overrides.)
LATEST_NEEDED=$(/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 - <<'EOF'
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
now = datetime.now(ZoneInfo("America/New_York"))
d = now.date()
if (now.hour, now.minute) < (16, 30):
    d -= timedelta(days=1)
while d.weekday() >= 5:
    d -= timedelta(days=1)
print(d.isoformat())
EOF
)
if [ "${1:-}" != "--force" ] && [ -f "output/screen_${LATEST_NEEDED}.csv" ]; then
    echo "=== Fresh output for ${LATEST_NEEDED} exists, skipping ===" >> "$LOG_FILE"
    exit 0
fi
```

Semantics: at the normal 16:30 fire, `LATEST_NEEDED` = today and today's file doesn't exist yet → runs. At a 10am boot fire, `LATEST_NEEDED` = previous trading day → runs only if that file is missing. Weekend logins skip. A market holiday causes one harmless extra run.

- [ ] **Step 2: Add RunAtLoad to screener plist and reload**

```bash
plutil -replace RunAtLoad -bool true ~/Library/LaunchAgents/com.arhanbarve.screener.plist
launchctl unload ~/Library/LaunchAgents/com.arhanbarve.screener.plist
launchctl load ~/Library/LaunchAgents/com.arhanbarve.screener.plist
```

- [ ] **Step 3: Verify gate logic manually**

```bash
bash run_screener.sh   # today's CSV exists -> must log "skipping" and exit 0
tail -2 logs/run_$(date +%Y-%m-%d).log
```

Expected: "Fresh output ... exists, skipping".

- [ ] **Step 4: Commit**

```bash
git add run_screener.sh
git commit -m "feat(screener): freshness gate enables RunAtLoad catch-up"
```

---

### Task 7: end-to-end dry run (blocked on user's Alpaca keys)

- [ ] **Step 1 (USER): create Alpaca paper account, put keys in `.env`**

```
ALPACA_API_KEY=PK...
ALPACA_SECRET_KEY=...
```

- [ ] **Step 2: verify connectivity**

```bash
set -a; source .env; set +a
/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m src.trader_cli status
/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m src.trader_cli gate
```

Expected: JSON with `"equity": "100000"`-ish account, clock, empty positions; gate true/false with sane reason.

- [ ] **Step 3: full dry-run session**

```bash
rm -f logs/trader_last_run
TRADER_DRY_RUN=1 ./run_trader.sh
cat trading/journal/$(date +%Y-%m-%d).md
```

Expected: session runs, journal written, all orders show `"dry_run": true`, nothing hit Alpaca's order endpoint (dashboard shows no orders).

- [ ] **Step 4 (USER): review journal, then arm**

Nothing to change — `run_trader.sh` doesn't set `TRADER_DRY_RUN`, so the next scheduled run trades live (paper). Reset the stamp only if you want a live session the same day: `rm logs/trader_last_run`.

---

## Test plan summary

- Unit: 20 mocked tests across broker + CLI (no network).
- Regression: full `pytest tests/` after Task 2.
- Integration: Task 7 dry-run against real Alpaca paper API with `TRADER_DRY_RUN=1`.
- Live validation: first real morning run reviewed via journal + Alpaca dashboard.

## Rollback / risk notes

- Disarm everything: `launchctl unload ~/Library/LaunchAgents/com.arhanbarve.trader.plist`.
- Paper account can be reset to $100k from the Alpaca dashboard at any time.
- Real-money risk: none — code is pinned to `paper-api.alpaca.markets`; there is no live endpoint anywhere in it.
- Runaway sessions: `--max-turns 120` bounds each session; stamp file bounds to one session/day.
- Token cost: one headless Claude session per trading day (~5 min) against the existing subscription.
