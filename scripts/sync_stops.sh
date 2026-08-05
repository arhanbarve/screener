#!/bin/bash
# Rest a protective stop under every position. Run by launchd during market hours.
#
# This closes the gap that made the watchdog nag: sync-stops existed as a command
# and as advice text, but nothing ever ran it. The stops that matter are the ones
# on a position bought this morning, and they cannot be placed before the buy
# fills — Alpaca reads a sell stop resting against an open buy limit on the same
# symbol as a wash trade and refuses it (403 / code 40310000).
#
# So this runs a few times through the session rather than once: an entry that
# fills at the open is covered by the first pass, one that fills at 14:00 by a
# later one. cmd_sync_stops is idempotent, so the extra passes place nothing when
# there is nothing to place.
#
# It does not notify. The watchdog owns alerting; this owns doing.
set -uo pipefail

SCREENER_DIR="/Users/arhanbarve/Code/screener"
LOG_FILE="$SCREENER_DIR/logs/stop_sync.log"
LOCKFILE="/tmp/stop_sync.lock"
PY="/Library/Frameworks/Python.framework/Versions/3.14/bin/python3"

mkdir -p "$SCREENER_DIR/logs"
cd "$SCREENER_DIR" || exit 0

# A sync overlapping itself could double-place a stop between the read and the
# write, so a slow pass makes the next one skip rather than queue.
if [ -f "$LOCKFILE" ]; then
    existing_pid=$(cat "$LOCKFILE" 2>/dev/null || echo "")
    if [ -n "$existing_pid" ] && kill -0 "$existing_pid" 2>/dev/null; then
        echo "=== stop sync already running (PID $existing_pid), skipping $(date) ===" >> "$LOG_FILE"
        exit 0
    fi
    rm -f "$LOCKFILE"
fi
echo $$ > "$LOCKFILE"
trap "rm -f '$LOCKFILE'" EXIT

if [ -f "$SCREENER_DIR/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$SCREENER_DIR/.env"
    set +a
fi

# Only act while the market is actually open. A stop placed at 04:00 rests
# untriggerable for hours, and the whole point is to cover shares that just filled.
SESSION=$("$PY" -c "
from src import broker, orders
from datetime import datetime
from zoneinfo import ZoneInfo
try:
    print(orders.market_session(broker.get_clock(), datetime.now(ZoneInfo('America/New_York'))))
except Exception as e:
    print(f'unknown:{type(e).__name__}')
" 2>&1 | tail -1)

if [ "$SESSION" != "open" ]; then
    echo "=== stop sync $(date) skipped: session=$SESSION ===" >> "$LOG_FILE"
    exit 0
fi

RESULT=$("$PY" -m src.trader_cli sync-stops --apply 2>&1)
RC=$?

# Log one line per outcome rather than the whole JSON — this log is read to answer
# "did the stops go on today", and the full plan is already in the sync output.
SUMMARY=$(printf '%s' "$RESULT" | "$PY" -c "
import json, sys
try:
    d = json.load(sys.stdin)
except Exception:
    print('unparseable output'); raise SystemExit
placed = [p.get('symbol') for p in d.get('placed') or []]
failed = d.get('failed') or []
keep = [k.get('symbol') or k.get('ticker') for k in d.get('keep') or []]
skip = [s.get('ticker') for s in d.get('skip') or []]
print(f\"placed={placed or '-'} kept={keep or '-'} excused={skip or '-'}\")
for f in failed:
    print(f\"  FAILED {f.get('ticker')} ({f.get('action')}): {str(f.get('error'))[:160]}\")
" 2>&1)

{
    echo "=== stop sync $(date) rc=$RC session=$SESSION ==="
    echo "$SUMMARY"
} >> "$LOG_FILE"

# Keep the log from growing without bound.
if [ -f "$LOG_FILE" ] && [ "$(wc -l < "$LOG_FILE")" -gt 5000 ]; then
    tail -2000 "$LOG_FILE" > "$LOG_FILE.tmp" && mv "$LOG_FILE.tmp" "$LOG_FILE"
fi

exit 0
