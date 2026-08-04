#!/bin/bash
# Trading-system health watchdog. Run by launchd every 30 minutes.
#
# Notifies only when something is not OK — a watchdog that pings on every green
# run gets muted, and a muted watchdog is the same as no watchdog.
set -uo pipefail

SCREENER_DIR="/Users/arhanbarve/Code/screener"
LOG_FILE="$SCREENER_DIR/logs/watchdog.log"
STATE_FILE="$SCREENER_DIR/logs/watchdog_last_status"
PY="/Library/Frameworks/Python.framework/Versions/3.14/bin/python3"

mkdir -p "$SCREENER_DIR/logs"
cd "$SCREENER_DIR" || exit 0

if [ -f "$SCREENER_DIR/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$SCREENER_DIR/.env"
    set +a
fi

# The pytest run is the slow part; skip it outside a nightly slot so the 30-minute
# cadence stays cheap. Full run once a day at the 22:00 tick.
HOUR=$(date +%H)
SKIP_TESTS="--skip-tests"
[ "$HOUR" = "22" ] && SKIP_TESTS=""

REPORT=$("$PY" -m src.watchdog $SKIP_TESTS 2>&1)
RC=$?

{
    echo "=== watchdog $(date) rc=$RC ==="
    echo "$REPORT"
} >> "$LOG_FILE"

STATUS=$(printf '%s' "$REPORT" | head -1 | awk '{print $2}')
PREV=$(cat "$STATE_FILE" 2>/dev/null || echo "")
printf '%s' "$STATUS" > "$STATE_FILE"

# Notify on a problem, and once more when it clears — the recovery is the part
# you actually want to know about after acting on an alert.
if [ "$RC" -ne 0 ]; then
    SUMMARY=$(printf '%s' "$REPORT" | grep -E '^ (FAIL|warn) ' | head -3 \
              | sed 's/^ *//' | tr '\n' ';' | cut -c1-240)
    osascript -e "display notification \"${SUMMARY//\"/}\" with title \"Screener watchdog: $STATUS\"" 2>/dev/null
elif [ -n "$PREV" ] && [ "$PREV" != "OK" ]; then
    osascript -e 'display notification "All checks green again" with title "Screener watchdog: recovered"' 2>/dev/null
fi

# Keep the log from growing without bound.
if [ -f "$LOG_FILE" ] && [ "$(wc -l < "$LOG_FILE")" -gt 5000 ]; then
    tail -2000 "$LOG_FILE" > "$LOG_FILE.tmp" && mv "$LOG_FILE.tmp" "$LOG_FILE"
fi

exit 0
