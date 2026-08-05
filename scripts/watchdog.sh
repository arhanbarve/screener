#!/bin/bash
# Trading-system health watchdog. Run by launchd every 30 minutes.
#
# Notifies only when something is not OK — a watchdog that pings on every green
# run gets muted, and a muted watchdog is the same as no watchdog.
#
# It also has to stay quiet about a problem it has already reported. The first
# version notified on every non-OK tick, so a warn that could not be cleared until
# the next open produced a notification every 30 minutes all night. Repetition is
# the same mute button as noise. So: a FAIL raises when it is new or every
# FAIL_REPEAT_HOURS, and warn-level findings collapse to one digest a day.
set -uo pipefail

SCREENER_DIR="/Users/arhanbarve/Code/screener"
LOG_FILE="$SCREENER_DIR/logs/watchdog.log"
STATE_FILE="$SCREENER_DIR/logs/watchdog_last_status"
NOTIFY_STATE="$SCREENER_DIR/logs/watchdog_notify_state"
PY="/Library/Frameworks/Python.framework/Versions/3.14/bin/python3"

# How long an unchanged FAIL waits before it raises again.
FAIL_REPEAT_HOURS=6

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

# What is wrong, and has it changed since the last time we said so?
PROBLEMS=$(printf '%s' "$REPORT" | grep -E '^ (FAIL|warn) ' || true)
HAS_FAIL=$(printf '%s' "$PROBLEMS" | grep -cE '^ FAIL ' || true)
DIGEST=$(printf '%s' "$PROBLEMS" | shasum | cut -c1-12)
NOW=$(date +%s)
TODAY_YMD=$(date +%Y-%m-%d)

# State line: "<digest> <epoch of last notify> <ymd of last warn digest>".
read -r LAST_DIGEST LAST_NOTIFY LAST_WARN_DAY < <(cat "$NOTIFY_STATE" 2>/dev/null || echo " 0 ")
LAST_NOTIFY=${LAST_NOTIFY:-0}

NOTIFY=""
TITLE="Screener watchdog: $STATUS"
if [ "$RC" -ne 0 ] && [ "$HAS_FAIL" -gt 0 ]; then
    # A FAIL means something is acting wrong now: raise when new, and re-raise on a
    # slow heartbeat so a standing failure cannot be forgotten entirely.
    AGE_HOURS=$(( (NOW - LAST_NOTIFY) / 3600 ))
    if [ "$DIGEST" != "$LAST_DIGEST" ] || [ "$AGE_HOURS" -ge "$FAIL_REPEAT_HOURS" ]; then
        NOTIFY=$(printf '%s' "$PROBLEMS" | head -3 | sed 's/^ *//' | tr '\n' ';' | cut -c1-240)
    fi
elif [ "$RC" -ne 0 ]; then
    # Warn-level only: one digest per calendar day. These are things worth knowing,
    # not things worth interrupting for — the log and the dashboard carry the detail.
    if [ "$TODAY_YMD" != "$LAST_WARN_DAY" ]; then
        COUNT=$(printf '%s' "$PROBLEMS" | grep -c . || true)
        NOTIFY="$COUNT warning(s) today: $(printf '%s' "$PROBLEMS" | head -2 \
                | sed 's/^ *//' | tr '\n' ';' | cut -c1-200)"
        TITLE="Screener watchdog: daily digest"
        LAST_WARN_DAY="$TODAY_YMD"
    fi
elif [ -n "$PREV" ] && [ "$PREV" != "OK" ]; then
    # The recovery is the part you actually want to know about after acting on an
    # alert, so it is never deduped away.
    NOTIFY="All checks green again"
    TITLE="Screener watchdog: recovered"
fi

if [ -n "$NOTIFY" ]; then
    osascript -e "display notification \"${NOTIFY//\"/}\" with title \"$TITLE\"" 2>/dev/null
    LAST_NOTIFY="$NOW"
fi
printf '%s %s %s\n' "$DIGEST" "$LAST_NOTIFY" "${LAST_WARN_DAY:-}" > "$NOTIFY_STATE"

# Keep the log from growing without bound.
if [ -f "$LOG_FILE" ] && [ "$(wc -l < "$LOG_FILE")" -gt 5000 ]; then
    tail -2000 "$LOG_FILE" > "$LOG_FILE.tmp" && mv "$LOG_FILE.tmp" "$LOG_FILE"
fi

exit 0
