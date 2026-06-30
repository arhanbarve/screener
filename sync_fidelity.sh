#!/bin/bash
# Fidelity positions sync — called by launchd on every login.
# Daily throttle: runs at most once per calendar day.
set -euo pipefail

SCREENER_DIR="/Users/arhanbarve/Code/screener"
LOG_FILE="$SCREENER_DIR/logs/fidelity_sync.log"
LOCK_FILE="/tmp/fidelity_sync_last_run"

mkdir -p "$SCREENER_DIR/logs"

TODAY=$(date +%Y-%m-%d)

# Already ran today — skip
if [ -f "$LOCK_FILE" ] && [ "$(cat "$LOCK_FILE" 2>/dev/null)" = "$TODAY" ]; then
    echo "$(date): already ran today, skipping" >> "$LOG_FILE"
    exit 0
fi

echo "$TODAY" > "$LOCK_FILE"

# Load API keys (needed in case positions page calls yfinance)
if [ -f "$SCREENER_DIR/.env" ]; then
    set -a
    source "$SCREENER_DIR/.env"
    set +a
fi

echo "=== Fidelity sync started: $(date) ===" >> "$LOG_FILE"

cd "$SCREENER_DIR"
/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m src.fidelity_sync >> "$LOG_FILE" 2>&1 \
    || echo "=== Fidelity sync FAILED (exit $?) ===" >> "$LOG_FILE"

echo "=== Fidelity sync finished: $(date) ===" >> "$LOG_FILE"
