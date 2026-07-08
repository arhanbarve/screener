#!/bin/bash
set -euo pipefail

SCREENER_DIR="/Users/arhanbarve/Code/screener"
LOG_DIR="$SCREENER_DIR/logs"
LOG_FILE="$LOG_DIR/run_$(date +%Y-%m-%d).log"
LOCKFILE="/tmp/screener_run.lock"

mkdir -p "$LOG_DIR"

# Belt-and-suspenders fd raise (plist SoftResourceLimits is the primary fix)
ulimit -n 4096 2>/dev/null || true

# Prevent overlapping manual + cron runs (concurrent SQLite writes cause lock contention)
if [ -f "$LOCKFILE" ]; then
    existing_pid=$(cat "$LOCKFILE" 2>/dev/null || echo "")
    if [ -n "$existing_pid" ] && kill -0 "$existing_pid" 2>/dev/null; then
        echo "=== Screener already running (PID $existing_pid), skipping ===" >> "$LOG_FILE"
        exit 0
    fi
    rm -f "$LOCKFILE"
fi
echo $$ > "$LOCKFILE"
trap "rm -f '$LOCKFILE'" EXIT

cd "$SCREENER_DIR"

# Load API keys
if [ -f "$SCREENER_DIR/.env" ]; then
    set -a
    source "$SCREENER_DIR/.env"
    set +a
fi

echo "=== Screener run started: $(date) ===" >> "$LOG_FILE"
/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m src.run >> "$LOG_FILE" 2>&1
echo "=== Screener run finished: $(date) ===" >> "$LOG_FILE"

# Commit and push today's output so watchdog/remote monitors can see results
cd "$SCREENER_DIR"
if git diff --quiet HEAD -- output/ && [ -z "$(git ls-files --others --exclude-standard output/)" ]; then
    echo "=== No new output to commit ===" >> "$LOG_FILE"
else
    TODAY=$(date +%Y-%m-%d)
    git add output/
    git commit -m "chore(output): screener results $TODAY" >> "$LOG_FILE" 2>&1
    git push >> "$LOG_FILE" 2>&1 && echo "=== Output committed and pushed ===" >> "$LOG_FILE" || echo "=== git push failed (non-fatal) ===" >> "$LOG_FILE"
fi
