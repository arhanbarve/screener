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

# Email the summary (no-op if GMAIL creds absent). Weekly digest on Fridays.
"$PY" -m src.notify daily >> "$LOG_FILE" 2>&1 || echo "=== daily email failed (non-fatal) ===" >> "$LOG_FILE"
if [ "$(date +%u)" = "5" ]; then
    "$PY" -m src.notify weekly >> "$LOG_FILE" 2>&1 || echo "=== weekly email failed (non-fatal) ===" >> "$LOG_FILE"
fi
