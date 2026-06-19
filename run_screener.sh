#!/bin/bash
set -euo pipefail

SCREENER_DIR="/Users/arhanbarve/Code/screener"
LOG_DIR="$SCREENER_DIR/logs"
LOG_FILE="$LOG_DIR/run_$(date +%Y-%m-%d).log"

mkdir -p "$LOG_DIR"

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
