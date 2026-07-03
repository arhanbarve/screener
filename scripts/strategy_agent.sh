#!/bin/bash
# Nightly strategy-queue agent — headless Claude Code run.
# Loaded by ~/Library/LaunchAgents/com.arhanbarve.screener.strategy-agent.plist
set -euo pipefail

REPO="/Users/arhanbarve/Code/screener"
LOG_DIR="$REPO/logs"
mkdir -p "$LOG_DIR"

cd "$REPO"

# Skip if a previous agent run is still going (backtests can run long).
LOCK="$LOG_DIR/strategy_agent.lock"
if [ -f "$LOCK" ] && kill -0 "$(cat "$LOCK")" 2>/dev/null; then
    echo "[$(date)] previous run still active, skipping" >> "$LOG_DIR/strategy_agent.log"
    exit 0
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

/opt/homebrew/bin/claude -p "$(cat docs/agent/STRATEGY-AGENT-PROMPT.md)" \
    --permission-mode acceptEdits \
    --allowedTools "Bash(python3:*)" "Bash(git:*)" "Bash(curl:*)" "Bash(ls:*)" "Bash(mkdir:*)" "Bash(unzip:*)" "Read" "Write" "Edit" "Glob" "Grep" \
    --max-turns 150 \
    >> "$LOG_DIR/strategy_agent.log" 2>&1
