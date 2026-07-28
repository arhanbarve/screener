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

echo "=== Screener run started: $(date) ===" >> "$LOG_FILE"
STARTED_ISO=$(date +%Y-%m-%dT%H:%M:%S)
STARTED_EPOCH=$(date +%s)

# Capture the exit code instead of letting `set -e` abort: a failed run still has
# to write run_status.json and fire the alert email.
set +e
/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m src.run >> "$LOG_FILE" 2>&1
RUN_RC=$?
set -e

echo "=== Screener run finished: $(date) ===" >> "$LOG_FILE"

DURATION=$(( $(date +%s) - STARTED_EPOCH ))
/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m src.run_status \
    --log "$LOG_FILE" --rc "$RUN_RC" --started "$STARTED_ISO" --duration "$DURATION" \
    >> "$LOG_FILE" 2>&1 || echo "=== run_status failed (non-fatal) ===" >> "$LOG_FILE"

# Standing exit plan: evaluate open positions against today's close and email
# any SELL/TRIM verdict. Runs as its own process so a screener failure above
# can't cost the user a sell instruction, and vice versa.
/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m src.exit_plan \
    >> "$LOG_FILE" 2>&1 || echo "=== exit-plan eval failed (non-fatal) ===" >> "$LOG_FILE"

# Commit and push today's output so watchdog/remote monitors can see results.
# run_status.json is committed on failure too — that's how the cloud dashboard
# reports a dead run instead of just going stale.
cd "$SCREENER_DIR"
COMMIT_PATHS="output/ positions.json data/fidelity/positions_data.json run_status.json"
if git diff --quiet HEAD -- $COMMIT_PATHS && [ -z "$(git ls-files --others --exclude-standard $COMMIT_PATHS)" ]; then
    echo "=== No new output to commit ===" >> "$LOG_FILE"
else
    TODAY=$(date +%Y-%m-%d)
    git add $COMMIT_PATHS
    git commit -m "chore(output): screener results $TODAY" >> "$LOG_FILE" 2>&1
    git push >> "$LOG_FILE" 2>&1 && echo "=== Output committed and pushed ===" >> "$LOG_FILE" || echo "=== git push failed (non-fatal) ===" >> "$LOG_FILE"
fi

# Surface the real result to launchd (`launchctl list` last-exit-status).
exit $RUN_RC
