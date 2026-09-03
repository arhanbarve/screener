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
ATTEMPT_FILE="$LOG_DIR/trader_attempts"
# TARGET is the trading day these orders are aimed at, filled in from the gate
# below. The evening session decides for tomorrow, so the stamp has to be keyed
# to the target session rather than the calendar date — otherwise an evening run
# would stamp today, and tomorrow morning would decide the same session again and
# double the position.
TARGET="$TODAY"
MAX_ATTEMPTS=5

# The stamp means "done for today, do not run again". It is written AFTER a
# session completes, or after a crashed session that already touched the book.
#
# It used to be written BEFORE the session, so a crash could never retry — the
# reasoning being that partial orders plus a retry is worse than a missed day.
# The reasoning is right; the consequence was not. launchd DarkWakes this Mac to
# run the job and macOS goes straight back to sleep, which kills the session's
# streaming connection ("Connection closed mid-response") on roughly half of
# runs. Stamp-first turned each of those into a permanently skipped trading day:
# 4 of the account's first 7 sessions, silently. Retry is now gated on whether
# the crashed session actually placed anything (see activity-today), which keeps
# the safety property while ending the permanent skip.
# NOTE: there is deliberately no stamp check here. The stamp names the trading
# session already decided, and in the evening that is tomorrow — comparing it to
# today would have let this morning's stamp block tonight's session for the next
# open, which is the whole point of the evening window. The only correct check is
# stamp == target_date, and target_date comes from the gate below.

# Bound the retries. This script is also invoked on a StartInterval poll so it
# picks up as soon as the Mac is next awake, which without a cap could relaunch
# a persistently failing session every few minutes for the whole trading window.
ATTEMPTS=0
if [ -f "$ATTEMPT_FILE" ]; then
    read -r stamped count < "$ATTEMPT_FILE" || true
    if [ "${stamped:-}" = "$TODAY" ]; then
        ATTEMPTS=${count:-0}
    fi
fi
if [ "$ATTEMPTS" -ge "$MAX_ATTEMPTS" ]; then
    echo "=== Gave up for today after $ATTEMPTS attempts ===" >> "$LOG_FILE"
    exit 0
fi

# Cheap local pre-check before the network gate. This script is polled every 15
# minutes so it can start as soon as the Mac wakes, and without this every poll
# would make an Alpaca call and append a "=== Gate:" line — ~96 a day, flooding
# the log the cadence report reads. 10# forces decimal so "0830" is not parsed as
# octal. The real gate below still owns holidays, half-days and window choice.
#
# Two windows: 08:30-15:45 intraday, and 16:15-23:59 for the evening session that
# decides for the next open. The 15:45-16:15 gap is deliberately dead.
NOW_ET=$((10#$(TZ=America/New_York date +%H%M)))
DOW_ET=$(TZ=America/New_York date +%u)
if [ "$DOW_ET" -gt 5 ]; then
    exit 0
fi
if { [ "$NOW_ET" -lt 830 ] || [ "$NOW_ET" -gt 1545 ]; } \
   && { [ "$NOW_ET" -lt 1615 ] || [ "$NOW_ET" -gt 2359 ]; }; then
    exit 0
fi

# Market-clock gate: trading day, 08:30-15:45 ET only
GATE=$("$PY" -m src.trader_cli gate 2>>"$LOG_FILE") || {
    echo "=== Gate check failed (missing keys / network?) ===" >> "$LOG_FILE"
    exit 0
}
# Only log a gate that says yes. Inside a window this script polls every 15
# minutes, and logging every refusal would bury the real entries in the file the
# cadence report reads.
if ! echo "$GATE" | grep -q '"run": true'; then
    exit 0
fi
echo "=== Gate: $GATE ===" >> "$LOG_FILE"

# Re-key the stamp to the session being decided for.
TARGET=$(printf '%s' "$GATE" | sed -n 's/.*"target_date": *"\([0-9-]*\)".*/\1/p')
[ -n "$TARGET" ] || TARGET="$TODAY"
WINDOW=$(printf '%s' "$GATE" | sed -n 's/.*"window": *"\([a-z]*\)".*/\1/p')
echo "=== Window: ${WINDOW:-unknown}, deciding for $TARGET ===" >> "$LOG_FILE"
if [ -f "$STAMP_FILE" ] && [ "$(cat "$STAMP_FILE")" = "$TARGET" ]; then
    echo "=== Session $TARGET already decided, skipping ===" >> "$LOG_FILE"
    exit 0
fi

echo "$TODAY $((ATTEMPTS + 1))" > "$ATTEMPT_FILE"

# Power state is the single most useful diagnostic for this failure: a run that
# begins on battery with the lid shut is the one macOS will sleep through, and
# caffeinate cannot override clamshell sleep.
echo "=== Power: $(pmset -g batt | head -1 | sed 's/^ *//') ===" >> "$LOG_FILE"
echo "=== Trader session started: $(date) (attempt $((ATTEMPTS + 1))/$MAX_ATTEMPTS) ===" >> "$LOG_FILE"

# caffeinate holds off idle, disk and system sleep for the session's lifetime.
# It fixes every case where the machine is on AC or the lid is open; macOS
# enforces clamshell sleep regardless, so a closed lid on battery can still fail.
#
# Write(trading/**) is intentionally absent: Claude Code reports it as unmatched
# by file permission checks, and Edit(trading/**) already covers every
# file-editing tool. Listing both only produced a warning on every run.
SESSION_RC=0
caffeinate -imsu "$CLAUDE" -p "$(cat trading/PROMPT.md)" \
    --model opus \
    --allowedTools "Bash($PY -m src.trader_cli:*),Read,Glob,Grep,WebSearch,WebFetch,Edit(trading/**)" \
    --max-turns 120 \
    >> "$LOG_FILE" 2>&1 || SESSION_RC=$?

echo "=== Trader session finished: $(date) ===" >> "$LOG_FILE"

if [ "$SESSION_RC" -eq 0 ]; then
    echo "$TARGET" > "$STAMP_FILE"
else
    echo "=== claude session exited nonzero (rc=$SESSION_RC) ===" >> "$LOG_FILE"
    # Retry only if the crashed session never touched the book.
    ACTIVITY=$("$PY" -m src.trader_cli activity-today 2>>"$LOG_FILE") || ACTIVITY=""
    if [ -z "$ACTIVITY" ]; then
        # The check itself failed — plausibly the same network fault that killed
        # the session. Stamp conservatively: an unverifiable book must be treated
        # as touched, because retrying blind could duplicate fills. Say so
        # accurately rather than claiming orders were found.
        echo "$TARGET" > "$STAMP_FILE"
        echo "=== Could not verify order activity — stamping conservatively, will NOT retry ===" >> "$LOG_FILE"
    elif echo "$ACTIVITY" | grep -q '"safe_to_retry": true'; then
        echo "=== No orders placed today — leaving unstamped, will retry ===" >> "$LOG_FILE"
    else
        echo "$TARGET" > "$STAMP_FILE"
        echo "=== Orders already placed today — stamping, will NOT retry ===" >> "$LOG_FILE"
        echo "=== Activity: $ACTIVITY ===" >> "$LOG_FILE"

        # PROMPT.md has the session write its rationale BEFORE trading, so a
        # crash between executing and recording leaves a journal still marked
        # PLANNED even though fills happened. The session is dead and cannot
        # correct it, so do it here — a journal that understates what it did is
        # exactly the kind of misleading record the write-first order exists to
        # prevent.
        JOURNAL="trading/journal/$TARGET.md"
        if [ -f "$JOURNAL" ] && grep -q 'Status:\*\* PLANNED' "$JOURNAL"; then
            # Fix the header too, not just the footnote. Someone scanning the top
            # of the file must not read PLANNED on a day that traded.
            sed -i '' 's|\*\*Status:\*\* PLANNED.*|**Status:** EXECUTED — orders filled but never recorded by the session; see the runner correction at the end of this file|' "$JOURNAL"
            {
                echo ""
                echo "> **STATUS CORRECTED BY THE RUNNER.** This entry was left marked PLANNED"
                echo "> because the session crashed between placing its orders and recording"
                echo "> them. Orders *did* fill on this date — \`run_trader.sh\` verified it via"
                echo "> \`trader_cli activity-today\`, which returned:"
                echo "> "
                echo "> \`\`\`"
                echo "> $ACTIVITY" | tr -d '\n' | sed 's/  */ /g'
                echo ""
                echo "> \`\`\`"
                echo "> "
                echo "> The decisions above are the session's own and were written before it"
                echo "> traded, so they are trustworthy. The fills, post-trade book and"
                echo "> scorecard were never written — reconcile against Alpaca order history."
            } >> "$JOURNAL"
            echo "=== Journal still marked PLANNED despite fills — appended runner correction ===" >> "$LOG_FILE"
        fi
    fi
fi

# Refresh the Alpaca snapshot the Paper page renders from. Runs after the
# session so it captures the day's fills. All-or-nothing inside: a fetch failure
# leaves the previous snapshot untouched, so a bad day never overwrites good
# data. Non-fatal — a stale snapshot is better than a failed trader run.
"$PY" -c "from src import paper; paper.refresh()" >> "$LOG_FILE" 2>&1 \
    || echo "=== paper snapshot refresh failed (non-fatal) ===" >> "$LOG_FILE"

# Commit journal + paper snapshot (same auto-commit policy as run_screener.sh).
# The snapshot must be committed for Streamlit Cloud to see it; logs/ is
# gitignored, so the cadence data embedded in the snapshot is the only way the
# cloud page can report whether the trader actually ran.
# The journals stay in this (public) repo deliberately. The Alpaca snapshot no
# longer does — it lists live positions — so it goes to the private data repo
# via publish_data.sh below.
if git diff --quiet HEAD -- trading/ \
   && [ -z "$(git ls-files --others --exclude-standard trading/)" ]; then
    echo "=== No journal changes to commit ===" >> "$LOG_FILE"
else
    git add trading/
    git commit -m "chore(trading): journal $TODAY" >> "$LOG_FILE" 2>&1
    git push >> "$LOG_FILE" 2>&1 || echo "=== git push failed (non-fatal) ===" >> "$LOG_FILE"
fi

# Fail-soft; cannot abort the trader run.
"$SCREENER_DIR/scripts/publish_data.sh" >> "$LOG_FILE" 2>&1

# Email the summary (no-op if GMAIL creds absent). Weekly digest on Fridays.
"$PY" -m src.notify daily >> "$LOG_FILE" 2>&1 || echo "=== daily email failed (non-fatal) ===" >> "$LOG_FILE"
if [ "$(date +%u)" = "5" ]; then
    "$PY" -m src.notify weekly >> "$LOG_FILE" 2>&1 || echo "=== weekly email failed (non-fatal) ===" >> "$LOG_FILE"
fi
