#!/bin/bash
# Exercises run_trader.sh's stamp / attempt / retry branches with stubbed
# `claude` and `trader_cli`, so the safety-critical decisions are tested without
# placing a single order.
#
# The property under test: a crashed session is retried ONLY when it never
# touched the book. Getting this wrong either re-runs after partial fills
# (duplicating positions) or permanently skips the day (the original bug).
#
# Run: bash tests/test_run_trader.sh
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PASS=0
FAIL=0

ok()   { printf '  ok   %s\n' "$1"; PASS=$((PASS + 1)); }
bad()  { printf '  FAIL %s\n     %s\n' "$1" "$2"; FAIL=$((FAIL + 1)); }

# ── harness ───────────────────────────────────────────────────────────────────
# Builds a sandbox with a copy of run_trader.sh repointed at it, plus stub
# executables whose behaviour each test controls.
setup() {
    SANDBOX=$(mktemp -d)
    mkdir -p "$SANDBOX/logs" "$SANDBOX/bin" "$SANDBOX/trading"
    echo "prompt" > "$SANDBOX/trading/PROMPT.md"

    # Stub claude: exit code from CLAUDE_RC.
    cat > "$SANDBOX/bin/claude" <<'STUB'
#!/bin/bash
echo "stub claude ran"
exit "${CLAUDE_RC:-0}"
STUB

    # Stub python: answers `gate` and `activity-today` from env, ignores the rest
    # (the paper-snapshot refresh is a -c invocation and must be a no-op here).
    cat > "$SANDBOX/bin/python3" <<'STUB'
#!/bin/bash
for a in "$@"; do
    case "$a" in
        gate)           echo "{\"run\": ${GATE_RUN:-true}, \"window\": \"${GATE_WINDOW:-evening}\", \"target_date\": \"${GATE_TARGET:-2099-01-01}\", \"reason\": \"stub\"}"; exit 0 ;;
        activity-today) echo "{\"count\": ${ACT_COUNT:-0}, \"safe_to_retry\": ${ACT_SAFE:-true}}"; exit 0 ;;
    esac
done
exit 0
STUB

    # caffeinate must not be a real sleep assertion in tests; just exec through.
    cat > "$SANDBOX/bin/caffeinate" <<'STUB'
#!/bin/bash
while [[ "${1:-}" == -* ]]; do shift; done
exec "$@"
STUB

    # git is called for the journal commit; make it inert.
    cat > "$SANDBOX/bin/git" <<'STUB'
#!/bin/bash
case "${1:-}" in
    diff) exit 0 ;;              # "no changes" so the commit branch is skipped
    ls-files) exit 0 ;;
esac
exit 0
STUB

    chmod +x "$SANDBOX/bin"/*

    # The two clock reads become env-injectable so the suite runs at any hour.
    # The comparison itself is left untouched, so the real condition is what
    # gets exercised — only its inputs are controlled.
    sed -e "s|^SCREENER_DIR=.*|SCREENER_DIR=\"$SANDBOX\"|" \
        -e "s|^PY=.*|PY=\"$SANDBOX/bin/python3\"|" \
        -e "s|^CLAUDE=.*|CLAUDE=\"$SANDBOX/bin/claude\"|" \
        -e "s|^LOCKFILE=.*|LOCKFILE=\"$SANDBOX/trader.lock\"|" \
        -e "s|^NOW_ET=.*|NOW_ET=\$((10#\${TEST_NOW_ET:-1000}))|" \
        -e "s|^DOW_ET=.*|DOW_ET=\${TEST_DOW_ET:-3}|" \
        "$REPO/run_trader.sh" > "$SANDBOX/run_trader.sh"
    chmod +x "$SANDBOX/run_trader.sh"

    # Fail loudly if the injection stopped matching the script.
    grep -q 'TEST_NOW_ET' "$SANDBOX/run_trader.sh" || {
        echo "HARNESS BROKEN: could not inject clock into run_trader.sh" >&2
        exit 2
    }
}

teardown() { rm -rf "$SANDBOX"; }

# Every invocation runs at an injected Wednesday 10:00 ET unless overridden.
# env, not a bare "VAR=x" prefix: "$@" expands after parsing, so bash would treat
# the first expanded word as the command name instead of an assignment.
trader() {
    ( cd "$SANDBOX" && env PATH="$SANDBOX/bin:$PATH" "$@" \
      bash "$SANDBOX/run_trader.sh" >/dev/null 2>&1 )
}

TODAY=$(date +%Y-%m-%d)
TARGET="2099-01-01"   # what the stubbed gate reports as target_date
STAMP() { cat "$SANDBOX/logs/trader_last_run" 2>/dev/null || echo "<none>"; }
LOGTXT() { cat "$SANDBOX/logs/trader_$TODAY.log" 2>/dev/null || echo ""; }

echo "run_trader.sh branch tests"

# ── 1. clean success stamps the day ───────────────────────────────────────────
setup
trader CLAUDE_RC=0
[ "$(STAMP)" = "$TARGET" ] \
    && ok "successful session stamps the target session" \
    || bad "successful session stamps the target session" "stamp=$(STAMP) want=$TARGET"
teardown

# ── 2. crash with an untouched book does NOT stamp ────────────────────────────
setup
trader CLAUDE_RC=1 ACT_SAFE=true ACT_COUNT=0
if [ "$(STAMP)" = "<none>" ] && LOGTXT | grep -q "will retry"; then
    ok "crash with no orders leaves the day unstamped (retryable)"
else
    bad "crash with no orders leaves the day unstamped" "stamp=$(STAMP)"
fi
teardown

# ── 3. crash AFTER placing orders stamps and refuses retry ────────────────────
setup
trader CLAUDE_RC=1 ACT_SAFE=false ACT_COUNT=3
if [ "$(STAMP)" = "$TARGET" ] && LOGTXT | grep -q "will NOT retry"; then
    ok "crash after placing orders stamps the day (no double-trade)"
else
    bad "crash after placing orders stamps the day" "stamp=$(STAMP)"
fi
teardown

# ── 3b. crash where the activity check ITSELF fails stamps conservatively ─────
# An unverifiable book must be treated as touched: retrying blind could duplicate
# fills. The log must say that, not claim orders were found.
setup
cat > "$SANDBOX/bin/python3" <<'STUB'
#!/bin/bash
for a in "$@"; do
    case "$a" in
        gate)           echo "{\"run\": true, \"window\": \"evening\", \"target_date\": \"2099-01-01\", \"reason\": \"stub\"}"; exit 0 ;;
        activity-today) exit 1 ;;   # the check itself fails
    esac
done
exit 0
STUB
chmod +x "$SANDBOX/bin/python3"
trader CLAUDE_RC=1
if [ "$(STAMP)" = "$TARGET" ] && LOGTXT | grep -q "Could not verify order activity"; then
    ok "unverifiable book stamps conservatively and says so"
else
    bad "unverifiable book stamps conservatively" "stamp=$(STAMP) log: $(LOGTXT | tr '\n' '|' | cut -c1-200)"
fi
if LOGTXT | grep -q "Orders already placed today"; then
    bad "unverifiable book must not claim orders were found" "misleading log line present"
else
    ok "unverifiable book does not claim orders were found"
fi
teardown

# ── 3c. crash after fills with a PLANNED journal gets corrected ───────────────
# PROMPT.md has the session journal before trading, so a crash between executing
# and recording leaves a journal marked PLANNED when fills happened. The dead
# session cannot fix that; the runner must.
setup
mkdir -p "$SANDBOX/trading/journal"
cat > "$SANDBOX/trading/journal/$TARGET.md" <<'J'
# Trading Journal — today
**Status:** PLANNED — no orders placed yet
## Decisions
Buy MYE because reasons.
J
trader CLAUDE_RC=1 ACT_SAFE=false ACT_COUNT=3
if grep -q "STATUS CORRECTED BY THE RUNNER" "$SANDBOX/trading/journal/$TARGET.md"; then
    ok "PLANNED journal is corrected when fills happened"
else
    bad "PLANNED journal is corrected when fills happened" \
        "journal: $(tr '\n' '|' < "$SANDBOX/trading/journal/$TARGET.md" | cut -c1-200)"
fi
# The session's own reasoning must survive untouched.
if grep -q "Buy MYE because reasons" "$SANDBOX/trading/journal/$TARGET.md"; then
    ok "runner correction preserves the session's reasoning"
else
    bad "runner correction preserves the session's reasoning" "original text gone"
fi
# The header must not still claim PLANNED on a day that traded.
if grep -q 'Status:\*\* EXECUTED' "$SANDBOX/trading/journal/$TARGET.md" \
   && ! grep -q 'Status:\*\* PLANNED' "$SANDBOX/trading/journal/$TARGET.md"; then
    ok "status header flipped to EXECUTED"
else
    bad "status header flipped to EXECUTED" \
        "header: $(grep -m1 'Status' "$SANDBOX/trading/journal/$TARGET.md")"
fi
teardown

# A clean crash with no fills must NOT touch the journal — it is honestly PLANNED.
setup
mkdir -p "$SANDBOX/trading/journal"
printf '# Trading Journal\n**Status:** PLANNED — no orders placed yet\n' \
    > "$SANDBOX/trading/journal/$TARGET.md"
before=$(cat "$SANDBOX/trading/journal/$TARGET.md")
trader CLAUDE_RC=1 ACT_SAFE=true ACT_COUNT=0
if [ "$(cat "$SANDBOX/trading/journal/$TARGET.md")" = "$before" ]; then
    ok "PLANNED journal untouched when no orders filled"
else
    bad "PLANNED journal untouched when no orders filled" "journal was modified"
fi
teardown

# An EXECUTED journal must never be second-guessed by the runner.
setup
mkdir -p "$SANDBOX/trading/journal"
printf '# Trading Journal\n**Status:** EXECUTED\n## Orders placed\nall filled\n' \
    > "$SANDBOX/trading/journal/$TARGET.md"
trader CLAUDE_RC=1 ACT_SAFE=false ACT_COUNT=3
if grep -q "STATUS CORRECTED" "$SANDBOX/trading/journal/$TARGET.md"; then
    bad "EXECUTED journal left alone" "runner appended a correction anyway"
else
    ok "EXECUTED journal left alone"
fi
teardown

# ── 4. an already-stamped day is skipped ──────────────────────────────────────
setup
echo "$TARGET" > "$SANDBOX/logs/trader_last_run"
trader CLAUDE_RC=0
if LOGTXT | grep -q "already decided, skipping" && ! LOGTXT | grep -q "session started"; then
    ok "already-decided target skips without starting a session"
else
    bad "already-decided target skips" "log: $(LOGTXT | tr '\n' '|' | cut -c1-160)"
fi
teardown

# ── 5. attempts are capped ────────────────────────────────────────────────────
setup
echo "$TODAY 5" > "$SANDBOX/logs/trader_attempts"
trader CLAUDE_RC=0
if LOGTXT | grep -q "Gave up for today" && ! LOGTXT | grep -q "session started"; then
    ok "attempt cap stops further sessions"
else
    bad "attempt cap stops further sessions" "log: $(LOGTXT | tr '\n' '|' | cut -c1-160)"
fi
teardown

# ── 6. yesterday's attempt count does not carry over ──────────────────────────
setup
echo "2020-01-01 5" > "$SANDBOX/logs/trader_attempts"
trader CLAUDE_RC=0
if LOGTXT | grep -q "session started" && ! LOGTXT | grep -q "Gave up"; then
    ok "stale attempt count is ignored"
else
    bad "stale attempt count is ignored" "log: $(LOGTXT | tr '\n' '|' | cut -c1-160)"
fi
teardown

# ── 7. attempts increment across retries ──────────────────────────────────────
setup
for _ in 1 2; do
    trader CLAUDE_RC=1 ACT_SAFE=true
done
read -r _ n < "$SANDBOX/logs/trader_attempts"
[ "${n:-0}" -eq 2 ] \
    && ok "attempt counter increments per session" \
    || bad "attempt counter increments per session" "count=${n:-unset} want=2"
teardown

# ── 8. gate refusal starts no session ─────────────────────────────────────────
setup
trader GATE_RUN=false
if ! LOGTXT | grep -q "session started"; then
    ok "gate refusal starts no session"
else
    bad "gate refusal starts no session" "log: $(LOGTXT | tr '\n' '|' | cut -c1-160)"
fi
teardown

# ── 9. the session runs under caffeinate ──────────────────────────────────────
setup
cat > "$SANDBOX/bin/caffeinate" <<'STUB'
#!/bin/bash
echo "CAFFEINATE_INVOKED $*" >> "$CAFF_LOG"
while [[ "${1:-}" == -* ]]; do shift; done
exec "$@"
STUB
chmod +x "$SANDBOX/bin/caffeinate"
trader CLAUDE_RC=0 CAFF_LOG="$SANDBOX/caff.log"
if grep -q "CAFFEINATE_INVOKED -imsu" "$SANDBOX/caff.log" 2>/dev/null; then
    ok "session runs under caffeinate -imsu"
else
    bad "session runs under caffeinate -imsu" "caff.log: $(cat "$SANDBOX/caff.log" 2>/dev/null)"
fi
teardown

# ── 10-13. the local pre-check (polling means this runs ~96x/day) ─────────────
# Without it, every 15-minute poll would make an Alpaca call and append a
# "=== Gate:" line, flooding the log the cadence report reads.

check_precheck() {  # desc, TEST_NOW_ET, TEST_DOW_ET, expect_run(yes|no)
    setup
    trader CLAUDE_RC=0 TEST_NOW_ET="$2" TEST_DOW_ET="$3"
    local started="no"
    LOGTXT | grep -q "session started" && started="yes"
    if [ "$started" = "$4" ]; then
        ok "$1"
    else
        bad "$1" "started=$started want=$4"
    fi
    teardown
}

check_precheck "08:29 ET is before the window"     0829 3 no
check_precheck "08:30 ET opens the window"          0830 3 yes
check_precheck "15:45 ET is still inside"           1545 3 yes
check_precheck "15:46 ET is past the cutoff"        1546 3 no
check_precheck "Saturday never runs"                1000 6 no
check_precheck "Sunday never runs"                  1000 7 no
check_precheck "Friday runs"                        1000 5 yes

# Leading-zero times must be read as decimal, not octal — "0830" is not a valid
# octal literal and would abort the script without the 10# prefix.
setup
trader CLAUDE_RC=0 TEST_NOW_ET=0900 TEST_DOW_ET=3
if LOGTXT | grep -q "session started"; then
    ok "leading-zero ET time parses as decimal"
else
    bad "leading-zero ET time parses as decimal" "log: $(LOGTXT | tr '\n' '|' | cut -c1-160)"
fi
teardown

# ── the removed Write() rule stays removed ────────────────────────────────────
# Grep the allowedTools line only — a comment explains why the rule is absent,
# and matching that comment would make this test pass for the wrong reason.
if grep -E '^\s+--allowedTools' "$REPO/run_trader.sh" | grep -q 'Write(trading'; then
    bad "Write(trading/**) stays out of allowedTools" "still present in allowedTools"
elif grep -E '^\s+--allowedTools' "$REPO/run_trader.sh" | grep -q 'Edit(trading'; then
    ok "allowedTools has Edit(trading/**) and not Write(trading/**)"
else
    bad "allowedTools has Edit(trading/**)" "Edit rule missing — journal writes would be denied"
fi

echo
echo "passed $PASS, failed $FAIL"
[ "$FAIL" -eq 0 ]

# ── the evening session must not be blocked by today's stamp ───────────────────
# This is the bug that would have stopped tonight from deciding tomorrow: the
# stamp names the target session, so comparing it to the calendar date let a
# crashed morning run block the evening session for the next open.

setup
echo "$(date +%Y-%m-%d)" > "$SANDBOX/logs/trader_last_run"   # today, not the target
trader CLAUDE_RC=0 GATE_WINDOW=evening GATE_TARGET=2099-01-01
if LOGTXT | grep -q "session started"; then
    ok "today's stamp does not block an evening session for the next open"
else
    bad "today's stamp does not block an evening session" \
        "log: $(LOGTXT | tr '\n' '|' | cut -c1-200)"
fi
teardown

setup
trader CLAUDE_RC=0 GATE_WINDOW=evening GATE_TARGET=2099-01-01
[ "$(STAMP)" = "2099-01-01" ] \
    && ok "evening session stamps the next session, not today" \
    || bad "evening session stamps the next session" "stamp=$(STAMP)"
teardown

setup
trader GATE_RUN=false
if ! LOGTXT | grep -q "=== Gate:"; then
    ok "a refusing gate is not logged (polls every 15min inside a window)"
else
    bad "a refusing gate is not logged" "gate line present"
fi
teardown

# A gate response without target_date must fall back to the calendar date rather
# than stamping an empty string — proved itself when a stub omitted the field.
setup
cat > "$SANDBOX/bin/python3" <<'STUB'
#!/bin/bash
for a in "$@"; do
    case "$a" in
        gate)           echo "{\"run\": true, \"reason\": \"no target_date\"}"; exit 0 ;;
        activity-today) echo "{\"count\": 0, \"safe_to_retry\": true}"; exit 0 ;;
    esac
done
exit 0
STUB
chmod +x "$SANDBOX/bin/python3"
trader CLAUDE_RC=0
[ "$(STAMP)" = "$TODAY" ] \
    && ok "missing target_date falls back to the calendar date" \
    || bad "missing target_date falls back to the calendar date" "stamp=$(STAMP)"
teardown
