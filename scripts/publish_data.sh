#!/bin/bash
# Publish private data snapshots to the private companion repo.
#
# The screener repo is public, so the holdings it renders cannot live in it.
# This copies the whitelist below into a clone of arhanbarve/screener-data and
# pushes, which is where the Streamlit Cloud app reads them from at runtime
# (see src/datastore.py).
#
# FAIL-SOFT BY CONTRACT: this always exits 0 and never blocks a trading run.
# The worst failure mode is a stale cloud dashboard, which is strictly better
# than a screener or trader run aborting because GitHub was briefly unreachable.
#
#   ./scripts/publish_data.sh

set -uo pipefail   # deliberately no -e: a failure here must not propagate

SCREENER_DIR="${SCREENER_DIR:-/Users/arhanbarve/Code/screener}"
DATA_REPO_DIR="${DATA_REPO_DIR:-$HOME/Code/screener-data}"
DATA_REPO_URL="${DATA_REPO_URL:-https://github.com/arhanbarve/screener-data.git}"
LOG_FILE="$SCREENER_DIR/logs/publish_data.log"

mkdir -p "$(dirname "$LOG_FILE")"
log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" >> "$LOG_FILE"; }

log "=== publish start ==="

# Clone on first use so a fresh machine needs no manual setup.
if [ ! -d "$DATA_REPO_DIR/.git" ]; then
    log "cloning $DATA_REPO_URL -> $DATA_REPO_DIR"
    if ! git clone --quiet "$DATA_REPO_URL" "$DATA_REPO_DIR" >>"$LOG_FILE" 2>&1; then
        log "clone failed — skipping publish"; exit 0
    fi
fi

# Another machine may have pushed since the last run; rebase rather than merge
# so the data repo keeps a flat, snapshot-per-commit history.
git -C "$DATA_REPO_DIR" pull --quiet --rebase >>"$LOG_FILE" 2>&1 \
    || log "pull failed (continuing; push may be rejected)"

copy() {  # copy <relative-path>
    local rel="$1" src="$SCREENER_DIR/$1" dst="$DATA_REPO_DIR/$1"
    [ -f "$src" ] || return 0
    mkdir -p "$(dirname "$dst")"
    cp "$src" "$dst" && log "  + $rel"
}

copy positions.json
copy run_status.json
copy data/fidelity/positions_data.json
copy data/alpaca/portfolio.json
copy logs/fidelity_sync_status.json

# Screener results: the dashboard lists every date, so publish them all. This is
# ~1.3 MB of CSV/markdown and grows by a few KB per trading day.
mkdir -p "$DATA_REPO_DIR/output"
for f in "$SCREENER_DIR"/output/screen_*.csv "$SCREENER_DIR"/output/screen_*.md; do
    [ -f "$f" ] && cp "$f" "$DATA_REPO_DIR/output/"
done

cd "$DATA_REPO_DIR" || { log "cannot cd to $DATA_REPO_DIR"; exit 0; }

if [ -z "$(git status --porcelain)" ]; then
    log "no changes to publish"
    log "=== publish done ==="
    exit 0
fi

git add -A >>"$LOG_FILE" 2>&1
if git commit -q -m "data: snapshot $(date +%Y-%m-%dT%H:%M)" >>"$LOG_FILE" 2>&1; then
    if git push --quiet >>"$LOG_FILE" 2>&1; then
        log "pushed"
    else
        log "push failed (non-fatal) — will retry next run"
    fi
else
    log "commit failed (non-fatal)"
fi

log "=== publish done ==="
exit 0
