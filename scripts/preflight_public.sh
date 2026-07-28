#!/bin/bash
# Run this BEFORE ever making this repo public, or before pushing it anywhere
# new. Scans the entire history — not just the current tree — because making a
# repo public exposes every commit ever pushed to it.
#
#   ./scripts/preflight_public.sh
#
# Exit 0 = history looks publishable. Exit 1 = do not publish.

set -uo pipefail
REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT"
source scripts/secret_scan.sh

echo "── scanning $(git rev-list --all | wc -l | tr -d ' ') commits ──"
FAILED=0

# 1. Sensitive content in any blob, in any commit, on any ref.
echo
echo "[1/3] history blob content"
for pat_idx in "${!SECRET_PATTERNS[@]}"; do
    hits=$(git grep -IlE "${SECRET_PATTERNS[$pat_idx]}" $(git rev-list --all) -- 2>/dev/null | head -5)
    if [ -n "$hits" ]; then
        echo "  ✗ ${PATTERN_NAMES[$pat_idx]}:"
        printf '%s\n' "$hits" | sed 's/^/      /'
        FAILED=1
    fi
done
[ "$FAILED" -eq 0 ] && echo "  ✓ no secret patterns in any commit"

# 2. Data files that should never have been tracked at all.
echo
echo "[2/3] data files ever added to history"
DATA_HITS=$(git log --all --pretty=format: --name-only --diff-filter=A \
    | sort -u | grep -E '(positions_data|positions\.json|\.env|fidelity/)' || true)
if [ -n "$DATA_HITS" ]; then
    echo "  ⚠ these paths exist in history and will be public:"
    printf '%s\n' "$DATA_HITS" | sed 's/^/      /'
    FAILED=1
else
    echo "  ✓ no brokerage/env data paths in history"
fi

# 3. Author identity leaks in commit metadata.
echo
echo "[3/3] commit metadata"
EMAILS=$(git log --all --format='%ae' | sort -u)
echo "  author emails that will be public:"
printf '%s\n' "$EMAILS" | sed 's/^/      /'

echo
if [ "$FAILED" -ne 0 ]; then
    echo "RESULT: DO NOT PUBLISH — findings above are in history, not just HEAD."
    echo "Publishing this repo exposes every one of them. See SECURITY.md for"
    echo "the clean-export path that does not require rewriting history."
    exit 1
fi
echo "RESULT: history looks publishable. Re-read SECURITY.md before you flip it."
exit 0
