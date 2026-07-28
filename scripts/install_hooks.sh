#!/bin/bash
# Installs the tracked hooks into .git/hooks (which git cannot version itself).
# Re-run after cloning, or any time the hooks change.
set -euo pipefail
REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT"

for hook in scripts/hooks/*; do
    name=$(basename "$hook")
    install -m 755 "$hook" ".git/hooks/$name"
    echo "installed .git/hooks/$name"
done
