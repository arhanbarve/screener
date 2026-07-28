#!/bin/bash
# Shared secret/PII patterns for the pre-commit hook and the publication preflight.
#
# Sourced, not executed. Defines:
#   scan_text <label>   — reads stdin, prints findings, returns 1 if any found.

# Extended-regex patterns. Keep these conservative: a noisy hook gets bypassed,
# and a bypassed hook protects nothing.
SECRET_PATTERNS=(
    'sk-ant-[A-Za-z0-9_-]{20,}'                 # Anthropic key
    'ghp_[A-Za-z0-9]{36}'                       # GitHub PAT (classic)
    'github_pat_[A-Za-z0-9_]{30,}'              # GitHub PAT (fine-grained)
    'AKIA[0-9A-Z]{16}'                          # AWS access key
    'AIza[0-9A-Za-z_-]{35}'                     # Google API key
    'xox[baprs]-[A-Za-z0-9-]{10,}'              # Slack token
    '-----BEGIN [A-Z ]*PRIVATE KEY'             # any private key
    '\b[A-Z][0-9]{8}\b'                         # Fidelity-style account number
    # Account field with a real value. Values starting with '*' are redaction
    # markers left in rewritten history and are not a leak.
    '"(account|account_number|accountNumber)"[[:space:]]*:[[:space:]]*"[^"*][^"]*"'
)

PATTERN_NAMES=(
    "Anthropic API key"
    "GitHub PAT (classic)"
    "GitHub PAT (fine-grained)"
    "AWS access key"
    "Google API key"
    "Slack token"
    "Private key block"
    "Brokerage account number"
    "JSON account field with a value"
)

scan_text() {
    local label="$1"
    local text
    text=$(cat)
    local found=0
    for i in "${!SECRET_PATTERNS[@]}"; do
        local hits
        hits=$(printf '%s' "$text" | grep -nE "${SECRET_PATTERNS[$i]}" 2>/dev/null | head -3)
        if [ -n "$hits" ]; then
            echo "  ✗ ${PATTERN_NAMES[$i]} in $label"
            printf '%s\n' "$hits" | sed 's/^/      /'
            found=1
        fi
    done
    return $found
}
