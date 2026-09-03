# Security

This repo is public. It runs against a real brokerage account and a paper
trading account, so the rule that matters is simple: **code is public, data is
not.** Everything below exists to keep that line from moving.

## What is deliberately not in this repo

None of the following is tracked, and none of it is in the git history:

| Not tracked | Why |
|---|---|
| `.env`, `.streamlit/secrets.toml` | API keys for Alpaca, Finnhub, OpenAI, Anthropic, and Gmail. |
| `positions.json`, `data/` | Real brokerage holdings — quantities, cost basis, account value. |
| `run_status.json` | Recorded `socket.gethostname()`, which on a campus or home network discloses the network and the machine. |
| `output/` | Run artifacts. Regenerable, and they disclose which strategies were tested. |
| `STRATEGY.md`, `docs/SYSTEM_WRITEUP.md`, `docs/strategy-specs.md`, `docs/strategy-registry.md`, `docs/agent/` | The strategy specification. The engine is public; the edge is not. |
| `docs/handoffs/` | Session handoffs containing personal fit constraints and real holdings. |

Test fixtures use synthetic round numbers, never real positions.

## Defenses

| Layer | Mechanism |
|---|---|
| Don't generate it | `src/fidelity_sync.py` does not capture Account Number, Account Name, or Description. `src/run_status.py` records a `SCREENER_HOST_LABEL` label, not the real hostname. |
| Don't commit it | `scripts/hooks/pre-commit` blocks staged secrets, account-number patterns, private-network hostnames, and `.env` files. Install with `./scripts/install_hooks.sh` — hooks live in `.git/hooks`, which git cannot version. |
| Don't publish it | `scripts/preflight_public.sh` scans the full history, not just `HEAD`. |
| Limit blast radius | No credential is ever required to render a page; pages degrade to empty rather than to a stack trace. |

## Before you push

```bash
./scripts/install_hooks.sh      # once per clone
./scripts/preflight_public.sh   # before any push to a new remote
```

`--no-verify` is for genuine false positives only. The hook flags nothing in a
clean tree, so a hit means something real.

## Running your own copy

Copy `.env.template` to `.env` and fill in your own keys. `SEC_USER_AGENT` must
be your own name and email — the SEC requires a real contact string and rate
limits anonymous traffic. No address is baked into the source.

## Reporting

Found something that leaks? Open an issue without including the sensitive value
itself, and it will be handled.
