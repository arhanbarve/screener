"""Read private data that is deliberately absent from this public repo.

The screener code is public; the holdings it renders are not. Locally every
file is on disk exactly as before. On Streamlit Cloud the repo carries no
private data at all, so reads fall back to a private companion repo
(`screener-data`) over the GitHub Contents API.

Local disk always wins. That keeps `python -m src.run`, the cron jobs and the
test suite on the identical code path they used before this module existed —
the remote branch only engages where the file genuinely is not there.

Writes are NOT routed through here. Everything that produces data
(`fidelity_sync`, `positions.save_positions`, the screener output writer) keeps
writing to local disk; `scripts/publish_data.sh` is what moves it off-machine.

Env:
  DATA_REPO_TOKEN  GitHub PAT with read access to the private data repo.
                   Absent locally, and absent on Cloud until it is set in
                   Streamlit secrets — in which case remote reads return None
                   and every caller degrades to its existing empty state.
  DATA_REPO        owner/name of that repo. Defaults to arhanbarve/screener-data.
"""
import fnmatch
import os
import time
from pathlib import Path

import requests

SCREENER_DIR = Path(__file__).parent.parent
DEFAULT_REPO = "arhanbarve/screener-data"
API_ROOT = "https://api.github.com"
TIMEOUT = 10

# The dashboard is read at human pace, and the data behind it changes at most
# once per trading session. A short TTL keeps a rerun from spending a GitHub
# API call per widget interaction without ever showing genuinely stale data.
CACHE_TTL_SECS = 300

_cache: dict[str, tuple[float, object]] = {}


def _cached(key: str, produce):
    """Memoise `produce()` under `key` for CACHE_TTL_SECS."""
    hit = _cache.get(key)
    if hit is not None and (time.monotonic() - hit[0]) < CACHE_TTL_SECS:
        return hit[1]
    value = produce()
    _cache[key] = (time.monotonic(), value)
    return value


def clear_cache() -> None:
    _cache.clear()


def _repo() -> str:
    return os.environ.get("DATA_REPO") or DEFAULT_REPO


def _token() -> str | None:
    return os.environ.get("DATA_REPO_TOKEN") or None


def _request(url: str, accept: str):
    token = _token()
    if not token:
        return None
    try:
        resp = requests.get(
            url,
            headers={"Authorization": f"Bearer {token}", "Accept": accept},
            timeout=TIMEOUT,
        )
    except requests.RequestException:
        return None  # a dashboard must degrade, never traceback
    if resp.status_code != 200:
        return None
    return resp


# ── public API ────────────────────────────────────────────────────────────────

def read_text(relpath: str) -> str | None:
    """File contents as text, or None if it exists in neither place."""
    local = SCREENER_DIR / relpath
    if local.exists():
        return local.read_text(errors="replace")
    return _cached(f"text:{relpath}", lambda: _remote_text(relpath))


def _remote_text(relpath: str) -> str | None:
    resp = _request(
        f"{API_ROOT}/repos/{_repo()}/contents/{relpath}",
        "application/vnd.github.raw",
    )
    return resp.text if resp is not None else None


def list_names(dirpath: str, pattern: str = "*") -> list[str]:
    """Base names in `dirpath` matching `pattern`, newest-sortable, sorted.

    Returns names only, so callers pair it with read_text(f"{dirpath}/{name}")
    rather than assuming a filesystem path exists.
    """
    local = SCREENER_DIR / dirpath
    if local.is_dir():
        return sorted(p.name for p in local.glob(pattern) if p.is_file())
    return _cached(f"list:{dirpath}:{pattern}", lambda: _remote_names(dirpath, pattern))


def _remote_names(dirpath: str, pattern: str) -> list[str]:
    resp = _request(
        f"{API_ROOT}/repos/{_repo()}/contents/{dirpath}",
        "application/vnd.github+json",
    )
    if resp is None:
        return []
    try:
        entries = resp.json()
    except ValueError:
        return []
    if not isinstance(entries, list):
        return []
    return sorted(
        e["name"] for e in entries
        if e.get("type") == "file" and fnmatch.fnmatch(e.get("name", ""), pattern)
    )
