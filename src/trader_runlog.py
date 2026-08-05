"""Paper-trader run cadence.

Answers one question the rest of the system cannot: did the trader actually run
and finish on every market day? The weekly review named session cadence as the
dominant error source, and nothing surfaced it — run_trader.sh writes per-day
logs under logs/, which is gitignored, so the cloud dashboard never sees them
and a missed or crashed day looks identical to a clean one.

Two signals are cross-checked per market day:

  logs/trader_<date>.log   what the runner did (local only)
  trading/journal/<date>.md whether the session produced its journal (tracked)

A run that started, crashed before writing its journal, and left the stamp file
set is the failure mode that matters: run_trader.sh stamps the date *before* the
session so a crash never retries, which is correct for order safety but means a
crashed day is silently skipped forever. Only this module makes that visible.

Pure functions take text and dates so they can be tested without a filesystem;
_collect() is the thin IO wrapper.
"""
from datetime import date, datetime
from pathlib import Path

SCREENER_DIR = Path(__file__).parent.parent
LOG_DIR      = SCREENER_DIR / "logs"
JOURNAL_DIR  = SCREENER_DIR / "trading" / "journal"

# Markers emitted by run_trader.sh. Keep in sync with that script.
_M_STARTED         = "Trader session started:"
_M_FINISHED        = "Trader session finished:"
_M_NONZERO         = "claude session exited nonzero"
_M_GATE_FAILED     = "Gate check failed"
_M_GATE            = "=== Gate:"
_M_ALREADY_RAN     = "Trader already ran today, skipping"
_M_ALREADY_DECIDED = "already decided, skipping"   # evening session pre-empted it
_M_ALREADY_RUNNING = "Trader already running"
_M_NO_JOURNAL      = "No journal changes to commit"

# Status vocabulary, worst first — the order the UI ranks and colors by.
STATUS_ORDER = [
    "no_log",       # market day, runner never wrote anything: launchd didn't fire
    "gate_failed",  # gate crashed (missing keys / network), session never attempted
    "crashed",      # session ran but exited nonzero
    "running",      # started, no finish marker yet
    "gate_blocked", # gate deliberately said no (holiday, outside 08:30-15:45 ET)
    "skipped",      # stamp file already set for the day
    "ok",           # started, finished, clean exit
]

_SEVERITY = {s: i for i, s in enumerate(STATUS_ORDER)}

# Human labels for the UI.
STATUS_LABEL = {
    "no_log":       "NEVER RAN",
    "gate_failed":  "GATE FAILED",
    "crashed":      "CRASHED",
    "running":      "RUNNING",
    "gate_blocked": "GATE SAID NO",
    "skipped":      "SKIPPED",
    "ok":           "COMPLETED",
}


def _sessions(text: str | None) -> list[str]:
    """Split a day's log into one chunk per launched session.

    run_trader.sh appends every attempt to the same per-day file, and since it is
    now retried on a poll, a single day's log can hold a crash followed by a
    success. Judging the file as a whole would find the first attempt's
    "exited nonzero" and report the day as crashed even though it later
    completed — so the outcome must come from the last session, not the file.
    """
    if not text:
        return []
    parts = text.split(_M_STARTED)
    return [_M_STARTED + p for p in parts[1:]]


def classify_log(text: str | None) -> str:
    """Map one day's trader log to a status. None/absent text means no_log.

    The outcome is that of the *final* session launched that day.
    """
    if text is None:
        return "no_log"

    sessions = _sessions(text)
    if sessions:
        last = sessions[-1]
        if _M_NONZERO in last:
            return "crashed"
        return "ok" if _M_FINISHED in last else "running"

    # Nothing ever started. A gate crash outranks the quieter reasons.
    if _M_GATE_FAILED in text:
        return "gate_failed"
    if (_M_ALREADY_RAN in text or _M_ALREADY_RUNNING in text
            or _M_ALREADY_DECIDED in text):
        return "skipped"
    if _M_GATE in text:
        return "gate_blocked"
    return "no_log"


def log_wrote_journal(text: str | None) -> bool:
    """Whether the day's final session committed a journal.

    Scoped to the last session for the same reason as classify_log: a failed
    first attempt leaves "No journal changes to commit" in the file, which would
    otherwise mask a successful retry that did write one.
    """
    if not text:
        return False
    sessions = _sessions(text)
    scope = sessions[-1] if sessions else text
    return _M_NO_JOURNAL not in scope


def _session_times(text: str | None) -> tuple[str | None, str | None]:
    """Extract the raw started/finished timestamp strings, if present."""
    if not text:
        return None, None
    started = finished = None
    for line in text.splitlines():
        if _M_STARTED in line and started is None:
            started = line.split(_M_STARTED, 1)[1].strip().rstrip("=").strip()
        elif _M_FINISHED in line:
            finished = line.split(_M_FINISHED, 1)[1].strip().rstrip("=").strip()
    return started, finished


def attempt_count(text: str | None) -> int:
    """How many sessions were launched that day.

    run_trader.sh appends to one log per day and is polled every 15 minutes, so a
    day that crashed and retried has several start markers. A day that burned all
    five attempts is a different problem from a day that failed once, and only
    this count distinguishes them.
    """
    return (text or "").count(_M_STARTED)


def build_day(
    day: str,
    log_text: str | None,
    journal_exists: bool,
) -> dict:
    """One market day's cadence record."""
    status = classify_log(log_text)
    started, finished = _session_times(log_text)
    return {
        "date": day,
        "status": status,
        "label": STATUS_LABEL[status],
        "severity": _SEVERITY[status],
        "started_at": started,
        "finished_at": finished,
        "attempts": attempt_count(log_text),
        "journal": journal_exists,
        # A run that completed but left no journal is still a partial failure:
        # the decisions it made were never recorded.
        "journal_missing": status in ("ok", "crashed") and not journal_exists,
    }


def summarize(days: list[dict]) -> dict:
    """Roll a list of day records into headline counts for the summary row."""
    counts: dict[str, int] = {}
    for d in days:
        counts[d["status"]] = counts.get(d["status"], 0) + 1
    healthy = counts.get("ok", 0)
    # Days the trader was supposed to work and didn't do so cleanly.
    expected = [d for d in days if d["status"] != "gate_blocked"]
    broken = [d for d in expected if d["status"] not in ("ok", "skipped")]
    return {
        "market_days": len(days),
        "completed": healthy,
        "broken": len(broken),
        "journals_missing": sum(1 for d in days if d["journal_missing"]),
        "counts": counts,
        "worst": min((d["severity"] for d in days), default=len(STATUS_ORDER)),
        "streak_ok": _trailing_ok_streak(days),
    }


def _trailing_ok_streak(days: list[dict]) -> int:
    """Consecutive clean days counting back from the most recent."""
    streak = 0
    for d in sorted(days, key=lambda x: x["date"], reverse=True):
        if d["status"] == "ok":
            streak += 1
        elif d["status"] in ("gate_blocked",):
            continue  # a non-trading gate decision doesn't break a streak
        else:
            break
    return streak


def _read_log(day: str, log_dir: Path) -> str | None:
    p = log_dir / f"trader_{day}.log"
    if not p.exists():
        return None
    try:
        return p.read_text(errors="replace")
    except OSError:
        return None


def collect(
    market_days: list[str],
    log_dir: Path | None = None,
    journal_dir: Path | None = None,
) -> dict:
    """Build the full cadence record for the given market days (ascending)."""
    log_dir = log_dir or LOG_DIR
    journal_dir = journal_dir or JOURNAL_DIR
    days = [
        build_day(
            d,
            _read_log(d, log_dir),
            (journal_dir / f"{d}.md").exists(),
        )
        for d in sorted(market_days)
    ]
    return {"days": days, "summary": summarize(days)}


def last_run(days: list[dict]) -> dict | None:
    """Most recent day the session actually started, whatever the outcome."""
    for d in sorted(days, key=lambda x: x["date"], reverse=True):
        if d["status"] in ("ok", "crashed", "running"):
            return d
    return None
