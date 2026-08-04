"""Trader run-cadence classification.

The point of this module is to make a *missed or crashed* trading day visible,
so the tests are mostly about not mistaking failure for success. Log fixtures
are the real marker sequences run_trader.sh emitted on those dates.
"""
import pytest

from src.trader_runlog import (
    STATUS_LABEL,
    STATUS_ORDER,
    build_day,
    classify_log,
    collect,
    last_run,
    log_wrote_journal,
    summarize,
    _session_times,
)

# ── real marker sequences ─────────────────────────────────────────────────────

LOG_OK = """=== Gate: {
  "run": true,
  "reason": "trading day, within 08:30-15:45 ET window"
} ===
=== Trader session started: Fri Jul 31 09:00:12 EDT 2026 ===
All six filled.
=== Trader session finished: Fri Jul 31 12:14:02 EDT 2026 ===
"""

LOG_CRASHED = """=== Gate: {
  "run": true,
  "reason": "trading day, within 08:30-15:45 ET window"
} ===
=== Trader session started: Mon Aug  3 09:00:15 EDT 2026 ===
API Error: Connection closed mid-response.
=== claude session exited nonzero ===
=== Trader session finished: Mon Aug  3 11:49:02 EDT 2026 ===
=== No journal changes to commit ===
=== daily email failed (non-fatal) ===
"""

LOG_GATE_FAILED = "=== Gate check failed (missing keys / network?) ===\n"

LOG_GATE_BLOCKED = """=== Gate: {
  "run": false,
  "reason": "not a trading day"
} ===
"""

LOG_RUNNING = """=== Gate: {
  "run": true
} ===
=== Trader session started: Tue Aug  4 09:08:01 EDT 2026 ===
"""

LOG_SKIPPED = "=== Trader already ran today, skipping ===\n"

LOG_ALREADY_RUNNING = "=== Trader already running (PID 4821), skipping ===\n"


# ── classification ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text,expected", [
    (LOG_OK,              "ok"),
    (LOG_CRASHED,         "crashed"),
    (LOG_GATE_FAILED,     "gate_failed"),
    (LOG_GATE_BLOCKED,    "gate_blocked"),
    (LOG_RUNNING,         "running"),
    (LOG_SKIPPED,         "skipped"),
    (LOG_ALREADY_RUNNING, "skipped"),
    (None,                "no_log"),
    ("",                  "no_log"),
])
def test_classify_log(text, expected):
    assert classify_log(text) == expected


def test_crash_beats_finish_marker():
    """The Aug 3 trap: the script prints 'finished' even after a nonzero exit.

    Reading only for 'finished' would score that day as a clean run, which is
    exactly how four crashed days went unnoticed.
    """
    assert "Trader session finished" in LOG_CRASHED
    assert classify_log(LOG_CRASHED) == "crashed"


def test_every_status_has_a_label():
    assert set(STATUS_LABEL) == set(STATUS_ORDER)


def test_status_order_is_worst_first():
    assert STATUS_ORDER[0] == "no_log"
    assert STATUS_ORDER[-1] == "ok"


# ── journal detection ─────────────────────────────────────────────────────────

def test_journal_marker_absent_means_written():
    assert log_wrote_journal(LOG_OK) is True


def test_no_journal_marker_detected():
    assert log_wrote_journal(LOG_CRASHED) is False


def test_no_log_never_wrote_journal():
    assert log_wrote_journal(None) is False


# ── session timestamps ────────────────────────────────────────────────────────

def test_session_times_extracted():
    started, finished = _session_times(LOG_OK)
    assert started == "Fri Jul 31 09:00:12 EDT 2026"
    assert finished == "Fri Jul 31 12:14:02 EDT 2026"


def test_session_times_partial_when_still_running():
    started, finished = _session_times(LOG_RUNNING)
    assert started == "Tue Aug  4 09:08:01 EDT 2026"
    assert finished is None


def test_session_times_none_without_log():
    assert _session_times(None) == (None, None)


# ── day records ───────────────────────────────────────────────────────────────

def test_completed_run_with_journal_is_clean():
    d = build_day("2026-07-31", LOG_OK, journal_exists=True)
    assert d["status"] == "ok"
    assert d["label"] == "COMPLETED"
    assert d["journal_missing"] is False


def test_completed_run_without_journal_flagged():
    """A session that finished but recorded nothing is a partial failure."""
    d = build_day("2026-07-31", LOG_OK, journal_exists=False)
    assert d["status"] == "ok"
    assert d["journal_missing"] is True


def test_gate_blocked_day_not_flagged_for_missing_journal():
    d = build_day("2026-07-04", LOG_GATE_BLOCKED, journal_exists=False)
    assert d["journal_missing"] is False


def test_no_log_day_not_flagged_for_missing_journal():
    d = build_day("2026-08-05", None, journal_exists=False)
    assert d["status"] == "no_log"
    assert d["journal_missing"] is False


# ── summary ───────────────────────────────────────────────────────────────────

def _real_week():
    """The account's actual first nine market days."""
    return [
        build_day("2026-07-23", LOG_GATE_FAILED, False),
        build_day("2026-07-24", LOG_OK,          True),
        build_day("2026-07-27", LOG_CRASHED,     False),
        build_day("2026-07-28", LOG_GATE_FAILED, False),
        build_day("2026-07-29", LOG_OK,          True),
        build_day("2026-07-30", LOG_CRASHED,     False),
        build_day("2026-07-31", LOG_OK,          True),
        build_day("2026-08-03", LOG_CRASHED,     False),
        build_day("2026-08-04", LOG_CRASHED,     False),
    ]


def test_summary_counts_real_history():
    s = summarize(_real_week())
    assert s["market_days"] == 9
    assert s["completed"] == 3
    assert s["broken"] == 6           # 4 crashed + 2 gate failures
    assert s["journals_missing"] == 4  # the crashed days that ran
    assert s["counts"]["crashed"] == 4
    assert s["counts"]["gate_failed"] == 2


def test_summary_streak_zero_when_latest_crashed():
    assert summarize(_real_week())["streak_ok"] == 0


def test_summary_streak_counts_back_from_latest():
    days = [
        build_day("2026-08-03", LOG_CRASHED, False),
        build_day("2026-08-04", LOG_OK, True),
        build_day("2026-08-05", LOG_OK, True),
    ]
    assert summarize(days)["streak_ok"] == 2


def test_gate_blocked_does_not_break_streak():
    days = [
        build_day("2026-08-03", LOG_OK, True),
        build_day("2026-08-04", LOG_GATE_BLOCKED, False),
        build_day("2026-08-05", LOG_OK, True),
    ]
    assert summarize(days)["streak_ok"] == 2


def test_gate_blocked_excluded_from_broken():
    days = [build_day("2026-07-04", LOG_GATE_BLOCKED, False)]
    assert summarize(days)["broken"] == 0


def test_summary_of_empty_is_safe():
    s = summarize([])
    assert s["market_days"] == 0
    assert s["broken"] == 0
    assert s["streak_ok"] == 0


# ── last run ──────────────────────────────────────────────────────────────────

def test_last_run_returns_most_recent_started_day():
    lr = last_run(_real_week())
    assert lr["date"] == "2026-08-04"
    assert lr["status"] == "crashed"


def test_last_run_ignores_days_that_never_started():
    days = [
        build_day("2026-08-03", LOG_OK, True),
        build_day("2026-08-04", LOG_GATE_FAILED, False),
        build_day("2026-08-05", None, False),
    ]
    assert last_run(days)["date"] == "2026-08-03"


def test_last_run_none_when_never_ran():
    assert last_run([build_day("2026-08-05", None, False)]) is None


# ── collect (filesystem wrapper) ──────────────────────────────────────────────

def test_collect_reads_logs_and_journals(tmp_path):
    logs = tmp_path / "logs"
    journals = tmp_path / "journal"
    logs.mkdir()
    journals.mkdir()
    (logs / "trader_2026-08-03.log").write_text(LOG_CRASHED)
    (logs / "trader_2026-08-04.log").write_text(LOG_OK)
    (journals / "2026-08-04.md").write_text("# journal")

    rec = collect(["2026-08-04", "2026-08-03", "2026-08-05"], logs, journals)
    assert [d["date"] for d in rec["days"]] == ["2026-08-03", "2026-08-04", "2026-08-05"]
    assert [d["status"] for d in rec["days"]] == ["crashed", "ok", "no_log"]
    assert rec["summary"]["completed"] == 1
    assert rec["summary"]["broken"] == 2  # the crash and the never-ran day


def test_collect_missing_dirs_yields_no_log(tmp_path):
    rec = collect(["2026-08-04"], tmp_path / "nope", tmp_path / "nada")
    assert rec["days"][0]["status"] == "no_log"
