"""Health floors, stats file, and the machine-readable screen_latest.json."""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import run as run_mod
from src.output import write_latest_json
from src.run_status import build_status, load_stats

CFG = {"health": {"min_adv_survivors": 800, "min_cap_survivors": 500, "min_ranked": 150}}


def test_evaluate_health_passes_when_all_floors_met():
    ok, reasons = run_mod.evaluate_health({"adv_survivors": 2400, "cap_survivors": 1900, "ranked": 600}, CFG)
    assert ok and reasons == []


def test_evaluate_health_reports_every_breach():
    ok, reasons = run_mod.evaluate_health({"adv_survivors": 53, "cap_survivors": 53, "ranked": 38}, CFG)
    assert not ok
    assert len(reasons) == 3
    assert "adv_survivors=53 below floor 800" in reasons


def test_evaluate_health_only_checks_stages_that_ran():
    """Called after stage 2 the ranked count does not exist yet — that is not a breach."""
    ok, reasons = run_mod.evaluate_health({"adv_survivors": 2400}, CFG)
    assert ok and reasons == []


def test_evaluate_health_without_config_never_fails():
    ok, reasons = run_mod.evaluate_health({"adv_survivors": 1}, {})
    assert ok


def test_gate_raises_and_writes_failed_screen(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_mod, "STATS_PATH", tmp_path / "data" / "last_run_stats.json")
    monkeypatch.setattr(run_mod, "OUTPUT_DIR", str(tmp_path / "output"))
    stats = {"date": "2026-09-04", "adv_survivors": 53}
    with pytest.raises(run_mod.PipelineHealthError) as exc:
        run_mod._gate(stats, CFG, allow_degraded=False, today="2026-09-04")
    assert "adv_survivors=53" in str(exc.value)
    written = json.loads((tmp_path / "data" / "last_run_stats.json").read_text())
    assert written["ok"] is False and written["reasons"]
    latest = json.loads((tmp_path / "output" / "screen_latest.json").read_text())
    assert latest["health"]["ok"] is False and latest["rows"] == []
    # No dated CSV must exist: a failed run publishes no ranking.
    assert not list((tmp_path / "output").glob("screen_2026-*.csv"))


def test_gate_allow_degraded_continues_and_flags(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_mod, "STATS_PATH", tmp_path / "data" / "last_run_stats.json")
    monkeypatch.setattr(run_mod, "OUTPUT_DIR", str(tmp_path / "output"))
    stats = {"date": "2026-09-04", "adv_survivors": 53}
    run_mod._gate(stats, CFG, allow_degraded=True, today="2026-09-04")
    assert stats["ok"] is False and stats["degraded"] is True


def test_gate_ok_leaves_no_files(tmp_path, monkeypatch):
    monkeypatch.setattr(run_mod, "STATS_PATH", tmp_path / "data" / "last_run_stats.json")
    monkeypatch.setattr(run_mod, "OUTPUT_DIR", str(tmp_path / "output"))
    stats = {"date": "2026-09-04", "adv_survivors": 2400}
    run_mod._gate(stats, CFG, allow_degraded=False, today="2026-09-04")
    assert stats["ok"] is True
    assert not (tmp_path / "output").exists()


# ── screen_latest.json ────────────────────────────────────────────────────────

def test_write_latest_json_rows_ranks_and_nan_to_null(tmp_path):
    df = pd.DataFrame({
        "ticker": ["AAA", "BBB"], "name": ["Alpha, Inc.", "Beta"], "sector": ["Tech", "Health"],
        "composite": [1.2, 0.9], "price": [10.0, np.nan], "atr_14": [0.5, 0.7],
        "entry": ["OK", "WAIT"], "conviction": np.array([7, 4], dtype="int64"),
        "stoch_cross": [True, False],
    })
    health = {"ok": True, "adv_survivors": 2400, "reasons": []}
    path = write_latest_json(df, str(tmp_path), "2026-09-04", health=health,
                             regime={"regime": "NORMAL", "scale_factor": 1.0, "reason": "all clear"},
                             ranking_tail=[{"rank": 1, "ticker": "AAA", "composite": 1.2}])
    blob = json.loads(Path(path).read_text())
    assert blob["date"] == "2026-09-04" and blob["health"]["ok"] is True
    assert blob["rows"][0]["rank"] == 1 and blob["rows"][1]["rank"] == 2
    assert blob["rows"][1]["price"] is None            # NaN -> null, never the string "nan"
    assert blob["rows"][0]["conviction"] == 7          # numpy int -> int
    assert blob["rows"][0]["stoch_cross"] is True
    assert blob["rows"][0]["name"] == "Alpha, Inc."    # commas are not a parsing problem in JSON
    assert blob["regime"]["regime"] == "NORMAL"
    assert blob["ranking_tail"][0]["ticker"] == "AAA"
    assert "rank" in blob["columns"]


def test_write_latest_json_empty_frame(tmp_path):
    path = write_latest_json(pd.DataFrame(), str(tmp_path), "2026-09-04",
                             health={"ok": False, "reasons": ["x"]},
                             regime={"regime": "UNKNOWN", "scale_factor": 1.0}, ranking_tail=[])
    blob = json.loads(Path(path).read_text())
    assert blob["rows"] == [] and blob["health"]["ok"] is False


def test_write_latest_json_overwrites_atomically(tmp_path):
    for i in range(2):
        write_latest_json(pd.DataFrame({"ticker": [f"T{i}"], "composite": [1.0]}), str(tmp_path),
                          "2026-09-04", health={"ok": True}, regime={"regime": "NORMAL"}, ranking_tail=[])
    blob = json.loads((tmp_path / "screen_latest.json").read_text())
    assert blob["rows"][0]["ticker"] == "T1"
    assert not (tmp_path / "screen_latest.json.tmp").exists()


# ── run_status carries stats ──────────────────────────────────────────────────

def test_load_stats_only_for_today(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"date": "2026-09-03", "ok": True}))
    assert load_stats("2026-09-04", p) is None
    assert load_stats("2026-09-03", p) == {"date": "2026-09-03", "ok": True}
    assert load_stats("2026-09-03", tmp_path / "missing.json") is None
    p.write_text("{not json")
    assert load_stats("2026-09-03", p) is None


def test_build_status_embeds_stats_and_health_error_line():
    log = ("=== Screener run started: x ===\n"
           "[health] adv_survivors=53 below floor 800\n"
           "src.run.PipelineHealthError: adv_survivors=53 below floor 800\n")
    st = build_status(log, rc=1, started_at="2026-09-04T16:30:00", duration_secs=90,
                      stats={"date": "2026-09-04", "ok": False, "reasons": ["adv_survivors=53 below floor 800"]})
    assert st["result"] == "failed"
    assert st["stats"]["ok"] is False
    assert "PipelineHealthError" in st["error"]


def test_build_status_without_stats_is_none():
    st = build_status("", rc=0, started_at="x", duration_secs=1)
    assert st["stats"] is None
