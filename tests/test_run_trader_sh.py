"""Runs the run_trader.sh branch suite as part of pytest.

The shell tests cover decisions no Python test can reach — stamp/retry/attempt
handling and the caffeinate wrapper live in bash. Without this wrapper they would
never run in CI and would rot.
"""
import subprocess
from pathlib import Path

import pytest

SUITE = Path(__file__).parent / "test_run_trader.sh"


@pytest.mark.skipif(not SUITE.exists(), reason="shell suite missing")
def test_run_trader_shell_suite():
    proc = subprocess.run(
        ["bash", str(SUITE)],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=SUITE.parent.parent,
    )
    # Print on failure so the failing case is visible in pytest output.
    assert proc.returncode == 0, f"\n{proc.stdout}\n{proc.stderr}"
    assert "failed 0" in proc.stdout, proc.stdout
