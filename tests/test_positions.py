import json
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch

from src import datastore
from src.positions import (
    load_positions,
    save_positions,
    add_position,
    remove_position,
)


@pytest.fixture(autouse=True)
def _isolate_datastore(tmp_path, monkeypatch):
    """Keep these tests off the real repo's data.

    load_positions() falls back to the private data repo when the local file is
    absent, so a test that chdirs into an empty directory has to neutralise that
    second source too — otherwise "missing file" quietly reads the developer's
    own positions.json from the repo root.
    """
    datastore.clear_cache()
    monkeypatch.setattr(datastore, "SCREENER_DIR", tmp_path)
    monkeypatch.delenv("DATA_REPO_TOKEN", raising=False)


# ── CRUD tests ────────────────────────────────────────────────────────────────

def test_load_positions_missing_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = load_positions()
    assert result == []


def test_save_and_load_roundtrip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    positions = [{"ticker": "AAPL", "entry_date": "2026-06-01", "entry_price": 150.0}]
    save_positions(positions)
    assert load_positions() == positions


def test_save_positions_no_tmp_file_left(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    positions = [{"ticker": "AAPL", "entry_date": "2026-06-01", "entry_price": 150.0}]
    save_positions(positions)
    assert not (tmp_path / "positions.tmp").exists()


def test_add_position(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    add_position("TSLA", "2026-06-01", 200.0)
    positions = load_positions()
    assert len(positions) == 1
    assert positions[0]["ticker"] == "TSLA"
    assert positions[0]["entry_price"] == 200.0


def test_add_position_duplicate_raises(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    add_position("AAPL", "2026-06-01", 150.0)
    with pytest.raises(ValueError, match="AAPL already in open positions"):
        add_position("aapl", "2026-06-02", 155.0)


def test_remove_position(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    add_position("AAPL", "2026-06-01", 150.0)
    add_position("MSFT", "2026-06-01", 300.0)
    remove_position("AAPL")
    positions = load_positions()
    assert len(positions) == 1
    assert positions[0]["ticker"] == "MSFT"


def test_remove_position_noop_if_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    add_position("AAPL", "2026-06-01", 150.0)
    remove_position("ZZZZ")  # should not raise
    assert len(load_positions()) == 1

