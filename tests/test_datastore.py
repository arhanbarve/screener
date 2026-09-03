"""Private-data reads.

The public repo carries no holdings, so on Streamlit Cloud these reads have to
reach a private companion repo. Local disk must still win outright, or the cron
jobs and the test suite would start depending on a network call.
"""
import json

import pytest

from src import datastore


@pytest.fixture(autouse=True)
def _clean(monkeypatch, tmp_path):
    datastore.clear_cache()
    monkeypatch.delenv("DATA_REPO_TOKEN", raising=False)
    monkeypatch.delenv("DATA_REPO", raising=False)
    monkeypatch.setattr(datastore, "SCREENER_DIR", tmp_path)
    return tmp_path


class _Resp:
    def __init__(self, status=200, text="", payload=None):
        self.status_code = status
        self.text = text
        self._payload = payload

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


# ── local disk wins ───────────────────────────────────────────────────────────

def test_local_file_read_without_network(_clean, monkeypatch):
    (_clean / "positions.json").write_text('[{"ticker": "AAA"}]')

    def _boom(*a, **k):
        raise AssertionError("must not hit the network when the file is local")

    monkeypatch.setattr(datastore.requests, "get", _boom)
    assert json.loads(datastore.read_text("positions.json"))[0]["ticker"] == "AAA"


def test_local_dir_listing_wins(_clean, monkeypatch):
    out = _clean / "output"
    out.mkdir()
    (out / "screen_2026-01-02.csv").write_text("x")
    (out / "screen_2026-01-01.csv").write_text("x")
    (out / "notes.md").write_text("x")

    monkeypatch.setattr(datastore.requests, "get",
                        lambda *a, **k: pytest.fail("no network"))
    assert datastore.list_names("output", "screen_*.csv") == [
        "screen_2026-01-01.csv", "screen_2026-01-02.csv",
    ]


# ── remote fallback ───────────────────────────────────────────────────────────

def test_remote_used_when_local_absent(_clean, monkeypatch):
    monkeypatch.setenv("DATA_REPO_TOKEN", "tok")
    seen = {}

    def _get(url, headers=None, timeout=None):
        seen["url"] = url
        seen["auth"] = headers["Authorization"]
        return _Resp(text='{"ok": true}')

    monkeypatch.setattr(datastore.requests, "get", _get)
    assert datastore.read_text("run_status.json") == '{"ok": true}'
    assert seen["url"].endswith("/repos/arhanbarve/screener-data/contents/run_status.json")
    assert seen["auth"] == "Bearer tok"


def test_remote_listing_filters_by_pattern_and_type(_clean, monkeypatch):
    monkeypatch.setenv("DATA_REPO_TOKEN", "tok")
    payload = [
        {"name": "screen_2026-01-02.csv", "type": "file"},
        {"name": "screen_2026-01-01.csv", "type": "file"},
        {"name": "screen_2026-01-01.md", "type": "file"},
        {"name": "subdir", "type": "dir"},
    ]
    monkeypatch.setattr(datastore.requests, "get",
                        lambda *a, **k: _Resp(payload=payload))
    assert datastore.list_names("output", "screen_*.csv") == [
        "screen_2026-01-01.csv", "screen_2026-01-02.csv",
    ]


def test_custom_repo_env_respected(_clean, monkeypatch):
    monkeypatch.setenv("DATA_REPO_TOKEN", "tok")
    monkeypatch.setenv("DATA_REPO", "someone/else")
    seen = {}
    monkeypatch.setattr(datastore.requests, "get",
                        lambda url, **k: (seen.setdefault("u", url), _Resp(text="x"))[1])
    datastore.read_text("a.json")
    assert "/repos/someone/else/contents/a.json" in seen["u"]


# ── degrade, never traceback ──────────────────────────────────────────────────

def test_no_token_returns_none_without_network(_clean, monkeypatch):
    monkeypatch.setattr(datastore.requests, "get",
                        lambda *a, **k: pytest.fail("must not call without a token"))
    assert datastore.read_text("positions.json") is None
    assert datastore.list_names("output", "*.csv") == []


def test_http_error_returns_none(_clean, monkeypatch):
    monkeypatch.setenv("DATA_REPO_TOKEN", "tok")
    monkeypatch.setattr(datastore.requests, "get", lambda *a, **k: _Resp(status=404))
    assert datastore.read_text("missing.json") is None
    assert datastore.list_names("nope", "*") == []


def test_network_exception_returns_none(_clean, monkeypatch):
    monkeypatch.setenv("DATA_REPO_TOKEN", "tok")

    def _raise(*a, **k):
        raise datastore.requests.RequestException("down")

    monkeypatch.setattr(datastore.requests, "get", _raise)
    assert datastore.read_text("positions.json") is None


def test_malformed_listing_json_returns_empty(_clean, monkeypatch):
    monkeypatch.setenv("DATA_REPO_TOKEN", "tok")
    monkeypatch.setattr(datastore.requests, "get", lambda *a, **k: _Resp(payload=None))
    assert datastore.list_names("output", "*") == []


# ── caching ───────────────────────────────────────────────────────────────────

def test_remote_reads_are_cached(_clean, monkeypatch):
    monkeypatch.setenv("DATA_REPO_TOKEN", "tok")
    calls = []
    monkeypatch.setattr(datastore.requests, "get",
                        lambda *a, **k: (calls.append(1), _Resp(text="v"))[1])
    assert datastore.read_text("x.json") == "v"
    assert datastore.read_text("x.json") == "v"
    assert len(calls) == 1


def test_clear_cache_forces_refetch(_clean, monkeypatch):
    monkeypatch.setenv("DATA_REPO_TOKEN", "tok")
    calls = []
    monkeypatch.setattr(datastore.requests, "get",
                        lambda *a, **k: (calls.append(1), _Resp(text="v"))[1])
    datastore.read_text("x.json")
    datastore.clear_cache()
    datastore.read_text("x.json")
    assert len(calls) == 2
