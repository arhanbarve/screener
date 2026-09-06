"""Alpaca tradability filter, SEC suffix exclusions, universe staleness."""
import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from src import universe as U


def _sec_df(*tickers):
    return pd.DataFrame({"ticker": list(tickers), "cik": ["0000000001"] * len(tickers),
                         "name": ["Some Company Inc"] * len(tickers)})


def test_filter_universe_drops_preferred_warrant_unit_right_suffixes():
    df = _sec_df("ACR-PC", "AGM-PI", "CLBR-WS", "ABLV-WT", "BEBE-U", "GFR-R", "GEF-B", "BRK-B", "AAPL")
    out = U.filter_universe(df)
    assert sorted(out["ticker"]) == ["AAPL", "BRK-B", "GEF-B"]


def test_to_alpaca_symbol():
    assert U.to_alpaca_symbol("GEF-B") == "GEF.B"
    assert U.to_alpaca_symbol("aapl") == "AAPL"


def _assets_raw():
    return [
        {"symbol": "AAPL", "exchange": "NASDAQ", "tradable": True, "fractionable": True, "shortable": True},
        {"symbol": "GEF.B", "exchange": "NYSE", "tradable": True, "fractionable": True, "shortable": False},
        {"symbol": "WCPRF", "exchange": "OTC", "tradable": True, "fractionable": False, "shortable": False},
        {"symbol": "DEAD", "exchange": "NYSE", "tradable": False, "fractionable": False, "shortable": False},
        {"symbol": "BAC.PRN", "exchange": "NYSE", "tradable": True, "fractionable": False, "shortable": False},
        {"symbol": "CLBR.WS", "exchange": "NYSE", "tradable": True, "fractionable": False, "shortable": False},
        {"symbol": "GLED.U", "exchange": "NASDAQ", "tradable": True, "fractionable": False, "shortable": False},
        {"symbol": "GFR.RT", "exchange": "NYSE", "tradable": True, "fractionable": False, "shortable": False},
        {"symbol": "HEI.A", "exchange": "NYSE", "tradable": True, "fractionable": True, "shortable": True},
    ]


def test_load_tradable_assets_filters_and_caches(tmp_path):
    cache = tmp_path / "assets.json"
    assets = U.load_tradable_assets(fetch=_assets_raw, cache_path=cache)
    assert set(assets) == {"AAPL", "GEF.B", "HEI.A"}
    assert assets["AAPL"]["fractionable"] is True and assets["AAPL"]["exchange"] == "NASDAQ"
    # Cached: a second call with a failing fetch still returns the same set.
    def boom():
        raise RuntimeError("down")
    assert set(U.load_tradable_assets(fetch=boom, cache_path=cache)) == {"AAPL", "GEF.B", "HEI.A"}


def test_load_tradable_assets_returns_none_when_unavailable_and_no_cache(tmp_path):
    def boom():
        raise RuntimeError("down")
    assert U.load_tradable_assets(fetch=boom, cache_path=tmp_path / "none.json") is None


def test_load_tradable_assets_uses_stale_cache_when_fetch_fails(tmp_path):
    cache = tmp_path / "assets.json"
    cache.write_text(json.dumps({"fetched_at": (datetime.utcnow() - timedelta(days=3)).isoformat(),
                                 "assets": {"AAPL": {"exchange": "NASDAQ", "fractionable": True, "shortable": True}}}))
    def boom():
        raise RuntimeError("down")
    assert set(U.load_tradable_assets(fetch=boom, cache_path=cache, ttl_hours=24)) == {"AAPL"}


def test_load_tradable_assets_without_credentials_returns_none(tmp_path, monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    assert U.load_tradable_assets(cache_path=tmp_path / "none.json") is None


def test_filter_tradable_keeps_listed_and_maps_symbols():
    df = _sec_df("AAPL", "GEF-B", "WCPRF", "PBR-A", "BHPLF")
    assets = {"AAPL": {"exchange": "NASDAQ", "fractionable": True},
              "GEF.B": {"exchange": "NYSE", "fractionable": True},
              "PBR.A": {"exchange": "NYSE", "fractionable": True}}
    out = U.filter_tradable(df, assets)
    assert list(out["ticker"]) == ["AAPL", "GEF-B", "PBR-A"]
    assert list(out["alpaca_symbol"]) == ["AAPL", "GEF.B", "PBR.A"]
    assert list(out["exchange"]) == ["NASDAQ", "NYSE", "NYSE"]
    assert out["fractionable"].all()


def test_filter_tradable_with_none_drops_nothing():
    df = _sec_df("AAPL", "WCPRF")
    out = U.filter_tradable(df, None)
    assert list(out["ticker"]) == ["AAPL", "WCPRF"]
    assert list(out["alpaca_symbol"]) == ["AAPL", "WCPRF"]
    assert "fractionable" in out.columns


def test_load_or_build_universe_uses_fresh_cache(tmp_path, monkeypatch):
    path = tmp_path / "u.parquet"
    _sec_df("AAPL").to_parquet(path, index=False)
    monkeypatch.setattr(U, "build_universe", lambda cfg, p: (_ for _ in ()).throw(AssertionError("must not rebuild")))
    out = U.load_or_build_universe({"universe": {}}, str(path), max_age_days=7)
    assert list(out["ticker"]) == ["AAPL"]


def test_load_or_build_universe_rebuilds_when_stale(tmp_path, monkeypatch):
    path = tmp_path / "u.parquet"
    _sec_df("OLD").to_parquet(path, index=False)
    old = time.time() - 10 * 86400
    os.utime(path, (old, old))
    monkeypatch.setattr(U, "build_universe", lambda cfg, p: _sec_df("NEW"))
    out = U.load_or_build_universe({"universe": {}}, str(path), max_age_days=7)
    assert list(out["ticker"]) == ["NEW"]


def test_load_or_build_universe_stale_rebuild_failure_falls_back(tmp_path, monkeypatch):
    path = tmp_path / "u.parquet"
    _sec_df("OLD").to_parquet(path, index=False)
    old = time.time() - 10 * 86400
    os.utime(path, (old, old))
    def boom(cfg, p):
        raise RuntimeError("SEC down")
    monkeypatch.setattr(U, "build_universe", boom)
    out = U.load_or_build_universe({"universe": {}}, str(path), max_age_days=7)
    assert list(out["ticker"]) == ["OLD"]


def test_load_or_build_universe_missing_and_failing_raises(tmp_path, monkeypatch):
    def boom(cfg, p):
        raise RuntimeError("SEC down")
    monkeypatch.setattr(U, "build_universe", boom)
    with pytest.raises(RuntimeError):
        U.load_or_build_universe({"universe": {}}, str(tmp_path / "missing.parquet"))
