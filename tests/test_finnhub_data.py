"""Finnhub-sourced caps, industries, ratios and earnings dates."""
import json
import os
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from src.cache import init_db, put_fh_metrics, put_fh_profile, put_market_cap
from src import finnhub_data as fd


def _db():
    f = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    f.close()
    init_db(f.name)
    return f.name


class _APIError(Exception):
    def __init__(self, status_code):
        super().__init__(f"status {status_code}")
        self.status_code = status_code


class FakeClient:
    """Scripted Finnhub client. `metrics[sym]` is the metric dict to return,
    `errors[sym]` an HTTP status to raise (once, unless persistent)."""

    def __init__(self, metrics=None, profiles=None, errors=None, persistent=False, calendar=None):
        self.metrics = metrics or {}
        self.profiles = profiles or {}
        self.errors = dict(errors or {})
        self.persistent = persistent
        self.calendar = calendar or {"earningsCalendar": []}
        self.calls = []

    def _maybe_raise(self, sym):
        if sym in self.errors:
            code = self.errors[sym]
            if not self.persistent:
                del self.errors[sym]
            raise _APIError(code)

    def company_basic_financials(self, sym, metric):
        self.calls.append(("metrics", sym))
        self._maybe_raise(sym)
        return {"metric": self.metrics.get(sym, {})} if sym in self.metrics else {}

    def company_profile2(self, symbol):
        self.calls.append(("profile", symbol))
        self._maybe_raise(symbol)
        return self.profiles.get(symbol, {})

    def earnings_calendar(self, _from, to, symbol):
        self.calls.append(("calendar", _from, to))
        return self.calendar


# ── symbols / sectors ─────────────────────────────────────────────────────────

def test_to_finnhub_symbol_maps_dash_to_dot():
    assert fd.to_finnhub_symbol("brk-b") == "BRK.B"
    assert fd.to_finnhub_symbol("AAPL") == "AAPL"


def test_sector_for_maps_known_and_passes_through_unknown():
    assert fd.sector_for("Biotechnology") == "Healthcare"
    assert fd.sector_for("Banking") == "Financials"
    assert fd.sector_for("Weird New Industry") == "Weird New Industry"
    assert fd.sector_for("") == "Unknown"
    assert fd.sector_for(None) == "Unknown"


def test_is_financial():
    assert fd.is_financial("Banking") and fd.is_financial("Real Estate")
    assert not fd.is_financial("Technology") and not fd.is_financial(None)


# ── fetch_metrics ─────────────────────────────────────────────────────────────

def test_fetch_metrics_caches_and_reuses():
    db = _db()
    client = FakeClient(metrics={"AAPL": {"peTTM": 30.0, "marketCapitalization": 3_000_000}})
    out, stats = fd.fetch_metrics(["AAPL"], db, client=client, sleep=lambda s: None)
    assert out["AAPL"]["peTTM"] == 30.0
    assert stats["fetched"] == 1 and stats["cached"] == 0
    out2, stats2 = fd.fetch_metrics(["AAPL"], db, client=client, sleep=lambda s: None)
    assert out2["AAPL"]["peTTM"] == 30.0
    assert stats2["cached"] == 1 and stats2["fetched"] == 0
    assert len([c for c in client.calls if c[0] == "metrics"]) == 1
    os.unlink(db)


def test_fetch_metrics_unknown_symbol_cached_as_empty():
    """Finnhub answers {} for an OTC line; cache that so it is not re-asked daily."""
    db = _db()
    client = FakeClient(metrics={})
    out, stats = fd.fetch_metrics(["WCPRF"], db, client=client, sleep=lambda s: None)
    assert out["WCPRF"] == {}
    _, stats2 = fd.fetch_metrics(["WCPRF"], db, client=client, sleep=lambda s: None)
    assert stats2["cached"] == 1 and stats2["fetched"] == 0
    os.unlink(db)


def test_fetch_metrics_respects_max_fetch():
    db = _db()
    client = FakeClient(metrics={t: {"peTTM": 1.0} for t in "ABCDE"})
    out, stats = fd.fetch_metrics(list("ABCDE"), db, max_fetch=2, client=client, sleep=lambda s: None)
    assert stats["fetched"] == 2 and stats["unfetched"] == 3
    assert set(out) == {"A", "B"}
    os.unlink(db)


def test_fetch_metrics_retries_once_on_429_then_succeeds():
    db = _db()
    slept = []
    client = FakeClient(metrics={"AAPL": {"peTTM": 30.0}}, errors={"AAPL": 429})
    out, stats = fd.fetch_metrics(["AAPL"], db, client=client, sleep=slept.append)
    assert out["AAPL"]["peTTM"] == 30.0
    assert 60.0 in slept
    os.unlink(db)


def test_fetch_metrics_403_stops_the_loop_without_poisoning_cache():
    db = _db()
    client = FakeClient(metrics={"AAPL": {}, "MSFT": {}}, errors={"AAPL": 403}, persistent=True)
    out, stats = fd.fetch_metrics(["AAPL", "MSFT"], db, client=client, sleep=lambda s: None)
    assert stats["forbidden"] is True
    assert "AAPL" not in out and "MSFT" not in out
    assert len(client.calls) == 1  # stopped after the 403
    os.unlink(db)


def test_fetch_metrics_other_error_counts_failed_and_continues():
    db = _db()
    client = FakeClient(metrics={"MSFT": {"peTTM": 20.0}}, errors={"AAPL": 500}, persistent=True)
    out, stats = fd.fetch_metrics(["AAPL", "MSFT"], db, client=client, sleep=lambda s: None)
    assert stats["failed"] == 1 and out["MSFT"]["peTTM"] == 20.0
    os.unlink(db)


def test_fetch_metrics_uses_finnhub_symbol_for_share_classes():
    db = _db()
    client = FakeClient(metrics={"BRK.B": {"peTTM": 10.0}})
    out, _ = fd.fetch_metrics(["BRK-B"], db, client=client, sleep=lambda s: None)
    assert out["BRK-B"]["peTTM"] == 10.0  # keyed by our ticker, asked with Finnhub's
    os.unlink(db)


# ── fetch_profiles ────────────────────────────────────────────────────────────

def test_fetch_profiles_converts_millions_and_keeps_industry():
    db = _db()
    client = FakeClient(profiles={"AAPL": {"finnhubIndustry": "Technology",
                                          "marketCapitalization": 3_000_000.0,   # $3.0T in millions
                                          "shareOutstanding": 15_000.0}})        # 15B shares in millions
    out, stats = fd.fetch_profiles(["AAPL"], db, client=client, sleep=lambda s: None)
    p = out["AAPL"]
    assert p["industry"] == "Technology"
    assert p["market_cap"] == pytest.approx(3.0e12)
    assert p["shares_out"] == pytest.approx(1.5e10)
    assert stats["fetched"] == 1
    os.unlink(db)


def test_fetch_profiles_refreshes_only_stale_caps():
    db = _db()
    put_fh_profile(db, "AAPL", "Technology", 2.0e12, 1.5e10)   # fresh row
    client = FakeClient(profiles={"AAPL": {"finnhubIndustry": "Technology", "marketCapitalization": 3_000_000.0}})
    out, stats = fd.fetch_profiles(["AAPL"], db, cap_ttl_days=7, client=client, sleep=lambda s: None)
    assert stats["fetched"] == 0 and out["AAPL"]["market_cap"] == pytest.approx(2.0e12)
    os.unlink(db)


def test_fetch_profiles_stale_row_still_returned_when_budget_exhausted():
    db = _db()
    put_fh_profile(db, "AAPL", "Technology", 2.0e12, None)
    client = FakeClient(profiles={})
    out, stats = fd.fetch_profiles(["AAPL"], db, cap_ttl_days=0, max_fetch=0, client=client, sleep=lambda s: None)
    assert out["AAPL"]["industry"] == "Technology"
    assert stats["unfetched"] == 1
    os.unlink(db)


# ── market cap resolution ─────────────────────────────────────────────────────

def test_resolve_market_cap_precedence_metrics_first():
    cap, src = fd.resolve_market_cap(
        "AAPL", 200.0,
        metrics={"AAPL": {"marketCapitalization": 3_000_000.0}},
        profiles={"AAPL": {"market_cap": 2.0e12, "shares_out": 1.5e10}})
    assert cap == pytest.approx(3.0e12) and src == "fh_metrics"


def test_resolve_market_cap_falls_to_profile_then_shares_times_price():
    cap, src = fd.resolve_market_cap("X", 10.0, metrics={}, profiles={"X": {"market_cap": 5.0e8, "shares_out": None}})
    assert cap == 5.0e8 and src == "fh_profile"
    cap, src = fd.resolve_market_cap("X", 10.0, metrics={}, profiles={"X": {"market_cap": None, "shares_out": 1.0e8}})
    assert cap == pytest.approx(1.0e9) and src == "shares_x_price"


def test_resolve_market_cap_legacy_cache_last_resort_and_none():
    db = _db()
    put_market_cap(db, "OLD", 7.5e8)
    cap, src = fd.resolve_market_cap("OLD", 10.0, {}, {}, db_path=db)
    assert cap == 7.5e8 and src == "legacy_cache"
    cap, src = fd.resolve_market_cap("NEW", 10.0, {}, {}, db_path=db)
    assert cap is None and src == "none"
    os.unlink(db)


def test_resolve_market_cap_rejects_implausible_values():
    # 3e18 is not a market cap; a wrong-unit source must not pass the gate.
    cap, src = fd.resolve_market_cap("X", 10.0, metrics={"X": {"marketCapitalization": 3e12}}, profiles={})
    assert cap is None and src == "none"
    cap, src = fd.resolve_market_cap("X", 10.0, metrics={"X": {"marketCapitalization": 0}}, profiles={})
    assert cap is None


def test_attach_market_caps_adds_columns_and_counts():
    df = pd.DataFrame({"ticker": ["AAPL", "NOCAP", "BANK"], "price": [200.0, 10.0, 50.0]})
    metrics = {"AAPL": {"marketCapitalization": 3_000_000.0}}
    profiles = {"AAPL": {"industry": "Technology", "market_cap": None, "shares_out": None},
                "BANK": {"industry": "Banking", "market_cap": 8.0e9, "shares_out": None}}
    out, stats = fd.attach_market_caps(df, metrics, profiles)
    assert out.loc[0, "market_cap"] == pytest.approx(3.0e12)
    assert pd.isna(out.loc[1, "market_cap"]) and out.loc[1, "cap_source"] == "none"
    assert out.loc[2, "sector"] == "Financials" and bool(out.loc[2, "is_financial"]) is True
    assert out.loc[0, "sector"] == "Technology" and bool(out.loc[0, "is_financial"]) is False
    assert stats["dropped_no_cap"] == 1 and stats["fh_metrics"] == 1 and stats["fh_profile"] == 1
    assert out.loc[1, "sector"] == "Unknown"


# ── earnings calendar ─────────────────────────────────────────────────────────

def test_fetch_earnings_calendar_keeps_earliest_date_and_normalises_symbols(tmp_path):
    client = FakeClient(calendar={"earningsCalendar": [
        {"symbol": "AAPL", "date": "2026-10-29"},
        {"symbol": "AAPL", "date": "2026-10-28"},
        {"symbol": "BRK.B", "date": "2026-11-01"},
        {"symbol": "", "date": "2026-11-01"},
        {"symbol": "BAD", "date": "2026-1-1"},
    ]})
    cal = fd.fetch_earnings_calendar(cache_path=tmp_path / "cal.json", client=client, today=date(2026, 9, 4))
    assert cal == {"AAPL": "2026-10-28", "BRK-B": "2026-11-01"}
    # Second call is served from the cache: no new client call.
    n = len(client.calls)
    fd.fetch_earnings_calendar(cache_path=tmp_path / "cal.json", client=client, today=date(2026, 9, 4))
    assert len(client.calls) == n


def test_fetch_earnings_calendar_falls_back_to_stale_cache_on_error(tmp_path):
    path = tmp_path / "cal.json"
    path.write_text(json.dumps({"fetched_at": (datetime.utcnow() - timedelta(days=3)).isoformat(),
                                "dates": {"AAPL": "2026-10-28"}}))

    class Broken(FakeClient):
        def earnings_calendar(self, _from, to, symbol):
            raise _APIError(500)

    cal = fd.fetch_earnings_calendar(cache_path=path, client=Broken(), today=date(2026, 9, 4))
    assert cal == {"AAPL": "2026-10-28"}


def test_days_to_earnings():
    cal = {"AAPL": "2026-09-10", "OLD": "2026-09-01"}
    assert fd.days_to_earnings("aapl", cal, today=date(2026, 9, 4)) == 6
    assert fd.days_to_earnings("OLD", cal, today=date(2026, 9, 4)) is None   # stale, never negative
    assert fd.days_to_earnings("MSFT", cal, today=date(2026, 9, 4)) is None
