"""Finalist enrichment: estimate revisions, caching, budget, composite_final."""
import os
import tempfile
from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.cache import init_db
from src import enrich


def _db():
    f = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    f.close()
    init_db(f.name)
    return f.name


def _eps_trend(cur_y=8.8, ago30_y=8.6, ago90_y=8.5, cur_q=1.97, ago30_q=1.99, ago90_q=2.0):
    return pd.DataFrame(
        {"current": [cur_q, 2.9, cur_y, 9.5], "7daysAgo": [cur_q, 2.9, cur_y, 9.5],
         "30daysAgo": [ago30_q, 2.9, ago30_y, 9.5], "60daysAgo": [ago30_q, 2.9, ago30_y, 9.5],
         "90daysAgo": [ago90_q, 2.9, ago90_y, 9.5]},
        index=pd.Index(["0q", "+1q", "0y", "+1y"], name="period"))


def test_eps_revision_prefers_fiscal_year_row():
    r30, r90 = enrich.eps_revision(_eps_trend())
    assert r30 == pytest.approx(8.8 / 8.6 - 1)
    assert r90 == pytest.approx(8.8 / 8.5 - 1)


def test_eps_revision_falls_back_to_quarter_when_year_missing():
    df = _eps_trend().drop(index="0y")
    r30, _ = enrich.eps_revision(df)
    assert r30 == pytest.approx(1.97 / 1.99 - 1)


def test_eps_revision_negative_eps_uses_abs_denominator():
    df = _eps_trend(cur_y=-0.40, ago30_y=-0.50, ago90_y=-0.60)
    r30, r90 = enrich.eps_revision(df)
    assert r30 == pytest.approx(0.20) and r90 == pytest.approx(1 / 3)


def test_eps_revision_tiny_base_is_nan_and_empty_is_nan():
    df = _eps_trend(cur_y=0.01, ago30_y=0.005, ago90_y=0.001, cur_q=0.01, ago30_q=0.005, ago90_q=0.001)
    r30, r90 = enrich.eps_revision(df)
    assert np.isnan(r30) and np.isnan(r90)
    assert all(np.isnan(v) for v in enrich.eps_revision(None))
    assert all(np.isnan(v) for v in enrich.eps_revision(pd.DataFrame()))


def test_target_pct_and_short_interest_helpers():
    assert enrich.target_pct({"mean": 110.0}, 100.0) == pytest.approx(0.10)
    assert np.isnan(enrich.target_pct({"mean": None}, 100.0))
    assert np.isnan(enrich.target_pct({"mean": 110.0}, 0.0))
    assert np.isnan(enrich.target_pct(None, 100.0))
    sf, dtc = enrich.short_interest({"sharesShort": 10, "floatShares": 100, "averageVolume": 5})
    assert sf == pytest.approx(0.10) and dtc == pytest.approx(2.0)
    assert all(np.isnan(v) for v in enrich.short_interest(None))


def test_composite_final_adds_rank_normalised_revision():
    df = pd.DataFrame({"ticker": list("ABCD"), "composite": [1.0, 1.0, 1.0, 1.0],
                       "eps_rev_30d": [0.10, 0.0, -0.10, np.nan]})
    out = enrich.composite_final(df, weight=0.05)
    assert out.loc[0, "composite_final"] > out.loc[1, "composite_final"] > out.loc[2, "composite_final"]
    assert out.loc[3, "z_eps_rev_30d"] == 0.0 and out.loc[3, "composite_final"] == 1.0


def test_composite_final_neutral_when_too_few_revisions():
    df = pd.DataFrame({"ticker": list("AB"), "composite": [1.0, 0.5], "eps_rev_30d": [0.3, np.nan]})
    out = enrich.composite_final(df)
    assert (out["composite_final"] == out["composite"]).all()


def _fake_fetch(payloads: dict, fail: set = frozenset()):
    calls = []
    def fetch(ticker):
        calls.append(ticker)
        if ticker in fail:
            raise RuntimeError("rate limited")
        return payloads[ticker]
    fetch.calls = calls
    return fetch


def test_enrich_finalists_attaches_columns_and_caches():
    db = _db()
    df = pd.DataFrame({"ticker": ["AAA", "BBB"], "price": [100.0, 50.0], "composite": [1.0, 0.9]})
    payloads = {
        "AAA": {"eps_trend": _eps_trend(), "targets": {"mean": 120.0},
                "info": {"sharesShort": 10, "floatShares": 100, "averageVolume": 5}},
        "BBB": {"eps_trend": None, "targets": None, "info": None},
    }
    cal = {"AAA": "2026-09-20"}
    fetch = _fake_fetch(payloads)
    out, stats = enrich.enrich_finalists(df, db, cal, today=date(2026, 9, 4), fetch=fetch, sleep=lambda s: None)
    assert stats["fetched"] == 2 and stats["cached"] == 0
    assert out.loc[0, "eps_rev_30d"] == pytest.approx(8.8 / 8.6 - 1)
    assert out.loc[0, "analyst_target_pct"] == pytest.approx(0.20)
    assert out.loc[0, "short_float"] == pytest.approx(0.10)
    assert out.loc[0, "days_to_earnings"] == 16
    assert np.isnan(out.loc[1, "eps_rev_30d"]) and np.isnan(out.loc[1, "days_to_earnings"])
    # Second call same day: served from cache, fetch not called again.
    out2, stats2 = enrich.enrich_finalists(df, db, cal, today=date(2026, 9, 4), fetch=fetch, sleep=lambda s: None)
    assert stats2["cached"] == 2 and len(fetch.calls) == 2
    assert out2.loc[0, "eps_rev_30d"] == pytest.approx(out.loc[0, "eps_rev_30d"])
    os.unlink(db)


def test_enrich_finalists_retries_once_then_counts_failure():
    db = _db()
    df = pd.DataFrame({"ticker": ["AAA"], "price": [100.0], "composite": [1.0]})
    slept = []
    fetch = _fake_fetch({}, fail={"AAA"})
    out, stats = enrich.enrich_finalists(df, db, {}, today=date(2026, 9, 4), fetch=fetch, sleep=slept.append)
    assert stats["failed"] == 1 and len(fetch.calls) == 2 and slept == [enrich.RETRY_SLEEP]
    assert np.isnan(out.loc[0, "eps_rev_30d"])
    os.unlink(db)


def test_enrich_finalists_respects_wall_clock_budget():
    db = _db()
    df = pd.DataFrame({"ticker": ["AAA", "BBB", "CCC"], "price": [1.0, 1.0, 1.0], "composite": [1, 1, 1]})
    payloads = {t: {"eps_trend": _eps_trend(), "targets": None, "info": None} for t in ["AAA", "BBB", "CCC"]}
    ticks = iter([0.0, 0.0, 100.0, 100.0, 100.0, 100.0])   # budget blown after the first fetch
    out, stats = enrich.enrich_finalists(df, db, {}, today=date(2026, 9, 4), fetch=_fake_fetch(payloads),
                                         budget_secs=10.0, sleep=lambda s: None, clock=lambda: next(ticks))
    assert stats["fetched"] == 1 and stats["budget_exhausted"] == 2
    # days_to_earnings still filled from the calendar even when yfinance is skipped
    assert "days_to_earnings" in out.columns
    os.unlink(db)
