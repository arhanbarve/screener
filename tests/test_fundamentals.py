# tests/test_fundamentals.py
import pytest
from unittest.mock import patch, MagicMock
from src.fundamentals import (
    parse_edgar_gp,
    parse_finnhub_surprise,
    parse_finnhub_revisions,
    parse_short_interest,
    parse_insider_buys,
)

SAMPLE_EDGAR = {
    "facts": {
        "us-gaap": {
            "Revenues": {
                "units": {"USD": [
                    {"end": "2025-09-27", "val": 383285000000, "form": "10-K", "filed": "2025-11-03"},
                    {"end": "2024-09-28", "val": 394328000000, "form": "10-K", "filed": "2024-10-28"},
                ]}
            },
            "CostOfGoodsAndServicesSold": {
                "units": {"USD": [
                    {"end": "2025-09-27", "val": 214137000000, "form": "10-K", "filed": "2025-11-03"},
                ]}
            },
            "Assets": {
                "units": {"USD": [
                    {"end": "2025-09-27", "val": 352583000000, "form": "10-K", "filed": "2025-11-03"},
                ]}
            },
        }
    }
}

def test_parse_edgar_gp():
    gp, rev, cogs, assets = parse_edgar_gp(SAMPLE_EDGAR)
    assert abs(gp - (383285e6 - 214137e6) / 352583e6) < 1e-4

def test_parse_edgar_gp_missing_tag_raises():
    with pytest.raises(KeyError):
        parse_edgar_gp({"facts": {"us-gaap": {}}})

SAMPLE_EARNINGS = [
    {"period": "2023-09-30", "actual": 1.46, "estimate": 1.39, "symbol": "AAPL"},
    {"period": "2023-06-30", "actual": 1.26, "estimate": 1.19, "symbol": "AAPL"},
    {"period": "2023-03-31", "actual": 1.52, "estimate": 1.43, "symbol": "AAPL"},
    {"period": "2022-12-31", "actual": 1.88, "estimate": 1.94, "symbol": "AAPL"},
]

def test_parse_finnhub_surprise():
    actuals, estimates = parse_finnhub_surprise(SAMPLE_EARNINGS)
    assert len(actuals) == 4
    assert actuals[-1] == 1.46   # most recent last

def test_parse_finnhub_revisions_empty():
    rev_b, rev_m = parse_finnhub_revisions({})
    assert rev_b == 0.0
    assert rev_m == 0.0

def test_parse_short_interest_from_info():
    info = {"sharesShort": 100_000_000, "floatShares": 500_000_000, "averageVolume": 20_000_000}
    short_float, dtc = parse_short_interest(info)
    assert abs(short_float - 0.20) < 1e-6
    assert abs(dtc - 5.0) < 1e-6

def test_parse_insider_buys():
    from datetime import datetime, timedelta
    recent = (datetime.utcnow() - timedelta(days=30)).strftime("%Y-%m-%d")
    transactions = [
        {"transactionCode": "P", "name": "Tim Cook", "officerTitle": "CEO", "transactionDate": recent, "share": 100, "transactionPrice": 150.0},
        {"transactionCode": "P", "name": "Luca Maestri", "officerTitle": "CFO", "transactionDate": recent, "share": 50, "transactionPrice": 150.0},
        {"transactionCode": "S", "name": "Tim Cook", "officerTitle": "CEO", "transactionDate": recent},
    ]
    result = parse_insider_buys(transactions, days=90)
    assert result["insider_buys_90d"] == 2   # 2 distinct insiders with purchases
    assert result["exec_buys_90d"] == 2      # both are executives (CEO + CFO)
    assert result["insider_buy_value"] > 0   # dollar value captured


# ── extended EDGAR parser ─────────────────────────────────────────────────────
from src.fundamentals import parse_edgar_facts, _annual_series, _latest_two


def _dur(start, end, val, form="10-K", filed=None):
    return {"start": start, "end": end, "val": val, "form": form, "filed": filed or end}


def _inst(end, val, form="10-K", filed=None):
    return {"end": end, "val": val, "form": form, "filed": filed or end}


FULL_EDGAR = {"facts": {"us-gaap": {
    "Revenues": {"units": {"USD": [
        _dur("2025-01-01", "2025-12-31", 1000.0),
        _dur("2025-10-01", "2025-12-31", 300.0),        # Q4 stub inside the 10-K — must be ignored
        _dur("2024-01-01", "2024-12-31", 800.0),
        _dur("2025-01-01", "2025-03-31", 200.0, form="10-Q"),
    ]}},
    "CostOfRevenue": {"units": {"USD": [_dur("2025-01-01", "2025-12-31", 400.0),
                                         _dur("2024-01-01", "2024-12-31", 350.0)]}},
    "NetIncomeLoss": {"units": {"USD": [_dur("2025-01-01", "2025-12-31", 120.0),
                                         _dur("2025-01-01", "2025-12-31", 118.0, filed="2025-02-01"),  # earlier filing, superseded
                                         _dur("2024-01-01", "2024-12-31", 90.0)]}},
    "NetCashProvidedByUsedInOperatingActivities": {"units": {"USD": [_dur("2025-01-01", "2025-12-31", 150.0)]}},
    "Assets": {"units": {"USD": [_inst("2025-12-31", 2000.0), _inst("2024-12-31", 1600.0),
                                  _inst("2025-06-30", 1900.0, form="10-Q")]}},
    "Liabilities": {"units": {"USD": [_inst("2025-12-31", 1200.0)]}},
    "StockholdersEquity": {"units": {"USD": [_inst("2025-12-31", 800.0)]}},
    "WeightedAverageNumberOfDilutedSharesOutstanding": {"units": {"shares": [
        _dur("2025-01-01", "2025-12-31", 110.0), _dur("2024-01-01", "2024-12-31", 100.0)]}},
}}}


def test_annual_series_filters_forms_durations_and_dedupes_by_latest_filing():
    ni = FULL_EDGAR["facts"]["us-gaap"]["NetIncomeLoss"]["units"]["USD"]
    assert _annual_series(ni, duration=True) == [("2025-12-31", 120.0), ("2024-12-31", 90.0)]
    rev = FULL_EDGAR["facts"]["us-gaap"]["Revenues"]["units"]["USD"]
    assert _annual_series(rev, duration=True) == [("2025-12-31", 1000.0), ("2024-12-31", 800.0)]
    assets = FULL_EDGAR["facts"]["us-gaap"]["Assets"]["units"]["USD"]
    assert _annual_series(assets, duration=False) == [("2025-12-31", 2000.0), ("2024-12-31", 1600.0)]


def test_latest_two_requires_a_real_prior_year():
    assert _latest_two([("2024-12-31", 1.0), ("2023-12-31", 2.0)]) == (1.0, 2.0)
    assert _latest_two([("2024-12-31", 1.0), ("2024-06-30", 2.0)]) == (1.0, None)   # stub, not a year
    assert _latest_two([("2024-12-31", 1.0)]) == (1.0, None)
    assert _latest_two([]) == (None, None)


def test_parse_edgar_facts_derives_ratios():
    f = parse_edgar_facts(FULL_EDGAR)
    assert f["gp_assets"] == pytest.approx((1000 - 400) / 2000)
    assert f["accruals"] == pytest.approx((120 - 150) / 2000)
    assert f["asset_growth"] == pytest.approx(2000 / 1600 - 1)
    assert f["net_issuance"] == pytest.approx(0.10)
    assert f["roa"] == pytest.approx(120 / 2000)
    assert f["leverage"] == pytest.approx(0.6)
    assert f["n_resolved"] == 10


def test_parse_edgar_facts_partial_data_is_nan_not_error():
    only_assets = {"facts": {"us-gaap": {"Assets": {"units": {"USD": [_inst("2025-12-31", 2000.0)]}}}}}
    f = parse_edgar_facts(only_assets)
    assert f["assets"] == 2000.0 and f["n_resolved"] == 1
    for k in ("gp_assets", "accruals", "asset_growth", "net_issuance", "roa", "leverage"):
        assert f[k] != f[k]   # NaN


def test_parse_edgar_facts_no_usgaap_returns_zero_resolved():
    assert parse_edgar_facts({"facts": {"ifrs-full": {}}})["n_resolved"] == 0
    assert parse_edgar_facts({})["n_resolved"] == 0


def test_parse_edgar_gp_ignores_q4_stub_with_same_end_date():
    gp, rev, cogs, assets = parse_edgar_gp(FULL_EDGAR)
    assert rev == 1000.0 and cogs == 400.0 and assets == 2000.0


def test_edgar_facts_cache_roundtrip_preserves_nan(tmp_path):
    from src.cache import init_db, put_edgar_facts, get_edgar_facts
    db = str(tmp_path / "c.db"); init_db(db)
    put_edgar_facts(db, "0000000001", {"gp_assets": 0.3, "accruals": float("nan"), "n_resolved": 3})
    back = get_edgar_facts(db, "0000000001", ttl_days=30)
    assert back["gp_assets"] == 0.3 and back["accruals"] != back["accruals"] and back["n_resolved"] == 3
    assert get_edgar_facts(db, "0000000001", ttl_days=0) is None


def test_resolve_pair_prefers_most_recent_tag_not_list_order():
    """MSFT: `Revenues` stopped in FY2010, the newer tag carries 2026. Priority
    order used to return the 2010 number."""
    from src.fundamentals import _resolve_pair
    from datetime import date
    facts = {"us-gaap": {
        "Revenues": {"units": {"USD": [_dur("2009-07-01", "2010-06-30", 62.0)]}},
        "RevenueFromContractWithCustomerExcludingAssessedTax": {"units": {"USD": [
            _dur("2025-07-01", "2026-06-30", 331.0), _dur("2024-07-01", "2025-06-30", 281.0)]}},
    }}
    latest, prev = _resolve_pair(facts, ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax"],
                                 duration=True, today=date(2026, 9, 4))
    assert (latest, prev) == (331.0, 281.0)


def test_resolve_pair_tie_on_end_takes_the_larger_total():
    from src.fundamentals import _resolve_pair
    from datetime import date
    facts = {"us-gaap": {
        "CostOfGoodsSold": {"units": {"USD": [_dur("2025-01-01", "2025-12-31", 15.0)]}},
        "CostOfRevenue": {"units": {"USD": [_dur("2025-01-01", "2025-12-31", 34.0)]}},
    }}
    latest, _ = _resolve_pair(facts, ["CostOfGoodsSold", "CostOfRevenue"], duration=True, today=date(2026, 9, 4))
    assert latest == 34.0


def test_resolve_pair_discards_stale_filers():
    from src.fundamentals import _resolve_pair
    from datetime import date
    facts = {"us-gaap": {"Assets": {"units": {"USD": [_inst("2023-12-31", 100.0)]}}}}
    assert _resolve_pair(facts, ["Assets"], duration=False, today=date(2026, 9, 4)) == (None, None)
    assert _resolve_pair(facts, ["Assets"], duration=False, today=date(2024, 9, 4)) == (100.0, None)


def test_parse_edgar_facts_gross_profit_fallback_when_no_cogs():
    facts = {"facts": {"us-gaap": {
        "GrossProfit": {"units": {"USD": [_dur("2025-01-01", "2025-12-31", 600.0)]}},
        "Assets": {"units": {"USD": [_inst("2025-12-31", 2000.0)]}},
    }}}
    assert parse_edgar_facts(facts)["gp_assets"] == pytest.approx(0.3)


# ── EDGAR fetch is paced — SEC throttles sustained unpaced requests ───────────
def test_fetch_edgar_paces_a_real_fetch_through_the_bucket(tmp_path):
    from src.fundamentals import fetch_edgar, _TokenBucket
    from src.cache import init_db
    db = str(tmp_path / "c.db"); init_db(db)
    consumed = []
    bucket = MagicMock()
    bucket.consume.side_effect = lambda: consumed.append(1)
    with patch("src.fundamentals.requests.get") as mock_get:
        mock_get.return_value.raise_for_status.return_value = None
        mock_get.return_value.json.return_value = {"facts": {"us-gaap": {}}}
        fetch_edgar("0000000001", db, ttl_days=30, bucket=bucket)
    assert consumed == [1]


def test_fetch_edgar_skips_the_bucket_entirely_on_a_cache_hit(tmp_path):
    from src.fundamentals import fetch_edgar
    from src.cache import init_db, put_edgar_facts
    db = str(tmp_path / "c.db"); init_db(db)
    put_edgar_facts(db, "0000000001", {"gp_assets": 0.3, "n_resolved": 1})
    bucket = MagicMock()
    with patch("src.fundamentals.requests.get") as mock_get:
        result = fetch_edgar("0000000001", db, ttl_days=30, bucket=bucket)
    bucket.consume.assert_not_called()
    mock_get.assert_not_called()
    assert result["gp_assets"] == 0.3


def test_fetch_edgar_without_a_bucket_still_works():
    """bucket is optional — standalone/backtest callers that never pass one
    must not break. An empty companyfacts payload resolves nothing, which
    fetch_edgar caches as a miss and reports as None (not a dict)."""
    from src.fundamentals import fetch_edgar
    with patch("src.fundamentals.requests.get") as mock_get, \
         patch("src.fundamentals.get_edgar_facts", return_value=None), \
         patch("src.fundamentals.put_edgar_facts") as mock_put:
        mock_get.return_value.raise_for_status.return_value = None
        mock_get.return_value.json.return_value = {"facts": {"us-gaap": {}}}
        result = fetch_edgar("0000000001", "unused.db", ttl_days=30)
    assert result is None
    mock_put.assert_called_once()


def test_token_bucket_sleeps_via_injected_sleep_when_exhausted():
    from src.fundamentals import _TokenBucket
    slept = []
    bucket = _TokenBucket(rate=1, sleep=slept.append)
    bucket.consume()   # first token is free
    bucket.consume()   # bucket now exhausted, must sleep
    assert slept and slept[0] > 0


def test_fetch_edgar_converts_a_stall_into_a_logged_miss_not_a_hang():
    """A live 2026-09-04 run saw requests.get's own timeout= not reliably
    fire on a stalled SEC connection. fetch_edgar must not hang the whole
    fundamentals loop waiting for it."""
    import time as _time
    from src.fundamentals import fetch_edgar, EDGAR_REQUEST_TIMEOUT_SEC
    def _hangs(*a, **k):
        _time.sleep(1.0)
        raise AssertionError("should never complete before the deadline fires")
    with patch("src.fundamentals.EDGAR_REQUEST_TIMEOUT_SEC", 0.05), \
         patch("src.fundamentals.requests.get", side_effect=_hangs), \
         patch("src.fundamentals.get_edgar_facts", return_value=None), \
         patch("src.fundamentals.put_edgar_facts"):
        result = fetch_edgar("0000000001", "unused.db", ttl_days=30)
    assert result is None
