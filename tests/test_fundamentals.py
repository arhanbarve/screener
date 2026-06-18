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
                    {"end": "2023-09-30", "val": 383285000000, "form": "10-K", "filed": "2023-11-03"},
                    {"end": "2022-09-24", "val": 394328000000, "form": "10-K", "filed": "2022-10-28"},
                ]}
            },
            "CostOfGoodsAndServicesSold": {
                "units": {"USD": [
                    {"end": "2023-09-30", "val": 214137000000, "form": "10-K", "filed": "2023-11-03"},
                ]}
            },
            "Assets": {
                "units": {"USD": [
                    {"end": "2023-09-30", "val": 352583000000, "form": "10-K", "filed": "2023-11-03"},
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
        {"transactionCode": "P", "name": "Tim Cook", "transactionDate": recent},
        {"transactionCode": "P", "name": "Luca Maestri", "transactionDate": recent},
        {"transactionCode": "S", "name": "Tim Cook", "transactionDate": recent},
    ]
    count = parse_insider_buys(transactions, days=90)
    assert count == 2  # 2 distinct insiders with purchases
