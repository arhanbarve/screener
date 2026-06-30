# tests/test_universe.py
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
import json
from src.universe import parse_sec_tickers, filter_universe

SAMPLE_SEC = {
    "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
    "1": {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft Corp"},
    "2": {"cik_str": 1018724, "ticker": "AMZN-WT", "title": "Amazon Warrant"},
    "3": {"cik_str": 1001085, "ticker": "SPXU", "title": "ProShares UltraPro Short S&P500"},
    "4": {"cik_str": 1234567, "ticker": "BX", "title": "Blackstone lp"},
    "5": {"cik_str": 2345678, "ticker": "KKR2", "title": "KKR & Co lp fund"},
}

def test_parse_sec_tickers():
    df = parse_sec_tickers(SAMPLE_SEC)
    assert "ticker" in df.columns
    assert "cik" in df.columns
    assert len(df) == 6

def test_filter_removes_warrants():
    df = parse_sec_tickers(SAMPLE_SEC)
    result = filter_universe(df)
    assert "AMZN-WT" not in result["ticker"].values

def test_filter_removes_etfs_by_name():
    df = parse_sec_tickers(SAMPLE_SEC)
    result = filter_universe(df)
    assert "SPXU" not in result["ticker"].values

def test_filter_removes_lp_funds():
    df = parse_sec_tickers(SAMPLE_SEC)
    result = filter_universe(df)
    assert "BX" not in result["ticker"].values    # name ends in " lp"
    assert "KKR2" not in result["ticker"].values  # name contains " lp "

def test_cik_formatted_as_10digit_string():
    df = parse_sec_tickers(SAMPLE_SEC)
    assert df["cik"].iloc[0] == "0000320193"
