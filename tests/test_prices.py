# tests/test_prices.py
import pandas as pd
import numpy as np
import tempfile, os
from unittest.mock import patch, MagicMock
from src.cache import init_db
from src.prices import compute_price_factors, apply_liquidity_gate

def make_ohlcv(n=252, start_price=50.0):
    prices = np.linspace(start_price, start_price * 1.1, n)
    idx = pd.date_range(end="2024-01-31", periods=n, freq="B")
    return pd.DataFrame({
        "open": prices * 0.99,
        "high": prices * 1.01,
        "low":  prices * 0.98,
        "close": prices,
        "volume": np.ones(n) * 2_000_000,
    }, index=idx)

def test_compute_price_factors_returns_required_columns():
    df = make_ohlcv(252)
    spy = make_ohlcv(252, start_price=450.0)
    result = compute_price_factors("AAPL", df, spy, market_cap=5e11)
    for col in ["mom_12_1", "rs_6m", "pct_from_high", "avg_dollar_vol_20d", "market_cap", "mom_1m"]:
        assert col in result, f"Missing column: {col}"

def test_compute_price_factors_skips_short_series():
    df = make_ohlcv(100)  # < 252
    spy = make_ohlcv(252, start_price=450.0)
    result = compute_price_factors("AAPL", df, spy, market_cap=5e11)
    assert result is None

def test_apply_liquidity_gate():
    rows = [
        {"ticker": "A", "market_cap": 400e6, "avg_dollar_vol_20d": 10e6},
        {"ticker": "B", "market_cap": 200e6, "avg_dollar_vol_20d": 10e6},  # mcap fail
        {"ticker": "C", "market_cap": 400e6, "avg_dollar_vol_20d": 2e6},   # vol fail
    ]
    df = pd.DataFrame(rows)
    cfg = {"liquidity_gate": {"min_market_cap": 300e6, "min_avg_dollar_vol_20d": 5e6}}
    result = apply_liquidity_gate(df, cfg)
    assert list(result["ticker"]) == ["A"]

def test_compute_price_factors_has_atr_and_true_sma200():
    df = make_ohlcv(300)
    spy = make_ohlcv(300, start_price=450.0)
    result = compute_price_factors("AAPL", df, spy)
    assert result["market_cap"] is None            # attached later from Finnhub
    assert result["atr_14"] > 0
    assert abs(result["sma_200"] - float(df["close"].iloc[-200:].mean())) < 1e-9


# ── batch fetch: exceptions split and retry, never quarantine ─────────────────
from src.prices import fetch_with_split, BatchFetchError, apply_adv_gate


def _scripted_fetch(fail_if):
    """fetch(tickers) raises BatchFetchError when fail_if(tickers) is True,
    otherwise returns a frame for every ticker except those named 'DEAD*'."""
    calls = []
    def fetch(tickers):
        calls.append(list(tickers))
        if fail_if(tickers):
            raise BatchFetchError("boom")
        return {t: pd.DataFrame({"close": [1.0]}) for t in tickers if not t.startswith("DEAD")}
    fetch.calls = calls
    return fetch


def test_fetch_with_split_clean_batch_reports_no_data_only_for_missing():
    fetch = _scripted_fetch(lambda ts: False)
    got, no_data, unfetched = fetch_with_split(["A", "DEAD1", "B"], fetch=fetch, min_batch=1, sleep=lambda s: None)
    assert set(got) == {"A", "B"} and no_data == ["DEAD1"] and unfetched == []


def test_fetch_with_split_halves_on_exception_and_recovers():
    """A batch of 8 raises; each half of 4 succeeds. Nothing is quarantined."""
    fetch = _scripted_fetch(lambda ts: len(ts) > 4)
    tickers = [f"T{i}" for i in range(8)]
    got, no_data, unfetched = fetch_with_split(tickers, fetch=fetch, min_batch=2, sleep=lambda s: None)
    assert set(got) == set(tickers) and no_data == [] and unfetched == []
    assert [len(c) for c in fetch.calls] == [8, 4, 4]


def test_fetch_with_split_gives_up_below_min_batch_without_quarantine():
    """Persistent failure: the tickers come back as unfetched, NOT as no_data.
    This is the 2026-08-24 bug — 4,344 names were marked dead for 30 days."""
    fetch = _scripted_fetch(lambda ts: True)
    tickers = [f"T{i}" for i in range(8)]
    got, no_data, unfetched = fetch_with_split(tickers, fetch=fetch, min_batch=4, sleep=lambda s: None)
    assert got == {} and no_data == []
    assert sorted(unfetched) == sorted(tickers)


def test_fetch_with_split_mixed_outcome():
    """Left half fails persistently, right half succeeds: only the right half is
    fetched, the left half is unfetched, and a dead ticker on the right is no_data."""
    def fail_if(ts):
        return any(t.startswith("L") for t in ts)
    fetch = _scripted_fetch(fail_if)
    tickers = ["L1", "L2", "R1", "DEADR"]
    got, no_data, unfetched = fetch_with_split(tickers, fetch=fetch, min_batch=2, sleep=lambda s: None)
    assert set(got) == {"R1"} and no_data == ["DEADR"] and sorted(unfetched) == ["L1", "L2"]


def test_fetch_with_split_sleeps_between_retries():
    slept = []
    fetch = _scripted_fetch(lambda ts: len(ts) > 2)
    fetch_with_split(["A", "B", "C", "D"], fetch=fetch, min_batch=1, sleep=slept.append)
    assert slept and all(s > 0 for s in slept)


# ── a stalled download must not hang the run forever ──────────────────────────
import time as _time
from src import prices as _prices_mod
from src.prices import _fetch_batch_yfinance


def test_fetch_batch_yfinance_converts_a_stall_into_batch_fetch_error():
    """yf.download() has no timeout of its own — a connection that never
    responds must not hang the run forever. A download that outlives the
    deadline must raise BatchFetchError, same as any other batch failure, so
    fetch_with_split's halving-retry can handle it."""
    def _hangs(*a, **k):
        _time.sleep(1.0)
        return pd.DataFrame({"close": [1.0]})
    with patch.object(_prices_mod, "DOWNLOAD_TIMEOUT_SEC", 0.05), \
         patch.object(_prices_mod.yf, "download", side_effect=_hangs):
        try:
            _fetch_batch_yfinance(["A", "B"])
            assert False, "expected BatchFetchError"
        except BatchFetchError as e:
            assert "timed out" in str(e)


def test_fetch_batch_yfinance_returns_normally_within_the_deadline():
    def _fast(*a, **k):
        return pd.DataFrame({"close": [1.0]})
    with patch.object(_prices_mod, "DOWNLOAD_TIMEOUT_SEC", 5), \
         patch.object(_prices_mod.yf, "download", side_effect=_fast):
        result = _fetch_batch_yfinance(["A"])
    assert "A" in result


def test_apply_adv_gate_uses_volume_and_min_price():
    df = pd.DataFrame({
        "ticker": ["A", "B", "C"],
        "avg_dollar_vol_20d": [10e6, 2e6, 10e6],
        "price": [50.0, 50.0, 3.0],
    })
    cfg = {"liquidity_gate": {"min_market_cap": 300e6, "min_avg_dollar_vol_20d": 5e6},
           "universe": {"min_price": 5.0}}
    assert list(apply_adv_gate(df, cfg)["ticker"]) == ["A"]


def test_apply_adv_gate_empty_frame():
    cfg = {"liquidity_gate": {"min_market_cap": 300e6, "min_avg_dollar_vol_20d": 5e6}, "universe": {}}
    assert len(apply_adv_gate(pd.DataFrame(), cfg)) == 0


def test_apply_liquidity_gate_nan_cap_fails():
    df = pd.DataFrame({"ticker": ["A", "B"], "market_cap": [400e6, float("nan")],
                       "avg_dollar_vol_20d": [10e6, 10e6]})
    cfg = {"liquidity_gate": {"min_market_cap": 300e6, "min_avg_dollar_vol_20d": 5e6}}
    assert list(apply_liquidity_gate(df, cfg)["ticker"]) == ["A"]


def test_apply_liquidity_gate_without_cap_column_returns_empty():
    df = pd.DataFrame({"ticker": ["A"], "avg_dollar_vol_20d": [10e6]})
    cfg = {"liquidity_gate": {"min_market_cap": 300e6, "min_avg_dollar_vol_20d": 5e6}}
    assert len(apply_liquidity_gate(df, cfg)) == 0


# ── Alpaca bars: yfinance's alternative, added after a 2026-09-05 Yahoo rate limit ─
from src.prices import _fetch_batch_alpaca, ALPACA_PAGE_LIMIT


def _bar(day, price=100.0):
    return {"t": f"2026-01-{day:02d}T04:00:00Z", "o": price, "h": price + 1,
            "l": price - 1, "c": price, "v": 1000}


def test_fetch_batch_alpaca_maps_dash_tickers_to_alpaca_dot_symbols():
    """The env's own env vars are used; the request must ask Alpaca for the
    dot-notation symbol but the result must key off the ORIGINAL ticker."""
    seen_params = {}
    def fake_get(url, headers, params, timeout):
        seen_params.update(params)
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"bars": {"BRK.B": [_bar(2)]}, "next_page_token": None}
        return resp
    with patch.dict(os.environ, {"ALPACA_API_KEY": "k", "ALPACA_SECRET_KEY": "s"}), \
         patch("src.prices.requests.get", side_effect=fake_get):
        result = _fetch_batch_alpaca(["BRK-B"], start="2026-01-01", end="2026-01-31")
    assert "BRK.B" in seen_params["symbols"]
    assert list(result.keys()) == ["BRK-B"]
    assert list(result["BRK-B"].columns) == ["open", "high", "low", "close", "volume"]


def test_fetch_batch_alpaca_pages_until_next_page_token_is_none():
    calls = []
    def fake_get(url, headers, params, timeout):
        calls.append(params.get("page_token"))
        resp = MagicMock()
        resp.status_code = 200
        if params.get("page_token") is None:
            resp.json.return_value = {"bars": {"AAPL": [_bar(2)]}, "next_page_token": "tok2"}
        else:
            resp.json.return_value = {"bars": {"AAPL": [_bar(3)]}, "next_page_token": None}
        return resp
    with patch.dict(os.environ, {"ALPACA_API_KEY": "k", "ALPACA_SECRET_KEY": "s"}), \
         patch("src.prices.requests.get", side_effect=fake_get):
        result = _fetch_batch_alpaca(["AAPL"], start="2026-01-01", end="2026-01-31")
    assert len(calls) == 2 and calls[0] is None and calls[1] == "tok2"
    assert len(result["AAPL"]) == 2   # bars from both pages accumulated


def test_fetch_batch_alpaca_429_raises_batch_fetch_error():
    def fake_get(url, headers, params, timeout):
        resp = MagicMock()
        resp.status_code = 429
        resp.text = "Too many requests"
        return resp
    with patch.dict(os.environ, {"ALPACA_API_KEY": "k", "ALPACA_SECRET_KEY": "s"}), \
         patch("src.prices.requests.get", side_effect=fake_get):
        try:
            _fetch_batch_alpaca(["AAPL"], start="2026-01-01", end="2026-01-31")
            assert False, "expected BatchFetchError"
        except BatchFetchError as e:
            assert "429" in str(e) or "rate limited" in str(e)


def test_fetch_batch_alpaca_missing_credentials_raises_batch_fetch_error():
    with patch.dict(os.environ, {}, clear=True):
        try:
            _fetch_batch_alpaca(["AAPL"], start="2026-01-01", end="2026-01-31")
            assert False, "expected BatchFetchError"
        except BatchFetchError as e:
            assert "ALPACA" in str(e)


def test_fetch_batch_alpaca_unknown_symbol_is_just_absent_not_an_error():
    def fake_get(url, headers, params, timeout):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"bars": {"AAPL": [_bar(2)]}, "next_page_token": None}
        return resp
    with patch.dict(os.environ, {"ALPACA_API_KEY": "k", "ALPACA_SECRET_KEY": "s"}), \
         patch("src.prices.requests.get", side_effect=fake_get):
        result = _fetch_batch_alpaca(["AAPL", "NOTAREALTICKERXYZ"], start="2026-01-01", end="2026-01-31")
    assert set(result.keys()) == {"AAPL"}


def test_fetch_batch_alpaca_stall_converts_to_batch_fetch_error():
    import time as _time
    def _hangs(*a, **k):
        _time.sleep(1.0)
        raise AssertionError("should never complete before the deadline fires")
    with patch.dict(os.environ, {"ALPACA_API_KEY": "k", "ALPACA_SECRET_KEY": "s"}), \
         patch("src.prices.ALPACA_BAR_REQUEST_TIMEOUT_SEC", 0.05), \
         patch("src.prices.requests.get", side_effect=_hangs):
        try:
            _fetch_batch_alpaca(["AAPL"], start="2026-01-01", end="2026-01-31")
            assert False, "expected BatchFetchError"
        except BatchFetchError as e:
            assert "timed out" in str(e)


def test_fetch_with_split_defaults_to_alpaca_now():
    from src.prices import fetch_with_split
    import inspect
    assert inspect.signature(fetch_with_split).parameters["fetch"].default is _fetch_batch_alpaca
