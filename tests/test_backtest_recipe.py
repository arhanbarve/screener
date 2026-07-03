import pandas as pd

from src.backtest_recipe import (
    dedupe_events,
    drop_earnings_contamination,
    filter_to_universe,
)


def _events(rows):
    df = pd.DataFrame(rows)
    df["file_date"] = pd.to_datetime(df["file_date"])
    return df


def test_filter_to_universe_drops_non_universe_ciks():
    events = pd.DataFrame([{"cik": "0000000001"}, {"cik": "0000000002"}])
    uni = pd.DataFrame([{"cik": "0000000001", "ticker": "AAA"}])
    out = filter_to_universe(events, uni)
    assert list(out["ticker"]) == ["AAA"]


def test_dedupe_keeps_first_within_window():
    events = _events([
        {"ticker": "AAA", "file_date": "2026-01-05"},
        {"ticker": "AAA", "file_date": "2026-01-15"},  # 10d gap, dropped
    ])
    out = dedupe_events(events, dedupe_days=20)
    assert len(out) == 1
    assert out.iloc[0]["file_date"] == pd.Timestamp("2026-01-05")


def test_dedupe_keeps_events_past_window():
    events = _events([
        {"ticker": "AAA", "file_date": "2026-01-05"},
        {"ticker": "AAA", "file_date": "2026-02-10"},  # 36d gap, kept
    ])
    out = dedupe_events(events, dedupe_days=20)
    assert len(out) == 2


def test_dedupe_is_per_ticker():
    events = _events([
        {"ticker": "AAA", "file_date": "2026-01-05"},
        {"ticker": "BBB", "file_date": "2026-01-06"},
    ])
    out = dedupe_events(events, dedupe_days=20)
    assert len(out) == 2


def test_drop_earnings_contamination_drops_within_window():
    events = pd.DataFrame([
        {"ticker": "AAA", "event_date": "2026-01-10"},   # 2d from earnings
        {"ticker": "AAA", "event_date": "2026-06-01"},   # far from earnings
    ])
    earnings = pd.DataFrame([{"ticker": "AAA", "announce_date": "2026-01-08"}])
    out = drop_earnings_contamination(events, earnings, window_days=3)
    assert list(out["event_date"]) == ["2026-06-01"]


def test_drop_earnings_contamination_no_earnings_data_keeps_event():
    events = pd.DataFrame([{"ticker": "ZZZ", "event_date": "2026-01-10"}])
    earnings = pd.DataFrame([{"ticker": "AAA", "announce_date": "2026-01-08"}])
    out = drop_earnings_contamination(events, earnings, window_days=3)
    assert len(out) == 1


from src.backtest_recipe import (
    attach_cap_proxy,
    filter_cap_proxy,
    split_by_earnings,
    split_by_news,
)


def _ts(s):
    return pd.Timestamp(s)


class TestSplitByEarnings:
    def test_splits_contaminated_from_clean(self):
        events = pd.DataFrame({
            "ticker": ["AAA", "BBB"],
            "event_date": ["2025-03-10", "2025-03-10"],
        })
        earnings = pd.DataFrame({
            "ticker": ["AAA"],
            "announce_date": ["2025-03-09"],  # within +/-3d of AAA event
        })
        clean, contaminated = split_by_earnings(events, earnings, window_days=3)
        assert list(clean["ticker"]) == ["BBB"]
        assert list(contaminated["ticker"]) == ["AAA"]

    def test_empty_earnings_means_all_clean(self):
        events = pd.DataFrame({"ticker": ["AAA"], "event_date": ["2025-03-10"]})
        earnings = pd.DataFrame(columns=["ticker", "announce_date"])
        clean, contaminated = split_by_earnings(events, earnings)
        assert len(clean) == 1 and len(contaminated) == 0


class TestSplitByNews:
    def test_filing_inside_window_is_news(self):
        events = pd.DataFrame({
            "ticker": ["AAA", "BBB", "CCC"],
            "trigger_date": [_ts("2025-03-10"), _ts("2025-03-10"), _ts("2025-03-10")],
        })
        filings = {
            "AAA": [_ts("2025-03-08")],   # 2d before -> news
            "BBB": [_ts("2025-03-11")],   # 1d after -> news (days_after=1)
            "CCC": [_ts("2025-03-01")],   # far before -> clean
        }
        clean, news = split_by_news(events, filings, days_before=3, days_after=1,
                                    date_col="trigger_date")
        assert sorted(news["ticker"]) == ["AAA", "BBB"]
        assert list(clean["ticker"]) == ["CCC"]

    def test_ticker_with_no_filings_is_clean(self):
        events = pd.DataFrame({"ticker": ["ZZZ"], "trigger_date": [_ts("2025-03-10")]})
        clean, news = split_by_news(events, {}, date_col="trigger_date")
        assert len(clean) == 1 and len(news) == 0


class TestCapProxy:
    def test_cap_proxy_scales_current_cap_by_price_ratio(self):
        idx = pd.to_datetime(["2025-01-02", "2026-07-01"])
        prices = {"AAA": pd.DataFrame({"close": [50.0, 100.0]}, index=idx)}
        uni = pd.DataFrame({"ticker": ["AAA"], "market_cap": [10e9]})
        events = pd.DataFrame({"ticker": ["AAA"], "event_date": ["2025-01-02"]})
        out = attach_cap_proxy(events, uni, prices)
        # cap_now 10B x (50 / 100) = 5B
        assert abs(out["cap_proxy"].iloc[0] - 5e9) < 1e6

    def test_filter_drops_below_min_and_missing(self):
        events = pd.DataFrame({
            "ticker": ["AAA", "BBB"],
            "event_date": ["2025-01-02", "2025-01-02"],
            "cap_proxy": [5e9, 1e9],
        })
        kept = filter_cap_proxy(events, min_cap=2.5e9)
        assert list(kept["ticker"]) == ["AAA"]

    def test_missing_price_gives_nan_proxy(self):
        events = pd.DataFrame({"ticker": ["NOPE"], "event_date": ["2025-01-02"]})
        uni = pd.DataFrame({"ticker": ["NOPE"], "market_cap": [10e9]})
        out = attach_cap_proxy(events, uni, {})
        assert pd.isna(out["cap_proxy"].iloc[0])
