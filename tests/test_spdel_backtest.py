import pandas as pd

from src.spdel_backtest import is_ma_reason, parse_changes


def _raw(rows):
    """Build a frame shaped like Wikipedia's changes table (MultiIndex cols)."""
    cols = pd.MultiIndex.from_tuples([
        ("Date", "Date"), ("Added", "Ticker"), ("Added", "Security"),
        ("Removed", "Ticker"), ("Removed", "Security"), ("Reason", "Reason"),
    ])
    return pd.DataFrame(rows, columns=cols)


class TestIsMaReason:
    def test_acquisition_is_ma(self):
        assert is_ma_reason("Acquired by Broadcom.")
        assert is_ma_reason("Merged with XYZ Corp")
        assert is_ma_reason("Taken private by KKR")
        assert is_ma_reason("Chapter 11 bankruptcy")

    def test_market_cap_change_is_not_ma(self):
        assert not is_ma_reason("Market capitalization change.")
        assert not is_ma_reason("No longer representative of large-cap space")


class TestParseChanges:
    def test_splits_deletions_and_additions(self):
        raw = _raw([
            ["June 10, 2025", "NEWCO", "New Co", "OLDCO", "Old Co",
             "Market capitalization change."],
            ["May 5, 2025", "BUYER", "Buyer Inc", "TARGET", "Target Inc",
             "Acquired by Buyer Inc."],
        ])
        out = parse_changes(raw, start="2025-01-01", end="2025-12-31")
        dels = out[out["category"] == "sp500_deletion"]
        adds = out[out["category"] == "sp500_addition"]
        # TARGET removed for M&A -> excluded from deletions
        assert list(dels["ticker"]) == ["OLDCO"]
        # both additions kept
        assert sorted(adds["ticker"]) == ["BUYER", "NEWCO"]
        # event_date = effective date + 1 calendar day
        assert dels["event_date"].iloc[0] == "2025-06-11"

    def test_window_filter(self):
        raw = _raw([
            ["June 10, 2019", "A", "A Co", "B", "B Co", "Market capitalization change."],
        ])
        out = parse_changes(raw, start="2025-01-01", end="2025-12-31")
        assert out.empty
