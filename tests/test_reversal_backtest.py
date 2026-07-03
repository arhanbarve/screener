import pandas as pd

from src.reversal_backtest import cap_per_day, detect_dislocations


def _mk_prices(closes, start="2024-01-02"):
    idx = pd.bdate_range(start, periods=len(closes))
    return pd.DataFrame({"close": closes}, index=idx)


def _alternating(base, n, step=0.005):
    """Deterministic +/-0.5% alternation — gives nonzero residual vol
    without randomness."""
    out, p = [], base
    for i in range(n):
        p = p * (1 + step if i % 2 == 0 else 1 - step)
        out.append(p)
    return out


class TestDetectDislocations:
    def test_big_idio_drop_triggers_drop_event(self):
        # 150 quiet days, then five straight -2.5% days (~ -11.9% 5d resid)
        closes = _alternating(100.0, 150)
        for _ in range(5):
            closes.append(closes[-1] * 0.975)
        closes += _alternating(closes[-1], 20)
        prices = {"AAA": _mk_prices(closes)}
        bench = _mk_prices([100.0] * len(closes))  # flat benchmark
        start, end = prices["AAA"].index[0], prices["AAA"].index[-1]
        ev = detect_dislocations(prices, bench, start, end)
        drops = ev[ev["direction"] == "drop"]
        assert not drops.empty
        assert set(drops["ticker"]) == {"AAA"}
        assert (drops["resid_5d"] <= -0.07).all()
        assert (drops["z"] <= -2.5).all()

    def test_small_move_does_not_trigger(self):
        closes = _alternating(100.0, 150)
        for _ in range(5):
            closes.append(closes[-1] * 0.994)  # only ~ -3% over 5d
        prices = {"AAA": _mk_prices(closes)}
        bench = _mk_prices([100.0] * len(closes))
        ev = detect_dislocations(prices, bench,
                                 prices["AAA"].index[0], prices["AAA"].index[-1])
        assert ev[ev["direction"] == "drop"].empty

    def test_big_spike_triggers_spike_event(self):
        closes = _alternating(100.0, 150)
        for _ in range(5):
            closes.append(closes[-1] * 1.026)
        closes += _alternating(closes[-1], 20)
        prices = {"AAA": _mk_prices(closes)}
        bench = _mk_prices([100.0] * len(closes))
        ev = detect_dislocations(prices, bench,
                                 prices["AAA"].index[0], prices["AAA"].index[-1])
        assert not ev[ev["direction"] == "spike"].empty


class TestCapPerDay:
    def test_keeps_most_extreme_z_per_day(self):
        d = pd.Timestamp("2025-03-10")
        events = pd.DataFrame({
            "ticker": [f"T{i}" for i in range(7)],
            "trigger_date": [d] * 7,
            "direction": ["drop"] * 7,
            "z": [-2.6, -2.7, -2.8, -2.9, -3.0, -3.1, -3.2],
            "resid_5d": [-0.08] * 7,
        })
        kept = cap_per_day(events, max_per_day=5)
        assert len(kept) == 5
        assert set(kept["z"]) == {-2.8, -2.9, -3.0, -3.1, -3.2}

    def test_directions_capped_independently(self):
        d = pd.Timestamp("2025-03-10")
        events = pd.DataFrame({
            "ticker": [f"T{i}" for i in range(6)],
            "trigger_date": [d] * 6,
            "direction": ["drop"] * 3 + ["spike"] * 3,
            "z": [-3.0, -2.9, -2.8, 3.0, 2.9, 2.8],
            "resid_5d": [-0.08] * 3 + [0.08] * 3,
        })
        kept = cap_per_day(events, max_per_day=5)
        assert len(kept) == 6
