import pandas as pd

from src.leadlag_backtest import detect_leadlag


def _mk_prices(closes, start="2024-01-02"):
    idx = pd.bdate_range(start, periods=len(closes))
    return pd.DataFrame({"close": closes}, index=idx)


def _flat_then_jump(n_flat, jump, n_after=10, base=100.0):
    closes = [base] * n_flat
    closes.append(closes[-1] * (1 + jump))
    closes += [closes[-1]] * n_after
    return closes


class TestDetectLeadlag:
    def _fixture(self, jump):
        n_flat, n_after = 30, 10
        prices = {
            "LEAD": _mk_prices(_flat_then_jump(n_flat, jump, n_after)),
            "LAG1": _mk_prices([100.0] * (n_flat + 1 + n_after)),  # cap 8B
            "LAG2": _mk_prices([100.0] * (n_flat + 1 + n_after)),  # cap 4B
            "OTHER": _mk_prices([100.0] * (n_flat + 1 + n_after)),  # other industry
        }
        bench = _mk_prices([100.0] * (n_flat + 1 + n_after))
        industry = {"LEAD": "Semis", "LAG1": "Semis", "LAG2": "Semis",
                    "OTHER": "Banks"}
        caps = {"LEAD": 20e9, "LAG1": 8e9, "LAG2": 4e9, "OTHER": 9e9}
        start = prices["LEAD"].index[0]
        end = prices["LEAD"].index[-1]
        return detect_leadlag(prices, bench, industry, caps, start, end)

    def test_up_leader_selects_largest_flat_peer(self):
        ev = self._fixture(jump=0.09)
        up = ev[ev["direction"] == "up"]
        assert len(up) == 1
        assert up["ticker"].iloc[0] == "LAG1"       # largest qualifying peer
        assert up["leader"].iloc[0] == "LEAD"

    def test_down_leader_makes_control_event(self):
        ev = self._fixture(jump=-0.09)
        down = ev[ev["direction"] == "down"]
        assert len(down) == 1
        assert down["ticker"].iloc[0] == "LAG1"

    def test_small_leader_move_no_event(self):
        ev = self._fixture(jump=0.04)
        assert ev.empty
