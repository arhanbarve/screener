# tests/test_sizing.py
import numpy as np
import pandas as pd
import pytest

from src.sizing import realized_vol, inverse_vol_weights, apply_caps


def _series_with_vol(daily_sigma: float, n: int = 300) -> pd.Series:
    """Deterministic close series whose daily log-returns have a known std.
    Alternating +/- daily_sigma gives sample std == daily_sigma exactly."""
    rets = np.array([daily_sigma if i % 2 == 0 else -daily_sigma for i in range(n)])
    close = 100.0 * np.exp(np.cumsum(rets))
    idx = pd.date_range(end="2024-06-28", periods=n, freq="B")
    return pd.Series(close, index=idx)


# ---- realized_vol ---------------------------------------------------------

def test_realized_vol_known_value():
    # Alternating +/-sigma returns: sample std (ddof=1) differs from sigma by
    # the sqrt(n/(n-1)) factor, so a ~1% tolerance is the honest bound here.
    close = _series_with_vol(0.01, n=300)
    got = realized_vol(close, window=63)
    assert got == pytest.approx(0.01 * np.sqrt(252), rel=1e-2)


def test_realized_vol_short_series_returns_nan():
    close = _series_with_vol(0.01, n=30)
    assert np.isnan(realized_vol(close, window=63))


def test_realized_vol_higher_vol_series_scores_higher():
    lo = realized_vol(_series_with_vol(0.005), window=63)
    hi = realized_vol(_series_with_vol(0.02), window=63)
    assert hi > lo


# ---- inverse_vol_weights --------------------------------------------------

def test_inverse_vol_weights_sum_to_one():
    w = inverse_vol_weights({"A": 0.2, "B": 0.4, "C": 0.6})
    assert sum(w.values()) == pytest.approx(1.0)


def test_inverse_vol_weights_higher_vol_gets_less():
    w = inverse_vol_weights({"LOW": 0.1, "HIGH": 0.5})
    assert w["LOW"] > w["HIGH"]


def test_inverse_vol_weights_drops_nan_and_nonpositive():
    w = inverse_vol_weights({"A": 0.2, "B": float("nan"), "C": 0.0, "D": -0.1})
    assert set(w.keys()) == {"A"}
    assert w["A"] == pytest.approx(1.0)


def test_inverse_vol_weights_empty_input():
    assert inverse_vol_weights({}) == {}


# ---- apply_caps -----------------------------------------------------------

def test_apply_caps_compliant_input_unchanged():
    weights = {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}
    sectors = {"A": "Tech", "B": "Health", "C": "Energy", "D": "Utilities"}
    out = apply_caps(weights, sectors, name_cap=0.30, sector_cap=0.50)
    for k in weights:
        assert out[k] == pytest.approx(weights[k])
    assert sum(out.values()) == pytest.approx(1.0)


def test_apply_caps_single_name_over_cap_clipped_and_redistributed():
    weights = {"BIG": 0.70, "B": 0.10, "C": 0.10, "D": 0.10}
    sectors = {"BIG": "Tech", "B": "Health", "C": "Energy", "D": "Utilities"}
    out = apply_caps(weights, sectors, name_cap=0.30, sector_cap=0.90)
    assert out["BIG"] == pytest.approx(0.30, abs=1e-9)
    assert out["BIG"] <= 0.30 + 1e-9
    assert sum(out.values()) == pytest.approx(1.0)
    # freed 0.40 spread pro-rata across the three equal others
    assert out["B"] == pytest.approx(0.70 / 3, rel=1e-6)


def test_apply_caps_sector_over_cap_scaled_down():
    # Tech = 0.80; cap to 0.50. Non-tech capacity (2 x 0.40) leaves room,
    # so this is feasible (a 0.25 sector cap here would not be — only two
    # non-tech names at 0.40 max = 0.60, +0.25 tech < 1.0).
    weights = {"T1": 0.40, "T2": 0.40, "H1": 0.10, "E1": 0.10}
    sectors = {"T1": "Tech", "T2": "Tech", "H1": "Health", "E1": "Energy"}
    out = apply_caps(weights, sectors, name_cap=0.40, sector_cap=0.50)
    tech = out["T1"] + out["T2"]
    assert tech <= 0.50 + 1e-9
    assert sum(out.values()) == pytest.approx(1.0)
    # weight moved out of Tech into the other sectors
    assert out["H1"] + out["E1"] > 0.20


def test_apply_caps_both_constraints_hold_at_convergence():
    weights = {"T1": 0.50, "T2": 0.20, "H1": 0.20, "E1": 0.10}
    sectors = {"T1": "Tech", "T2": "Tech", "H1": "Health", "E1": "Energy"}
    out = apply_caps(weights, sectors, name_cap=0.30, sector_cap=0.40)
    assert max(out.values()) <= 0.30 + 1e-9
    sector_tot = {}
    for t, w in out.items():
        sector_tot[sectors[t]] = sector_tot.get(sectors[t], 0.0) + w
    assert max(sector_tot.values()) <= 0.40 + 1e-9
    assert sum(out.values()) == pytest.approx(1.0)


def test_apply_caps_infeasible_single_sector_best_effort_no_crash():
    # All one sector: sector cap of 0.25 is impossible (they must sum to 1.0).
    weights = {"T1": 0.5, "T2": 0.3, "T3": 0.2}
    sectors = {"T1": "Tech", "T2": "Tech", "T3": "Tech"}
    out = apply_caps(weights, sectors, name_cap=0.40, sector_cap=0.25)
    # no crash; still a valid distribution; name cap still respected
    assert sum(out.values()) == pytest.approx(1.0)
    assert max(out.values()) <= 0.40 + 1e-9
