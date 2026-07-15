# Walk-Forward Harness + Regime Ladder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a portfolio-level walk-forward backtest of the price-block composite (weekly band rebalance, costs) plus a 4-signal graded regime exposure ladder with breadth-thrust re-entry, per `docs/superpowers/specs/2026-07-15-walkforward-harness-regime-ladder.md`.

**Architecture:** Two-stage. Stage A (`src/factor_panel.py`) fetches max-history prices (reusing `event_backtest`'s cached fetch layer), computes point-in-time price factors + gates + daily breadth series, and saves parquet artifacts. Stage B (`src/portfolio_sim.py`) is a fast simulator over the cached panel — band portfolio, costs, exposure overlay, baselines, metrics, report, verdict. `src/regime.py` holds the ladder/thrust signal logic, shared later by the live screener.

**Tech Stack:** Python 3, pandas/numpy/scipy, yfinance (via existing `src/prices.py` + `src/event_backtest.py` helpers), sqlite cache (`src/cache.py`), pytest. No new dependencies.

**Spec amendments locked here:**
1. Reuse `event_backtest_prices` table + `get_history_bulk` instead of a new `portfolio_backtest_prices` table (spec's Part-1 data layer) — same payload shape, already covers arbitrary start/end ranges. DRY.
2. De-risking trades **pro-rata across all positions** (spec's "lowest-rank first" wording was internally inconsistent with "scalar on total equity"; pro-rata matches the scalar model and is deterministic).
3. Breadth series computed over all candidate tickers with ≥200 bars of data at t (not gate survivors — gates depend on breadth-free factors anyway; documented in module docstring).

**Conventions for every task:** run tests with `python3 -m pytest <file> -v` from repo root. All test data is synthetic — no network in tests. Commit after each task's tests pass.

---

### Task 1: Extract monthly-core from `residual_momentum`

`factors.residual_momentum` resamples daily→monthly internally. The panel needs to resample **once per ticker** then evaluate at ~520 dates, so the regression core must be callable on pre-resampled monthly series. Surgical refactor, no behavior change.

**Files:**
- Modify: `src/factors.py:61-83` (`residual_momentum`)
- Test: `tests/test_factors.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_factors.py`:

```python
def test_residual_momentum_from_monthly_matches_wrapper():
    """Core on pre-resampled monthly data == wrapper on daily data."""
    import numpy as np
    import pandas as pd
    from src.factors import residual_momentum, residual_momentum_from_monthly

    rng = np.random.default_rng(42)
    idx = pd.bdate_range("2022-01-03", periods=420)
    mkt = pd.Series(100 * np.cumprod(1 + rng.normal(0.0004, 0.01, len(idx))), index=idx)
    stk = pd.Series(50 * np.cumprod(1 + rng.normal(0.0006, 0.02, len(idx))), index=idx)

    expected = residual_momentum(stk, mkt)
    monthly_stk = stk.resample("ME").last()
    monthly_mkt = mkt.resample("ME").last()
    got = residual_momentum_from_monthly(monthly_stk, monthly_mkt)
    assert abs(got - expected) < 1e-12


def test_residual_momentum_from_monthly_insufficient_data_nan():
    import numpy as np
    import pandas as pd
    from src.factors import residual_momentum_from_monthly

    idx = pd.date_range("2023-01-31", periods=6, freq="ME")
    s = pd.Series(np.linspace(10, 12, 6), index=idx)
    assert np.isnan(residual_momentum_from_monthly(s, s))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_factors.py -k residual_momentum_from_monthly -v`
Expected: FAIL — `ImportError: cannot import name 'residual_momentum_from_monthly'`

- [ ] **Step 3: Implement the refactor**

In `src/factors.py`, replace the body of `residual_momentum` and add the core function directly below it:

```python
def residual_momentum(close: pd.Series, market_close: pd.Series) -> float:
    """
    Risk-adjusted momentum (Blitz, Huij, Martens 2011).
    Regress monthly stock returns on monthly market returns,
    sum residuals over 12 months (skip last), scale by residual vol.
    Lower crash risk than raw 12-1 momentum.
    """
    monthly_stock = close.resample("ME").last()
    monthly_mkt   = market_close.resample("ME").last()
    return residual_momentum_from_monthly(monthly_stock, monthly_mkt)


def residual_momentum_from_monthly(monthly_stock: pd.Series, monthly_mkt: pd.Series) -> float:
    """Core of residual_momentum, taking already-resampled month-end closes.
    Exists so the backtest factor panel can resample each ticker once and
    evaluate at many as-of dates without paying resample cost per date."""
    r = monthly_stock.pct_change().dropna()
    m = monthly_mkt.pct_change().dropna()
    df = pd.DataFrame({"r": r, "m": m}).dropna()
    if len(df) < 13:
        return float("nan")
    df = df.iloc[-13:-1]  # 12 months, skip most recent
    m_var = float(df["m"].var())
    if m_var < 1e-12:
        return float("nan")
    beta = float(df["r"].cov(df["m"])) / m_var
    residuals = df["r"] - beta * df["m"]
    std = float(residuals.std())
    if std < 1e-12:
        return float("nan")
    return float(residuals.sum() / std)
```

- [ ] **Step 4: Run the full factors test file**

Run: `python3 -m pytest tests/test_factors.py -v`
Expected: all PASS (new tests + every pre-existing test — proves no behavior change).

- [ ] **Step 5: Commit**

```bash
git add src/factors.py tests/test_factors.py
git commit -m "refactor(factors): extract residual_momentum_from_monthly core"
```

---

### Task 2: Regime signals (`src/regime.py`)

Four boolean daily series. NaN-safe: missing data → signal False (no point), so early history with no ^VIX3M/HYG simply contributes 0 points rather than poisoning the ladder.

**Files:**
- Create: `src/regime.py`
- Create: `tests/test_regime.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_regime.py`:

```python
"""Regime ladder signal tests. All series synthetic; index = business days."""
import numpy as np
import pandas as pd
import pytest

from src.regime import (
    trend_signal, breadth_signal, vol_signal, credit_signal,
)


def _bdays(n, start="2020-01-01"):
    return pd.bdate_range(start, periods=n)


def test_trend_signal_fires_below_sma200():
    idx = _bdays(260)
    # 210 flat days at 100 establish SMA200 ~100, then close drops to 90
    vals = np.concatenate([np.full(210, 100.0), np.full(50, 90.0)])
    sig = trend_signal(pd.Series(vals, index=idx))
    assert bool(sig.iloc[-1]) is True
    assert bool(sig.iloc[209]) is False        # at 100, not below SMA
    assert bool(sig.iloc[100]) is False        # SMA200 not formed yet -> False


def test_breadth_signal_thresholds():
    idx = _bdays(3)
    pct = pd.Series([0.55, 0.39, np.nan], index=idx)
    sig = breadth_signal(pct)
    assert list(sig) == [False, True, False]   # NaN -> False


def test_vol_signal_level_or_inversion():
    idx = _bdays(4)
    vix = pd.Series([15.0, 26.0, 18.0, 15.0], index=idx)
    vix3m = pd.Series([17.0, 28.0, 17.0, np.nan], index=idx)
    sig = vol_signal(vix, vix3m)
    # day0: level ok, 15<17 no inversion -> False
    # day1: 26>25 -> True
    # day2: 18>17 inverted -> True
    # day3: level ok, vix3m NaN -> inversion unknown -> False
    assert list(sig) == [False, True, True, False]


def test_credit_signal_ratio_below_sma100():
    idx = _bdays(160)
    hyg = pd.Series(np.concatenate([np.full(120, 80.0), np.full(40, 72.0)]), index=idx)
    ief = pd.Series(np.full(160, 100.0), index=idx)
    sig = credit_signal(hyg, ief)
    assert bool(sig.iloc[-1]) is True          # ratio dropped below its SMA100
    assert bool(sig.iloc[110]) is False        # flat ratio == SMA -> not below
    assert bool(sig.iloc[50]) is False         # SMA100 not formed -> False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_regime.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.regime'`

- [ ] **Step 3: Implement the signals**

Create `src/regime.py`:

```python
"""Regime exposure ladder (spec: docs/superpowers/specs/2026-07-15-walkforward-harness-regime-ladder.md).

Four independent stress signals, each worth 1 point; points map to a target
equity exposure. Shared by the portfolio backtest and (after validation) the
live screener, so thresholds live here as module constants.

All signal functions are NaN-safe: any day where the underlying data is
missing or the lookback window hasn't formed yet contributes False (0 points),
never NaN. Early-history gaps (e.g. ^VIX3M before 2008) therefore weaken the
ladder rather than break it.
"""
import pandas as pd

SPY_SMA_WINDOW = 200
BREADTH_200D_THRESHOLD = 0.40
VIX_LEVEL_THRESHOLD = 25.0
CREDIT_SMA_WINDOW = 100

EXPOSURE_MAP = {0: 1.00, 1: 1.00, 2: 0.66, 3: 0.33, 4: 0.00}

THRUST_LOW = 0.20      # breadth-50d must have been below this...
THRUST_HIGH = 0.55     # ...and cross above this...
THRUST_WINDOW = 10     # ...within this many sessions
THRUST_HOLD = 20       # override lasts this many sessions
THRUST_FLOOR = 0.66    # forced minimum exposure while active


def trend_signal(spy_close: pd.Series) -> pd.Series:
    """SPY close < its SMA200. False until the SMA has formed."""
    sma = spy_close.rolling(SPY_SMA_WINDOW).mean()
    return ((spy_close < sma) & sma.notna()).fillna(False)


def breadth_signal(pct_above_200: pd.Series) -> pd.Series:
    """Fraction of universe above own SMA200 < threshold. NaN -> False."""
    return (pct_above_200 < BREADTH_200D_THRESHOLD).fillna(False)


def vol_signal(vix_close: pd.Series, vix3m_close: pd.Series) -> pd.Series:
    """VIX above absolute threshold, or term structure inverted (VIX > VIX3M)."""
    vix, v3 = vix_close.align(vix3m_close, join="left")
    level = (vix > VIX_LEVEL_THRESHOLD).fillna(False)
    inverted = (vix > v3).fillna(False)
    return level | inverted


def credit_signal(hyg_close: pd.Series, ief_close: pd.Series) -> pd.Series:
    """HYG/IEF ratio below its own SMA100 — credit risk-off leads equities."""
    hyg, ief = hyg_close.align(ief_close, join="inner")
    ratio = hyg / ief
    sma = ratio.rolling(CREDIT_SMA_WINDOW).mean()
    return ((ratio < sma) & sma.notna()).fillna(False)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_regime.py -v`
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add src/regime.py tests/test_regime.py
git commit -m "feat(regime): four daily stress signals for exposure ladder"
```

---

### Task 3: Ladder points → exposure, thrust override, combined exposure

**Files:**
- Modify: `src/regime.py` (append)
- Modify: `tests/test_regime.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_regime.py`:

```python
from src.regime import (
    ladder_points, ladder_exposure, thrust_override, combined_exposure,
    THRUST_FLOOR,
)


def test_ladder_points_sums_aligned_signals():
    idx = _bdays(3)
    a = pd.Series([True, False, True], index=idx)
    b = pd.Series([True, False, False], index=idx)
    c = pd.Series([False, False, True], index=idx)
    d = pd.Series([False, False, True], index=idx)
    pts = ladder_points(a, b, c, d)
    assert list(pts) == [2, 0, 3]


def test_ladder_exposure_mapping():
    idx = _bdays(5)
    pts = pd.Series([0, 1, 2, 3, 4], index=idx)
    exp = ladder_exposure(pts)
    assert list(exp) == [1.00, 1.00, 0.66, 0.33, 0.00]


def test_thrust_fires_on_cross_and_expires():
    # breadth50: 12 days low (0.10), then jumps to 0.60 -> fires for THRUST_HOLD days
    idx = _bdays(60)
    vals = np.concatenate([np.full(12, 0.10), np.full(48, 0.60)])
    active = thrust_override(pd.Series(vals, index=idx))
    assert bool(active.iloc[11]) is False       # before cross
    assert bool(active.iloc[12]) is True        # cross day
    assert bool(active.iloc[12 + 19]) is True   # last day of hold window
    assert bool(active.iloc[12 + 20]) is False  # expired (no re-fire: no dip below LOW)


def test_thrust_does_not_fire_without_prior_low():
    idx = _bdays(30)
    vals = np.concatenate([np.full(15, 0.45), np.full(15, 0.60)])  # never below 0.20
    active = thrust_override(pd.Series(vals, index=idx))
    assert not active.any()


def test_combined_exposure_applies_floor_only_when_thrust_active():
    idx = _bdays(4)
    pts = pd.Series([4, 4, 2, 0], index=idx)
    thrust = pd.Series([True, False, True, False], index=idx)
    exp = combined_exposure(pts, thrust)
    assert list(exp) == [THRUST_FLOOR, 0.00, 0.66, 1.00]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_regime.py -v`
Expected: new tests FAIL — `ImportError: cannot import name 'ladder_points'`

- [ ] **Step 3: Implement**

Append to `src/regime.py`:

```python
def ladder_points(trend: pd.Series, breadth: pd.Series,
                  vol: pd.Series, credit: pd.Series) -> pd.Series:
    """Sum the four boolean signals into 0-4 daily points (int)."""
    df = pd.concat(
        {"t": trend, "b": breadth, "v": vol, "c": credit}, axis=1
    ).fillna(False)
    return df.sum(axis=1).astype(int)


def ladder_exposure(points: pd.Series) -> pd.Series:
    """Map daily points to target equity exposure per EXPOSURE_MAP."""
    return points.map(EXPOSURE_MAP).astype(float)


def thrust_override(pct_above_50: pd.Series) -> pd.Series:
    """Breadth-thrust re-entry: True while the override is active.

    Fires when pct_above_50 crosses above THRUST_HIGH having been below
    THRUST_LOW within the prior THRUST_WINDOW sessions; stays active for
    THRUST_HOLD sessions from the firing day (re-fires reset the clock).
    """
    vals = pct_above_50.to_numpy()
    active = pd.Series(False, index=pct_above_50.index)
    fire_until = -1
    for i in range(len(vals)):
        crossed = (
            i > 0
            and vals[i] > THRUST_HIGH
            and not (vals[i - 1] > THRUST_HIGH)
        )
        if crossed:
            lo = max(0, i - THRUST_WINDOW)
            window = vals[lo:i]
            if (window < THRUST_LOW).any():
                fire_until = i + THRUST_HOLD - 1
        if i <= fire_until:
            active.iloc[i] = True
    return active


def combined_exposure(points: pd.Series, thrust_active: pd.Series) -> pd.Series:
    """Ladder exposure with the thrust floor applied on active days."""
    exp = ladder_exposure(points)
    floored = exp.clip(lower=THRUST_FLOOR)
    return exp.where(~thrust_active.reindex(exp.index, fill_value=False), floored)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_regime.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/regime.py tests/test_regime.py
git commit -m "feat(regime): exposure ladder mapping and breadth-thrust override"
```

---

### Task 4: Factor panel — rebalance dates and per-ticker factor frame

Vectorized per-ticker computation over the full daily index, evaluated later at rebalance dates. Correctness anchor: values must match the scalar functions in `src/factors.py` on sliced history.

**Files:**
- Create: `src/factor_panel.py`
- Create: `tests/test_factor_panel.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_factor_panel.py`:

```python
"""Factor panel tests — synthetic prices, no network."""
import numpy as np
import pandas as pd
import pytest

from src.factor_panel import rebalance_dates, ticker_factor_frame
from src import factors


def _synthetic_prices(n=600, seed=7, drift=0.0005, vol=0.02, start="2021-01-04"):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    close = pd.Series(100 * np.cumprod(1 + rng.normal(drift, vol, n)), index=idx)
    volume = pd.Series(rng.integers(200_000, 2_000_000, n).astype(float), index=idx)
    return pd.DataFrame({"close": close, "volume": volume})


def test_rebalance_dates_last_trading_day_of_week():
    idx = pd.bdate_range("2024-01-01", "2024-01-31")
    dates = rebalance_dates(idx)
    # January 2024: Fridays are 5, 12, 19, 26; month ends Wed 31st
    got = [d.strftime("%Y-%m-%d") for d in dates]
    assert got == ["2024-01-05", "2024-01-12", "2024-01-19", "2024-01-26", "2024-01-31"]


def test_ticker_factor_frame_matches_scalar_functions():
    df = _synthetic_prices()
    spy = _synthetic_prices(seed=99, drift=0.0003, vol=0.01)
    dates = rebalance_dates(df.index)[-5:]

    frame = ticker_factor_frame(df, spy["close"], dates)
    assert list(frame.index) == list(dates)

    t = dates[-1]
    hist = df.loc[:t]
    spy_hist = spy.loc[:t, "close"]
    row = frame.loc[t]

    assert row["mom_12_1"] == pytest.approx(factors.mom_12_1(hist["close"]), rel=1e-9)
    assert row["rs_6m"] == pytest.approx(
        factors.rs_vs_spy(hist["close"], spy_hist, window=126), rel=1e-9)
    rs_3m = factors.rs_vs_spy(hist["close"], spy_hist, window=63)
    assert row["rs_accel"] == pytest.approx(2 * rs_3m - row["rs_6m"], rel=1e-9)
    assert row["rs_slope"] == pytest.approx(
        factors.rs_slope(hist["close"], spy_hist, window=63), rel=1e-6)
    assert row["residual_mom"] == pytest.approx(
        factors.residual_momentum(hist["close"], spy_hist), rel=1e-9)
    assert row["pct_from_high"] == pytest.approx(
        factors.pct_from_52w_high(hist["close"]), rel=1e-9)
    assert row["dollar_vol_20d"] == pytest.approx(
        factors.avg_dollar_vol(hist["close"], hist["volume"]), rel=1e-9)
    # gate ingredients
    assert row["above_sma200"] == (hist["close"].iloc[-1] >= hist["close"].rolling(200).mean().iloc[-1])
    assert row["close"] == pytest.approx(float(hist["close"].iloc[-1]))


def test_ticker_factor_frame_nan_when_insufficient_history():
    df = _synthetic_prices(n=100)
    spy = _synthetic_prices(n=100, seed=99)
    dates = rebalance_dates(df.index)[-2:]
    frame = ticker_factor_frame(df, spy["close"], dates)
    assert np.isnan(frame.loc[dates[-1], "mom_12_1"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_factor_panel.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.factor_panel'`

- [ ] **Step 3: Implement**

Create `src/factor_panel.py`:

```python
"""Stage A of the portfolio backtest: point-in-time price-block factor panel.

SURVIVORSHIP BIAS WARNING (carried from the retired backtest/backtest.py):
yfinance contains only currently-listed tickers. Delisted names
(bankruptcies, mergers) are absent, which inflates measured returns.
Treat all results as an upper bound / directional sanity check, NOT a
reliable estimate of live performance. The spec compensates by demanding a
wide pass margin and judging the regime ladder on relative drawdown
reduction vs our own unhedged portfolio.

LOOK-AHEAD NOTE: every factor at date t uses only bars <= t. Market cap at t
is approximated as (cached market cap today) x (close_t / close_today) —
i.e. constant share count. Breadth series use all candidate tickers with
enough data at t, not gate survivors.

Only the price momentum block is computed (mom_12_1, residual_mom, rs_6m,
rs_accel, rs_slope, pct_from_high); fundamental factors cannot be
reconstructed point-in-time from free sources. Weights renormalized to 1.
"""
import argparse
import logging

import numpy as np
import pandas as pd

from src.cache import get_market_cap_stale
from src.event_backtest import get_history, get_history_bulk
from src.factors import residual_momentum_from_monthly

logger = logging.getLogger(__name__)

# config.yaml price-block weights (sum 0.50), renormalized to sum 1.0
PRICE_BLOCK_WEIGHTS = {
    "mom_12_1": 0.24,
    "residual_mom": 0.28,
    "rs_6m": 0.20,
    "rs_accel": 0.12,
    "rs_slope": 0.08,
    "pct_from_high": 0.08,
}

MIN_MCAP_TODAY = 150e6        # candidate floor (half the live $300M gate)
GATE_MIN_MCAP = 300e6         # point-in-time liquidity gate
GATE_MIN_DOLLAR_VOL = 5e6
GATE_MAX_BELOW_HIGH = 0.35    # config.yaml confirmation.max_pct_below_52w_high
WARMUP_DAYS = 400             # calendar days of history before sim start

PANEL_PATH = "output/factor_panel.parquet"
BREADTH_PATH = "output/breadth.parquet"


def rebalance_dates(index: pd.DatetimeIndex) -> list[pd.Timestamp]:
    """Last trading day of each ISO week present in `index`."""
    s = pd.Series(index, index=index)
    iso = index.isocalendar()
    key = list(zip(iso.year, iso.week))
    return list(s.groupby(key).max().sort_values())


def _rolling_slope(y: pd.Series, window: int) -> pd.Series:
    """Least-squares slope of y against x=0..window-1, rolling. Matches
    scipy.stats.linregress slope used by factors.rs_slope."""
    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    x_var_sum = ((x - x_mean) ** 2).sum()

    def _slope(vals: np.ndarray) -> float:
        return float(((x - x_mean) * (vals - vals.mean())).sum() / x_var_sum)

    return y.rolling(window).apply(_slope, raw=True)


def ticker_factor_frame(prices: pd.DataFrame, spy_close: pd.Series,
                        dates: list[pd.Timestamp]) -> pd.DataFrame:
    """All price-block factors + gate ingredients for one ticker, evaluated
    at `dates`. `prices` needs 'close' and 'volume' columns, daily index."""
    close = prices["close"]
    volume = prices["volume"]
    spy = spy_close.reindex(close.index).ffill()

    mom_12_1 = close.shift(21) / close.shift(252) - 1.0
    stock_6m = close / close.shift(126) - 1.0
    spy_6m = spy / spy.shift(126) - 1.0
    rs_6m = stock_6m - spy_6m
    stock_3m = close / close.shift(63) - 1.0
    spy_3m = spy / spy.shift(63) - 1.0
    rs_3m = stock_3m - spy_3m
    rs_accel = 2 * rs_3m - rs_6m
    rs_slope = _rolling_slope(close / spy, 63)
    high_52w = close.rolling(252).max()
    pct_from_high = close / high_52w - 1.0
    dollar_vol = (close * volume).rolling(20).mean()
    sma200 = close.rolling(200).mean()

    # residual momentum: resample once, evaluate per date on month-ends <= t
    monthly_close = close.resample("ME").last()
    monthly_spy = spy.resample("ME").last()

    rows = []
    for t in dates:
        if t not in close.index or len(close.loc[:t]) < 252:
            rows.append({c: np.nan for c in
                         ["mom_12_1", "residual_mom", "rs_6m", "rs_accel",
                          "rs_slope", "pct_from_high", "dollar_vol_20d",
                          "close", "above_sma200"]})
            continue
        res_mom = residual_momentum_from_monthly(
            monthly_close.loc[:t], monthly_spy.loc[:t])
        rows.append({
            "mom_12_1": float(mom_12_1.loc[t]),
            "residual_mom": res_mom,
            "rs_6m": float(rs_6m.loc[t]),
            "rs_accel": float(rs_accel.loc[t]),
            "rs_slope": float(rs_slope.loc[t]),
            "pct_from_high": float(pct_from_high.loc[t]),
            "dollar_vol_20d": float(dollar_vol.loc[t]),
            "close": float(close.loc[t]),
            "above_sma200": bool(close.loc[t] >= sma200.loc[t]),
        })
    return pd.DataFrame(rows, index=pd.DatetimeIndex(dates))
```

Note: `above_sma200` NaN case — `close.rolling(200)` is NaN before 200 bars, but the 252-bar guard above already covers it.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_factor_panel.py -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add src/factor_panel.py tests/test_factor_panel.py
git commit -m "feat(backtest): per-ticker point-in-time factor frame + weekly rebalance dates"
```

---

### Task 5: Panel assembly — gates, composite z-scoring, breadth series

**Files:**
- Modify: `src/factor_panel.py` (append)
- Modify: `tests/test_factor_panel.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_factor_panel.py`:

```python
from src.factor_panel import (
    apply_gates, composite_scores, breadth_series,
    PRICE_BLOCK_WEIGHTS, GATE_MIN_MCAP, GATE_MIN_DOLLAR_VOL, GATE_MAX_BELOW_HIGH,
)


def _panel_row(ticker, date, **overrides):
    row = {
        "ticker": ticker, "date": pd.Timestamp(date), "close": 50.0,
        "mom_12_1": 0.2, "residual_mom": 1.0, "rs_6m": 0.05, "rs_accel": 0.01,
        "rs_slope": 0.001, "pct_from_high": -0.05, "dollar_vol_20d": 10e6,
        "above_sma200": True, "mcap": 1e9,
    }
    row.update(overrides)
    return row


def test_apply_gates():
    d = "2023-06-02"
    panel = pd.DataFrame([
        _panel_row("PASS", d),
        _panel_row("SMLL", d, mcap=100e6),                    # below $300M
        _panel_row("THIN", d, dollar_vol_20d=1e6),            # below $5M ADV
        _panel_row("BELW", d, above_sma200=False),            # below SMA200
        _panel_row("DEEP", d, pct_from_high=-0.50),           # >35% off high
        _panel_row("NODA", d, mom_12_1=np.nan),               # missing factor
    ])
    gated = apply_gates(panel)
    assert list(gated.loc[gated["passes_gates"], "ticker"]) == ["PASS"]


def test_composite_scores_zscore_and_weights():
    d = pd.Timestamp("2023-06-02")
    # three passing tickers, factors constructed so only mom_12_1 differs
    base = dict(residual_mom=1.0, rs_6m=0.05, rs_accel=0.01,
                rs_slope=0.001, pct_from_high=-0.05)
    panel = pd.DataFrame([
        _panel_row("A", d, mom_12_1=0.30, **base),
        _panel_row("B", d, mom_12_1=0.20, **base),
        _panel_row("C", d, mom_12_1=0.10, **base),
    ])
    panel["passes_gates"] = True
    scored = composite_scores(panel)
    by_t = scored.set_index("ticker")
    # identical factors z to 0; mom_12_1 z-scores are +1.09.., 0, -1.09.. (ddof=1... pandas std default ddof=1)
    assert by_t.loc["A", "composite"] > by_t.loc["B", "composite"] > by_t.loc["C", "composite"]
    assert by_t.loc["B", "composite"] == pytest.approx(0.0, abs=1e-12)
    # composite = w_mom * z_mom exactly, since all other z are 0
    z_a = (0.30 - 0.20) / pd.Series([0.30, 0.20, 0.10]).std()
    assert by_t.loc["A", "composite"] == pytest.approx(
        PRICE_BLOCK_WEIGHTS["mom_12_1"] * z_a, rel=1e-9)


def test_breadth_series_fractions():
    idx = pd.bdate_range("2021-01-04", periods=300)
    up = pd.DataFrame({"close": np.linspace(10, 40, 300),
                       "volume": np.full(300, 1e6)}, index=idx)
    down = pd.DataFrame({"close": np.linspace(40, 10, 300),
                         "volume": np.full(300, 1e6)}, index=idx)
    b = breadth_series({"UP": up, "DOWN": down})
    last = b.iloc[-1]
    assert last["pct_above_200"] == pytest.approx(0.5)   # UP above, DOWN below
    assert last["pct_above_50"] == pytest.approx(0.5)
    # before any ticker has 200 bars, the 200d breadth is undefined
    assert np.isnan(b.iloc[100]["pct_above_200"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_factor_panel.py -v`
Expected: new tests FAIL — `ImportError: cannot import name 'apply_gates'`

- [ ] **Step 3: Implement**

Append to `src/factor_panel.py`:

```python
FACTOR_COLS = list(PRICE_BLOCK_WEIGHTS)


def apply_gates(panel: pd.DataFrame) -> pd.DataFrame:
    """Point-in-time liquidity + confirmation gates -> `passes_gates` column.
    Rows with any missing factor fail (price-block factors are all-or-nothing
    from the same OHLCV history, so partial coverage means short history)."""
    p = panel.copy()
    has_factors = p[FACTOR_COLS].notna().all(axis=1)
    p["passes_gates"] = (
        has_factors
        & (p["mcap"] >= GATE_MIN_MCAP)
        & (p["dollar_vol_20d"] >= GATE_MIN_DOLLAR_VOL)
        & p["above_sma200"].fillna(False)
        & (p["pct_from_high"] >= -GATE_MAX_BELOW_HIGH)
    )
    return p


def _winsorize(s: pd.Series, pct: float = 0.01) -> pd.Series:
    if s.notna().sum() < 3:
        return s
    lo, hi = s.quantile(pct), s.quantile(1 - pct)
    return s.clip(lo, hi)


def composite_scores(panel: pd.DataFrame) -> pd.DataFrame:
    """Winsorize + z-score each factor cross-sectionally per date among gate
    survivors; composite = weighted sum. Non-survivors get composite NaN."""
    p = panel.copy()
    p["composite"] = np.nan

    for date, idx in p.groupby("date").groups.items():
        rows = p.loc[idx]
        surv = rows.index[rows["passes_gates"]]
        if len(surv) < 3:
            continue
        comp = pd.Series(0.0, index=surv)
        for col, w in PRICE_BLOCK_WEIGHTS.items():
            vals = _winsorize(p.loc[surv, col])
            std = vals.std()
            if std < 1e-12 or np.isnan(std):
                continue
            comp += w * (vals - vals.mean()) / std
        p.loc[surv, "composite"] = comp
    return p


def breadth_series(prices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Daily fraction of tickers above their own SMA200 / SMA50.
    Denominator = tickers whose SMA window has formed by that day."""
    above200, above50 = [], []
    for t, df in prices.items():
        c = df["close"]
        above200.append((c >= c.rolling(200).mean()).where(c.rolling(200).mean().notna()))
        above50.append((c >= c.rolling(50).mean()).where(c.rolling(50).mean().notna()))
    a200 = pd.concat(above200, axis=1)
    a50 = pd.concat(above50, axis=1)
    return pd.DataFrame({
        "pct_above_200": a200.mean(axis=1),
        "pct_above_50": a50.mean(axis=1),
    })
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_factor_panel.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/factor_panel.py tests/test_factor_panel.py
git commit -m "feat(backtest): gates, cross-sectional composite, breadth series"
```

---

### Task 6: Panel build CLI (fetch → compute → save parquet)

Orchestration only — every piece is already tested. The CLI itself gets a smoke test with monkeypatched fetchers.

**Files:**
- Modify: `src/factor_panel.py` (append)
- Modify: `tests/test_factor_panel.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_factor_panel.py`:

```python
def test_build_panel_end_to_end_synthetic(tmp_path, monkeypatch):
    """Full build with fetchers monkeypatched — proves wiring, not math."""
    import src.factor_panel as fp

    n = 320
    idx = pd.bdate_range("2022-01-03", periods=n)
    rng = np.random.default_rng(3)

    def synth(seed, base):
        r = np.random.default_rng(seed)
        return pd.DataFrame({
            "close": base * np.cumprod(1 + r.normal(0.0005, 0.015, n)),
            "volume": np.full(n, 1_000_000.0),
        }, index=idx)

    tickers = ["AAA", "BBB", "CCC"]
    prices = {t: synth(i, 50 + 10 * i) for i, t in enumerate(tickers)}
    spy = synth(99, 400)

    monkeypatch.setattr(fp, "candidate_tickers", lambda db_path: tickers)
    monkeypatch.setattr(fp, "get_history_bulk",
                        lambda ts, db_path, start, end: {t: prices[t] for t in ts})
    monkeypatch.setattr(fp, "get_history",
                        lambda t, db_path, start, end: spy)
    monkeypatch.setattr(fp, "get_market_cap_stale", lambda db, t: 2e9)

    panel_path = tmp_path / "panel.parquet"
    breadth_path = tmp_path / "breadth.parquet"
    fp.build_panel(db_path="unused.db", start="2022-06-01", end="2023-03-01",
                   panel_path=str(panel_path), breadth_path=str(breadth_path))

    panel = pd.read_parquet(panel_path)
    breadth = pd.read_parquet(breadth_path)
    assert set(panel["ticker"]) == set(tickers)
    assert {"composite", "passes_gates", "mcap", "close"} <= set(panel.columns)
    assert {"pct_above_200", "pct_above_50"} <= set(breadth.columns)
    # only rebalance dates inside [start, end]
    assert panel["date"].min() >= pd.Timestamp("2022-06-01")
    assert panel["date"].max() <= pd.Timestamp("2023-03-01")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_factor_panel.py::test_build_panel_end_to_end_synthetic -v`
Expected: FAIL — `AttributeError: ... has no attribute 'candidate_tickers'`

- [ ] **Step 3: Implement**

Append to `src/factor_panel.py`:

```python
def candidate_tickers(db_path: str) -> list[str]:
    """Universe tickers with a cached market cap >= MIN_MCAP_TODAY. Names the
    live screener has never cached a cap for are excluded (documented
    universe-reconstruction crudeness — see module docstring)."""
    uni = pd.read_parquet("data/universe.parquet")
    out = []
    for t in uni["ticker"]:
        mc = get_market_cap_stale(db_path, t)
        if mc is not None and mc >= MIN_MCAP_TODAY:
            out.append(t)
    logger.info(f"[candidates] {len(out)} tickers with cached mcap >= {MIN_MCAP_TODAY:,.0f}")
    return out


def build_panel(db_path: str, start: str, end: str,
                panel_path: str = PANEL_PATH,
                breadth_path: str = BREADTH_PATH) -> None:
    """Fetch history, compute the factor panel at weekly rebalance dates in
    [start, end], compute daily breadth, save both to parquet."""
    fetch_start = (pd.Timestamp(start) - pd.Timedelta(days=WARMUP_DAYS)).strftime("%Y-%m-%d")
    tickers = candidate_tickers(db_path)
    spy = get_history("SPY", db_path, start=fetch_start, end=end)
    if spy is None or spy.empty:
        raise RuntimeError("Could not fetch SPY history")
    prices = get_history_bulk(tickers, db_path, start=fetch_start, end=end)
    prices = {t: df for t, df in prices.items() if df is not None and len(df) >= 260}
    logger.info(f"[panel] {len(prices)} tickers with usable history")

    sim_index = spy.loc[start:end].index
    dates = rebalance_dates(sim_index)

    frames = []
    for t, df in prices.items():
        f = ticker_factor_frame(df, spy["close"], dates)
        f["ticker"] = t
        mc_today = get_market_cap_stale(db_path, t)
        last_close = float(df["close"].iloc[-1])
        f["mcap"] = (f["close"] / last_close) * mc_today if mc_today else np.nan
        frames.append(f)

    panel = pd.concat(frames).rename_axis("date").reset_index()
    panel = apply_gates(panel)
    panel = composite_scores(panel)
    panel.to_parquet(panel_path, index=False)

    breadth = breadth_series(prices).loc[start:end]
    breadth.to_parquet(breadth_path)
    logger.info(f"[panel] saved {len(panel)} rows -> {panel_path}, breadth -> {breadth_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description="Build the portfolio-backtest factor panel")
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default="2023-12-31",
                    help="dev-window default; 2024+ is holdout, run once at the end")
    ap.add_argument("--db", default="data/cache.db")
    args = ap.parse_args()
    build_panel(args.db, args.start, args.end)
```

- [ ] **Step 4: Run the whole test file**

Run: `python3 -m pytest tests/test_factor_panel.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/factor_panel.py tests/test_factor_panel.py
git commit -m "feat(backtest): panel build CLI — fetch, compute, save parquet"
```

---

### Task 7: Simulator — metrics functions

Pure math first; the engine (Task 8) reports through these.

**Files:**
- Create: `src/portfolio_sim.py`
- Create: `tests/test_portfolio_sim.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_portfolio_sim.py`:

```python
"""Portfolio simulator tests — synthetic panels, no network."""
import numpy as np
import pandas as pd
import pytest

from src.portfolio_sim import cagr, max_drawdown, sharpe, per_year


def test_cagr_known_curve():
    idx = pd.bdate_range("2020-01-01", periods=504)  # ~2 years
    equity = pd.Series(np.linspace(100.0, 144.0, 504), index=idx)
    years = (idx[-1] - idx[0]).days / 365.25
    assert cagr(equity) == pytest.approx((144 / 100) ** (1 / years) - 1, rel=1e-9)


def test_max_drawdown_known_curve():
    idx = pd.bdate_range("2020-01-01", periods=5)
    equity = pd.Series([100, 120, 90, 110, 130], index=idx, dtype=float)
    dd, dd_days = max_drawdown(equity)
    assert dd == pytest.approx(-0.25)  # 120 -> 90
    # convention: dd_days = consecutive trading days spent below the running
    # peak. 90 and 110 are below the 120 peak; 130 recovers -> 2 days.
    assert dd_days == 2


def test_sharpe_zero_vol_is_nan():
    idx = pd.bdate_range("2020-01-01", periods=10)
    flat = pd.Series(100.0, index=idx)
    assert np.isnan(sharpe(flat))


def test_per_year_table():
    idx = pd.bdate_range("2020-06-01", "2021-06-01")
    equity = pd.Series(np.linspace(100, 120, len(idx)), index=idx)
    table = per_year(equity)
    assert set(table.index) == {2020, 2021}
    assert (table["return"] > 0).all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_portfolio_sim.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.portfolio_sim'`

- [ ] **Step 3: Implement**

Create `src/portfolio_sim.py`:

```python
"""Stage B of the portfolio backtest: band-portfolio simulator + report.

Consumes the parquet artifacts produced by src/factor_panel.py. See the
survivorship-bias warning in that module's docstring — it applies to every
number this simulator prints, and is repeated in the report output.
"""
import argparse
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRANSACTION_COST_BPS = 20.0   # per side: ~10 commission-equivalent + ~10 spread/slippage


def cagr(equity: pd.Series) -> float:
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    if years <= 0 or equity.iloc[0] <= 0:
        return float("nan")
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1)


def max_drawdown(equity: pd.Series) -> tuple[float, int]:
    """Returns (max drawdown as negative fraction, longest drawdown length in
    trading days measured peak-to-recovery; unrecovered runs count to end)."""
    peak = equity.cummax()
    dd = equity / peak - 1.0
    max_dd = float(dd.min()) if len(dd) else float("nan")

    below = equity < peak
    longest = current = 0
    for b in below:
        current = current + 1 if b else 0
        longest = max(longest, current)
    return max_dd, longest


def sharpe(equity: pd.Series) -> float:
    rets = equity.pct_change().dropna()
    std = rets.std()
    if std < 1e-12 or np.isnan(std):
        return float("nan")
    return float(rets.mean() / std * np.sqrt(252))


def per_year(equity: pd.Series) -> pd.DataFrame:
    """Calendar-year return and max drawdown."""
    rows = {}
    for year, eq in equity.groupby(equity.index.year):
        dd, _ = max_drawdown(eq)
        rows[year] = {"return": float(eq.iloc[-1] / eq.iloc[0] - 1), "max_dd": dd}
    return pd.DataFrame(rows).T
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_portfolio_sim.py -v`
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add src/portfolio_sim.py tests/test_portfolio_sim.py
git commit -m "feat(backtest): portfolio metrics — cagr, drawdown, sharpe, per-year"
```

---

### Task 8: Simulator — band-portfolio engine with exposure overlay

The core. Daily mark-to-market; weekly band trades at rebalance-date close; exposure changes trade at any daily close; 20 bps per side on traded notional.

**Files:**
- Modify: `src/portfolio_sim.py` (append)
- Modify: `tests/test_portfolio_sim.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_portfolio_sim.py`:

```python
from src.portfolio_sim import simulate


def _mini_market():
    """5 tickers, 15 business days, 3 weekly rebalances (Fridays).
    Prices constant except winners drift; panel makes ranks deterministic."""
    idx = pd.bdate_range("2024-01-01", periods=15)  # Mon 1/1 .. Fri 1/19
    fridays = [pd.Timestamp("2024-01-05"), pd.Timestamp("2024-01-12"),
               pd.Timestamp("2024-01-19")]
    tickers = ["T1", "T2", "T3", "T4", "T5"]
    closes = pd.DataFrame(
        {t: 100.0 * (1.01 ** np.arange(15)) if t in ("T1", "T2")
         else np.full(15, 100.0) for t in tickers},
        index=idx,
    )

    rows = []
    for d in fridays:
        for rank, t in enumerate(tickers, start=1):
            rows.append({
                "date": d, "ticker": t, "passes_gates": True,
                "composite": float(len(tickers) - rank),  # T1 best ... T5 worst
                "close": float(closes.loc[d, t]),
            })
    return pd.DataFrame(rows), closes, fridays


def test_simulate_enters_top_ranked_equal_weight():
    panel, closes, fridays = _mini_market()
    res = simulate(panel, closes, max_positions=2, entry_band=2, exit_band=3,
                   cost_bps=0.0)
    first_trades = res["trades"][res["trades"]["date"] == fridays[0]]
    assert set(first_trades["ticker"]) == {"T1", "T2"}
    assert (first_trades["side"] == "buy").all()
    # equal weight: each buy ~50% of equity
    assert first_trades["notional"].iloc[0] == pytest.approx(50_000, rel=1e-6)


def test_simulate_band_exit_only_below_exit_band():
    panel, closes, fridays = _mini_market()
    # At second rebalance, demote T2 to rank 3 (within exit band of 3 -> hold),
    # then at third rebalance to rank 4 (beyond band -> sell).
    p = panel.copy()
    p.loc[(p["date"] == fridays[1]) & (p["ticker"] == "T2"), "composite"] = 1.5  # rank 3
    p.loc[(p["date"] == fridays[2]) & (p["ticker"] == "T2"), "composite"] = 0.5  # rank 4
    res = simulate(p, closes, max_positions=2, entry_band=2, exit_band=3, cost_bps=0.0)
    t2_sells = res["trades"][(res["trades"]["ticker"] == "T2") &
                             (res["trades"]["side"] == "sell")]
    assert list(t2_sells["date"]) == [fridays[2]]


def test_simulate_costs_reduce_equity():
    panel, closes, _ = _mini_market()
    free = simulate(panel, closes, max_positions=2, entry_band=2, exit_band=3,
                    cost_bps=0.0)
    costly = simulate(panel, closes, max_positions=2, entry_band=2, exit_band=3,
                      cost_bps=20.0)
    assert costly["equity"].iloc[-1] < free["equity"].iloc[-1]
    assert costly["costs"] > 0


def test_simulate_exposure_scales_invested_fraction():
    panel, closes, fridays = _mini_market()
    exposure = pd.Series(1.0, index=closes.index)
    exposure.loc[exposure.index >= fridays[1]] = 0.5   # de-risk halfway through
    res = simulate(panel, closes, max_positions=2, entry_band=2, exit_band=3,
                   cost_bps=0.0, exposure=exposure)
    # day after de-risking: invested value ~= 50% of equity
    day_after = closes.index[closes.index.get_loc(fridays[1]) + 1]
    snap = res["daily"].loc[day_after]
    assert snap["invested"] / snap["equity"] == pytest.approx(0.5, abs=0.02)
    # and average exposure < 1
    assert res["avg_exposure"] < 0.85


def test_simulate_full_exposure_zero_goes_all_cash():
    panel, closes, fridays = _mini_market()
    exposure = pd.Series(0.0, index=closes.index)
    res = simulate(panel, closes, max_positions=2, entry_band=2, exit_band=3,
                   cost_bps=0.0, exposure=exposure)
    assert (res["daily"]["invested"] < 1e-9).all()
    assert res["equity"].iloc[-1] == pytest.approx(100_000.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_portfolio_sim.py -v`
Expected: new tests FAIL — `ImportError: cannot import name 'simulate'`

- [ ] **Step 3: Implement the engine**

Append to `src/portfolio_sim.py`:

```python
def simulate(
    panel: pd.DataFrame,
    closes: pd.DataFrame,
    exposure: pd.Series | None = None,
    score_col: str = "composite",
    entry_band: int = 20,
    exit_band: int = 35,
    max_positions: int = 15,
    cost_bps: float = TRANSACTION_COST_BPS,
    initial_capital: float = 100_000.0,
) -> dict:
    """Band-portfolio simulation.

    - Daily mark-to-market on `closes` (columns = tickers).
    - On each panel date (weekly): rank gate-survivors by `score_col` desc;
      sell holdings ranked beyond `exit_band` (or absent from survivors);
      buy best-ranked names within `entry_band` until `max_positions` held.
      New positions sized equal-weight against target invested capital;
      existing positions drift (no weight rebalancing).
    - Exposure (daily target fraction, default 1.0): on days the target
      changes, all positions scale pro-rata toward target invested value.
    - Costs: `cost_bps` per side on traded notional. Cash earns 0%.
    """
    exposure = (pd.Series(1.0, index=closes.index) if exposure is None
                else exposure.reindex(closes.index).ffill().fillna(1.0))
    cost_rate = cost_bps / 1e4

    rebal_by_date = {d: g for d, g in panel.groupby("date")}
    positions: dict[str, float] = {}     # ticker -> current dollar value
    cash = initial_capital
    trades, daily_rows = [], []
    equity_curve = {}
    total_costs = 0.0
    turnover_notional = 0.0
    prev_exposure = None

    rets = closes.pct_change().fillna(0.0)

    for day in closes.index:
        # 1) mark to market
        for t in list(positions):
            positions[t] *= 1.0 + float(rets.loc[day].get(t, 0.0))
        equity = cash + sum(positions.values())

        def _trade(ticker: str, notional: float, side: str):
            nonlocal cash, total_costs, turnover_notional
            fee = abs(notional) * cost_rate
            if side == "buy":
                positions[ticker] = positions.get(ticker, 0.0) + notional
                cash -= notional + fee
            else:
                positions[ticker] = positions.get(ticker, 0.0) - notional
                if positions[ticker] < 1e-9:
                    positions.pop(ticker, None)
                cash += notional - fee
            total_costs += fee
            turnover_notional += abs(notional)
            trades.append({"date": day, "ticker": ticker, "side": side,
                           "notional": abs(notional)})

        target_exposure = float(exposure.loc[day])

        # 2) weekly band rebalance at this close
        if day in rebal_by_date:
            g = rebal_by_date[day]
            surv = g[g["passes_gates"]].dropna(subset=[score_col])
            ranked = surv.sort_values(score_col, ascending=False)["ticker"].tolist()
            rank_of = {t: i + 1 for i, t in enumerate(ranked)}

            for t in list(positions):
                r = rank_of.get(t)
                if r is None or r > exit_band:
                    _trade(t, positions[t], "sell")

            equity = cash + sum(positions.values())
            target_invested = equity * target_exposure
            slot = target_invested / max_positions if max_positions else 0.0
            for t in ranked:
                if len(positions) >= max_positions:
                    break
                if t in positions or rank_of[t] > entry_band:
                    continue
                # fees come out of cash too: shrink the final slot to what
                # cash affords rather than skipping it over a fee-sized
                # shortfall, but don't open dust positions
                notional = min(slot, cash / (1.0 + cost_rate))
                if notional <= 1e-9 or notional < slot * 0.5:
                    break
                _trade(t, notional, "buy")

        # 3) daily exposure adjustment (pro-rata) when target changed
        invested = sum(positions.values())
        if prev_exposure is not None and target_exposure != prev_exposure and invested > 0:
            equity = cash + invested
            target_invested = equity * target_exposure
            scale = target_invested / invested
            for t in list(positions):
                delta = positions[t] * (scale - 1.0)
                if delta > 0:
                    _trade(t, delta, "buy")
                elif delta < 0:
                    _trade(t, -delta, "sell")
        prev_exposure = target_exposure

        invested = sum(positions.values())
        equity = cash + invested
        equity_curve[day] = equity
        daily_rows.append({"date": day, "equity": equity, "invested": invested,
                           "n_positions": len(positions),
                           "exposure_target": target_exposure})

    equity_s = pd.Series(equity_curve).sort_index()
    daily = pd.DataFrame(daily_rows).set_index("date")
    avg_exposure = float((daily["invested"] / daily["equity"]).mean())
    return {
        "equity": equity_s,
        "daily": daily,
        "trades": pd.DataFrame(trades, columns=["date", "ticker", "side", "notional"]),
        "costs": total_costs,
        "turnover_notional": turnover_notional,
        "avg_exposure": avg_exposure,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_portfolio_sim.py -v`
Expected: all PASS. If `test_simulate_exposure_scales_invested_fraction` is off by more than tolerance, check that step 3 (exposure adjust) runs on rebalance days too — the rebalance already sizes to `target_exposure`, so the scale factor should be ~1.0 there.

- [ ] **Step 5: Commit**

```bash
git add src/portfolio_sim.py tests/test_portfolio_sim.py
git commit -m "feat(backtest): band-portfolio simulator with exposure overlay and costs"
```

---

### Task 9: Verdict logic + markdown report writer

**Files:**
- Modify: `src/portfolio_sim.py` (append)
- Modify: `tests/test_portfolio_sim.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_portfolio_sim.py`:

```python
from src.portfolio_sim import verdict, write_report


def _fake_run(final=150_000.0, dd_curve=None, avg_exposure=1.0, n=756,
              start="2019-01-01"):
    idx = pd.bdate_range(start, periods=n)
    if dd_curve is None:
        equity = pd.Series(np.linspace(100_000, final, n), index=idx)
    else:
        equity = pd.Series(dd_curve, index=idx[:len(dd_curve)])
    return {"equity": equity, "avg_exposure": avg_exposure, "costs": 0.0,
            "turnover_notional": 0.0,
            "daily": pd.DataFrame({"equity": equity, "invested": equity,
                                   "n_positions": 10, "exposure_target": 1.0})}


def _curve_with_dd(n, dd_frac, seed=0):
    """Linear up, one crash of dd_frac in the middle, recovery."""
    third = n // 3
    up1 = np.linspace(100_000, 130_000, third)
    crash = np.linspace(130_000, 130_000 * (1 - dd_frac), third)
    up2 = np.linspace(130_000 * (1 - dd_frac), 160_000, n - 2 * third)
    return np.concatenate([up1, crash, up2])


def test_verdict_pass_when_all_criteria_met():
    n = 2016  # ~8 years covering 2019-2026, includes 2020 and 2022
    unhedged = _fake_run(dd_curve=_curve_with_dd(n, 0.30))
    laddered = _fake_run(dd_curve=_curve_with_dd(n, 0.15), avg_exposure=0.8)
    naive = _fake_run(dd_curve=_curve_with_dd(n, 0.35))
    v = verdict(unhedged, laddered, naive)
    assert v["dd_reduced_third"] is True
    assert isinstance(v["cagr_giveup_ok"], bool)
    assert isinstance(v["sharpe_vs_naive_ok"], bool)
    assert v["overall"] == (v["dd_reduced_third"] and v["dd_2020_ok"]
                            and v["dd_2022_ok"] and v["cagr_giveup_ok"]
                            and v["sharpe_vs_naive_ok"])


def test_write_report_renders(tmp_path):
    runs = {"composite": _fake_run(), "composite+ladder": _fake_run(final=140_000.0),
            "naive_momentum": _fake_run(final=130_000.0), "SPY": _fake_run(final=120_000.0)}
    v = verdict(runs["composite"], runs["composite+ladder"], runs["naive_momentum"])
    path = tmp_path / "report.md"
    write_report(str(path), runs, v, meta={"start": "2019-01-01", "end": "2026-01-01",
                                           "cost_bps": 20.0, "note": "test"})
    text = path.read_text()
    assert "SURVIVORSHIP" in text.upper()
    assert "composite+ladder" in text
    assert "CAGR" in text
    assert "Verdict" in text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_portfolio_sim.py -k "verdict or report" -v`
Expected: FAIL — `ImportError: cannot import name 'verdict'`

- [ ] **Step 3: Implement**

Append to `src/portfolio_sim.py`:

```python
CRASH_WINDOWS = {"2020": ("2020-01-01", "2020-12-31"),
                 "2022": ("2022-01-01", "2022-12-31")}
DD_REDUCTION_REQUIRED = 1 / 3      # spec pass criterion 1
CAGR_GIVEUP_MAX = 0.02             # spec pass criterion 2


def _window_dd(run: dict, start: str, end: str) -> float:
    eq = run["equity"].loc[start:end]
    if len(eq) < 2:
        return float("nan")
    dd, _ = max_drawdown(eq)
    return dd


def verdict(unhedged: dict, laddered: dict, naive: dict) -> dict:
    """Spec pass criteria, all a priori. NaN window (period doesn't cover a
    crash year) -> that check passes vacuously but is flagged in the value."""
    dd_u, _ = max_drawdown(unhedged["equity"])
    dd_l, _ = max_drawdown(laddered["equity"])
    reduced = (abs(dd_l) <= abs(dd_u) * (1 - DD_REDUCTION_REQUIRED))

    window_ok = {}
    for name, (s, e) in CRASH_WINDOWS.items():
        wu, wl = _window_dd(unhedged, s, e), _window_dd(laddered, s, e)
        if np.isnan(wu) or np.isnan(wl):
            window_ok[name] = True   # window outside sim period
        else:
            window_ok[name] = abs(wl) <= abs(wu) * (1 - DD_REDUCTION_REQUIRED)

    giveup = cagr(unhedged["equity"]) - cagr(laddered["equity"])
    cagr_ok = giveup <= CAGR_GIVEUP_MAX
    sharpe_ok = sharpe(laddered["equity"]) >= sharpe(naive["equity"])

    overall = (reduced and window_ok["2020"] and window_ok["2022"]
               and cagr_ok and sharpe_ok)
    return {
        "dd_unhedged": dd_u, "dd_laddered": dd_l,
        "dd_reduced_third": reduced,
        "dd_2020_ok": window_ok["2020"], "dd_2022_ok": window_ok["2022"],
        "cagr_giveup": giveup, "cagr_giveup_ok": cagr_ok,
        "sharpe_vs_naive_ok": sharpe_ok,
        "overall": overall,
    }


def write_report(path: str, runs: dict[str, dict], v: dict, meta: dict) -> None:
    lines = [
        "# Portfolio Backtest Report",
        "",
        f"Period: {meta.get('start')} → {meta.get('end')} · "
        f"costs {meta.get('cost_bps')} bps/side · cash earns 0%",
        "",
        "> **SURVIVORSHIP BIAS WARNING:** universe contains only currently-",
        "> listed tickers (yfinance). Delisted losers are absent; every CAGR",
        "> here is an upper bound. Ladder value is judged on *relative*",
        "> drawdown reduction, which is far less bias-sensitive.",
        "",
        "## Summary",
        "",
        "| Strategy | CAGR | Vol | Sharpe | MaxDD | LongestDD (d) | AvgExp | ExpAdj CAGR | Costs $ |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, run in runs.items():
        eq = run["equity"]
        dd, dd_days = max_drawdown(eq)
        c = cagr(eq)
        ae = run.get("avg_exposure", 1.0)
        vol = eq.pct_change().std() * np.sqrt(252)
        exp_adj = c / ae if ae > 0 else float("nan")
        lines.append(
            f"| {name} | {c:.2%} | {vol:.2%} | {sharpe(eq):.2f} | {dd:.2%} "
            f"| {dd_days} | {ae:.2f} | {exp_adj:.2%} | {run.get('costs', 0):,.0f} |")

    lines += ["", "## Per-year", ""]
    for name, run in runs.items():
        table = per_year(run["equity"])
        lines += [f"### {name}", "", "| Year | Return | MaxDD |", "|---|---|---|"]
        for year, row in table.iterrows():
            lines.append(f"| {year} | {row['return']:.2%} | {row['max_dd']:.2%} |")
        lines.append("")

    lines += [
        "## Verdict (spec pass criteria, a priori)",
        "",
        f"- Max drawdown cut ≥ 1/3 overall: **{v['dd_reduced_third']}** "
        f"(unhedged {v['dd_unhedged']:.2%} → laddered {v['dd_laddered']:.2%})",
        f"- 2020 window cut ≥ 1/3: **{v['dd_2020_ok']}**",
        f"- 2022 window cut ≥ 1/3: **{v['dd_2022_ok']}**",
        f"- CAGR give-up ≤ 2pts: **{v['cagr_giveup_ok']}** ({v['cagr_giveup']:.2%})",
        f"- Ladder Sharpe ≥ naive momentum Sharpe: **{v['sharpe_vs_naive_ok']}**",
        "",
        f"**OVERALL: {'PASS' if v['overall'] else 'FAIL'}**",
        "",
        meta.get("note", ""),   # raw markdown block (e.g. sensitivity table)
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"[report] wrote {path}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_portfolio_sim.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/portfolio_sim.py tests/test_portfolio_sim.py
git commit -m "feat(backtest): a-priori verdict logic and markdown report"
```

---

### Task 10: CLI — wire panel + regime + sims + sensitivity table

Orchestration with monkeypatched smoke test.

**Files:**
- Modify: `src/portfolio_sim.py` (append)
- Modify: `tests/test_portfolio_sim.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_portfolio_sim.py`:

```python
def test_run_backtest_end_to_end_synthetic(tmp_path, monkeypatch):
    """Wiring smoke test: panel/breadth/instruments synthetic, full CLI path."""
    import src.portfolio_sim as ps

    n = 300
    idx = pd.bdate_range("2023-01-02", periods=n)
    rng = np.random.default_rng(11)
    tickers = [f"T{i}" for i in range(30)]
    closes = pd.DataFrame(
        {t: 100 * np.cumprod(1 + rng.normal(0.0004, 0.015, n)) for t in tickers},
        index=idx)

    fridays = [d for d in idx if d.dayofweek == 4]
    rows = []
    for d in fridays:
        for i, t in enumerate(tickers):
            rows.append({"date": d, "ticker": t, "passes_gates": True,
                         "composite": float(rng.normal()), "mom_12_1": float(rng.normal()),
                         "close": float(closes.loc[d, t])})
    panel = pd.DataFrame(rows)
    breadth = pd.DataFrame({"pct_above_200": np.full(n, 0.6),
                            "pct_above_50": np.full(n, 0.6)}, index=idx)

    def synth_instr(base, vol):
        r = np.random.default_rng(hash(base) % 2**31)
        return pd.DataFrame({"close": base * np.cumprod(1 + r.normal(0.0003, vol, n)),
                             "volume": np.full(n, 1e6)}, index=idx)

    instruments = {"SPY": synth_instr(400, 0.01), "^VIX": synth_instr(18, 0.05),
                   "^VIX3M": synth_instr(20, 0.04), "HYG": synth_instr(75, 0.005),
                   "IEF": synth_instr(95, 0.004)}
    monkeypatch.setattr(ps, "_fetch_instrument",
                        lambda name, db_path, start, end: instruments[name])

    panel_path = tmp_path / "panel.parquet"
    breadth_path = tmp_path / "breadth.parquet"
    panel.to_parquet(panel_path, index=False)
    breadth.to_parquet(breadth_path)
    report_path = tmp_path / "report.md"

    ps.run_backtest(panel_path=str(panel_path), breadth_path=str(breadth_path),
                    db_path="unused.db", report_path=str(report_path),
                    sensitivity=True)

    text = report_path.read_text()
    assert "composite+ladder" in text
    assert "naive_momentum" in text
    assert "Sensitivity" in text
    assert "OVERALL" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_portfolio_sim.py::test_run_backtest_end_to_end_synthetic -v`
Expected: FAIL — `AttributeError: ... no attribute '_fetch_instrument'`

- [ ] **Step 3: Implement**

Append to `src/portfolio_sim.py`:

```python
from src import regime
from src.event_backtest import get_history

SENSITIVITY_MAPS = [
    {0: 1.00, 1: 1.00, 2: 0.66, 3: 0.33, 4: 0.00},   # spec (chosen)
    {0: 1.00, 1: 0.75, 2: 0.50, 3: 0.25, 4: 0.00},   # linear
    {0: 1.00, 1: 1.00, 2: 0.50, 3: 0.00, 4: 0.00},   # aggressive
    {0: 1.00, 1: 0.66, 2: 0.33, 3: 0.00, 4: 0.00},   # early
]


def _fetch_instrument(name: str, db_path: str, start: str, end: str) -> pd.DataFrame:
    df = get_history(name, db_path, start=start, end=end)
    if df is None or df.empty:
        raise RuntimeError(f"Could not fetch {name}")
    return df


def compute_exposure(breadth: pd.DataFrame, db_path: str, start: str, end: str,
                     exposure_map: dict | None = None) -> pd.Series:
    """Daily combined ladder+thrust exposure over breadth's index."""
    spy = _fetch_instrument("SPY", db_path, start, end)["close"]
    vix = _fetch_instrument("^VIX", db_path, start, end)["close"]
    vix3m = _fetch_instrument("^VIX3M", db_path, start, end)["close"]
    hyg = _fetch_instrument("HYG", db_path, start, end)["close"]
    ief = _fetch_instrument("IEF", db_path, start, end)["close"]

    idx = breadth.index
    pts = regime.ladder_points(
        regime.trend_signal(spy).reindex(idx, fill_value=False),
        regime.breadth_signal(breadth["pct_above_200"]),
        regime.vol_signal(vix, vix3m).reindex(idx, fill_value=False),
        regime.credit_signal(hyg, ief).reindex(idx, fill_value=False),
    )
    thrust = regime.thrust_override(breadth["pct_above_50"])
    if exposure_map is None:
        return regime.combined_exposure(pts, thrust)
    exp = pts.map(exposure_map).astype(float)
    floored = exp.clip(lower=regime.THRUST_FLOOR)
    return exp.where(~thrust.reindex(exp.index, fill_value=False), floored)


def run_backtest(panel_path: str = "output/factor_panel.parquet",
                 breadth_path: str = "output/breadth.parquet",
                 db_path: str = "data/cache.db",
                 report_path: str | None = None,
                 sensitivity: bool = False) -> dict:
    panel = pd.read_parquet(panel_path)
    panel["date"] = pd.to_datetime(panel["date"])
    breadth = pd.read_parquet(breadth_path)
    breadth.index = pd.to_datetime(breadth.index)

    start = str(breadth.index.min().date())
    end = str(breadth.index.max().date())
    # warm-up for SMA200/SMA100 in regime signals
    fetch_start = str((breadth.index.min() - pd.Timedelta(days=400)).date())

    closes = (panel.pivot_table(index="date", columns="ticker", values="close")
              .reindex(breadth.index).ffill())

    spy = _fetch_instrument("SPY", db_path, fetch_start, end)
    spy_close = spy["close"].reindex(breadth.index).ffill()
    spy_run = {"equity": 100_000.0 * spy_close / spy_close.iloc[0],
               "avg_exposure": 1.0, "costs": 0.0, "turnover_notional": 0.0,
               "daily": pd.DataFrame({"equity": spy_close, "invested": spy_close,
                                      "n_positions": 1, "exposure_target": 1.0})}

    exposure = compute_exposure(breadth, db_path, fetch_start, end)

    runs = {
        "composite": simulate(panel, closes),
        "composite+ladder": simulate(panel, closes, exposure=exposure),
        "naive_momentum": simulate(panel, closes, score_col="mom_12_1"),
        "SPY": spy_run,
    }
    v = verdict(runs["composite"], runs["composite+ladder"], runs["naive_momentum"])

    note = ""
    if sensitivity:
        note_lines = ["", "## Sensitivity (alternative exposure maps — robustness, not tuning)", "",
                      "| Map | CAGR | MaxDD | Sharpe |", "|---|---|---|---|"]
        for m in SENSITIVITY_MAPS:
            exp_m = compute_exposure(breadth, db_path, fetch_start, end, exposure_map=m)
            r = simulate(panel, closes, exposure=exp_m)
            dd, _ = max_drawdown(r["equity"])
            note_lines.append(f"| {m} | {cagr(r['equity']):.2%} | {dd:.2%} "
                              f"| {sharpe(r['equity']):.2f} |")
        note = "\n".join(note_lines)

    if report_path is None:
        report_path = f"output/portfolio_backtest_{pd.Timestamp.today().date()}.md"
    write_report(report_path, runs, v,
                 meta={"start": start, "end": end,
                       "cost_bps": TRANSACTION_COST_BPS, "note": note})
    for name, run in runs.items():
        run["equity"].to_csv(report_path.replace(".md", f"_{name.replace('+','_')}_equity.csv"))
    return {"runs": runs, "verdict": v, "report_path": report_path}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description="Run the portfolio backtest + regime ladder")
    ap.add_argument("--panel", default="output/factor_panel.parquet")
    ap.add_argument("--breadth", default="output/breadth.parquet")
    ap.add_argument("--db", default="data/cache.db")
    ap.add_argument("--report", default=None)
    ap.add_argument("--sensitivity", action="store_true")
    args = ap.parse_args()
    out = run_backtest(panel_path=args.panel, breadth_path=args.breadth,
                       db_path=args.db, report_path=args.report,
                       sensitivity=args.sensitivity)
    print(f"OVERALL: {'PASS' if out['verdict']['overall'] else 'FAIL'}"
          f" -> {out['report_path']}")
```

The "Sensitivity" heading lives in `meta["note"]`, which `write_report` appends as a raw markdown block — that satisfies the test's `"Sensitivity" in text` assertion.

- [ ] **Step 4: Run the whole test file**

Run: `python3 -m pytest tests/test_portfolio_sim.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/portfolio_sim.py tests/test_portfolio_sim.py
git commit -m "feat(backtest): CLI wiring — regime exposure, baselines, sensitivity, verdict"
```

---

### Task 11: Retire `backtest/backtest.py`

Superseded by the new harness; its survivorship warning already migrated to `src/factor_panel.py`'s docstring (Task 4).

**Files:**
- Delete: `backtest/backtest.py`, `backtest/__init__.py`

- [ ] **Step 1: Verify nothing imports it**

Run: `grep -rn "from backtest\|import backtest" src/ tests/ app*.py run_screener.sh`
Expected: no output. If anything matches, stop and surface it before deleting.

- [ ] **Step 2: Delete**

```bash
git rm -r backtest/
```

- [ ] **Step 3: Run the full test suite**

Run: `python3 -m pytest tests/ -v`
Expected: all PASS, nothing referencing `backtest/`.

- [ ] **Step 4: Commit**

```bash
git commit -m "chore(backtest): retire toy monthly backtest, superseded by portfolio harness"
```

---

### Task 12: Build the real panel (operational, long-running)

No new code. First real data pull: ~3,000+ tickers × ~10.5 years. Expect 30–90 minutes on first run (yfinance batches of 200 + 1s sleeps); re-runs hit the `event_backtest_prices` cache.

- [ ] **Step 1: Dev-window panel build**

Run: `python3 -m src.factor_panel --start 2015-01-01 --end 2023-12-31 2>&1 | tee output/panel_build_$(date +%F).log`
Expected: log lines `[candidates] N tickers...`, `[panel] N tickers with usable history`, ends with `[panel] saved ... -> output/factor_panel.parquet`.

- [ ] **Step 2: Sanity-check the artifacts**

Run:
```bash
python3 - <<'EOF'
import pandas as pd
p = pd.read_parquet("output/factor_panel.parquet")
b = pd.read_parquet("output/breadth.parquet")
print("panel rows:", len(p), "tickers:", p["ticker"].nunique(),
      "dates:", p["date"].nunique())
print("gate pass rate:", p["passes_gates"].mean().round(3))
print("breadth range:", b["pct_above_200"].min().round(2),
      "-", b["pct_above_200"].max().round(2))
EOF
```
Expected: hundreds of dates (~470 weeks), 1,500+ tickers, gate pass rate roughly 0.1–0.4, breadth spanning a wide range (should dip near 0.1–0.2 in the 2020 and 2022 windows — eyeball check).

- [ ] **Step 3: Commit the build log only (parquet artifacts are outputs, not source)**

```bash
echo "output/*.parquet" >> .gitignore
git add .gitignore output/panel_build_*.log
git commit -m "chore(backtest): dev-window factor panel built; ignore parquet artifacts"
```

---

### Task 13: Dev-window run + report review

- [ ] **Step 1: Run with sensitivity table**

Run: `python3 -m src.portfolio_sim --sensitivity 2>&1 | tee output/sim_run_$(date +%F).log`
Expected: final line `OVERALL: PASS -> output/portfolio_backtest_<date>.md` (or FAIL — the report ships either way, per spec).

- [ ] **Step 2: Review report with the user before any holdout run**

Present `output/portfolio_backtest_<date>.md` to the user: summary table, per-year, verdict, sensitivity. **STOP here.** The holdout run (2024–2026) happens exactly once, only after the user has reviewed dev-window results and agreed no further iteration is wanted. Do not run it as part of this plan's execution.

- [ ] **Step 3: Commit report + log**

```bash
git add output/portfolio_backtest_*.md output/sim_run_*.log output/*_equity.csv
git commit -m "chore(backtest): dev-window portfolio backtest report"
```

---

## Explicitly out of scope (per spec)

- **Holdout run (2024–2026):** one-shot, user-initiated after dev-window review. Command when the time comes: rebuild panel with `--start 2015-01-01 --end $(date +%F)`, rerun sim, compare.
- **Part 3 live wiring** (regime block in daily output/app, `regime:` config key): separate small plan, only if verdict PASS.
- Event sleeve, XBRL factors, sector caps: own specs.

## Self-review notes

- Spec coverage: data layer (T4/T6, amended to reuse `event_backtest_prices`), sim engine + band mechanics + costs + cash-at-0% (T8), baselines SPY + naive momentum (T10), metrics incl. exposure-adjusted CAGR (T7/T9 report), anti-overfit holdout discipline (T6 default `--end 2023-12-31`, T13 stop-gate), 4 regime signals (T2), exposure map + a-priori sensitivity framing (T3/T10), thrust override (T3), pass criteria verbatim (T9), report + equity CSVs (T9/T10), old backtest deletion with warning-text migration (T4 docstring + T11).
- Deviations from spec: three amendments listed in the header, all argued there.
- Type consistency: `simulate` returns dict with keys `equity/daily/trades/costs/turnover_notional/avg_exposure` — consumed identically in T9 `_fake_run` fixtures and T10 `run_backtest`. `ticker_factor_frame` column names match `apply_gates`/`composite_scores`/`FACTOR_COLS` and `PRICE_BLOCK_WEIGHTS` keys throughout.
