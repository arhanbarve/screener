# Position Sizing + Concentration Caps Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add inverse-volatility position weights with per-name and per-GICS-sector concentration caps to the screener's top-N output.

**Architecture:** A new pure-function module `src/sizing.py` (realized vol → inverse-vol weights → iterative cap redistribution) is wired into `src/run.py` after the composite ranking, adding a `weight_pct` column that `src/output.py` writes to CSV and markdown. Config lives under a new `sizing:` block. Correctness is arithmetic (caps hold, weights sum to 1.0), so no backtest is needed to validate.

**Tech Stack:** Python 3.14, pandas, numpy, pytest.

Spec: `docs/superpowers/specs/2026-07-16-position-sizing-concentration-caps-design.md`

---

## File Structure

- `src/sizing.py` (NEW) — `realized_vol`, `inverse_vol_weights`, `apply_caps`. Pure, no I/O.
- `tests/test_sizing.py` (NEW) — unit tests for the three functions.
- `config.yaml` (MODIFY) — new `sizing:` block.
- `src/run.py` (MODIFY) — wire sizing between ranking and output.
- `src/output.py` (MODIFY) — add `weight_pct` to CSV columns and the markdown table.
- `tests/test_output.py` (MODIFY or CREATE) — assert `weight_pct` renders.

---

## Task 1: Core sizing module — ALREADY COMPLETE ✅

**Files:**
- Created: `src/sizing.py`
- Test: `tests/test_sizing.py`

This task was implemented TDD-first earlier in the session and is green
(12/12 passing). `realized_vol`, `inverse_vol_weights`, and `apply_caps` exist
with the sum-preserving `_redistribute` helper. **Do not re-implement.** Verify
only:

- [ ] **Step 1: Confirm the module is green**

Run: `python3 -m pytest tests/test_sizing.py -q`
Expected: `12 passed`

If it does not pass, STOP and report — the rest of the plan builds on it.

---

## Task 2: Add the `sizing:` config block

**Files:**
- Modify: `config.yaml` (add block after the `output:` block, ~line 57)
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_config.py`:

```python
def test_config_has_sizing_block():
    cfg = load_config()
    sizing = cfg["sizing"]
    assert sizing["enabled"] is True
    assert sizing["vol_window"] == 63
    assert sizing["name_cap"] == 0.10
    assert sizing["sector_cap"] == 0.25
```

If `test_config.py` does not already import `load_config`, add at the top:
`from src.config import load_config`

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_config.py::test_config_has_sizing_block -v`
Expected: FAIL with `KeyError: 'sizing'`

- [ ] **Step 3: Add the config block**

In `config.yaml`, after the `output:` block (which ends with `exit_band: 35`)
and before the `cache:` block, insert:

```yaml
sizing:
  enabled: true
  vol_window: 63        # ~3mo realized-vol lookback (trading days)
  name_cap: 0.10        # no single stock above 10% of the book
  sector_cap: 0.25      # no GICS sector above 25% of the book
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_config.py::test_config_has_sizing_block -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add config.yaml tests/test_config.py
git commit -m "feat(sizing): add sizing config block"
```

---

## Task 3: Wire sizing into the run pipeline

**Files:**
- Create helper: `src/sizing.py` gets one new function `attach_weights`
- Modify: `src/run.py:93-97` (after composite + news overlay, before output)
- Test: `tests/test_sizing.py`

Rationale for a dedicated `attach_weights`: keeps `run.py` a thin orchestrator
and makes the DataFrame-level glue testable without running the full pipeline.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_sizing.py`:

```python
def _fake_ranked_df():
    # two low-vol names, one high-vol name, mixed sectors
    idx = pd.date_range(end="2024-06-28", periods=300, freq="B")
    def series(sig):
        rets = np.array([sig if i % 2 == 0 else -sig for i in range(300)])
        return pd.Series(100.0 * np.exp(np.cumsum(rets)), index=idx)
    return pd.DataFrame({
        "ticker": ["A", "B", "C"],
        "sector": ["Tech", "Health", "Energy"],
        "close_series": [series(0.005), series(0.01), series(0.05)],
    })


def test_attach_weights_adds_weight_pct_summing_to_100():
    from src.sizing import attach_weights
    cfg = {"sizing": {"enabled": True, "vol_window": 63,
                      "name_cap": 0.60, "sector_cap": 0.90}}
    out = attach_weights(_fake_ranked_df(), cfg)
    assert "weight_pct" in out.columns
    assert out["weight_pct"].sum() == pytest.approx(100.0, abs=1e-6)
    # lowest-vol name (A) gets the largest weight
    assert out.set_index("ticker").loc["A", "weight_pct"] > \
           out.set_index("ticker").loc["C", "weight_pct"]


def test_attach_weights_respects_name_cap():
    from src.sizing import attach_weights
    cfg = {"sizing": {"enabled": True, "vol_window": 63,
                      "name_cap": 0.40, "sector_cap": 0.90}}
    out = attach_weights(_fake_ranked_df(), cfg)
    assert out["weight_pct"].max() <= 40.0 + 1e-6


def test_attach_weights_disabled_returns_df_unchanged():
    from src.sizing import attach_weights
    df = _fake_ranked_df()
    cfg = {"sizing": {"enabled": False}}
    out = attach_weights(df, cfg)
    assert "weight_pct" not in out.columns


def test_attach_weights_empty_df_noop():
    from src.sizing import attach_weights
    cfg = {"sizing": {"enabled": True, "vol_window": 63,
                      "name_cap": 0.10, "sector_cap": 0.25}}
    out = attach_weights(pd.DataFrame(columns=["ticker"]), cfg)
    assert len(out) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_sizing.py -k attach_weights -v`
Expected: FAIL with `ImportError: cannot import name 'attach_weights'`

- [ ] **Step 3: Implement `attach_weights` in `src/sizing.py`**

Append to `src/sizing.py`:

```python
def attach_weights(ranked_df, cfg):
    """Add a `weight_pct` column (inverse-vol, cap-constrained) to the ranked
    top-N. No-op when sizing is disabled, the frame is empty, or the required
    `close_series`/`sector` columns are absent."""
    sizing = cfg.get("sizing", {})
    if not sizing.get("enabled", False):
        return ranked_df
    if len(ranked_df) == 0 or "close_series" not in ranked_df.columns:
        return ranked_df

    window = sizing.get("vol_window", 63)
    name_cap = sizing.get("name_cap", 0.10)
    sector_cap = sizing.get("sector_cap", 0.25)

    vols = {}
    sectors = {}
    for _, row in ranked_df.iterrows():
        t = row["ticker"]
        vols[t] = realized_vol(row["close_series"], window=window)
        sectors[t] = str(row.get("sector", "") or "Unknown")

    raw = inverse_vol_weights(vols)
    capped = apply_caps(raw, sectors, name_cap, sector_cap)

    df = ranked_df.copy()
    df["weight_pct"] = df["ticker"].map(
        lambda t: round(capped.get(t, 0.0) * 100.0, 4)
    )
    return df
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_sizing.py -k attach_weights -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Wire into `run.py`**

In `src/run.py`, add the import near the other src imports (after line 13,
`from src.spy_analysis import compute_market_stress_overlay`):

```python
from src.sizing import attach_weights
```

Then, in `run()`, immediately after the news-overlay block (currently lines
95-97, ending with the `attach_news_overlay` call) and before the squeeze
screen block (line 99 `# Squeeze screen`), insert:

```python
    # Stage 4.6: Position sizing + concentration caps (advisory weights)
    ranked_df = attach_weights(ranked_df, cfg)
```

- [ ] **Step 6: Verify nothing broke**

Run: `python3 -m pytest tests/test_sizing.py tests/test_config.py -q`
Expected: all pass

- [ ] **Step 7: Commit**

```bash
git add src/sizing.py src/run.py tests/test_sizing.py
git commit -m "feat(sizing): attach inverse-vol capped weights in run pipeline"
```

---

## Task 4: Render `weight_pct` in CSV and markdown output

**Files:**
- Modify: `src/output.py` (CSV_COLUMNS ~line 5; markdown table header + rows ~line 85-98)
- Test: `tests/test_output.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_output.py` (create the file if it does not exist, with
`import pandas as pd` and `from src.output import write_csv, write_markdown`
at the top):

```python
def test_weight_pct_written_to_csv(tmp_path):
    df = pd.DataFrame({
        "ticker": ["AAA", "BBB"],
        "name": ["Alpha", "Beta"],
        "sector": ["Tech", "Health"],
        "composite": [1.2, 0.9],
        "weight_pct": [60.0, 40.0],
    })
    path = write_csv(df, str(tmp_path), "2026-07-16")
    text = open(path).read()
    assert "weight_pct" in text.splitlines()[0]
    assert "60" in text


def test_weight_pct_column_in_markdown(tmp_path):
    df = pd.DataFrame({
        "ticker": ["AAA"],
        "name": ["Alpha"],
        "sector": ["Tech"],
        "composite": [1.2],
        "conviction": [8],
        "weight_pct": [60.0],
    })
    path = write_markdown(df, str(tmp_path), "2026-07-16", squeeze_df=None)
    text = open(path).read()
    assert "Weight" in text
    assert "60.0%" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_output.py -k weight -v`
Expected: FAIL (`weight_pct` not in CSV header / `Weight` not in markdown)

- [ ] **Step 3: Add `weight_pct` to CSV_COLUMNS**

In `src/output.py`, the `CSV_COLUMNS` list (starts line ~5 with
`"ticker", "name", "sector", "composite", "conviction", "factor_coverage",`).
Add `"weight_pct"` immediately after `"composite"`:

```python
CSV_COLUMNS = [
    "ticker", "name", "sector", "composite", "weight_pct", "conviction", "factor_coverage",
    # ... rest unchanged
```

(Keep every other entry in the list exactly as it was; only insert
`"weight_pct"`.)

- [ ] **Step 4: Add the Weight column to the markdown table**

In `write_markdown`, change the header row and separator (currently):

```python
        "| Rank | Ticker | Name | Sector | Composite | Conv | Streak | Signal | Entry | Rationale |",
        "|------|--------|------|--------|-----------|------|--------|--------|-------|-----------|",
```

to:

```python
        "| Rank | Ticker | Name | Sector | Composite | Weight | Conv | Streak | Signal | Entry | Rationale |",
        "|------|--------|------|--------|-----------|--------|------|--------|--------|-------|-----------|",
```

Then, in the row-building loop, after the `comp = f"{row.get('composite', 0):.3f}"`
line, add:

```python
        wt = row.get("weight_pct")
        wt_str = f"{float(wt):.1f}%" if wt is not None and pd.notna(wt) else "—"
```

and change the `lines.append(...)` row to include `{wt_str}` after `{comp}`:

```python
        lines.append(f"| {i} | {row['ticker']} | {name} | {sector} | {comp} | {wt_str} | {conv}/10 | {streak_str} | {es_str} | {entry} | {rationale} |")
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python3 -m pytest tests/test_output.py -k weight -v`
Expected: PASS (2 tests)

- [ ] **Step 6: Commit**

```bash
git add src/output.py tests/test_output.py
git commit -m "feat(sizing): render weight_pct in CSV and markdown output"
```

---

## Task 5: Full-suite regression + final verification

**Files:** none (verification only)

- [ ] **Step 1: Run the full test suite**

Run: `python3 -m pytest -q`
Expected: all tests pass (prior suite + new sizing/config/output tests). If any
pre-existing test fails, confirm it also fails on `main` before this branch —
do not "fix" unrelated failures.

- [ ] **Step 2: Sanity-check the invariants on synthetic data**

Run:
```bash
python3 -c "
import pandas as pd, numpy as np
from src.sizing import realized_vol, inverse_vol_weights, apply_caps
vols = {f't{i}': 0.2 + 0.05*i for i in range(20)}
w = apply_caps(inverse_vol_weights(vols), {t:'Tech' for t in vols}, 0.10, 0.25)
print('sum', round(sum(w.values()),6), 'max', round(max(w.values()),4))
"
```
Expected: `sum 1.0` and `max <= 0.1` (name cap holds; single-sector so the
sector cap logs a best-effort warning — that is the documented behavior).

- [ ] **Step 3: Final commit if any docs need updating**

If the spec's "Known limitations" need a note about observed behavior, update
`docs/superpowers/specs/2026-07-16-position-sizing-concentration-caps-design.md`
and commit. Otherwise no action.

---

## Self-Review Notes

- **Spec coverage:** module (Task 1), config (Task 2), wire point + inverse-vol
  + caps (Task 3), output column (Task 4), invariants (Task 5). All spec
  sections mapped.
- **Type consistency:** `attach_weights(ranked_df, cfg)`, `realized_vol(series,
  window)`, `inverse_vol_weights(vols) -> dict`, `apply_caps(weights, sectors,
  name_cap, sector_cap) -> dict`, output column `weight_pct` (float, percent
  units). Consistent across tasks.
- **Known dependency:** `close_series` and `sector` must survive into
  `ranked_df`; verified no column drops in `build_composite`. `attach_weights`
  degrades to a no-op if `close_series` is absent, so a future refactor that
  drops it fails safe (no weights) rather than crashing.
