# Phases 3–4 — Replay Harness, Dashboard, Documentation (Plan 04)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure what the daily screens did afterwards; show pipeline health and the engine plan on the dashboard; make the docs describe the system that now exists.

**Architecture:** `src/replay.py` reads `output/screen_*.csv` and `data/cache.db` prices (every fetched bar is kept), computes forward returns vs SPY per horizon, flags degraded screens (< 15 rows — the 08-31…09-03 junk-universe screens). `app_shared.py` gains `_render_pipeline_health` (Monitor), a fundamentals expander + AVG FUND cell (Screener) and `_render_engine_plan` (Paper). Docs: STRATEGY.md, README.md, docs/SYSTEM_WRITEUP.md.

---

### Task 3.1: `src/replay.py`

**Files:**
- Create: `src/replay.py`
- Create: `tests/test_replay.py`

A plumbing check and small-sample sanity read, not a backtest: ~30 daily screens with overlapping windows are a handful of independent observations. The point-in-time fundamentals walk-forward (EDGAR `filed` dates) is a separate future project.

- [ ] **Step 1:** **Create `tests/test_replay.py` with exactly this content:**

```python
"""Replay harness: forward returns, degraded flags, breakdowns."""
import numpy as np
import pandas as pd
import pytest

from src import replay


def _closes(start_price, n=30, step=1.0, start="2026-08-03"):
    idx = pd.bdate_range(start=start, periods=n)
    return pd.Series(start_price + step * np.arange(n), index=idx)


def test_forward_return_from_as_of_or_previous_bar():
    c = _closes(100.0)
    # 2026-08-03 is a Monday; 5 bars later the close is 105.
    assert replay.forward_return(c, "2026-08-03", 5) == pytest.approx(0.05)
    # A weekend as_of uses the last bar before it.
    sat = replay.forward_return(c, "2026-08-08", 5)
    fri = replay.forward_return(c, "2026-08-07", 5)
    assert sat == fri
    assert np.isnan(replay.forward_return(c, "2026-09-30", 5))     # not enough forward bars
    assert np.isnan(replay.forward_return(c, "2026-07-01", 5))     # before the series
    assert np.isnan(replay.forward_return(None, "2026-08-03", 5))


def test_replay_excess_vs_spy_and_degraded_flag():
    screens = [
        ("2026-08-03", pd.DataFrame({"ticker": ["A", "B"] + [f"F{i}" for i in range(18)], "entry": ["OK"] * 20,
                                     "conviction": [8] * 20})),
        ("2026-08-04", pd.DataFrame({"ticker": ["A"], "entry": ["OK"], "conviction": [3]})),   # degraded
    ]
    closes = {"SPY": _closes(100.0, step=1.0), "A": _closes(100.0, step=2.0), "B": _closes(100.0, step=0.0)}
    table = replay.replay(screens, closes, horizons=(5,), top_n=20, min_rows=15)
    first = table[(table["date"] == "2026-08-03") & (table["horizon"] == 5)].iloc[0]
    assert first["n"] == 2                                  # F* tickers have no prices → excluded
    assert first["spy_ret"] == pytest.approx(0.05)
    assert first["mean_excess"] == pytest.approx(np.mean([0.10 - 0.05, 0.0 - 0.05]))
    assert first["hit_rate"] == 0.5 and not first["degraded"]
    second = table[table["date"] == "2026-08-04"].iloc[0]
    assert bool(second["degraded"]) is True
    summ = replay.summary(table)
    assert list(summ["screens"]) == [1]                    # degraded screen excluded


def test_breakdown_by_entry_and_conviction_bucket():
    df = pd.DataFrame({"ticker": ["A", "B"] + [f"F{i}" for i in range(18)],
                       "entry": ["STRONG", "WAIT"] + ["OK"] * 18, "conviction": [9, 2] + [5] * 18})
    closes = {"SPY": _closes(100.0), "A": _closes(100.0, step=2.0), "B": _closes(100.0, step=0.0)}
    by_entry = replay.breakdown([("2026-08-03", df)], closes, "entry", horizon=5, min_rows=15)
    assert set(by_entry["group"]) == {"STRONG", "WAIT"}
    strong = by_entry[by_entry["group"] == "STRONG"].iloc[0]
    assert strong["mean_excess"] == pytest.approx(0.05) and strong["hit_rate"] == 1.0
    by_conv = replay.breakdown([("2026-08-03", df)], closes, "conviction", horizon=5, min_rows=15)
    assert set(by_conv["group"]) == {"8-10", "1-4"}
    assert replay.breakdown([("2026-08-03", df)], closes, "missing_col", 5).empty


def test_load_screens_and_closes(tmp_path):
    (tmp_path / "screen_2026-08-03.csv").write_text("ticker,composite\nA,1.0\n")
    (tmp_path / "screen_bad.csv").write_text("not,a,symbol,file\n1,2,3,4\n")
    screens = replay.load_screens(str(tmp_path))
    assert [d for d, _ in screens] == ["2026-08-03"]
    import sqlite3
    db = tmp_path / "c.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE prices (ticker TEXT, date TEXT, open REAL, high REAL, low REAL, close REAL, volume INTEGER, fetched_at TEXT)")
    conn.executemany("INSERT INTO prices VALUES (?,?,?,?,?,?,?,?)",
                     [("A", "2026-08-03", 1, 1, 1, 10.0, 1, ""), ("A", "2026-08-04", 1, 1, 1, 11.0, 1, "")])
    conn.commit(); conn.close()
    closes = replay.load_closes(str(db), {"A", "ZZZ"})
    assert list(closes) == ["A"] and closes["A"].iloc[-1] == 11.0
    assert replay.load_closes(str(db), set()) == {}
```

- [ ] **Step 2:** Run:

```bash
/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m pytest tests/test_replay.py -q
```

Expected: FAIL at import.

- [ ] **Step 3:** **Create `src/replay.py` with exactly this content:**

```python
"""Replay harness: what did the daily screens actually do afterwards?

    python -m src.replay [--top 20] [--horizons 5,10,20] [--min-rows 15]

For every output/screen_YYYY-MM-DD.csv, take the top N tickers and measure the
average forward return over each horizon from that day's close, minus SPY over
the same window, using the OHLCV already in data/cache.db (the `prices` table
keeps every fetched bar, so two months of daily runs give two months of
forward returns for free). Also breaks results down by entry grade and by
conviction bucket, and flags screens whose row count is suspiciously small —
the 2026-08-31 → 09-03 screens came from a ~40-name universe and must not be
read as evidence about the strategy.

Pure functions take frames; `main` does the IO. This is a plumbing check and a
small-sample sanity read, not a backtest: ~30 daily screens with overlapping
20-day windows are a handful of independent observations.
"""
import argparse
import glob
import os
import sqlite3
from datetime import date

import numpy as np
import pandas as pd

DEFAULT_HORIZONS = (5, 10, 20)


def load_screens(output_dir: str) -> list[tuple[str, pd.DataFrame]]:
    out = []
    for path in sorted(glob.glob(os.path.join(output_dir, "screen_*.csv"))):
        d = os.path.basename(path)[len("screen_"):-len(".csv")]
        try:
            df = pd.read_csv(path)
        except Exception:  # noqa: BLE001 — a corrupt file is skipped, not fatal
            continue
        if "ticker" in df.columns:
            out.append((d, df))
    return out


def load_closes(db_path: str, tickers: set[str]) -> dict[str, pd.Series]:
    """ticker -> close series indexed by date (Timestamp), from cache.db."""
    if not tickers:
        return {}
    conn = sqlite3.connect(db_path)
    marks = ",".join("?" * len(tickers))
    rows = conn.execute(
        f"SELECT ticker, date, close FROM prices WHERE ticker IN ({marks}) ORDER BY ticker, date",
        sorted(tickers)).fetchall()
    conn.close()
    df = pd.DataFrame(rows, columns=["ticker", "date", "close"])
    if df.empty:
        return {}
    df["date"] = pd.to_datetime(df["date"])
    return {t: g.set_index("date")["close"] for t, g in df.groupby("ticker")}


def forward_return(closes: pd.Series, as_of: str, horizon: int) -> float:
    """Return from the close ON `as_of` (or the last bar before it) to the close
    `horizon` trading bars later. NaN if either end is missing."""
    if closes is None or closes.empty:
        return float("nan")
    ts = pd.Timestamp(as_of)
    idx = closes.index
    start_pos = idx.searchsorted(ts, side="right") - 1
    if start_pos < 0:
        return float("nan")
    end_pos = start_pos + horizon
    if end_pos >= len(idx):
        return float("nan")
    p0, p1 = float(closes.iloc[start_pos]), float(closes.iloc[end_pos])
    if p0 <= 0:
        return float("nan")
    return p1 / p0 - 1.0


def replay(screens: list[tuple[str, pd.DataFrame]], closes: dict[str, pd.Series],
           horizons=DEFAULT_HORIZONS, top_n: int = 20, min_rows: int = 15) -> pd.DataFrame:
    """One row per screen date per horizon: n, mean excess return vs SPY,
    hit rate (share of names beating SPY), plus a `degraded` flag."""
    spy = closes.get("SPY")
    records = []
    for d, df in screens:
        head = df.head(top_n)
        degraded = len(df) < min_rows
        for h in horizons:
            rets, excess = [], []
            spy_r = forward_return(spy, d, h) if spy is not None else float("nan")
            for t in head["ticker"].astype(str):
                r = forward_return(closes.get(t), d, h)
                if r == r:
                    rets.append(r)
                    if spy_r == spy_r:
                        excess.append(r - spy_r)
            if not rets:
                continue
            records.append({
                "date": d, "horizon": h, "n": len(rets), "degraded": degraded,
                "mean_ret": float(np.mean(rets)),
                "spy_ret": spy_r,
                "mean_excess": float(np.mean(excess)) if excess else float("nan"),
                "hit_rate": float(np.mean([e > 0 for e in excess])) if excess else float("nan"),
            })
    return pd.DataFrame(records)


def breakdown(screens: list[tuple[str, pd.DataFrame]], closes: dict[str, pd.Series],
              by: str, horizon: int = 20, top_n: int = 20, min_rows: int = 15) -> pd.DataFrame:
    """Mean excess return grouped by a screen column (entry grade, conviction
    bucket, entry_signal). Degraded screens are excluded."""
    spy = closes.get("SPY")
    rows = []
    for d, df in screens:
        if len(df) < min_rows or by not in df.columns:
            continue
        spy_r = forward_return(spy, d, horizon) if spy is not None else float("nan")
        for _, r in df.head(top_n).iterrows():
            fr = forward_return(closes.get(str(r["ticker"])), d, horizon)
            if fr != fr or spy_r != spy_r:
                continue
            key = r[by]
            if by == "conviction":
                key = "8-10" if key >= 8 else ("5-7" if key >= 5 else "1-4")
            rows.append({"group": str(key), "excess": fr - spy_r})
    if not rows:
        return pd.DataFrame(columns=["group", "n", "mean_excess", "hit_rate"])
    g = pd.DataFrame(rows).groupby("group")["excess"]
    return pd.DataFrame({"n": g.size(), "mean_excess": g.mean(),
                         "hit_rate": g.apply(lambda s: float((s > 0).mean()))}).reset_index()


def summary(table: pd.DataFrame) -> pd.DataFrame:
    """Per-horizon aggregate over non-degraded screens."""
    if table.empty:
        return table
    ok = table[~table["degraded"]]
    if ok.empty:
        return pd.DataFrame()
    g = ok.groupby("horizon")
    return pd.DataFrame({"screens": g.size(), "mean_excess": g["mean_excess"].mean(),
                         "median_excess": g["mean_excess"].median(),
                         "share_positive": g["mean_excess"].apply(lambda s: float((s > 0).mean()))}).reset_index()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="replay", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", default="output")
    p.add_argument("--db", default="data/cache.db")
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--horizons", default="5,10,20")
    p.add_argument("--min-rows", type=int, default=15)
    a = p.parse_args(argv)
    horizons = tuple(int(h) for h in a.horizons.split(","))

    screens = load_screens(a.output_dir)
    tickers = {"SPY"} | {str(t) for _, df in screens for t in df.head(a.top)["ticker"]}
    closes = load_closes(a.db, tickers)
    table = replay(screens, closes, horizons, a.top, a.min_rows)
    pd.set_option("display.width", 200)
    print(f"{len(screens)} screens, {sum(1 for _, d in screens if len(d) < a.min_rows)} flagged degraded (< {a.min_rows} rows)\n")
    print(table.to_string(index=False, float_format=lambda x: f"{x:+.2%}" if abs(x) < 5 else f"{x:.0f}"))
    print("\nSummary (non-degraded screens):")
    print(summary(table).to_string(index=False, float_format=lambda x: f"{x:+.2%}" if abs(x) < 5 else f"{x:.0f}"))
    for by in ("entry", "conviction", "entry_signal"):
        print(f"\nBy {by} (h={horizons[-1]}):")
        print(breakdown(screens, closes, by, horizons[-1], a.top, a.min_rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4:** Run:

```bash
/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m pytest tests/test_replay.py -q
```

Expected: 4 passed.

- [ ] **Step 5:** Run:

```bash
/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m src.replay --top 20 --horizons 5,10,20 | head -60
```

Expected: a table with one row per screen date × horizon, `degraded True` for 2026-08-31…09-03, a per-horizon summary over non-degraded screens, and breakdowns by `entry`, `conviction`, `entry_signal`. Report the h20 mean excess to the user without interpretation beyond 'small sample'.

- [ ] **Step 6:** Commit as `feat(replay): forward-return harness over the daily screens`. Per the user's global CLAUDE.md §6: run `git status` first and **ask the user before committing**. Never add a Co-Authored-By line.

**Edge cases this task must handle (each has a test):**
- Forward return uses the last bar on/before `as_of`; NaN when the window runs past the data or before it.
- Tickers without prices are excluded and reduce `n`; degraded screens are flagged and excluded from the summary.
- Breakdown buckets conviction into 1-4/5-7/8-10; missing column → empty frame.


### Task 4.1: Dashboard — health strip, fundamentals, engine plan

**Files:**
- Modify: `app_shared.py`

Three additive renderers; no layout changes. The health strip is what would have shown `ADV GATE 53` in red instead of `✓ LAST RUN OK`.

- [ ] **Step 1:** Apply:

**Apply this unified diff to `app_shared.py`** (from the reference patch; `patch -p1 < docs/superpowers/plans/2026-09-04-overhaul-reference.patch` applies all of them at once):

```diff
--- a/app_shared.py	2026-09-03 18:00:00
+++ b/app_shared.py	2026-09-04 12:51:18
@@ -1718,6 +1718,7 @@
     avg        = float(df["composite"].mean()) if "composite" in df.columns else 0.0
     confirms   = int((df["entry_signal"] == "confirm_entry").sum()) if "entry_signal" in df.columns else 0
     avoids     = int((df["entry_signal"] == "avoid").sum()) if "entry_signal" in df.columns else 0
+    fund_avg   = float(df["fund_score"].mean()) if "fund_score" in df.columns and df["fund_score"].notna().any() else None
 
     # ── Summary strip ─────────────────────────────────────────────────────────
     st.markdown(f"""
@@ -1747,10 +1748,22 @@
   <div class="summary-cell">
     <div class="summary-label">AVOIDS</div>
     <div class="summary-value bear">{avoids}</div>
+  </div>
+  <div class="summary-cell">
+    <div class="summary-label">AVG FUND</div>
+    <div class="summary-value">{f"{fund_avg:.2f}" if fund_avg is not None else "—"}</div>
   </div>
 </div>
 """, unsafe_allow_html=True)
 
+    # ── Fundamentals table (block sub-scores) ────────────────────────────────
+    _fund_cols = [c for c in ["ticker", "fund_score", "fund_quality", "fund_growth", "fund_value",
+                              "fund_invest", "fund_strength", "peTTM", "roeTTM", "revenueGrowthTTMYoy",
+                              "eps_rev_30d", "days_to_earnings"] if c in df.columns]
+    if len(_fund_cols) > 2:
+        with st.expander("Fundamentals — block sub-scores (percentile 0–1) and key ratios"):
+            st.dataframe(df[_fund_cols].round(3), hide_index=True, use_container_width=True)
+
     # ── Top 3 cards ───────────────────────────────────────────────────────────
     cols = st.columns(3)
     for i, col in enumerate(cols):
@@ -1776,7 +1789,9 @@
     # ── Glossary ──────────────────────────────────────────────────────────────
     with st.expander("📖 What do these indicators mean?"):
         st.markdown("""
-**Composite Score** — Weighted z-score across 7 factors: 28% 12-month momentum, 20% analyst revision breadth, 17% earnings surprise, 15% 6-month RS vs SPY, 10% technical alignment, 5% RS slope, 5% streak bonus. Higher = stronger setup.
+**Composite Score** — Weighted z-score across four blocks: price momentum 40% (12-1 momentum, residual momentum, RS vs SPY, RS acceleration/slope, distance from 52-week high), earnings momentum 20% (SUE, analyst rating breadth and shift), **fundamentals 32%** (quality, growth, value, investment, balance-sheet strength as percentile sub-scores, plus insider buying), technical confirmation 8%. Names below the 30th percentile on fundamentals are not ranked at all. Finalists are re-ranked by 30-day EPS estimate revisions. Higher = stronger setup.
+
+**Fund (0–1)** — the fundamentals block score: the mean of up to five percentile sub-scores (quality, growth, value, investment, strength). 0.70+ is the top 30% of the investable universe on fundamentals; below 0.30 is excluded from the ranking.
 
 **Conviction (1–10)** — Synthesis of four layers: rank position (top 3 = 3pts), streak consistency (≥7 days = 3pts), technical alignment across 8 indicators (≥6 green = 2pts), and fundamental quality (gross profitability, insider buying, short float). Use this to decide position sizing — high conviction = larger starter position.
 
@@ -2982,6 +2997,54 @@
     return paper.live_view()
 
 
+def _render_engine_plan() -> None:
+    """The portfolio engine's most recent plan: legs, reviewer decisions,
+    executions. Read through datastore so the cloud deploy sees the copy
+    publish_data.sh pushed to the private repo."""
+    names = sorted(datastore.list_names("trading/plans", "*.json"), reverse=True)
+    if not names:
+        return
+    raw = datastore.read_text(f"trading/plans/{names[0]}")
+    try:
+        plan = json.loads(raw) if raw else None
+    except json.JSONDecodeError:
+        plan = None
+    if not plan:
+        return
+    _page_title(f"ENGINE PLAN · {plan.get('target_date', '—')}")
+    reg = (plan.get("regime") or {}).get("regime", "—")
+    tgt = (plan.get("targets") or {})
+    brk = plan.get("breaker") or {}
+    st.markdown(
+        f'<div style="font-family:var(--mono);font-size:0.7rem;color:var(--muted);margin-bottom:0.5rem">'
+        f'regime <b>{_html.escape(str(reg))}</b> · alpha target {tgt.get("alpha_pct", 0):.0%} · '
+        f'screen {"usable" if plan.get("screen_usable") else "NOT usable — " + _html.escape(str(plan.get("screen_note", "")))}'
+        f'{" · <span style=color:var(--bear)>BREAKER ACTIVE</span>" if brk.get("active") else ""}</div>',
+        unsafe_allow_html=True,
+    )
+    rows = []
+    reviews = plan.get("reviews") or {}
+    executed = plan.get("executed") or {}
+    failures = plan.get("failures") or {}
+    for l in plan.get("legs") or []:
+        lid = l["id"]
+        rv = reviews.get(lid, {}).get("decision", "")
+        ex = executed.get(lid) or failures.get(lid) or {}
+        status = ex.get("status") or ("skipped" if rv in ("SKIP", "DEFER") else "pending")
+        rows.append({"leg": lid, "kind": l["kind"], "symbol": l["symbol"], "side": l["side"],
+                     "size": f"${l['notional']:,.0f}" if l.get("notional") else f"{l['qty']:g} sh",
+                     "fixed": "" if l["overridable"] else "✓", "review": rv, "status": status,
+                     "reason": l["reason"][:110]})
+    if rows:
+        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
+    else:
+        st.markdown('<div style="font-family:var(--mono);font-size:0.65rem;color:var(--muted)">no legs — hold</div>',
+                    unsafe_allow_html=True)
+    for w in plan.get("warnings") or []:
+        st.markdown(f'<div style="font-family:var(--mono);font-size:0.62rem;color:var(--accent)">! {_html.escape(str(w))}</div>',
+                    unsafe_allow_html=True)
+
+
 def _render_paper() -> None:
     from src import paper
 
@@ -3210,6 +3273,7 @@
 
     # ── run cadence ──────────────────────────────────────────────────────────
     st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
+    _render_engine_plan()
     _render_paper_cadence(snap.get("cadence") or {})
 
 
@@ -3447,6 +3511,45 @@
 
 
 @st.fragment(run_every="5s")
+def _render_pipeline_health(rs: dict) -> None:
+    """Stage counts and the health verdict from run_status.json.stats.
+
+    This is the strip that would have shown "ADV 53" in red from 2026-08-31 to
+    09-03 instead of a green "LAST RUN OK". Renders nothing when the run wrote
+    no stats (older runs, or a run that died before stage 2).
+    """
+    stats = (rs or {}).get("stats") or {}
+    if not stats:
+        return
+    ok = stats.get("ok")
+    color = "var(--bull)" if ok else "var(--bear)"
+    verdict = "HEALTH OK" if ok else "HEALTH FAILED"
+    if stats.get("degraded"):
+        verdict, color = "DEGRADED (forced)", "var(--accent)"
+    cells = ""
+    for label, key in [("UNIVERSE", "universe"), ("TRADABLE", "tradable"), ("PRICED", "priced"),
+                       ("ADV GATE", "adv_survivors"), ("CAP GATE", "cap_survivors"),
+                       ("FUND GATE", None), ("RANKED", "ranked"), ("SELECTED", "selected")]:
+        if key is None:
+            fg = (stats.get("fund_gate") or {})
+            val = fg.get("after") if fg.get("after") is not None else "—"
+            if fg.get("skipped"):
+                val = f"skip"
+        else:
+            v = stats.get(key)
+            val = f"{int(v):,}" if isinstance(v, (int, float)) else "—"
+        cells += (f'<div class="summary-cell"><div class="summary-label">{label}</div>'
+                  f'<div class="summary-value">{_html.escape(str(val))}</div></div>')
+    reasons = "; ".join(str(r) for r in (stats.get("reasons") or []) + (stats.get("warnings") or []))
+    st.markdown(
+        f'<div style="font-family:var(--mono);font-size:0.7rem;font-weight:700;color:{color};'
+        f'margin:0.6rem 0 0.3rem">{verdict}'
+        f'{" · " + _html.escape(reasons[:220]) if reasons else ""}</div>'
+        f'<div class="summary-strip">{cells}</div>',
+        unsafe_allow_html=True,
+    )
+
+
 def _render_monitor():
     log_path = _find_todays_log()
     lock_active = Path("/tmp/screener_run.lock").exists()
@@ -3485,6 +3588,7 @@
             headline = ('<div style="font-family:var(--mono);font-size:1rem;color:var(--muted)">'
                         '— NO RUN TODAY</div>')
         st.markdown(headline, unsafe_allow_html=True)
+        _render_pipeline_health(_rs)
         _out = sorted(datastore.list_names("output", "screen_*.csv"), reverse=True)
         if _out:
             st.markdown(
@@ -3603,6 +3707,9 @@
                 unsafe_allow_html=True,
             )
 
+    # ── Health strip (from run_status.json, once the run has written it) ────────
+    _render_pipeline_health(_load_run_status())
+
     # ── Funnel panel ─────────────────────────────────────────────────────────
     def _fv(v): return f"{v:,}" if v is not None else "—"
     universe_n  = _fv(s["universe"])
```

- [ ] **Step 2:** Run:

```bash
/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -c "import ast;ast.parse(open('app_shared.py').read());import app_shared;print('ok')"
```

Expected: `ok`.

- [ ] **Step 3:** Visual check (the user's CLAUDE.md asks for browser verification of UI changes): `streamlit run app.py`, open Monitor (health strip under the run headline), Screener (AVG FUND cell + 'Fundamentals' expander once a Phase 1 CSV exists), Paper (ENGINE PLAN table once a plan file exists). Use the Chrome tools to screenshot each and confirm with the user.

- [ ] **Step 4:** Commit as `feat(ui): pipeline health strip, fundamentals table, engine plan view`. Per the user's global CLAUDE.md §6: run `git status` first and **ask the user before committing**. Never add a Co-Authored-By line.


### Task 4.2: Documentation

- [ ] **Step 1:** `STRATEGY.md` (private, gitignored — edit freely). Replace §3 "Pre-Scoring Gates → Quality Gate" with the fundamentals floor gate (fund_score ≥ 0.30, coverage guard), replace §4's factor tables with the four-block table from the spec (weights 0.40/0.20/0.32/0.08; list the five sub-scores and their inputs with signs), rename "Revision Breadth / Revision Magnitude" to "Analyst rating breadth / rating shift" and add the finalist `eps_rev_30d` stage, add a §5.2 "Portfolio engine (paper account)" summarising the policy table from the spec, and correct §10 ("It is not a value screener … does not consider P/E ratios") to say valuation, growth, quality, investment and balance-sheet strength now decide eligibility and a third of the score. Add a dated changelog line at the top: `2026-09-04 — data pipeline restored; fundamentals block; portfolio engine`.

- [ ] **Step 2:** `README.md` (public). In "Factors": replace the composite table with the four blocks and add one paragraph on the fundamentals gate. In "Gates": Liquidity now reads "Alpaca-tradable, ≥ $5M ADV, ≥ $300M cap (Finnhub)"; Quality reads "fundamentals block ≥ 30th percentile". Add a "Pipeline health" paragraph under "Automated daily runs": floors, `screen_latest.json`, failed-run behaviour. In "Paper trading": describe the engine + reviewer split in three sentences and list the new `trader_cli` subcommands. In "Cache": add `Finnhub metrics/profiles — 7 days`, `Alpaca assets — 24 h`, `EDGAR facts — 30 days`, `earnings calendar — 20 h`. Update the test count (795) and the project-structure block (`finnhub_data.py`, `fundamentals_block.py`, `enrich.py`, `portfolio_engine.py`, `trader_plan.py`, `replay.py`, `cache_maint.py`).

- [ ] **Step 3:** `docs/SYSTEM_WRITEUP.md` (private). Replace the pipeline ASCII diagram with the one in the spec's Architecture section and add a short "2026-09 incident" subsection describing the quarantine bug and the health floors that now prevent it.

- [ ] **Step 4:** `docs/agent/RUNLOG.md` (private): add a dated entry summarising this overhaul and pointing at the spec, the vetting report and the reference patch.

- [ ] **Step 5:** Ask the user before committing (`docs: describe the fundamentals block, health floors and portfolio engine`).

### Task 4.3: Post-rollout checks (one week)

- [ ] Each evening for five sessions, read `logs/run_<date>.log` for the `[health]`/`[fund_gate]`/`[enrich]` lines and `trading/plans/<target>.json` for legs and rejections. Note any rejection reason that repeats every night (a policy parameter that is too tight) and any Finnhub `forbidden`/`failed` counts.
- [ ] After five sessions run `/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m src.replay` again; the new screens should no longer be flagged degraded.
- [ ] Review `config.yaml → trader` with the user: the aggressive parameters were chosen before seeing the engine's first week of legs.
