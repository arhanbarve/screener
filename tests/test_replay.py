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
