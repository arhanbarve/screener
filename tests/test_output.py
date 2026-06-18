# tests/test_output.py
import os, tempfile, pandas as pd, numpy as np
from src.output import write_csv, write_markdown

def make_ranked_df(n=5):
    return pd.DataFrame({
        "ticker":        [f"T{i}" for i in range(n)],
        "name":          [f"Company {i}" for i in range(n)],
        "sector":        ["Tech"] * n,
        "composite":     np.linspace(2.0, 0.5, n),
        "z_mom_12_1":    np.linspace(1.5, 0.0, n),
        "z_rev_breadth": np.linspace(1.0, -0.5, n),
        "z_sue":         np.linspace(0.8, -0.2, n),
        "z_rs_6m":       np.linspace(0.6, 0.0, n),
        "mom_12_1":      np.linspace(0.30, 0.05, n),
        "rev_breadth":   np.linspace(0.6, 0.0, n),
        "sue":           np.linspace(3.0, 0.0, n),
        "rs_6m":         np.linspace(0.15, 0.0, n),
        "gp_assets":     np.linspace(0.40, 0.20, n),
        "pct_from_high": np.linspace(-0.02, -0.10, n),
        "short_float":   np.linspace(0.05, 0.20, n),
        "insider_buys_90d": [2, 1, 0, 1, 3],
        "price":         np.linspace(200.0, 50.0, n),
        "market_cap":    np.linspace(50e9, 1e9, n),
    })

def test_write_csv_creates_file():
    df = make_ranked_df()
    tmpdir = tempfile.mkdtemp()
    path = write_csv(df, tmpdir, "2024-01-31")
    assert os.path.exists(path)
    loaded = pd.read_csv(path)
    assert "composite" in loaded.columns
    assert len(loaded) == 5

def test_write_markdown_creates_file():
    df = make_ranked_df()
    tmpdir = tempfile.mkdtemp()
    path = write_markdown(df, tmpdir, "2024-01-31", squeeze_df=None)
    assert os.path.exists(path)
    content = open(path, "r")
    text = content.read()
    content.close()
    assert "T0" in text
    assert "composite" in text.lower() or "Rank" in text
