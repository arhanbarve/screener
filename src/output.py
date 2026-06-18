import os
import pandas as pd

CSV_COLUMNS = [
    "ticker", "name", "sector", "composite",
    "z_mom_12_1", "z_rev_breadth", "z_sue", "z_rs_6m",
    "mom_12_1", "rev_breadth", "sue", "rs_6m",
    "gp_assets", "pct_from_high", "short_float", "insider_buys_90d",
    "price", "market_cap",
]


def _rationale(row: pd.Series) -> str:
    parts = []
    if row.get("z_mom_12_1", 0) > 1.0:
        parts.append(f"Top-decile 12-1 momentum ({row['mom_12_1']:.1%})")
    if row.get("z_rev_breadth", 0) > 1.0:
        rev = row.get("rev_breadth", 0)
        parts.append(f"Analysts revised up (breadth={rev:.2f})")
    if row.get("z_sue", 0) > 1.0:
        parts.append(f"Beat estimate ({row['sue']:.1f} SUE)")
    if row.get("z_rs_6m", 0) > 1.0:
        parts.append(f"Strong 6m RS vs SPY ({row['rs_6m']:.1%})")
    if row.get("insider_buys_90d", 0) >= 2:
        parts.append(f"{int(row['insider_buys_90d'])} insiders bought last 90d")
    return "; ".join(parts) if parts else "Composite score"


def write_csv(df: pd.DataFrame, out_dir: str, date_str: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"screen_{date_str}.csv")
    cols = [c for c in CSV_COLUMNS if c in df.columns]
    out = open(path, "w")
    df[cols].to_csv(out, index=False, float_format="%.6f")
    out.close()
    return path


def write_markdown(
    df: pd.DataFrame,
    out_dir: str,
    date_str: str,
    squeeze_df: pd.DataFrame | None,
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"screen_{date_str}.md")
    lines = [
        f"# Stock Screen — {date_str}",
        "",
        f"**Universe screened:** {len(df)} finalists from US common-stock universe",
        "",
        "## Top Ranked Names",
        "",
        "| Rank | Ticker | Name | Sector | Composite | Rationale |",
        "|------|--------|------|--------|-----------|-----------|",
    ]
    for i, (_, row) in enumerate(df.iterrows(), 1):
        name    = str(row.get("name", ""))[:30]
        sector  = str(row.get("sector", ""))[:20]
        comp    = f"{row.get('composite', 0):.3f}"
        rationale = _rationale(row)
        lines.append(f"| {i} | {row['ticker']} | {name} | {sector} | {comp} | {rationale} |")

    if squeeze_df is not None and len(squeeze_df) > 0:
        lines += [
            "",
            "## Short Squeeze Candidates",
            "",
            "| Ticker | Short Float | Days to Cover | 1M Mom |",
            "|--------|-------------|---------------|--------|",
        ]
        for _, row in squeeze_df.iterrows():
            sf  = f"{row.get('short_float', 0):.1%}"
            dtc = f"{row.get('days_to_cover', 0):.1f}"
            m1  = f"{row.get('mom_1m', 0):.1%}"
            lines.append(f"| {row['ticker']} | {sf} | {dtc} | {m1} |")

    lines += ["", "---", "*Research tool only. Not investment advice.*"]
    f = open(path, "w")
    f.write("\n".join(lines) + "\n")
    f.close()
    return path


def print_top10(df: pd.DataFrame):
    print("\n=== TOP 10 ===")
    for i, (_, row) in enumerate(df.head(10).iterrows(), 1):
        comp = row.get("composite", 0)
        price = row.get("price", 0)
        print(f"  {i:2d}. {row['ticker']:<8} composite={comp:+.3f}  price=${price:.2f}  {_rationale(row)}")
    print()
