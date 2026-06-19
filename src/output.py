import os
import pandas as pd

CSV_COLUMNS = [
    "ticker", "name", "sector", "composite",
    "z_mom_12_1", "z_rev_breadth", "z_sue", "z_rs_6m",
    "mom_12_1", "rev_breadth", "sue", "rs_6m",
    "gp_assets", "pct_from_high", "short_float", "insider_buys_90d",
    "price", "market_cap",
    "rsi_14", "macd", "vol_surge", "above_sma20", "above_sma50",
    "stoch_k", "stoch_d", "stoch_cross", "bb_pct_b", "bb_width", "adx", "mfi",
    "entry",
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
        "| Rank | Ticker | Name | Sector | Composite | Entry | Rationale |",
        "|------|--------|------|--------|-----------|-------|-----------|",
    ]
    for i, (_, row) in enumerate(df.iterrows(), 1):
        name      = str(row.get("name", ""))[:30]
        sector    = str(row.get("sector", ""))[:20]
        comp      = f"{row.get('composite', 0):.3f}"
        entry     = str(row.get("entry", ""))
        rationale = _rationale(row)
        lines.append(f"| {i} | {row['ticker']} | {name} | {sector} | {comp} | {entry} | {rationale} |")

    lines += [
        "",
        "## Entry Timing Details",
        "",
        "| Ticker | Price | RSI(14) | MFI(14) | Stoch %K/%D | MACD | BB %B | ADX | Vol Surge | >SMA20 | >SMA50 | Entry |",
        "|--------|-------|---------|---------|-------------|------|-------|-----|-----------|--------|--------|-------|",
    ]
    for _, row in df.iterrows():
        nan = float("nan")
        def _f(v, fmt, fallback="—"):
            return fmt.format(v) if v == v and v is not None else fallback

        price    = f"${row.get('price', 0):.2f}"
        rsi      = _f(row.get("rsi_14",   nan), "{:.1f}")
        mfi      = _f(row.get("mfi",      nan), "{:.1f}")
        sk       = row.get("stoch_k", nan)
        sd       = row.get("stoch_d", nan)
        cross    = row.get("stoch_cross", False)
        stoch_str = f"{sk:.0f}/{sd:.0f}{'*' if cross else ''}" if sk == sk and sd == sd else "—"
        macd     = str(row.get("macd", ""))
        bb       = _f(row.get("bb_pct_b", nan), "{:.2f}")
        adx_v    = _f(row.get("adx",      nan), "{:.1f}")
        vol      = row.get("vol_surge", nan)
        vol_str  = f"{vol:.2f}x" if vol == vol else "—"
        sma20    = "Y" if row.get("above_sma20") else "N"
        sma50    = "Y" if row.get("above_sma50") else "N"
        entry    = str(row.get("entry", ""))
        lines.append(
            f"| {row['ticker']} | {price} | {rsi} | {mfi} | {stoch_str} | {macd} | {bb} | {adx_v} | {vol_str} | {sma20} | {sma50} | {entry} |"
        )

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
    nan = float("nan")
    for i, (_, row) in enumerate(df.head(10).iterrows(), 1):
        comp  = row.get("composite", 0)
        price = row.get("price", 0)
        entry = row.get("entry", "")
        rsi   = row.get("rsi_14", nan)
        mfi   = row.get("mfi",    nan)
        adx   = row.get("adx",    nan)
        sk    = row.get("stoch_k", nan)
        sd    = row.get("stoch_d", nan)
        macd  = row.get("macd", "")
        rsi_s   = f"RSI={rsi:.0f}"         if rsi == rsi else "RSI=—"
        mfi_s   = f"MFI={mfi:.0f}"         if mfi == mfi else "MFI=—"
        adx_s   = f"ADX={adx:.0f}"         if adx == adx else "ADX=—"
        stoch_s = f"Stoch={sk:.0f}/{sd:.0f}" if sk == sk and sd == sd else "Stoch=—"
        print(
            f"  {i:2d}. {row['ticker']:<8} composite={comp:+.3f}  ${price:.2f}"
            f"  entry={entry:<6}  {rsi_s}  {mfi_s}  {stoch_s}  {adx_s}  macd={macd}"
        )
    print()
