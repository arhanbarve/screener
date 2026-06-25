import html as _html
import math
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as _components
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
load_dotenv()

from src.positions import (
    add_position,
    compute_exit_signals,
    fetch_ohlcv,
    get_current_price,
    load_positions,
    remove_position,
)
from src.spy_analysis import compute_spy_regime
from src.news import (
    get_market_news,
    get_stock_news,
    analyze_market_news,
    analyze_stock_news,
)

st.set_page_config(
    page_title="Stock Screener",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stSidebar"] { background: #1e293b; }
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stRadio label { font-size: 15px; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 📊 Screener")
    st.markdown("---")
    if st.button("🔄 Refresh", use_container_width=True,
                 help="Clear all cached data and reload (keyboard shortcut: R)"):
        st.cache_data.clear()
        st.session_state.pop("news_cache", None)
        st.rerun()
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["📈 Screener Results", "🌍 Market Regime", "📋 Open Positions", "🧾 Filing Edge"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("*Momentum factor strategy*")


# ── Cached data loaders ───────────────────────────────────────────────────────

@st.cache_data(ttl=900)
def _cached_position_data(ticker: str) -> tuple[dict, float | None]:
    df = fetch_ohlcv(ticker, days=60)
    signals = compute_exit_signals(df)
    price = get_current_price(ticker)
    return signals, price


@st.cache_data(ttl=64800)  # 18h — recompute once per trading day
def _cached_spy_regime() -> dict:
    return compute_spy_regime()


@st.cache_data(ttl=14400)  # 4h
def _cached_market_news() -> tuple[list, dict]:
    articles = get_market_news(20)
    analysis = analyze_market_news(articles)
    return articles, analysis


@st.cache_data(ttl=14400)  # 4h
def _cached_stock_news(ticker: str) -> tuple[list, dict]:
    articles = get_stock_news(ticker)
    analysis = analyze_stock_news(ticker, articles)
    return articles, analysis


# ── Shared helpers ────────────────────────────────────────────────────────────

_SENT_COLORS = {
    "bullish":     ("#166534", "#dcfce7"),
    "bearish":     ("#991b1b", "#fee2e2"),
    "neutral":     ("#374151", "#f3f4f6"),
    "mixed":       ("#92400e", "#fef3c7"),
    "unavailable": ("#374151", "#f3f4f6"),
}


def _news_card(analysis: dict, articles: list[dict], headline_count: int = 5) -> None:
    sentiment = analysis.get("sentiment", "neutral")
    summary = _html.escape(analysis.get("summary", ""))
    risks = analysis.get("key_risks", [])

    fg, bg = _SENT_COLORS.get(sentiment, _SENT_COLORS["neutral"])
    risks_html = "".join(
        f'<div style="color:#64748b;font-size:12px;margin-top:4px">'
        f'⚠ {_html.escape(r)}</div>' for r in risks
    )
    st.markdown(f"""
    <div style="background:{bg};border-radius:8px;padding:12px 16px;margin-bottom:8px">
      <span style="background:{fg};color:white;padding:2px 9px;border-radius:6px;
                   font-size:12px;font-weight:700">{sentiment.upper()}</span>
      <div style="color:#1e293b;font-size:14px;margin-top:8px;line-height:1.5">{summary}</div>
      {risks_html}
    </div>
    """, unsafe_allow_html=True)

    visible = [a for a in articles if a.get("headline")][:headline_count]
    if visible:
        st.caption("**Recent headlines:**")
        for a in visible:
            st.caption(f"• {_html.escape(a['headline'])}")


# ── Screener Results ──────────────────────────────────────────────────────────

def _fmt_pct(x: float) -> str:
    if pd.isna(x):
        return "—"
    return f"+{x:.0%}" if x >= 0 else f"{x:.0%}"


def _top3_card(medal: str, row: pd.Series) -> str:
    mom = row.get("mom_12_1", float("nan"))
    rs = row.get("rs_6m", float("nan"))
    _s = row.get("sector", "")
    sector = "—" if pd.isna(_s) or str(_s).strip() == "" else str(_s)
    name = str(row.get("name", "") or "")[:32]
    ticker = str(row.get("ticker", ""))
    score = row.get("composite", 0)
    conviction = int(row.get("conviction", 0) or 0)
    streak_cons = int(row.get("streak_consecutive", 0) or 0)

    streak_badge = (
        f'<span style="background:#fef3c7;color:#92400e;padding:3px 9px;'
        f'border-radius:8px;font-size:12px;font-weight:600">🔥{streak_cons}d</span>'
        if streak_cons >= 2 else ""
    )
    if conviction >= 7:
        conv_bg, conv_fg = "#dcfce7", "#166534"
    elif conviction >= 4:
        conv_bg, conv_fg = "#fef3c7", "#92400e"
    else:
        conv_bg, conv_fg = "#fee2e2", "#991b1b"
    conv_raw = row.get("conviction_news")
    conv_label = f"Conv {conviction}/10" + (f" (was {int(conv_raw)})" if conv_raw is not None and int(conv_raw or conviction) != conviction else "")
    conv_badge = (
        f'<span style="background:{conv_bg};color:{conv_fg};padding:3px 9px;'
        f'border-radius:8px;font-size:12px;font-weight:600">{conv_label}</span>'
    )
    es = str(row.get("entry_signal", "") or "")
    _es_styles = {
        "confirm_entry": ("#166534", "#dcfce7", "✅ Confirm"),
        "wait":          ("#92400e", "#fef3c7", "⏳ Wait"),
        "avoid":         ("#991b1b", "#fee2e2", "🚫 Avoid"),
    }
    if es in _es_styles:
        es_fg, es_bg, es_label = _es_styles[es]
        signal_badge = (
            f'<span style="background:{es_bg};color:{es_fg};padding:3px 9px;'
            f'border-radius:8px;font-size:12px;font-weight:600">{es_label}</span>'
        )
    else:
        signal_badge = ""

    return f"""
    <div style="border:1px solid #e2e8f0;border-radius:12px;padding:18px;background:#f8fafc;height:100%">
      <div style="font-size:28px;margin-bottom:4px">{medal}</div>
      <div style="font-weight:800;font-size:20px;color:#1e293b;letter-spacing:-0.5px">{_html.escape(ticker)}</div>
      <div style="color:#64748b;font-size:12px;margin-bottom:10px">{_html.escape(name)}</div>
      <div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:8px">
        <span style="background:#dcfce7;color:#166534;padding:3px 9px;border-radius:8px;font-size:12px;font-weight:600">
          📈 Mom: {_fmt_pct(mom)}
        </span>
        <span style="background:#dbeafe;color:#1e40af;padding:3px 9px;border-radius:8px;font-size:12px;font-weight:600">
          ⚡ RS: {_fmt_pct(rs)}
        </span>
      </div>
      <div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:8px">
        {conv_badge}
        {streak_badge}
        {signal_badge}
      </div>
      <div style="display:flex;gap:6px">
        <span style="background:#f1f5f9;color:#475569;padding:3px 9px;border-radius:8px;font-size:11px">{_html.escape(sector)}</span>
        <span style="background:#ede9fe;color:#6d28d9;padding:3px 9px;border-radius:8px;font-size:11px;font-weight:600">Score {score:.2f}</span>
      </div>
    </div>
    """


def _render_screener() -> None:
    st.title("📈 Screener Results")

    output_dir = Path("output")
    csv_files = sorted(output_dir.glob("screen_*.csv"), reverse=True)

    if not csv_files:
        st.info("No screen output files found in output/")
        return

    dates = [f.stem.replace("screen_", "") for f in csv_files]
    selected_date = st.selectbox("Screen date", dates)

    try:
        df = pd.read_csv(output_dir / f"screen_{selected_date}.csv")
    except Exception as e:
        st.error(f"Could not read {selected_date}: {e}")
        return

    # ── Summary metrics ───────────────────────────────────────────────────────
    m1, m2, m3 = st.columns(3)
    with m1:
        top_ticker = df.iloc[0]["ticker"] if len(df) > 0 else "—"
        top_score = df.iloc[0]["composite"] if len(df) > 0 and "composite" in df.columns else 0
        st.metric("🥇 Top Pick", top_ticker, f"Score {top_score:.2f}")
    with m2:
        st.metric("📊 Stocks Ranked", len(df))
    with m3:
        avg = df["composite"].mean() if "composite" in df.columns else 0
        st.metric("📈 Avg Score", f"{avg:.2f}")

    st.markdown("")

    # ── Top 3 medal cards ─────────────────────────────────────────────────────
    medals = ["🥇", "🥈", "🥉"]
    cols = st.columns(3)
    for i, (col, medal) in enumerate(zip(cols, medals)):
        if i < len(df):
            with col:
                st.markdown(_top3_card(medal, df.iloc[i]), unsafe_allow_html=True)

    st.markdown("---")

    # ── Full ranked table ─────────────────────────────────────────────────────
    df2 = df.copy()
    df2.insert(0, "Rank", range(1, len(df2) + 1))

    display_df = pd.DataFrame()
    display_df["Rank"] = df2["Rank"]
    display_df["Ticker"] = df2["ticker"]
    display_df["Name"] = df2["name"].str[:35] if "name" in df2.columns else "—"
    display_df["Sector"] = df2["sector"].fillna("—") if "sector" in df2.columns else "—"
    display_df["Score"] = df2["composite"] if "composite" in df2.columns else None
    if "conviction" in df2.columns and "conviction_news" in df2.columns:
        def _conv_label(row):
            c = int(row.get("conviction", 0) or 0)
            raw = row.get("conviction_news")
            if raw is not None and pd.notna(raw) and int(raw) != c:
                delta = c - int(raw)
                return f"{c} ({'+' if delta > 0 else ''}{delta})"
            return str(c)
        display_df["Conviction"] = df2.apply(_conv_label, axis=1)
    else:
        display_df["Conviction"] = df2["conviction"].fillna(0).astype(int) if "conviction" in df2.columns else 0
    display_df["Streak"] = (
        df2["streak_consecutive"].map(lambda x: f"🔥{int(x)}d" if pd.notna(x) and int(x) >= 2 else "—")
        if "streak_consecutive" in df2.columns else "—"
    )
    _es_map = {"confirm_entry": "✅ Confirm", "wait": "⏳ Wait", "avoid": "🚫 Avoid"}
    display_df["Signal"] = (
        df2["entry_signal"].map(lambda x: _es_map.get(str(x), "—") if pd.notna(x) else "—")
        if "entry_signal" in df2.columns else "—"
    )
    _cat_map = {"estimate_up": "📈 Est ↑", "estimate_down": "📉 Est ↓", "none": "—"}
    display_df["Catalyst"] = (
        df2["catalyst"].map(lambda x: _cat_map.get(str(x), "—") if pd.notna(x) else "—")
        if "catalyst" in df2.columns else "—"
    )
    display_df["Mom 12-1"] = df2["mom_12_1"].map(_fmt_pct) if "mom_12_1" in df2.columns else "—"
    display_df["RS vs SPY 6M"] = df2["rs_6m"].map(_fmt_pct) if "rs_6m" in df2.columns else "—"
    display_df["Price"] = (
        df2["price"].map(lambda x: f"${x:,.2f}" if pd.notna(x) else "—")
        if "price" in df2.columns else "—"
    )
    display_df["Mkt Cap"] = (
        df2["market_cap"].map(lambda x: f"${x / 1e9:.1f}B" if pd.notna(x) else "—")
        if "market_cap" in df2.columns else "—"
    )

    max_score = float(df["composite"].max()) if "composite" in df.columns else 5.0

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Rank": st.column_config.NumberColumn(width="small"),
            "Score": st.column_config.ProgressColumn(
                "Score",
                help=(
                    "Composite factor score (weighted z-score):\n"
                    "28% 12-month momentum · 20% analyst revision breadth · "
                    "17% earnings surprise · 15% 6-month RS vs SPY · "
                    "10% technical alignment · 5% RS slope · 5% streak bonus"
                ),
                min_value=0,
                max_value=max_score,
                format="%.2f",
            ),
            "Conviction": st.column_config.ProgressColumn(
                "Conviction",
                help=(
                    "1–10 score synthesizing rank position, streak consistency, "
                    "technical alignment (8 signals), and fundamental quality gates.\n"
                    "≥7 = high conviction  4–6 = moderate  ≤3 = low"
                ),
                min_value=1,
                max_value=10,
                format="%d",
            ),
            "Streak": st.column_config.TextColumn(
                "Streak",
                help="Consecutive trading days this stock appeared in the top results. 🔥 = sustained momentum.",
            ),
            "Signal": st.column_config.TextColumn(
                "Signal",
                help=(
                    "News entry signal overlay:\n"
                    "✅ Confirm = news supports the momentum thesis\n"
                    "⏳ Wait = mixed or uncertain — no edge from news\n"
                    "🚫 Avoid = news contradicts or undermines the thesis"
                ),
            ),
            "Catalyst": st.column_config.TextColumn(
                "Catalyst",
                help="📈 = news likely to drive analyst estimate upgrades  📉 = likely downgrades",
            ),
            "Mom 12-1": st.column_config.TextColumn(
                "Mom 12-1",
                help=(
                    "12-month price return, skipping the most recent month.\n"
                    "Why skip last month? 1-month returns tend to reverse.\n"
                    "The 12-1 window captures durable institutional momentum."
                ),
            ),
            "RS vs SPY 6M": st.column_config.TextColumn(
                "RS vs SPY 6M",
                help=(
                    "Outperformance vs S&P 500 over 6 months.\n"
                    "+300% = the stock beat the index by 300 percentage points.\n"
                    "Filters stocks that rose only because the market rose."
                ),
            ),
        },
    )

    # ── Sustained Movers ─────────────────────────────────────────────────────
    if "streak_consecutive" in df.columns:
        sustained = df[df["streak_consecutive"] >= 3].copy()
        if len(sustained) > 0:
            with st.expander(f"🔥 Sustained Movers — {len(sustained)} stock{'s' if len(sustained) != 1 else ''} with ≥3-day streak"):
                sus_display = sustained[
                    [c for c in ["ticker", "name", "composite", "conviction", "streak_consecutive", "streak_count"] if c in sustained.columns]
                ].rename(columns={"streak_consecutive": "Streak Days", "conviction": "Conviction", "composite": "Score"})
                st.dataframe(sus_display, hide_index=True, use_container_width=True)

    # ── Glossary ──────────────────────────────────────────────────────────────
    with st.expander("📖 What do these indicators mean?"):
        st.markdown("""
**Composite Score** — Weighted z-score across 7 factors: 28% 12-month momentum, 20% analyst revision breadth, 17% earnings surprise, 15% 6-month RS vs SPY, 10% technical alignment, 5% RS slope, 5% streak bonus. Higher = stronger setup.

**Conviction (1–10)** — Synthesis of four layers: rank position (top 3 = 3pts), streak consistency (≥7 days = 3pts), technical alignment across 8 indicators (≥6 green = 2pts), and fundamental quality (gross profitability, insider buying, short float). Use this to decide position sizing — high conviction = larger starter position.

**Streak** — How many consecutive trading days this stock appeared in the screener's top results. A stock ranking in the top 20 for 5+ days has proven staying power; a 1-day appearance may be noise.

**Mom 12-1 (12-month momentum)** — Price return over the past year, *excluding* the most recent month.
Skipping last month removes short-term mean reversion noise. What remains is the slow-moving
institutional momentum that academic research shows persists for 3–12 months.

**RS vs SPY 6M (Relative Strength)** — How much the stock beat the S&P 500 over 6 months.
+300% = outperformed by 300 percentage points — not just a rising tide lift.

**Rev Breadth (Analyst Revision Breadth)** — Net % of sell-side analysts raising EPS estimates
vs cutting. Rising estimates → institutions are likely accumulating.

**SUE (Standardized Unexpected Earnings)** — How much the last earnings beat surprised vs
the stock's own historical surprise volatility. Consistently beating = durable edge.

---
**Entry / exit timing indicators** (used in Open Positions — also guide when to enter after screening):

| Indicator | What it measures | Good entry zone | Exit trigger |
|---|---|---|---|
| **RSI (14)** | Momentum — overbought/oversold on 0–100 scale | 40–65 (not stretched) | >70 + declining |
| **MACD** | Trend direction via 12/26 EMA crossover | Bullish cross | Bearish cross |
| **Stochastic %K/%D** | Short-term price position in recent range | %K < 70, above %D | %K > 80 then crosses below %D |
| **ADX (14)** | Trend *strength* (not direction) — >25 = real trend | >20 | Peaked, then drops >5 pts |
| **MFI (14)** | Volume-weighted RSI — tracks smart money flow | 40–65 | <50 (money leaving) |

Exit rule: **3 or more of 5 signals triggered = exit**.
        """)

    # ── News Entry Signals — top picks ────────────────────────────────────────
    if "entry_signal" in df.columns and df["entry_signal"].notna().any():
        st.markdown("---")
        st.subheader("📰 News Entry Signals — Top Picks")
        st.caption("Context-aware analysis: does today's news confirm or contradict the momentum thesis?")
        _es_colors = {
            "confirm_entry": ("#166534", "#dcfce7"),
            "wait":          ("#92400e", "#fef3c7"),
            "avoid":         ("#991b1b", "#fee2e2"),
        }
        _cat_icons = {"estimate_up": "📈 Estimate revision UP", "estimate_down": "📉 Estimate revision DOWN"}
        for _, row in df.head(10).iterrows():
            es = str(row.get("entry_signal", "") or "")
            if not es or es == "wait" and not row.get("news_reasoning"):
                continue
            ticker = str(row.get("ticker", ""))
            fg, bg = _es_colors.get(es, ("#374151", "#f3f4f6"))
            es_label = {"confirm_entry": "✅ CONFIRM ENTRY", "wait": "⏳ WAIT", "avoid": "🚫 AVOID"}.get(es, es)
            cat = str(row.get("catalyst", "") or "")
            cat_str = _cat_icons.get(cat, "")
            tc = str(row.get("thesis_consistency", "") or "")
            tc_badge = {"confirms": "📐 Confirms thesis", "contradicts": "⚠️ Contradicts thesis"}.get(tc, "")
            reasoning = str(row.get("news_reasoning", "") or "")
            with st.expander(f"{ticker}  —  {es_label}"):
                cols = st.columns([1, 1, 1])
                if cat_str:
                    cols[0].markdown(f"**Catalyst:** {cat_str}")
                if tc_badge:
                    cols[1].markdown(f"**Thesis:** {tc_badge}")
                dur = str(row.get("duration", "") or "")
                if dur and dur != "noise":
                    cols[2].markdown(f"**Impact:** {dur}")
                if reasoning:
                    st.markdown(
                        f'<div style="background:{bg};color:{fg};padding:10px 14px;'
                        f'border-radius:8px;font-size:13px;margin-top:6px">{_html.escape(reasoning)}</div>',
                        unsafe_allow_html=True,
                    )


# ── Market Regime ─────────────────────────────────────────────────────────────

def _render_regime() -> None:
    st.title("🌍 SPY Market Regime")
    st.caption("Composite signal from 12+ technical, macro, and momentum indicators. Cached 18h.")

    with st.spinner("Computing market regime…"):
        data = _cached_spy_regime()

    if data.get("error"):
        st.error(f"Could not compute regime: {data['error']}")
        return

    regime = data["regime"]
    score = data["score"]
    signals = data["signals"]
    as_of = data["as_of"]
    bull_count = data["bull_count"]
    total_count = data["total_count"]

    # Regime banner
    if regime == "BULL":
        bg, fg, icon = "#dcfce7", "#166534", "📈"
        desc = "Favorable conditions — momentum strategies have tailwind"
    elif regime == "BEAR":
        bg, fg, icon = "#fee2e2", "#991b1b", "📉"
        desc = "Risk-off environment — reduce or hedge long exposure"
    else:
        bg, fg, icon = "#fef3c7", "#92400e", "⚠️"
        desc = "Mixed signals — selective entries only, size down"

    st.markdown(f"""
    <div style="background:{bg};border:2px solid {fg}30;border-radius:16px;
                padding:28px;text-align:center;margin-bottom:24px">
      <div style="font-size:52px;margin-bottom:8px">{icon}</div>
      <div style="font-size:36px;font-weight:800;color:{fg};letter-spacing:-1px">{regime}</div>
      <div style="color:{fg};font-size:15px;margin-top:4px;opacity:0.85">{desc}</div>
      <div style="color:{fg};font-size:13px;margin-top:8px;opacity:0.65">
        Score {score}/10 · {bull_count}/{total_count} signals bullish · As of {_html.escape(as_of)}
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Score bar + key metrics
    bar_color = "#10b981" if regime == "BULL" else ("#ef4444" if regime == "BEAR" else "#f59e0b")
    bar_pct = int(score / 10 * 100)
    st.markdown(f"""
    <div style="background:#e2e8f0;border-radius:8px;height:10px;margin-bottom:20px">
      <div style="background:{bar_color};width:{bar_pct}%;height:10px;border-radius:8px"></div>
    </div>
    """, unsafe_allow_html=True)

    # Quick key-metric cards
    vix_sig = next((s for s in signals if s["name"] == "vix"), None)
    yld_sig = next((s for s in signals if s["name"] == "yield_curve"), None)
    rsi_sig = next((s for s in signals if s["name"] == "rsi"), None)
    gc_sig  = next((s for s in signals if s["name"] == "golden_cross"), None)

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Composite", f"{score}/10")
    with k2:
        if vix_sig:
            d = "Low fear ✅" if vix_sig["is_bull"] else "Elevated ⚠️"
            st.metric("VIX", vix_sig["value"], delta=d,
                      delta_color="normal" if vix_sig["is_bull"] else "inverse")
    with k3:
        if yld_sig:
            d = "Normal ✅" if yld_sig["is_bull"] else "Inverted 🔴"
            st.metric("Yield Curve", yld_sig["value"], delta=d,
                      delta_color="normal" if yld_sig["is_bull"] else "inverse")
    with k4:
        if gc_sig:
            d = "Golden cross ✅" if gc_sig["is_bull"] else "Death cross 🔴"
            st.metric("SMA Cross", "Bullish" if gc_sig["is_bull"] else "Bearish", delta=d,
                      delta_color="normal" if gc_sig["is_bull"] else "inverse")

    # Signal breakdown table
    st.subheader("Signal Breakdown")
    rows_html = ""
    for s in signals:
        icon2 = "✅" if s["is_bull"] else "🔴"
        row_bg = "#f0fdf4" if s["is_bull"] else "#fff5f5"
        rows_html += (
            f'<tr style="background:{row_bg}">'
            f'<td style="padding:8px 14px;font-size:14px">{icon2} {_html.escape(s["label"])}</td>'
            f'<td style="padding:8px 14px;color:#64748b;text-align:right;font-size:13px">'
            f'{_html.escape(str(s["value"]))}</td>'
            f'</tr>'
        )
    st.markdown(f"""
    <table style="width:100%;border-collapse:collapse;border:1px solid #e2e8f0;
                  border-radius:8px;overflow:hidden">
      <thead>
        <tr style="background:#f8fafc">
          <th style="padding:10px 14px;text-align:left;color:#64748b;font-size:12px;
                     text-transform:uppercase;letter-spacing:.5px">Indicator</th>
          <th style="padding:10px 14px;text-align:right;color:#64748b;font-size:12px;
                     text-transform:uppercase;letter-spacing:.5px">Value</th>
        </tr>
      </thead>
      <tbody>{rows_html}</tbody>
    </table>
    """, unsafe_allow_html=True)

    st.markdown("")

    # Sector signals from news overlay (if available)
    try:
        import sqlite3, json as _json
        _db = "data/cache.db"
        _conn = sqlite3.connect(_db)
        _row = _conn.execute(
            "SELECT payload FROM news_sentiment WHERE ticker='__MARKET__' ORDER BY fetched_at DESC LIMIT 1"
        ).fetchone()
        _conn.close()
        if _row:
            _market_payload = _json.loads(_row[0])
            _sector_sigs = _market_payload.get("sector_signals") or {}
            _regime_note = _market_payload.get("regime_note", "")
            if _sector_sigs:
                st.subheader("📡 Sector Signals")
                if _regime_note:
                    st.caption(_regime_note)
                chips_html = ""
                for _sec, _sig in _sector_sigs.items():
                    _dir = _sig.get("direction", "")
                    _str = _sig.get("strength", "")
                    _rsn = _sig.get("reason", "")
                    _chip_bg = "#fee2e2" if _dir == "headwind" else "#dcfce7"
                    _chip_fg = "#991b1b" if _dir == "headwind" else "#166534"
                    _icon = "⬇" if _dir == "headwind" else "⬆"
                    chips_html += (
                        f'<span title="{_html.escape(_rsn)}" style="display:inline-block;'
                        f'background:{_chip_bg};color:{_chip_fg};padding:4px 12px;'
                        f'border-radius:20px;font-size:13px;font-weight:600;margin:3px">'
                        f'{_icon} {_html.escape(_sec)} ({_str})</span>'
                    )
                st.markdown(f'<div style="margin-bottom:12px">{chips_html}</div>', unsafe_allow_html=True)
    except Exception:
        pass

    # Market news analysis
    st.subheader("📰 Market News Analysis")
    with st.spinner("Loading market news…"):
        articles, news_analysis = _cached_market_news()

    _news_card(news_analysis, articles, headline_count=8)

    with st.expander("📑 All headlines"):
        for a in articles:
            h = a.get("headline", "")
            src = a.get("source", "")
            if h:
                st.caption(f"• {_html.escape(h)}  *({_html.escape(src)})*")


# ── Open Positions ────────────────────────────────────────────────────────────

def _signal_badge(label: str, triggered: bool | None, val: float | None = None) -> str:
    if triggered is None:
        bg, fg, icon = "#e2e8f0", "#64748b", "—"
    elif triggered:
        bg, fg, icon = "#fee2e2", "#991b1b", "🔴"
    else:
        bg, fg, icon = "#dcfce7", "#166534", "✓"
    val_str = ""
    if val is not None and not (isinstance(val, float) and math.isnan(val)):
        val_str = f" {val:.0f}"
    return (
        f'<span style="background:{bg};color:{fg};padding:3px 10px;'
        f'border-radius:12px;font-size:12px;margin-right:6px">'
        f'{icon} {label}{val_str}</span>'
    )


def _render_position_card(pos: dict) -> None:
    ticker = pos["ticker"]
    entry_price = pos["entry_price"]
    entry_date = pos["entry_date"]
    signals, current_price = _cached_position_data(ticker)
    score = signals.get("score", 0)

    ticker_safe = _html.escape(ticker)
    entry_date_safe = _html.escape(str(entry_date))

    try:
        held_days = (date.today() - datetime.strptime(entry_date, "%Y-%m-%d").date()).days
    except Exception:
        held_days = "?"

    if current_price and entry_price:
        pnl = (current_price - entry_price) / entry_price
        pnl_str = f"+{pnl:.1%}" if pnl >= 0 else f"{pnl:.1%}"
        pnl_color = "#10b981" if pnl >= 0 else "#dc2626"
    else:
        pnl_str, pnl_color = "N/A", "#94a3b8"

    price_str = f"${current_price:,.2f}" if current_price else "N/A"

    if score >= 3:
        border, bg = "#dc2626", "#fff5f5"
    elif score >= 1:
        border, bg = "#f59e0b", "#fffbeb"
    else:
        border, bg = "#10b981", "#f0fdf4"

    badges = (
        _signal_badge("RSI", signals.get("rsi"), signals.get("rsi_val"))
        + _signal_badge("MACD", signals.get("macd"))
        + _signal_badge("Stoch", signals.get("stoch"), signals.get("stoch_k"))
        + _signal_badge("ADX", signals.get("adx"), signals.get("adx_val"))
        + _signal_badge("MFI", signals.get("mfi"), signals.get("mfi_val"))
    )

    exit_banner = ""
    if score >= 3:
        exit_banner = (
            f'<div style="margin-top:10px;color:#dc2626;font-weight:600;font-size:13px">'
            f'🚨 EXIT SIGNAL — {score}/5 momentum signals triggered</div>'
        )

    card_html = f"""
    <div style="border:2px solid {border};border-radius:10px;padding:16px;
                margin-bottom:4px;background:{bg}">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
        <div>
          <span style="font-weight:700;font-size:18px;color:#1e293b">{ticker_safe}</span>
          <span style="color:#64748b;font-size:13px;margin-left:8px">
            · entered {entry_date_safe} @ ${entry_price:,.2f}
            · {held_days}d held
            · now {price_str} ·
          </span>
          <span style="color:{pnl_color};font-weight:600;font-size:14px">{pnl_str}</span>
        </div>
      </div>
      <div>{badges}</div>
      {exit_banner}
    </div>
    """

    col_card, col_btn = st.columns([11, 1])
    with col_card:
        st.markdown(card_html, unsafe_allow_html=True)
    with col_btn:
        st.markdown("<div style='margin-top:12px'></div>", unsafe_allow_html=True)
        if st.button("✕", key=f"close_{ticker}", help=f"Close {ticker} position"):
            remove_position(ticker)
            st.rerun()


def _render_positions() -> None:
    st.title("📋 Open Positions")

    st.caption(
        "Enter a ticker and the date you bought it — the closing price on that date "
        "is fetched automatically and used as your entry price for P&L tracking."
    )

    with st.form("add_position_form", clear_on_submit=True):
        c1, c2, c3 = st.columns([3, 3, 1])
        with c1:
            ticker_input = st.text_input("Ticker", placeholder="AAPL").strip().upper()
        with c2:
            entry_date_input = st.date_input("Entry date", value=date.today())
        with c3:
            st.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)
            submitted = st.form_submit_button("＋ Add", use_container_width=True)

    if submitted:
        if not ticker_input:
            st.error("Ticker required")
        else:
            with st.spinner(f"Fetching {ticker_input} close price on {entry_date_input}…"):
                try:
                    price_used = add_position(ticker_input, str(entry_date_input))
                    st.success(
                        f"Added {ticker_input} — entry price ${price_used:,.2f} "
                        f"(close on {entry_date_input})"
                    )
                    st.rerun()
                except ValueError as e:
                    st.error(str(e))

    positions = load_positions()
    if not positions:
        st.info("No open positions. Add one above.")
        return

    st.markdown(f"**{len(positions)} open position{'s' if len(positions) != 1 else ''}**")
    st.markdown("---")

    enriched = []
    for p in positions:
        signals, _ = _cached_position_data(p["ticker"])
        enriched.append({**p, "_score": signals.get("score", 0)})
    enriched.sort(key=lambda x: x["_score"], reverse=True)

    for pos in enriched:
        _render_position_card(pos)


# ── Filing Edge page ─────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)  # 1h — filings don't change intraday
def _cached_filing_edge() -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Load latest filing_edge_*.csv from output/. Returns (longs, watch, date_str)."""
    output_dir = Path("output")
    files = sorted(output_dir.glob("filing_edge_*.csv"), reverse=True)
    if not files:
        return pd.DataFrame(), pd.DataFrame(), ""
    path = files[0]
    date_str = path.stem.replace("filing_edge_", "")
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame(), pd.DataFrame(), date_str
    longs = df[df.get("__list__", pd.Series("long", index=df.index)) == "long"].drop(
        columns=["__list__"], errors="ignore"
    ).reset_index(drop=True)
    watch = df[df.get("__list__", pd.Series("long", index=df.index)) == "watch"].drop(
        columns=["__list__"], errors="ignore"
    ).reset_index(drop=True)
    return longs, watch, date_str


def _stab_badge(val: float) -> str:
    if pd.isna(val):
        return "—"
    if val >= 0.97:
        return f"🟢 {val:.4f}"
    if val >= 0.90:
        return f"🟡 {val:.4f}"
    return f"🔴 {val:.4f}"


def _render_filing_edge():
    longs, watch, date_str = _cached_filing_edge()

    st.markdown("## 🧾 Filing Edge Screen")
    st.markdown(
        "> **Strategy:** stable 10-K/10-Q language vs prior year "
        "([Cohen, Malloy & Nguyen 2020 *Lazy Prices*](https://doi.org/10.1111/jofi.12885)) "
        "in neglected small/micro caps ($50M–$2B) where institutional arbitrage can't reach. "
        "High `text_stability` = language unchanged = bullish signal."
    )

    if longs.empty:
        st.info("No filing-edge screen found. Run `python -m src.lazy_run` to generate one.")
        with st.expander("How to run"):
            st.code("python -m src.lazy_run --limit 200  # quick test\npython -m src.lazy_run         # full run")
        return

    st.caption(f"Screen date: **{date_str}** · {len(longs)} longs · {len(watch)} on watch list")

    # ── Top-3 cards ──────────────────────────────────────────────────────────
    top3 = longs.head(3)
    cols = st.columns(min(3, len(top3)))
    medals = ["🥇", "🥈", "🥉"]
    for i, (_, row) in enumerate(top3.iterrows()):
        with cols[i]:
            stab = row.get("text_stability")
            stab_s = f"{stab:.4f}" if pd.notna(stab) else "—"
            mcap = row.get("market_cap")
            mcap_s = f"${mcap/1e6:.0f}M" if pd.notna(mcap) and mcap else "—"
            conv = int(row.get("conviction", 0) or 0)
            cd = row.get("change_direction")
            cd_badge = {1: "🔼 improving", 0: "", -1: "🔽 deteriorating"}.get(
                int(cd) if pd.notna(cd) else 0, ""
            )
            st.markdown(
                f"""<div style='background:#1e293b;border-radius:10px;padding:14px;text-align:center;'>
                <div style='font-size:28px'>{medals[i]}</div>
                <div style='font-size:22px;font-weight:700;color:#f1f5f9'>{row['ticker']}</div>
                <div style='font-size:12px;color:#94a3b8'>{str(row.get('sector',''))}</div>
                <div style='margin:8px 0;font-size:14px;color:#38bdf8'>Stability: {stab_s}</div>
                <div style='font-size:12px;color:#94a3b8'>Conv {conv}/5 · {mcap_s}</div>
                {f'<div style="font-size:11px;color:#94a3b8">{cd_badge}</div>' if cd_badge else ''}
                </div>""",
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # ── Stable-filing longs table ────────────────────────────────────────────
    st.markdown("### Stable-Filing Longs")
    st.caption("Ranked by composite score (text_stability × 0.55 + gp_assets × 0.25 + neglect × 0.20)")

    display = pd.DataFrame()
    display["#"] = range(1, len(longs) + 1)
    display["Ticker"] = longs["ticker"]
    display["Name"] = longs.get("name", "").str[:25]
    display["Sector"] = longs.get("sector", "").fillna("—").str[:18]
    display["Score"] = longs.get("composite", pd.Series(dtype=float))
    display["Conv"] = longs.get("conviction", pd.Series(dtype=float)).fillna(0).astype(int).astype(str) + "/5"
    display["Stability"] = longs.get("text_stability", pd.Series(dtype=float)).apply(
        lambda v: _stab_badge(v) if pd.notna(v) else "—"
    )
    display["Doc Sim"] = longs.get("doc_sim", pd.Series(dtype=float)).apply(
        lambda v: f"{v:.4f}" if pd.notna(v) else "—"
    )
    display["Mkt Cap"] = longs.get("market_cap", pd.Series(dtype=float)).apply(
        lambda v: f"${v/1e6:.0f}M" if pd.notna(v) and v else "—"
    )
    display["ADV"] = longs.get("avg_dollar_vol_20d", pd.Series(dtype=float)).apply(
        lambda v: f"${v/1e3:.0f}K" if pd.notna(v) and v else "—"
    )
    if "change_direction" in longs.columns:
        display["Change"] = longs["change_direction"].apply(
            lambda v: {1: "🔼", 0: "—", -1: "🔽"}.get(int(v) if pd.notna(v) else 0, "—")
        )

    st.dataframe(display, use_container_width=True, hide_index=True)

    # ── Change-direction detail (Claude layer, if populated) ─────────────────
    if "change_reason" in longs.columns and longs["change_reason"].notna().any():
        with st.expander("Claude change-characterization details"):
            for _, row in longs[longs["change_reason"].notna()].iterrows():
                cd = int(row.get("change_direction", 0) or 0)
                icon = {1: "🔼", 0: "➡️", -1: "🔽"}.get(cd, "—")
                ct = str(row.get("change_type", ""))
                cr = str(row.get("change_reason", ""))
                st.markdown(f"**{row['ticker']}** {icon} `{ct}` — {cr}")

    st.markdown("---")

    # ── Deteriorating-language watch list ────────────────────────────────────
    st.markdown("### ⚠️ Deteriorating-Language Watch List")
    st.caption(
        "Lowest text_stability — language changed most vs prior year. "
        "Per the paper: changes are *on average* bearish. Use as avoid/short-watch, "
        "or inspect `change_reason` to confirm direction."
    )
    if watch.empty:
        st.info("No watch-list data.")
    else:
        wdisplay = pd.DataFrame()
        wdisplay["#"] = range(1, len(watch) + 1)
        wdisplay["Ticker"] = watch["ticker"]
        wdisplay["Name"] = watch.get("name", "").str[:25]
        wdisplay["Sector"] = watch.get("sector", "").fillna("—").str[:18]
        wdisplay["Stability"] = watch.get("text_stability", pd.Series(dtype=float)).apply(
            lambda v: _stab_badge(v) if pd.notna(v) else "—"
        )
        wdisplay["Doc Sim"] = watch.get("doc_sim", pd.Series(dtype=float)).apply(
            lambda v: f"{v:.4f}" if pd.notna(v) else "—"
        )
        wdisplay["Risk Sim"] = watch.get("risk_sim", pd.Series(dtype=float)).apply(
            lambda v: f"{v:.4f}" if pd.notna(v) else "—"
        )
        wdisplay["MDA Sim"] = watch.get("mda_sim", pd.Series(dtype=float)).apply(
            lambda v: f"{v:.4f}" if pd.notna(v) else "—"
        )
        if "change_direction" in watch.columns:
            wdisplay["Change"] = watch["change_direction"].apply(
                lambda v: {1: "🔼", 0: "—", -1: "🔽"}.get(int(v) if pd.notna(v) else 0, "—")
            )
        st.dataframe(wdisplay, use_container_width=True, hide_index=True)

    # ── Methodology expander ─────────────────────────────────────────────────
    with st.expander("📖 Methodology & edge rationale"):
        st.markdown("""
**Why this is different from the momentum screen:**

The momentum screen fishes the most-arbitraged pond with published factors every quant desk
already computes. This screen exploits two structural moats large funds *cannot* copy:

1. **Smallness** — you can hold $50M–$2B names; a fund running billions cannot build a position
   without moving the stock. Capacity-constrained anomalies survive *because* arbitrage can't reach.
2. **Document-reading at scale** — no retail investor reads 200-page 10-K filings annually.
   Claude can, and uses that to flag *primary-source* changes vs press headlines.

**The anomaly (Cohen, Malloy & Nguyen 2020):**
Firms that materially *change* the language of their 10-K/10-Q (Risk Factors + MD&A) vs the
prior comparable filing subsequently **underperform**; stable-language "non-changers"
**outperform**. The effect is strongest in small caps with low analyst coverage — exactly this universe.

**Score components:**
- `text_stability` (55%) — cosine similarity of current vs prior comparable filing sections
- `gp_assets` (25%) — gross profitability quality gate (avoid value traps)
- `neglect_score` (20%) — inverse dollar-volume rank (more neglected = more potential edge)

**Claude precision layer** (when `ANTHROPIC_API_KEY` is set):
Runs only on names *below* the similarity threshold — the small subset that actually changed.
Classifies whether the change is positive (+1), neutral (0), or negative (-1) to lift precision
over the raw deterministic signal.

**Risk controls:**
- Tradeable floor: ≥$200K ADV (must be exitable)
- Quality gate: gp_assets ≥ Q25 of universe (avoid structurally unprofitable names)
- No leverage, no shorts recommended — use watch list as *avoid* signal only
        """)


# ── Router ────────────────────────────────────────────────────────────────────

if page == "📈 Screener Results":
    _render_screener()
elif page == "🌍 Market Regime":
    _render_regime()
elif page == "🧾 Filing Edge":
    _render_filing_edge()
else:
    _render_positions()

# ── Keyboard shortcut: R = Refresh ───────────────────────────────────────────
_components.html("""
<script>
(function() {
    function _onKey(e) {
        var tag = (document.activeElement || {}).tagName || '';
        if ((e.key === 'r' || e.key === 'R') &&
            tag !== 'INPUT' && tag !== 'TEXTAREA' && tag !== 'SELECT') {
            try {
                var btns = window.parent.document.querySelectorAll('button');
                for (var i = 0; i < btns.length; i++) {
                    if (btns[i].innerText.indexOf('Refresh') !== -1) {
                        btns[i].click();
                        return;
                    }
                }
            } catch(ex) {}
        }
    }
    try { window.parent.addEventListener('keydown', _onKey); }
    catch(ex) {}
})();
</script>
""", height=0, width=0)
