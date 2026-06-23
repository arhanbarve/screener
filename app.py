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
        ["📈 Screener Results", "🌍 Market Regime", "📋 Open Positions"],
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
    conv_badge = (
        f'<span style="background:{conv_bg};color:{conv_fg};padding:3px 9px;'
        f'border-radius:8px;font-size:12px;font-weight:600">Conv {conviction}/10</span>'
    )

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
    display_df["Conviction"] = df2["conviction"].fillna(0).astype(int) if "conviction" in df2.columns else 0
    display_df["Streak"] = (
        df2["streak_consecutive"].map(lambda x: f"🔥{int(x)}d" if pd.notna(x) and int(x) >= 2 else "—")
        if "streak_consecutive" in df2.columns else "—"
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

    # ── AI News Sentiment — top picks ─────────────────────────────────────────
    st.markdown("---")
    st.subheader("📰 AI News Sentiment — Top Picks")
    st.caption("Finnhub headlines + Claude analysis. Loaded on demand, cached 4h.")

    if "news_cache" not in st.session_state:
        st.session_state["news_cache"] = {}

    top_tickers = list(df["ticker"].head(5).astype(str))
    for ticker in top_tickers:
        with st.expander(f"📊 {ticker}"):
            if ticker in st.session_state["news_cache"]:
                articles, analysis = st.session_state["news_cache"][ticker]
                _news_card(analysis, articles)
            else:
                if st.button(f"Load AI analysis for {ticker}", key=f"news_btn_{ticker}"):
                    with st.spinner(f"Fetching {ticker} news…"):
                        articles, analysis = _cached_stock_news(ticker)
                    st.session_state["news_cache"][ticker] = (articles, analysis)
                    st.rerun()
                else:
                    st.caption("Click above to fetch and analyze recent news.")


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


# ── Router ────────────────────────────────────────────────────────────────────

if page == "📈 Screener Results":
    _render_screener()
elif page == "🌍 Market Regime":
    _render_regime()
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
