import html as _html
import math
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from src.positions import (
    add_position,
    compute_exit_signals,
    fetch_ohlcv,
    get_current_price,
    load_positions,
    remove_position,
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
    page = st.radio(
        "Navigate",
        ["📈 Screener Results", "📋 Open Positions"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("*Momentum factor strategy*")


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
    df.insert(0, "Rank", range(1, len(df) + 1))

    col_map = {
        "ticker": "Ticker", "name": "Name", "sector": "Sector",
        "composite": "Score", "mom_12_1": "Mom 12-1",
        "rs_6m": "RS 6M", "price": "Price", "market_cap": "Mkt Cap",
    }
    display_cols = ["Rank"] + [c for c in col_map if c in df.columns]
    display_df = df[display_cols].rename(columns=col_map)

    if "Score" in display_df.columns:
        display_df["Score"] = display_df["Score"].map(
            lambda x: f"{x:.2f}" if pd.notna(x) else "—"
        )
    if "Mom 12-1" in display_df.columns:
        display_df["Mom 12-1"] = display_df["Mom 12-1"].map(
            lambda x: f"{x:.1%}" if pd.notna(x) else "—"
        )
    if "RS 6M" in display_df.columns:
        display_df["RS 6M"] = display_df["RS 6M"].map(
            lambda x: f"{x:.1%}" if pd.notna(x) else "—"
        )
    if "Price" in display_df.columns:
        display_df["Price"] = display_df["Price"].map(
            lambda x: f"${x:,.2f}" if pd.notna(x) else "—"
        )
    if "Mkt Cap" in display_df.columns:
        display_df["Mkt Cap"] = display_df["Mkt Cap"].map(
            lambda x: f"${x / 1e9:.1f}B" if pd.notna(x) else "—"
        )

    st.metric("Stocks in results", len(df))
    st.dataframe(display_df, use_container_width=True, hide_index=True)


@st.cache_data(ttl=900)
def _cached_position_data(ticker: str) -> tuple[dict, float | None]:
    df = fetch_ohlcv(ticker, days=60)
    signals = compute_exit_signals(df)
    price = get_current_price(ticker)
    return signals, price


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
            · {held_days}d
            · {price_str} ·
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

    with st.form("add_position_form", clear_on_submit=True):
        c1, c2, c3, c4 = st.columns([2, 2, 2, 1])
        with c1:
            ticker_input = st.text_input("Ticker", placeholder="AAPL").strip().upper()
        with c2:
            entry_date = st.date_input("Entry date", value=date.today())
        with c3:
            entry_price = st.number_input("Entry price ($)", min_value=0.01, step=0.01, format="%.2f")
        with c4:
            st.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)
            submitted = st.form_submit_button("＋ Add", use_container_width=True)

    if submitted:
        if not ticker_input:
            st.error("Ticker required")
        elif entry_price <= 0:
            st.error("Entry price must be > 0")
        else:
            try:
                add_position(ticker_input, str(entry_date), entry_price)
                st.success(f"Added {ticker_input}")
                st.rerun()
            except ValueError as e:
                st.error(str(e))

    positions = load_positions()
    if not positions:
        st.info("No open positions. Add one above.")
        return

    st.markdown(f"**{len(positions)} open position{'s' if len(positions) != 1 else ''}**")
    st.markdown("---")

    # Sort: most signals first (most urgent at top)
    enriched = []
    for p in positions:
        signals, _ = _cached_position_data(p["ticker"])
        enriched.append({**p, "_score": signals.get("score", 0)})
    enriched.sort(key=lambda x: x["_score"], reverse=True)

    for pos in enriched:
        _render_position_card(pos)


if page == "📈 Screener Results":
    _render_screener()
else:
    _render_positions()
