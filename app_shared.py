import html as _html
import json
import re
import sys


def _nans(v) -> str:
    """NaN-safe string coerce — returns '' for None, NaN, 'nan', 'None'."""
    if v is None:
        return ""
    try:
        import math as _m
        if isinstance(v, float) and _m.isnan(v):
            return ""
    except (TypeError, ValueError):
        pass
    s = str(v)
    return "" if s in ("nan", "None", "none", "NaN") else s
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

_GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

/* ── Design tokens ──────────────────────────────────────────────────────── */
:root {
  --bg:         #020209;
  --surface-1:  #07090f;
  --surface-2:  #0c0e1a;
  --surface-3:  #111420;
  --sidebar-bg: #040408;
  --border:     #161824;
  --border-hi:  #242c42;
  --text:       #e2e8f0;
  --muted:      #64748b;
  --dim:        #2a3a54;
  --accent:     #f59e0b;
  --accent-dim: rgba(245,158,11,0.10);
  --bull:       #22c55e;
  --bull-dim:   rgba(34,197,94,0.10);
  --bear:       #ef4444;
  --bear-dim:   rgba(239,68,68,0.10);
  --wait:       #f59e0b;
  --wait-dim:   rgba(245,158,11,0.10);
  --mono:       'IBM Plex Mono', 'Courier New', monospace;
  --sans:       'IBM Plex Sans', system-ui, sans-serif;
  --radius:     10px;
  --radius-sm:  6px;
  --radius-lg:  14px;
}

/* ── Reset ──────────────────────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }

/* ── App backgrounds ────────────────────────────────────────────────────── */
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main,
[data-testid="stMainBlockContainer"],
[data-testid="stBottom"],
.main .block-container {
  background: var(--bg) !important;
}

/* ── Header ─────────────────────────────────────────────────────────────── */
[data-testid="stHeader"] {
  background: var(--sidebar-bg) !important;
  border-bottom: 1px solid var(--border) !important;
}
[data-testid="stDecoration"] { display: none !important; }
[data-testid="stToolbar"]    { opacity: 0.35 !important; }

/* ── Sidebar ─────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div:first-child {
  background: var(--sidebar-bg) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] small { font-family: var(--mono) !important; }
[data-testid="stSidebar"] hr {
  border: none !important;
  border-top: 1px solid var(--border) !important;
  opacity: 0.5 !important;
}

/* ── Typography ─────────────────────────────────────────────────────────── */
body, p, div, li, td, th, label {
  font-family: var(--sans) !important;
}
h1, h2, h3, h4, h5 {
  font-family: var(--mono) !important;
  color: var(--text) !important;
  letter-spacing: -0.02em !important;
  font-weight: 600 !important;
}
h1 { font-size: 1.35rem !important; margin-bottom: 0.4rem !important; }
h2 { font-size: 1.1rem  !important; }
h3 { font-size: 0.95rem !important; }

/* ── Metric widgets ─────────────────────────────────────────────────────── */
[data-testid="metric-container"] {
  background: var(--surface-1) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  padding: 14px 16px !important;
}
[data-testid="stMetricLabel"] div {
  color: var(--muted) !important;
  font-family: var(--mono) !important;
  font-size: 0.67rem !important;
  letter-spacing: 0.08em !important;
  text-transform: uppercase !important;
}
[data-testid="stMetricValue"] div {
  color: var(--text) !important;
  font-family: var(--mono) !important;
  font-size: 1.35rem !important;
}
[data-testid="stMetricDelta"] {
  font-family: var(--mono) !important;
  font-size: 0.72rem !important;
}

/* ── DataFrames ─────────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
  background: var(--surface-1) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  overflow: hidden !important;
}
[data-testid="stDataFrame"] * { font-family: var(--mono) !important; }

/* ── Buttons ────────────────────────────────────────────────────────────── */
.stButton > button {
  background: var(--surface-2) !important;
  color: var(--muted) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-sm) !important;
  font-family: var(--mono) !important;
  font-size: 0.72rem !important;
  letter-spacing: 0.07em !important;
  font-weight: 500 !important;
  text-transform: uppercase !important;
  transition: border-color 160ms ease, color 160ms ease, background 160ms ease !important;
  cursor: pointer !important;
}
.stButton > button:hover {
  border-color: var(--accent) !important;
  color: var(--accent) !important;
  background: var(--accent-dim) !important;
}
[data-testid="stFormSubmitButton"] > button {
  color: var(--accent) !important;
  border-color: rgba(245,158,11,0.4) !important;
  background: var(--accent-dim) !important;
}

/* ── Radio nav ──────────────────────────────────────────────────────────── */
[data-testid="stRadio"] > div { gap: 2px !important; }
[data-testid="stRadio"] label {
  color: var(--muted) !important;
  font-family: var(--mono) !important;
  font-size: 0.75rem !important;
  letter-spacing: 0.06em !important;
  padding: 7px 10px !important;
  border-radius: var(--radius-sm) !important;
  transition: color 140ms ease, background 140ms ease !important;
  cursor: pointer !important;
}
[data-testid="stRadio"] label:hover {
  color: var(--text) !important;
  background: var(--surface-2) !important;
}
[data-testid="stRadio"] [data-baseweb="radio"] [aria-checked="true"] ~ div,
[data-testid="stRadio"] label:has(input:checked) {
  color: var(--accent) !important;
}
/* Active radio dot → amber */
[data-testid="stRadio"] [data-baseweb="radio"] > div:first-child {
  border-color: var(--border) !important;
  background: transparent !important;
}
[data-testid="stRadio"] [data-baseweb="radio"][aria-checked="true"] > div:first-child {
  border-color: var(--accent) !important;
  background: var(--accent) !important;
}

/* ── Form controls ──────────────────────────────────────────────────────── */
[data-testid="stTextInput"] input,
[data-testid="stDateInput"] input {
  background: var(--surface-2) !important;
  color: var(--text) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-sm) !important;
  font-family: var(--mono) !important;
  font-size: 0.85rem !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stDateInput"] input:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 2px rgba(245,158,11,0.18) !important;
  outline: none !important;
}
[data-testid="stSelectbox"] > div > div {
  background: var(--surface-2) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-sm) !important;
  color: var(--text) !important;
  font-family: var(--mono) !important;
  font-size: 0.82rem !important;
}
[data-testid="stForm"] {
  background: var(--surface-1) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  padding: 1rem !important;
}

/* ── Expanders ──────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  background: var(--surface-1) !important;
}
[data-testid="stExpander"] summary {
  background: transparent !important;
  color: var(--muted) !important;
  font-family: var(--mono) !important;
  font-size: 0.78rem !important;
  letter-spacing: 0.04em !important;
  padding: 10px 14px !important;
}
[data-testid="stExpander"] summary:hover { color: var(--text) !important; }
[data-testid="stExpander"] [data-testid="stExpanderDetails"] {
  background: transparent !important;
  padding: 4px 14px 14px !important;
}

/* ── Alerts ─────────────────────────────────────────────────────────────── */
[data-testid="stAlert"] {
  background: var(--surface-1) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  color: var(--text) !important;
}

/* ── Spinner ────────────────────────────────────────────────────────────── */
.stSpinner > div > div { border-top-color: var(--accent) !important; }

/* ── Divider ────────────────────────────────────────────────────────────── */
hr {
  border: none !important;
  border-top: 1px solid var(--border) !important;
  opacity: 0.55 !important;
  margin: 1.1rem 0 !important;
}

/* ── Captions ───────────────────────────────────────────────────────────── */
.stCaption p, [data-testid="stCaptionContainer"] p {
  color: var(--muted) !important;
  font-family: var(--mono) !important;
  font-size: 0.71rem !important;
  letter-spacing: 0.025em !important;
}

/* ── Selectbox dropdown items ───────────────────────────────────────────── */
[data-baseweb="popover"] [role="option"] {
  background: var(--surface-2) !important;
  color: var(--text) !important;
  font-family: var(--mono) !important;
  font-size: 0.82rem !important;
}
[data-baseweb="popover"] [role="option"]:hover,
[data-baseweb="popover"] [aria-selected="true"] {
  background: var(--surface-3) !important;
  color: var(--accent) !important;
}
[data-baseweb="popover"] { background: var(--surface-2) !important; border: 1px solid var(--border) !important; }

/* ── Tabs (if any) ──────────────────────────────────────────────────────── */
[data-testid="stTabs"] [data-baseweb="tab"] {
  font-family: var(--mono) !important;
  font-size: 0.75rem !important;
  letter-spacing: 0.05em !important;
  color: var(--muted) !important;
}
[data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"] {
  color: var(--accent) !important;
  border-bottom-color: var(--accent) !important;
}

/* ── Markdown links ─────────────────────────────────────────────────────── */
a { color: var(--accent) !important; text-decoration: none !important; }
a:hover { text-decoration: underline !important; }

/* ── Scrollbar ──────────────────────────────────────────────────────────── */
::-webkit-scrollbar       { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border-hi); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--muted); }

/* ─────────────────────────────────────────────────────────────────────────
   KEYFRAMES
   ───────────────────────────────────────────────────────────────────────── */

/* Page content entrance */
@keyframes fadeInUp {
  from { opacity: 0; transform: translateY(10px); }
  to   { opacity: 1; transform: translateY(0); }
}

/* Score bar fill left→right */
@keyframes barFill {
  from { width: 0 !important; }
}

/* Table row stagger slide-in */
@keyframes rowReveal {
  from { opacity: 0; }
  to   { opacity: 1; }
}

/* Conviction dot pop */
@keyframes convDotPop {
  0%   { transform: scale(0.4); opacity: 0; }
  60%  { transform: scale(1.45); opacity: 1; filter: brightness(1.6); }
  100% { transform: scale(1);    opacity: 1; filter: brightness(1); }
}

/* Radar ping — single expanding ring for CONFIRM signals */
@keyframes radarPing {
  0%   { box-shadow: 0 0 0 0   rgba(34,197,94,0.55); }
  70%  { box-shadow: 0 0 0 16px rgba(34,197,94,0);   }
  100% { box-shadow: 0 0 0 16px rgba(34,197,94,0);   }
}

/* Exit shockwave — red ripple for danger cards */
@keyframes shockwave {
  0%   { box-shadow: 0 0 0 0   rgba(239,68,68,0.65), 0 0 18px 2px rgba(239,68,68,0.12); }
  70%  { box-shadow: 0 0 0 42px rgba(239,68,68,0),   0 0 18px 2px rgba(239,68,68,0.12); }
  100% { box-shadow: 0 0 0 42px rgba(239,68,68,0),   0 0 18px 2px rgba(239,68,68,0.12); }
}

/* Horizontal scan sweep across a table row on hover */
@keyframes rowScan {
  from { left: -20%; }
  to   { left: 120%; }
}

/* Top card #1 amber pulse — fires once on load */
@keyframes scanPulse {
  0%   { box-shadow: 0 0 0 0   rgba(245,158,11,0.50); }
  65%  { box-shadow: 0 0 0 18px rgba(245,158,11,0);   }
  100% { box-shadow: 0 0 0 0   rgba(245,158,11,0);    }
}

/* ─────────────────────────────────────────────────────────────────────────
   COMPONENT CLASSES  (used in injected HTML across pages)
   ───────────────────────────────────────────────────────────────────────── */

/* Tilt card base — 3D perspective ready */
.tilt-card {
  transform-style: preserve-3d;
  transition: transform 0.08s ease-out, box-shadow 0.25s ease;
  will-change: transform;
  position: relative;
  cursor: default;
}
.card-shine {
  position: absolute;
  inset: 0;
  border-radius: inherit;
  opacity: 0;
  pointer-events: none;
  transition: opacity 0.18s ease;
  z-index: 2;
}

/* Signal badges */
.sig-confirm {
  display: inline-flex; align-items: center; gap: 5px;
  background: var(--bull-dim);
  color: var(--bull);
  border: 1px solid rgba(34,197,94,0.28);
  padding: 3px 10px;
  border-radius: var(--radius-sm);
  font-family: var(--mono); font-size: 0.7rem; font-weight: 600;
  letter-spacing: 0.06em;
  animation: radarPing 1.6s ease-out 0.4s 1 both;
}
.sig-wait {
  display: inline-flex; align-items: center; gap: 5px;
  background: var(--wait-dim);
  color: var(--wait);
  border: 1px solid rgba(245,158,11,0.28);
  padding: 3px 10px;
  border-radius: var(--radius-sm);
  font-family: var(--mono); font-size: 0.7rem; font-weight: 600;
  letter-spacing: 0.06em;
}
.sig-avoid {
  display: inline-flex; align-items: center; gap: 5px;
  background: var(--bear-dim);
  color: var(--bear);
  border: 1px solid rgba(239,68,68,0.28);
  padding: 3px 10px;
  border-radius: var(--radius-sm);
  font-family: var(--mono); font-size: 0.7rem; font-weight: 600;
  letter-spacing: 0.06em;
}

/* ── Tooltip system ──────────────────────────────────────────────────────── */
.th-tip {
  cursor: help;
  border-bottom: 1px dashed var(--border-hi);
  padding-bottom: 1px;
}
.sig-tip-wrap { cursor: help; }
#tip-panel {
  position: fixed;
  z-index: 99999;
  pointer-events: none;
  opacity: 0;
  transform: translateY(8px) scale(0.97);
  transition: opacity 160ms cubic-bezier(0.16,1,0.3,1),
              transform 160ms cubic-bezier(0.16,1,0.3,1);
  width: 280px;
  background: var(--surface-1);
  border: 1px solid var(--border-hi);
  border-radius: var(--radius-lg);
  padding: 12px 14px 10px;
  box-shadow: 0 12px 32px rgba(0,0,0,0.65), 0 0 0 1px rgba(36,44,66,0.4);
}
#tip-panel.tip-visible {
  opacity: 1;
  transform: translateY(0) scale(1);
}
#tip-panel .tip-title {
  font-family: var(--mono);
  font-weight: 700;
  font-size: 0.6rem;
  letter-spacing: 0.13em;
  color: var(--accent);
  margin-bottom: 6px;
}
#tip-panel .tip-body {
  font-family: var(--sans);
  font-size: 0.71rem;
  line-height: 1.55;
  color: var(--muted);
}
#tip-panel .tip-hint {
  margin-top: 7px;
  padding-top: 7px;
  border-top: 1px solid var(--border);
  font-family: var(--mono);
  font-size: 0.59rem;
  color: var(--dim);
  letter-spacing: 0.07em;
}
#tip-panel .tip-sig-title {
  font-family: var(--mono);
  font-weight: 700;
  font-size: 0.6rem;
  letter-spacing: 0.12em;
  margin-bottom: 8px;
}
#tip-panel .tip-reasoning {
  font-family: var(--sans);
  font-size: 0.72rem;
  line-height: 1.55;
  padding: 8px 10px;
  border-radius: var(--radius-sm);
  margin-bottom: 4px;
}
#tip-panel .tip-meta {
  font-family: var(--mono);
  font-size: 0.59rem;
  letter-spacing: 0.05em;
  color: var(--muted);
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  margin-top: 6px;
}
#tip-panel .tip-meta b { color: var(--text); }
#tip-panel::after {
  content: '';
  position: absolute;
  left: var(--caret-x, 50%);
  transform: translateX(-50%);
  border: 5px solid transparent;
}
#tip-panel.tip-above::after {
  top: 100%;
  border-top-color: var(--border-hi);
}
#tip-panel.tip-below::after {
  bottom: 100%;
  border-bottom-color: var(--border-hi);
}

/* Conviction dot row */
.conv-dot-row { display: flex; align-items: center; gap: 4px; }
.conv-dot {
  display: inline-block;
  width: 7px; height: 7px;
  border-radius: 50%;
  opacity: 0;
}
.conv-dot.bull  { background: var(--bull); }
.conv-dot.mid   { background: var(--wait); }
.conv-dot.low   { background: var(--bear); }
.conv-dot.empty { background: var(--dim); border: 1px solid var(--border-hi); }

/* Animated score number */
.slot-score {
  font-family: var(--mono) !important;
  font-weight: 700 !important;
  font-variant-numeric: tabular-nums;
  letter-spacing: -0.02em;
}

/* Animated fill bar */
.fill-bar-track {
  width: 100%; height: 4px;
  background: var(--surface-3);
  border-radius: 2px;
  overflow: hidden;
}
.fill-bar-fill {
  height: 100%;
  border-radius: 2px;
  animation: barFill 0.8s cubic-bezier(0.16,1,0.3,1) both;
}

/* Summary strip (replaces st.metric top row) */
.summary-strip {
  display: flex; gap: 1px;
  background: var(--border);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  overflow: hidden;
  margin-bottom: 1.25rem;
}
.summary-cell {
  flex: 1;
  background: var(--surface-1);
  padding: 12px 16px;
  display: flex; flex-direction: column; gap: 2px;
}
.summary-cell:first-child { border-radius: var(--radius-sm) 0 0 var(--radius-sm); }
.summary-cell:last-child  { border-radius: 0 var(--radius-sm) var(--radius-sm) 0; }
.summary-label {
  font-family: var(--mono); font-size: 0.62rem;
  color: var(--muted); letter-spacing: 0.08em; text-transform: uppercase;
}
.summary-value {
  font-family: var(--mono); font-size: 1.1rem; font-weight: 600;
  color: var(--text);
}
.summary-value.accent { color: var(--accent); }
.summary-value.bull   { color: var(--bull); }
.summary-value.bear   { color: var(--bear); }

/* Custom HTML table */
.q-table {
  width: 100%;
  border-collapse: collapse;
  font-family: var(--mono);
  font-size: 0.78rem;
}
.q-table th {
  background: var(--surface-2);
  color: var(--muted);
  font-size: 0.62rem;
  letter-spacing: 0.09em;
  text-transform: uppercase;
  padding: 8px 12px;
  text-align: left;
  border-bottom: 1px solid var(--border);
  font-weight: 500;
}
.q-table td {
  padding: 9px 12px;
  border-bottom: 1px solid rgba(26,37,64,0.5);
  color: var(--text);
  vertical-align: middle;
}
.q-table tbody tr {
  background: var(--surface-1);
  animation: rowReveal 0.3s ease-out both;
}
.q-table tbody tr:nth-child(even) { background: var(--surface-2); }
.q-table tbody tr:hover td { background: rgba(245,158,11,0.06); color: var(--text); }
.q-table .mono  { font-family: var(--mono); }
.q-table .dim   { color: var(--muted); font-size: 0.7rem; }
.q-table .rank  { color: var(--dim); font-size: 0.65rem; letter-spacing: 0.04em; }
.q-table .ticker { font-weight: 600; font-size: 0.85rem; color: var(--text); }

/* ── Sidebar chrome overrides ───────────────────────────────────────────── */
.sidebar-header {
  font-family: var(--mono) !important;
  font-size: 0.82rem !important;
  font-weight: 700 !important;
  letter-spacing: 0.22em !important;
  color: var(--text) !important;
  padding: 4px 0 20px !important;
  display: block !important;
}

/* ALL sidebar buttons default: transparent, text-left, nav style */
[data-testid="stSidebar"] .stButton > button {
  background: transparent !important;
  border: none !important;
  color: var(--muted) !important;
  text-align: left !important;
  font-family: var(--mono) !important;
  font-size: 0.72rem !important;
  font-weight: 500 !important;
  letter-spacing: 0.11em !important;
  text-transform: uppercase !important;
  padding: 8px 10px 8px 14px !important;
  border-radius: var(--radius-sm) !important;
  transition: color 130ms ease, background 130ms ease !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
  color: var(--text) !important;
  background: rgba(255,255,255,0.04) !important;
  border: none !important;
}

/* Refresh button: override to keep bordered style */
.refresh-btn-wrap .stButton > button {
  border: 1px solid var(--border) !important;
  background: var(--surface-2) !important;
  color: var(--muted) !important;
  padding: 7px 12px !important;
  margin-bottom: 8px !important;
}
.refresh-btn-wrap .stButton > button:hover {
  border-color: var(--accent) !important;
  color: var(--accent) !important;
  background: var(--accent-dim) !important;
}

/* Active nav item (rendered as div, not button) */
.nav-active-item {
  display: flex !important;
  align-items: stretch !important;
  gap: 10px !important;
  padding: 7px 10px 7px 0 !important;
  margin-bottom: 2px !important;
  cursor: default !important;
  user-select: none !important;
}
.nav-active-tick {
  width: 2px !important;
  flex-shrink: 0 !important;
  background: var(--accent) !important;
  border-radius: 0 1px 1px 0 !important;
  min-height: 30px !important;
}
.nav-active-text { padding-left: 4px !important; }
.nav-active-label {
  font-family: var(--mono) !important;
  font-size: 0.72rem !important;
  font-weight: 600 !important;
  letter-spacing: 0.11em !important;
  color: var(--accent) !important;
  text-transform: uppercase !important;
  display: block !important;
}
.nav-active-sub {
  font-family: var(--mono) !important;
  font-size: 0.56rem !important;
  letter-spacing: 0.08em !important;
  color: var(--muted) !important;
  text-transform: uppercase !important;
  display: block !important;
  margin-top: 2px !important;
}

/* Sidebar footer */
.sidebar-footer {
  font-family: var(--mono) !important;
  font-size: 0.57rem !important;
  letter-spacing: 0.10em !important;
  color: var(--dim) !important;
  text-transform: uppercase !important;
  line-height: 1.9 !important;
  padding-top: 8px !important;
  border-top: 1px solid var(--border) !important;
}
.sidebar-footer span {
  color: var(--muted) !important;
  font-size: 0.64rem !important;
}

/* ── Reduced motion fallbacks ────────────────────────────────────────────── */
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
  }
  .tilt-card { transform: none !important; }
  .tilt-card:hover { transform: none !important; }
}

/* ── Hide Streamlit auto-generated page nav ─────────────────────────────── */
[data-testid="stSidebarNav"],
[data-testid="stSidebarNavItems"],
[data-testid="stSidebarNavSeparator"] {
  display: none !important;
}
</style>
"""



def _inject_global_css() -> None:
    st.markdown(_GLOBAL_CSS, unsafe_allow_html=True)


# ── JS animation block (uses window.parent — iframe pattern, proven) ──────────

_JS_BLOCK = r"""
<script>
(function() {
'use strict';

var P = window.parent;
var PD = P.document;
var _noMotion = P.matchMedia('(prefers-reduced-motion: reduce)').matches;

// ── 1. Ghost canvas — atmospheric financial data stream ──────────────────────
// DOM created once; setInterval + resize handler always fresh from current realm.
if (!_noMotion) {
  if (P._ghostIntervalId) clearInterval(P._ghostIntervalId);
  if (P._ghostResizeHandler) P.removeEventListener('resize', P._ghostResizeHandler);

  var cv = PD.getElementById('ghost-canvas');
  if (!cv) {
    cv = PD.createElement('canvas');
    cv.id = 'ghost-canvas';
    cv.style.cssText = 'position:fixed;top:0;left:0;width:100vw;height:100vh;pointer-events:none;z-index:0;opacity:1;';
    PD.body.appendChild(cv);
  }

  var ctx = cv.getContext('2d');
  var chars = '01%$.+-×∑≈∂';
  var cols = Math.floor(P.innerWidth / 22);
  var drops = Array.from({length: cols}, function() { return Math.random() * -60; });

  P._ghostResizeHandler = function() {
    cv.width = P.innerWidth; cv.height = P.innerHeight;
    cols = Math.floor(P.innerWidth / 22);
    drops = Array.from({length: cols}, function() { return Math.random() * -60; });
  };
  P._ghostResizeHandler();
  P.addEventListener('resize', P._ghostResizeHandler);

  function drawCanvas() {
    ctx.clearRect(0, 0, cv.width, cv.height);
    ctx.font = '12px "IBM Plex Mono", monospace';
    drops.forEach(function(y, i) {
      var baseAlpha = 0.015 + (i % 3) * 0.004;
      ctx.fillStyle = 'rgba(226,232,240,' + baseAlpha + ')';
      var char = chars[Math.floor(Math.random() * chars.length)];
      ctx.fillText(char, i * 22, y * 22);
      if (y * 22 > cv.height && Math.random() > 0.977) drops[i] = 0;
      drops[i] += 0.12;
    });
  }
  P._ghostIntervalId = setInterval(drawCanvas, 130);
}

// ── 1b. Custom pulsating cursor ──────────────────────────────────────────────
// DOM + CSS created once; all event listeners + rAF always fresh from current realm.
if (!_noMotion) {
  // DOM elements — create once and persist
  if (!PD.getElementById('cx-dot')) {
    var cursorStyle = PD.createElement('style');
    cursorStyle.id = 'cursor-style';
    cursorStyle.textContent = [
      '* { cursor: none !important; }',
      '#cx-dot {',
      '  position:fixed;width:8px;height:8px;border-radius:50%;',
      '  background:#f59e0b;',
      '  box-shadow:0 0 6px 2px rgba(245,158,11,0.9),0 0 18px 6px rgba(245,158,11,0.35);',
      '  pointer-events:none;z-index:99999;',
      '  transform:translate(-50%,-50%);',
      '  transition:transform 0.06s ease,background 0.2s ease,box-shadow 0.2s ease;',
      '}',
      '#cx-ring {',
      '  position:fixed;width:28px;height:28px;border-radius:50%;',
      '  border:1.5px solid rgba(245,158,11,0.55);',
      '  pointer-events:none;z-index:99998;',
      '  transform:translate(-50%,-50%);',
      '  animation:cursorPulse 1.6s ease-in-out infinite;',
      '}',
      '#cx-ring2 {',
      '  position:fixed;width:48px;height:48px;border-radius:50%;',
      '  border:1px solid rgba(245,158,11,0.18);',
      '  pointer-events:none;z-index:99997;',
      '  transform:translate(-50%,-50%);',
      '  animation:cursorPulse 1.6s ease-in-out infinite 0.5s;',
      '}',
      '@keyframes cursorPulse {',
      '  0%,100%{opacity:1;transform:translate(-50%,-50%) scale(1);}',
      '  50%{opacity:0.4;transform:translate(-50%,-50%) scale(1.35);}',
      '}',
    ].join('');
    PD.head.appendChild(cursorStyle);
    var _d = PD.createElement('div'); _d.id = 'cx-dot';
    var _r = PD.createElement('div'); _r.id = 'cx-ring';
    var _r2= PD.createElement('div'); _r2.id= 'cx-ring2';
    PD.body.appendChild(_d);
    PD.body.appendChild(_r);
    PD.body.appendChild(_r2);
  }
  var cxDot  = PD.getElementById('cx-dot');
  var cxRing = PD.getElementById('cx-ring');
  var cxRing2= PD.getElementById('cx-ring2');

  // Cancel old rAF loop before starting a new one
  if (P._cursorRafId) P.cancelAnimationFrame(P._cursorRafId);

  // Remove old listeners — always swap in fresh functions from current realm
  if (P._cxMove)  PD.removeEventListener('mousemove',  P._cxMove,  { passive: true });
  if (P._cxDown)  PD.removeEventListener('mousedown',  P._cxDown);
  if (P._cxUp)    PD.removeEventListener('mouseup',    P._cxUp);
  if (P._cxOver)  PD.removeEventListener('mouseover',  P._cxOver);

  // Carry forward last known position so rings don't jump to 0,0
  var _mx = P._cxMX || 0, _my = P._cxMY || 0;
  var _rx = _mx, _ry = _my, _r2x = _mx, _r2y = _my;

  P._cxMove = function(e) {
    _mx = P._cxMX = e.clientX; _my = P._cxMY = e.clientY;
    cxDot.style.left = _mx + 'px'; cxDot.style.top = _my + 'px';
  };
  P._cxDown = function() { cxDot.style.transform = 'translate(-50%,-50%) scale(0.6)'; };
  P._cxUp   = function() { cxDot.style.transform = 'translate(-50%,-50%) scale(1)'; };
  P._cxOver = function(e) {
    var t = e.target;
    var hit = t && (t.tagName === 'BUTTON' || t.tagName === 'A' ||
      t.tagName === 'INPUT' || t.tagName === 'SELECT' ||
      t.closest('.tilt-card') || t.closest('button'));
    if (hit) { cxDot.style.transform = 'translate(-50%,-50%) scale(1.7)'; cxDot.style.background = '#fcd34d'; }
    else      { cxDot.style.transform = 'translate(-50%,-50%) scale(1)';   cxDot.style.background = '#f59e0b'; }
  };

  PD.addEventListener('mousemove', P._cxMove,  { passive: true });
  PD.addEventListener('mousedown', P._cxDown);
  PD.addEventListener('mouseup',   P._cxUp);
  PD.addEventListener('mouseover', P._cxOver);

  (function loop() {
    _rx  += (_mx - _rx)  * 0.18; _ry  += (_my - _ry)  * 0.18;
    _r2x += (_mx - _r2x) * 0.09; _r2y += (_my - _r2y) * 0.09;
    cxRing.style.left  = _rx  + 'px'; cxRing.style.top  = _ry  + 'px';
    cxRing2.style.left = _r2x + 'px'; cxRing2.style.top = _r2y + 'px';
    P._cursorRafId = P.requestAnimationFrame(loop);
  })();
}

// ── 2. Terminal boot scan line ───────────────────────────────────────────────
if (!P._bootScanEl) {
  var scanEl = PD.createElement('div');
  scanEl.id = 'boot-scanner';
  scanEl.style.cssText = [
    'position:fixed;top:0;left:0;right:0;height:2px;',
    'background:linear-gradient(90deg, transparent 0%, #f59e0b 40%, #fcd34d 50%, #f59e0b 60%, transparent 100%);',
    'box-shadow:0 0 22px 4px rgba(245,158,11,0.7);',
    'pointer-events:none;z-index:9999;',
    'opacity:0;transform:translateY(0);',
    'transition:transform 1.15s cubic-bezier(0.4,0,0.2,1), opacity 0.25s ease;'
  ].join('');
  PD.body.appendChild(scanEl);
  P._bootScanEl = scanEl;
}

function triggerBootScan() {
  var el = P._bootScanEl;
  el.style.opacity = '1';
  el.style.transform = 'translateY(0)';
  P.requestAnimationFrame(function() {
    P.requestAnimationFrame(function() {
      el.style.transform = 'translateY(' + P.innerHeight + 'px)';
      setTimeout(function() {
        el.style.transition = 'none';
        el.style.opacity = '0';
        el.style.transform = 'translateY(0)';
        setTimeout(function() {
          el.style.transition = 'transform 1.15s cubic-bezier(0.4,0,0.2,1), opacity 0.25s ease';
        }, 50);
      }, 1200);
    });
  });
}

if (!P._bootFired) { P._bootFired = true; setTimeout(triggerBootScan, 80); }

// ── 3. 3D card tilt — always re-register from current realm ─────────────────
// Remove old handlers (possibly from a dead realm) then add live ones.
var _tiltCurrent = null;

function _tiltReset(card) {
  if (!card) return;
  card.style.transition = 'transform 0.7s cubic-bezier(0.23,1,0.32,1)';
  card.style.transform  = 'perspective(920px) rotateX(0deg) rotateY(0deg) scale3d(1,1,1)';
  var sh = card.querySelector('.card-shine');
  if (sh) sh.style.opacity = '0';
}

if (P._tiltMove)  PD.removeEventListener('mousemove',  P._tiltMove,  { passive: true });
if (P._tiltLeave) PD.removeEventListener('mouseleave', P._tiltLeave);

P._tiltMove = function(e) {
  if (_noMotion) return;
  var card = e.target && e.target.closest ? e.target.closest('.tilt-card') : null;
  if (!card) { _tiltReset(_tiltCurrent); _tiltCurrent = null; return; }
  if (_tiltCurrent && _tiltCurrent !== card) _tiltReset(_tiltCurrent);
  if (_tiltCurrent !== card) { card.style.transition = 'transform 0.07s ease-out'; _tiltCurrent = card; }
  var r  = card.getBoundingClientRect();
  var dx = (e.clientX - (r.left + r.width  * 0.5)) / (r.width  * 0.5);
  var dy = (e.clientY - (r.top  + r.height * 0.5)) / (r.height * 0.5);
  card.style.transform =
    'perspective(920px) rotateX(' + (-dy*9) + 'deg) rotateY(' + (dx*13) + 'deg) scale3d(1.024,1.024,1.024)';
  var shine = card.querySelector('.card-shine');
  if (shine) {
    shine.style.opacity = '1';
    shine.style.background =
      'radial-gradient(circle at '+(50+dx*36)+'% '+(50+dy*36)+'%, '+
      'rgba(245,158,11,0.13) 0%, rgba(255,255,255,0.04) 40%, transparent 68%)';
  }
};
P._tiltLeave = function() { _tiltReset(_tiltCurrent); _tiltCurrent = null; };

PD.addEventListener('mousemove',  P._tiltMove,  { passive: true });
PD.addEventListener('mouseleave', P._tiltLeave);

function initTiltCards() {} // delegation handles all cards — no per-card init

// ── 4. Slot-machine score countup ────────────────────────────────────────────
function initSlotScores() {
  PD.querySelectorAll('.slot-score:not([data-slot-init])').forEach(function(el) {
    el.dataset.slotInit = '1';
    var target   = parseFloat(el.dataset.target   || '0');
    var decimals = parseInt(  el.dataset.decimals || '2');
    var dur      = 920;
    var flickEnd = dur * 0.62;
    var t0 = P.performance.now();
    function easeOutQuart(t) { return 1 - Math.pow(1 - t, 4); }
    (function tick(now) {
      var elapsed  = now - t0;
      var progress = Math.min(elapsed / dur, 1);
      var val;
      if (elapsed < flickEnd) {
        val = Math.random() * target * (1.8 - (elapsed / flickEnd) * 1.1);
      } else {
        val = easeOutQuart((elapsed - flickEnd) / (dur - flickEnd)) * target;
      }
      el.textContent = val.toFixed(decimals);
      if (progress < 1) P.requestAnimationFrame(tick);
      else el.textContent = target.toFixed(decimals);
    })(t0);
  });
}

// ── 5. Conviction dots — sequential fill with heartbeat pop ─────────────────
function initConvDots() {
  PD.querySelectorAll('.conv-dot-row:not([data-conv-init])').forEach(function(row) {
    row.dataset.convInit = '1';
    var dots = row.querySelectorAll('.conv-dot');
    dots.forEach(function(dot, i) {
      setTimeout(function() {
        dot.style.animation = 'convDotPop 200ms cubic-bezier(0.34,1.56,0.64,1) forwards';
      }, i * 90);
    });
  });
}

// ── 6. Keyboard shortcut: R = refresh — always re-register ──────────────────
if (P._keyHandler) P.removeEventListener('keydown', P._keyHandler);
P._keyHandler = function(e) {
  var tag = (PD.activeElement || {}).tagName || '';
  if ((e.key === 'r' || e.key === 'R') &&
      tag !== 'INPUT' && tag !== 'TEXTAREA' && tag !== 'SELECT') {
    var btns = PD.querySelectorAll('button');
    for (var i = 0; i < btns.length; i++) {
      if (btns[i].innerText.indexOf('Refresh') !== -1) { btns[i].click(); return; }
    }
  }
};
P.addEventListener('keydown', P._keyHandler);

// ── 7. Tooltip system ────────────────────────────────────────────────────────
var _TH_TIPS = {
  score:  { title: 'COMPOSITE SCORE', body: 'Weighted z-score: 28% 12-mo momentum, 20% analyst revision breadth, 17% earnings surprise, 15% 6-mo RS vs SPY, 10% technical alignment, 5% RS slope, 5% streak bonus.', hint: 'Higher = stronger setup' },
  conv:   { title: 'CONVICTION 1–10', body: 'Four layers: rank position (top 3 = 3pts), streak ≥7d = 3pts, technical alignment ≥6/8 green = 2pts, fundamental quality — gross profitability, insider buying, short float.', hint: 'Use for position sizing — high conviction = larger starter' },
  streak: { title: 'STREAK', body: "Consecutive trading days in the screener's top results. 5+ days = proven staying power. 1 day = may be noise.", hint: 'Longer = higher confidence in the setup' },
  signal: { title: 'NEWS SIGNAL', body: 'AI-analyzed entry signal. CONFIRM = news supports thesis. WAIT = mixed signals. AVOID = news contradicts thesis or negative catalyst detected.', hint: 'Hover the badge for full reasoning' },
  cat:    { title: 'CATALYST', body: 'Primary news catalyst. EST↑ = analyst estimate raised. EST↓ = estimate cut. Drives institutional accumulation or distribution.', hint: 'Estimate revisions precede price moves' },
  mom:    { title: 'MOM 12-1 — 12-MONTH MOMENTUM', body: 'Price return over the past year, excluding the most recent month. Skipping the last month removes mean-reversion noise — what remains is slow-moving institutional momentum that persists 3–12 months.', hint: 'Academic consensus: strongest near-term return predictor' },
  rs:     { title: 'RS 6M — RELATIVE STRENGTH vs SPY', body: 'How much the stock outperformed the S&P 500 over 6 months. +300% = beat the index by 300 percentage points — not a rising tide lift.', hint: 'Positive = beating the market on its own merit' }
};

function _escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function initTooltips() {
  // Always ensure panel exists in PD.body (may have been removed by Streamlit nav)
  if (!PD.getElementById('tip-panel')) {
    var _tp = PD.createElement('div');
    _tp.id = 'tip-panel';
    PD.body.appendChild(_tp);
  }

  // Follow existing pattern: remove stale listeners from dead realm, add fresh ones
  if (P._tipMouseover) PD.removeEventListener('mouseover', P._tipMouseover);
  if (P._tipMouseout)  PD.removeEventListener('mouseout',  P._tipMouseout);

  var _SIG_COLORS = {
    confirm: { fg: '#22c55e', bg: 'rgba(34,197,94,0.10)', label: 'CONFIRM ENTRY' },
    wait:    { fg: '#f59e0b', bg: 'rgba(245,158,11,0.10)', label: 'WAIT' },
    avoid:   { fg: '#ef4444', bg: 'rgba(239,68,68,0.10)', label: 'AVOID' }
  };

  // Always look up panel dynamically — captured ref becomes stale after Streamlit nav
  function _panel() {
    var p = PD.getElementById('tip-panel');
    if (!p) { p = PD.createElement('div'); p.id = 'tip-panel'; PD.body.appendChild(p); }
    return p;
  }

  function showPanel(el) {
    var key  = el.getAttribute('data-tip-key');
    var type = el.getAttribute('data-tip-type');
    var html = '';

    if (key && _TH_TIPS[key]) {
      var d = _TH_TIPS[key];
      html = '<div class="tip-title">' + d.title + '</div>' +
             '<div class="tip-body">'  + d.body  + '</div>' +
             '<div class="tip-hint">'  + d.hint  + '</div>';
    } else if (type) {
      var c = _SIG_COLORS[type] || { fg: '#94a3b8', bg: 'rgba(148,163,184,0.1)', label: type.toUpperCase() };
      var reasoning = el.getAttribute('data-tip-reasoning') || '';
      var catalyst  = el.getAttribute('data-tip-catalyst')  || '';
      var thesis    = el.getAttribute('data-tip-thesis')    || '';
      var duration  = el.getAttribute('data-tip-duration')  || '';
      html = '<div class="tip-sig-title" style="color:' + c.fg + '">' + c.label + '</div>';
      if (reasoning) {
        html += '<div class="tip-reasoning" style="background:' + c.bg + ';border:1px solid ' + c.fg + '44;color:' + c.fg + '">' + _escHtml(reasoning) + '</div>';
      } else {
        html += '<div class="tip-body" style="color:' + c.fg + '99">No news analysis for this stock.</div>';
      }
      var meta = [];
      if (catalyst && catalyst !== '')  meta.push('<b>CAT</b> ' + _escHtml(catalyst.replace(/_/g,' ').toUpperCase()));
      if (thesis && thesis !== '')      meta.push('<b>THESIS</b> ' + _escHtml(thesis));
      if (duration && duration !== '' && duration !== 'noise') meta.push('<b>IMPACT</b> ' + _escHtml(duration));
      if (meta.length) html += '<div class="tip-meta">' + meta.join(' &nbsp;·&nbsp; ') + '</div>';
    }

    if (!html) return;
    var panel = _panel();
    panel.innerHTML = html;

    var r  = el.getBoundingClientRect();
    var cx = r.left + r.width / 2;
    var W  = 280;
    var x  = Math.max(8, Math.min(cx - W / 2, P.innerWidth - W - 8));
    panel.style.left  = x + 'px';
    panel.style.width = W + 'px';
    panel.style.setProperty('--caret-x', (cx - x) + 'px');

    if (r.top > 150) {
      panel.style.top    = '';
      panel.style.bottom = (P.innerHeight - r.top + 8) + 'px';
      panel.classList.remove('tip-below');
      panel.classList.add('tip-above');
    } else {
      panel.style.top    = (r.bottom + 8) + 'px';
      panel.style.bottom = '';
      panel.classList.remove('tip-above');
      panel.classList.add('tip-below');
    }
    requestAnimationFrame(function() { panel.classList.add('tip-visible'); });
  }

  function hidePanel() {
    var panel = _panel();
    panel.classList.remove('tip-visible');
    setTimeout(function() { if (!P._tipEl) panel.innerHTML = ''; }, 200);
  }

  P._tipMouseover = function(e) {
    if (!e.target || !e.target.closest) return;
    var el = e.target.closest('[data-tip-key]') || e.target.closest('[data-tip-type]');
    if (el === P._tipEl) return;
    clearTimeout(P._tipHideT);
    P._tipEl = el;
    if (el) showPanel(el);
    else hidePanel();
  };

  P._tipMouseout = function(e) {
    if (!e.target || !e.target.closest) return;
    var el = e.target.closest('[data-tip-key]') || e.target.closest('[data-tip-type]');
    if (!el) return;
    if (e.relatedTarget && el.contains(e.relatedTarget)) return;
    P._tipEl = null;
    P._tipHideT = setTimeout(function() { if (!P._tipEl) hidePanel(); }, 50);
  };

  PD.addEventListener('mouseover', P._tipMouseover);
  PD.addEventListener('mouseout',  P._tipMouseout);
}

// ── MutationObserver: always replace so callback lives in current realm ────
function runAll() {
  initTiltCards();
  initSlotScores();
  initConvDots();
  initTooltips();
  // Boot scan triggered by a hidden span injected by Python on page nav
  var bt = PD.getElementById('x-boot-trigger');
  if (bt) { bt.parentNode && bt.parentNode.removeChild(bt); P._bootFired = false; }
  if (!P._bootFired) { P._bootFired = true; triggerBootScan(); }
}

runAll();

if (P._animObserver) { P._animObserver.disconnect(); }
var _mutDebounce;
P._animObserver = new P.MutationObserver(function(mutations) {
  var added = mutations.some(function(m) { return m.addedNodes.length > 0; });
  if (!added) return;
  clearTimeout(_mutDebounce);
  _mutDebounce = setTimeout(runAll, 60);
});
P._animObserver.observe(PD.body, { childList: true, subtree: true });

})();
</script>
"""


def _inject_js_animations() -> None:
    import streamlit.components.v1 as _cv1
    _cv1.html(_JS_BLOCK, height=0, width=0)


from src.positions import (
    get_live_quote,
    load_positions,
)
from src.exit_plan import health_label
from src.spy_analysis import compute_spy_regime
from src.news import (
    get_market_news,
    get_stock_news,
    analyze_market_news,
    analyze_stock_news,
)

_NAV = [
    ("SCREENER",   "screener",   "MOMENTUM RANKINGS"),
    ("REGIME",     "regime",     "MARKET STATE"),
    ("POSITIONS",  "positions",  "OPEN P&L"),
    ("MONITOR",    "monitor",    "RUN STATUS"),
]


_PAGE_FILES = {
    "screener":   "app.py",
    "regime":     "pages/2_Regime.py",
    "positions":  "pages/3_Positions.py",
    "monitor":    "pages/6_Monitor.py",
}


def render_sidebar(current_page_id: str) -> None:
    with st.sidebar:
        st.markdown('<span class="sidebar-header">▲ SCREENER</span>', unsafe_allow_html=True)

        st.markdown('<div class="refresh-btn-wrap">', unsafe_allow_html=True)
        if st.button("↺  REFRESH", use_container_width=True, key="refresh_btn",
                     help="Clear all cached data and reload  ·  keyboard: R"):
            st.cache_data.clear()
            st.session_state.pop("news_cache", None)
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)

        for _label, _pid, _sub in _NAV:
            if current_page_id == _pid:
                st.markdown(f"""
            <div class="nav-active-item">
              <div class="nav-active-tick"></div>
              <div class="nav-active-text">
                <span class="nav-active-label">{_label}</span>
                <span class="nav-active-sub">{_sub}</span>
              </div>
            </div>""", unsafe_allow_html=True)
            else:
                if st.button(_label, key=f"nav_{_pid}", use_container_width=True):
                    st.session_state._trigger_boot = True
                    st.switch_page(_PAGE_FILES[_pid])

        st.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)
        _out_files = sorted(Path("output").glob("screen_*.csv"), reverse=True)
        _last_run  = _out_files[0].stem.replace("screen_", "") if _out_files else "—"
        st.markdown(
            f'<div class="sidebar-footer">LAST RUN<br><span>{_last_run}</span></div>',
            unsafe_allow_html=True,
        )


def setup_page(page_id: str, title: str = "Screener") -> None:
    st.set_page_config(
        page_title=title,
        page_icon="▲",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_global_css()
    _inject_js_animations()
    if st.session_state.pop("_trigger_boot", False):
        st.markdown('<span id="x-boot-trigger" style="display:none"></span>', unsafe_allow_html=True)
    render_sidebar(page_id)


# ── Cached data loaders ───────────────────────────────────────────────────────

@st.cache_data(ttl=900)
def _batch_position_data(tickers: tuple[str, ...]) -> tuple[dict[str, tuple[dict, float | None, float | None]], datetime]:
    import concurrent.futures

    def _fetch_one(ticker: str) -> tuple[str, tuple[dict, float | None, float | None]]:
        try:
            price, prev_close = get_live_quote(ticker)
            if price is None:
                print(f"[positions] live quote fetch returned no price for {ticker} — falling back to Fidelity snapshot", flush=True)
            return ticker, ({}, price, prev_close)
        except Exception as e:
            print(f"[positions] live quote fetch failed for {ticker}: {e!r}", flush=True)
            return ticker, ({}, None, None)

    workers = min(len(tickers), 10)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(_fetch_one, tickers))
    return dict(results), datetime.now()


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

def _page_title(text: str) -> None:
    st.markdown(
        f'<h2 style="font-family:var(--mono);font-size:1.1rem;font-weight:700;'
        f'letter-spacing:0.12em;color:var(--muted);margin-bottom:0.5rem">{text}</h2>',
        unsafe_allow_html=True,
    )


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


def _top3_card(rank: int, row: pd.Series) -> str:
    mom        = row.get("mom_12_1", float("nan"))
    rs         = row.get("rs_6m",    float("nan"))
    _s         = row.get("sector", "")
    sector     = "—" if pd.isna(_s) or str(_s).strip() == "" else str(_s)
    name       = str(row.get("name", "") or "")[:30]
    ticker     = str(row.get("ticker", ""))
    score      = float(row.get("composite", 0) or 0)
    conviction = int(row.get("conviction", 0) or 0)
    streak     = int(row.get("streak_consecutive", 0) or 0)
    es         = str(row.get("entry_signal", "") or "")
    reasoning  = _nans(row.get("news_reasoning"))
    thesis     = _nans(row.get("thesis_consistency"))
    duration   = _nans(row.get("duration"))
    cat_card   = _nans(row.get("catalyst"))

    rank_str = f"0{rank}" if rank < 10 else str(rank)

    def _dot(i: int) -> str:
        if i > conviction:
            return '<span class="conv-dot empty"></span>'
        if conviction >= 7:
            return '<span class="conv-dot bull"></span>'
        if conviction >= 4:
            return '<span class="conv-dot mid"></span>'
        return '<span class="conv-dot low"></span>'
    conv_dots = "".join(_dot(i) for i in range(1, 11))

    _sig_map = {
        "confirm_entry": ("CONFIRM", "sig-confirm", "confirm"),
        "wait":          ("WAIT",    "sig-wait",    "wait"),
        "avoid":         ("AVOID",   "sig-avoid",   "avoid"),
    }
    if es in _sig_map:
        sig_label, sig_cls, tip_type = _sig_map[es]
        inner_badge = f'<span class="{sig_cls}">{sig_label}</span>'
        signal_html = (
            f'<span class="sig-tip-wrap" data-tip-type="{tip_type}"'
            f' data-tip-reasoning="{_html.escape(reasoning)}"'
            f' data-tip-catalyst="{_html.escape(cat_card)}"'
            f' data-tip-thesis="{_html.escape(thesis)}"'
            f' data-tip-duration="{_html.escape(duration)}">'
            f'{inner_badge}</span>'
        )
    else:
        signal_html = ""

    streak_html = (
        f'<span style="font-family:var(--mono);font-size:0.63rem;'
        f'letter-spacing:0.05em;color:var(--accent)">{streak}d streak</span>'
        if streak >= 2 else ""
    )

    def _pct_color(v: float) -> str:
        if pd.isna(v): return "var(--muted)"
        return "var(--bull)" if v > 0 else "var(--bear)"

    top_border  = ["var(--accent)", "var(--border-hi)", "var(--border)"][rank - 1]
    score_color = "var(--accent)" if rank == 1 else "var(--text)"
    pulse_anim  = "animation:scanPulse 2s ease-out 1.4s 1 both;" if rank == 1 else ""

    card_style  = (
        f"background:var(--surface-1);border:1px solid var(--border);"
        f"border-top:2px solid {top_border};border-radius:var(--radius-lg);"
        f"padding:20px 18px 16px;height:100%;position:relative;overflow:hidden;{pulse_anim}"
    )

    return f"""
<div class="tilt-card" style="{card_style}">
  <div class="card-shine"></div>
  <div style="position:absolute;top:8px;right:14px;font-family:var(--mono);
              font-size:3.6rem;font-weight:700;color:var(--dim);letter-spacing:-0.04em;
              line-height:1;pointer-events:none;user-select:none">{rank_str}</div>
  <div style="font-family:var(--mono);font-size:1.55rem;font-weight:700;
              color:var(--text);letter-spacing:-0.01em;line-height:1;
              position:relative;z-index:1">{_html.escape(ticker)}</div>
  <div style="font-size:0.71rem;color:var(--muted);margin-top:3px;
              margin-bottom:14px">{_html.escape(name)}</div>
  <div style="display:flex;align-items:baseline;gap:7px;margin-bottom:12px">
    <span class="slot-score" data-target="{score:.4f}" data-decimals="2"
          style="font-size:1.85rem;font-weight:700;color:{score_color}">{score:.2f}</span>
    <span style="font-family:var(--mono);font-size:0.58rem;color:var(--muted);
                 letter-spacing:0.08em">COMPOSITE</span>
  </div>
  <div style="margin-bottom:14px">
    <div style="font-family:var(--mono);font-size:0.57rem;color:var(--muted);
                letter-spacing:0.09em;margin-bottom:6px">CONVICTION {conviction}/10</div>
    <div class="conv-dot-row">{conv_dots}</div>
  </div>
  <div style="display:flex;gap:18px;margin-bottom:14px">
    <div>
      <div style="font-family:var(--mono);font-size:0.56rem;color:var(--muted);letter-spacing:0.07em"><span class="th-tip" data-tip-key="mom">MOM 12-1</span></div>
      <div style="font-family:var(--mono);font-size:0.85rem;font-weight:600;color:{_pct_color(mom)}">{_fmt_pct(mom)}</div>
    </div>
    <div>
      <div style="font-family:var(--mono);font-size:0.56rem;color:var(--muted);letter-spacing:0.07em"><span class="th-tip" data-tip-key="rs">RS 6M</span></div>
      <div style="font-family:var(--mono);font-size:0.85rem;font-weight:600;color:{_pct_color(rs)}">{_fmt_pct(rs)}</div>
    </div>
    <div>
      <div style="font-family:var(--mono);font-size:0.56rem;color:var(--muted);letter-spacing:0.07em">SECTOR</div>
      <div style="font-family:var(--mono);font-size:0.7rem;color:var(--text)">{_html.escape(sector[:18])}</div>
    </div>
  </div>
  <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap">
    {signal_html}{streak_html}
  </div>
</div>"""


def _screener_html_table(df: pd.DataFrame) -> str:
    max_score = float(df["composite"].max()) if "composite" in df.columns and df["composite"].notna().any() else 1.0
    if max_score <= 0:
        max_score = 1.0

    _sig_badge = {
        "confirm_entry": '<span class="sig-confirm">CONFIRM</span>',
        "wait":          '<span class="sig-wait">WAIT</span>',
        "avoid":         '<span class="sig-avoid">AVOID</span>',
    }
    _cat_badge = {
        "estimate_up":   '<span style="font-family:var(--mono);font-size:0.7rem;color:var(--bull)">EST&#8593;</span>',
        "estimate_down": '<span style="font-family:var(--mono);font-size:0.7rem;color:var(--bear)">EST&#8595;</span>',
    }

    rows = []
    for i, (_, row) in enumerate(df.iterrows()):
        rank      = i + 1
        ticker    = _html.escape(str(row.get("ticker", "")))
        name      = _html.escape(str(row.get("name", "") or "")[:28])
        _sec      = row.get("sector", "")
        sector    = _html.escape(("—" if pd.isna(_sec) or str(_sec).strip() == "" else str(_sec))[:14])
        score     = float(row.get("composite", 0) or 0)
        score_pct = min(score / max_score * 100, 100)
        conv      = int(row.get("conviction", 0) or 0)
        streak    = int(row.get("streak_consecutive", 0) or 0)
        es        = str(row.get("entry_signal", "") or "")
        cat       = str(row.get("catalyst", "") or "")
        reasoning = _nans(row.get("news_reasoning"))
        thesis    = _nans(row.get("thesis_consistency"))
        duration  = _nans(row.get("duration"))
        mom       = row.get("mom_12_1", float("nan"))
        rs        = row.get("rs_6m", float("nan"))
        price     = row.get("price")
        mcap      = row.get("market_cap")

        def _c(v: float) -> str:
            if pd.isna(v): return "var(--muted)"
            return "var(--bull)" if v > 0 else "var(--bear)"

        _tip_type_map = {"confirm_entry": "confirm", "wait": "wait", "avoid": "avoid"}
        if es in _sig_badge:
            _inner = _sig_badge[es]
            _ttype = _tip_type_map.get(es, "")
            sig_html = (
                f'<span class="sig-tip-wrap" data-tip-type="{_ttype}"'
                f' data-tip-reasoning="{_html.escape(reasoning)}"'
                f' data-tip-catalyst="{_html.escape(cat)}"'
                f' data-tip-thesis="{_html.escape(thesis)}"'
                f' data-tip-duration="{_html.escape(duration)}">'
                f'{_inner}</span>'
            )
        else:
            sig_html = '<span style="color:var(--dim)">—</span>'
        cat_html    = _cat_badge.get(cat, '<span style="color:var(--dim)">—</span>')
        streak_html = (f'<span style="color:var(--accent);font-family:var(--mono);font-size:0.72rem">{streak}d</span>'
                       if streak >= 2 else '<span style="color:var(--dim)">—</span>')
        price_str   = f"${price:,.2f}" if price is not None and not pd.isna(price) else "—"
        mcap_str    = f"${mcap/1e9:.1f}B" if mcap is not None and not pd.isna(mcap) else "—"

        rows.append(f"""
<tr style="animation-delay:{i * 20}ms">
  <td class="rank">#{rank:02d}</td>
  <td><span class="ticker">{ticker}</span></td>
  <td class="dim">{name}</td>
  <td class="dim">{sector}</td>
  <td>
    <div style="display:flex;align-items:center;gap:7px">
      <div class="fill-bar-track" style="width:52px">
        <div class="fill-bar-fill" style="width:{score_pct:.0f}%;background:var(--accent)"></div>
      </div>
      <span style="font-family:var(--mono);font-size:0.72rem;color:var(--text)">{score:.2f}</span>
    </div>
  </td>
  <td style="font-family:var(--mono);font-size:0.78rem;color:var(--text)">{conv}<span style="color:var(--dim)">/10</span></td>
  <td>{streak_html}</td>
  <td>{sig_html}</td>
  <td>{cat_html}</td>
  <td style="font-family:var(--mono);font-size:0.78rem;color:{_c(mom)}">{_fmt_pct(mom)}</td>
  <td style="font-family:var(--mono);font-size:0.78rem;color:{_c(rs)}">{_fmt_pct(rs)}</td>
  <td style="font-family:var(--mono);font-size:0.75rem;color:var(--muted)">{price_str}</td>
  <td style="font-family:var(--mono);font-size:0.72rem;color:var(--muted)">{mcap_str}</td>
</tr>""")

    return f"""
<div style="overflow-x:auto;border:1px solid var(--border);border-radius:var(--radius);overflow:hidden;margin-bottom:1rem">
<table class="q-table">
  <thead><tr>
    <th>#</th><th>TICKER</th><th>NAME</th><th>SECTOR</th>
    <th><span class="th-tip" data-tip-key="score">SCORE</span></th>
    <th><span class="th-tip" data-tip-key="conv">CONV</span></th>
    <th><span class="th-tip" data-tip-key="streak">STREAK</span></th>
    <th><span class="th-tip" data-tip-key="signal">SIGNAL</span></th>
    <th><span class="th-tip" data-tip-key="cat">CAT</span></th>
    <th><span class="th-tip" data-tip-key="mom">MOM 12-1</span></th>
    <th><span class="th-tip" data-tip-key="rs">RS 6M</span></th>
    <th>PRICE</th><th>MCAP</th>
  </tr></thead>
  <tbody>{"".join(rows)}</tbody>
</table>
</div>"""


def _render_screener() -> None:
    _page_title("SCREENER RESULTS")

    output_dir = Path("output")
    csv_files = sorted(output_dir.glob("screen_*.csv"), reverse=True)

    if not csv_files:
        st.info("No screen output files found in output/")
        return

    dates = [f.stem.replace("screen_", "") for f in csv_files]
    selected_date = st.selectbox("Screen date", dates, label_visibility="collapsed")

    try:
        df = pd.read_csv(output_dir / f"screen_{selected_date}.csv")
    except Exception as e:
        st.error(f"Could not read {selected_date}: {e}")
        return

    top_ticker = str(df.iloc[0]["ticker"]) if len(df) > 0 else "—"
    top_score  = float(df.iloc[0]["composite"]) if len(df) > 0 and "composite" in df.columns else 0.0
    avg        = float(df["composite"].mean()) if "composite" in df.columns else 0.0
    confirms   = int((df["entry_signal"] == "confirm_entry").sum()) if "entry_signal" in df.columns else 0
    avoids     = int((df["entry_signal"] == "avoid").sum()) if "entry_signal" in df.columns else 0

    # ── Summary strip ─────────────────────────────────────────────────────────
    st.markdown(f"""
<div class="summary-strip">
  <div class="summary-cell">
    <div class="summary-label">TOP PICK</div>
    <div class="summary-value accent">{_html.escape(top_ticker)}</div>
  </div>
  <div class="summary-cell">
    <div class="summary-label">SCORE</div>
    <div class="summary-value">
      <span class="slot-score" data-target="{top_score:.4f}" data-decimals="2">{top_score:.2f}</span>
    </div>
  </div>
  <div class="summary-cell">
    <div class="summary-label">RANKED</div>
    <div class="summary-value">{len(df)}</div>
  </div>
  <div class="summary-cell">
    <div class="summary-label">AVG SCORE</div>
    <div class="summary-value">{avg:.2f}</div>
  </div>
  <div class="summary-cell">
    <div class="summary-label">CONFIRMS</div>
    <div class="summary-value bull">{confirms}</div>
  </div>
  <div class="summary-cell">
    <div class="summary-label">AVOIDS</div>
    <div class="summary-value bear">{avoids}</div>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── Top 3 cards ───────────────────────────────────────────────────────────
    cols = st.columns(3)
    for i, col in enumerate(cols):
        if i < len(df):
            with col:
                st.markdown(_top3_card(i + 1, df.iloc[i]), unsafe_allow_html=True)

    st.markdown('<div style="height:1px;background:var(--border);margin:1.5rem 0"></div>', unsafe_allow_html=True)

    # ── Full ranked table ─────────────────────────────────────────────────────
    st.markdown(_screener_html_table(df), unsafe_allow_html=True)

    # ── Sustained Movers ─────────────────────────────────────────────────────
    if "streak_consecutive" in df.columns:
        sustained = df[df["streak_consecutive"] >= 3].copy()
        if len(sustained) > 0:
            with st.expander(f"Sustained Movers — {len(sustained)} stock{'s' if len(sustained) != 1 else ''} with 3d+ streak"):
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
| **ADX / DMI (14)** | Trend *strength* + direction | >20 | Weakening >5 pts AND −DI > +DI |
| **MFI (14)** | Volume-weighted RSI — tracks smart money flow | 40–65 | <50 (money leaving) |
| **OBV / SMA20-50-200** | Volume accumulation + trend structure | OBV rising, price > MAs | OBV distribution, close < SMA |
| **Chandelier / ATR stop** | Volatility-scaled trailing stop | — | close < high(22) − 3·ATR |

**Standing exit plan** — Each open position is evaluated once per day, on the closing price, during the 4:30pm pipeline run — never on page load — so a verdict can't flip with intraday noise or an ordinary market-wide down/up day the way a live-recomputed snapshot would.
- **SELL** (terminal — persists until you act) fires on any one of: close below the trailing stop (peak close − trail-mult × ATR14), close below the max-loss floor (entry − 2×ATR14, which ratchets up to breakeven once the de-risk trim fires), or 3+ consecutive closes below the 50-day SMA.
- **TRIM** rungs each fire at most once, ever, and each means sell 1/3 of the position: **de-risk** at +2R or +20% gain (moves the floor up to breakeven) and **blowoff** extension (>25% above the 50-day, weekly RSI > 80, or a >3×ATR burst within 5 sessions).
- **Weekly health check** can only tighten the trailing multiplier (3.0 → 2.5 → 2.0), never loosen it, and never issues a SELL by itself.
- Stops only ratchet up, trims are append-only, and a SELL is terminal — so the verdict can't oscillate day to day the way the old recomputed-every-load grade did.
- **EARN Nd** chip = earnings within 14 days (risk flag only, shown alongside the verdict — deliberately not part of it).
        """)

    # ── News Entry Signals — top picks ────────────────────────────────────────
    if "entry_signal" in df.columns and df["entry_signal"].notna().any():
        st.markdown('<div style="height:1px;background:var(--border);margin:1rem 0"></div>', unsafe_allow_html=True)
        st.markdown(
            '<div style="font-family:var(--mono);font-size:0.72rem;font-weight:700;'
            'letter-spacing:0.1em;color:var(--muted);margin-bottom:0.75rem">NEWS ENTRY SIGNALS</div>',
            unsafe_allow_html=True,
        )
        _es_colors = {
            "confirm_entry": ("#22c55e", "rgba(34,197,94,0.08)"),
            "wait":          ("#f59e0b", "rgba(245,158,11,0.08)"),
            "avoid":         ("#ef4444", "rgba(239,68,68,0.08)"),
        }
        _cat_icons = {"estimate_up": "EST UP", "estimate_down": "EST DOWN"}
        for _, row in df.head(10).iterrows():
            es = str(row.get("entry_signal", "") or "")
            if not es or (es == "wait" and not row.get("news_reasoning")):
                continue
            ticker = str(row.get("ticker", ""))
            fg, bg = _es_colors.get(es, ("#94a3b8", "rgba(148,163,184,0.08)"))
            es_label = {"confirm_entry": "CONFIRM", "wait": "WAIT", "avoid": "AVOID"}.get(es, es)
            cat = str(row.get("catalyst", "") or "")
            cat_str = _cat_icons.get(cat, "")
            tc = str(row.get("thesis_consistency", "") or "")
            tc_badge = {"confirms": "Confirms thesis", "contradicts": "Contradicts thesis"}.get(tc, "")
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
                        f'<div style="background:{bg};border:1px solid {fg}30;color:{fg};padding:10px 14px;'
                        f'border-radius:8px;font-size:13px;margin-top:6px;font-family:var(--sans)">'
                        f'{_html.escape(reasoning)}</div>',
                        unsafe_allow_html=True,
                    )


# ── Market Regime ─────────────────────────────────────────────────────────────

def _render_regime() -> None:
    _page_title("MARKET REGIME")

    with st.spinner("Computing market regime…"):
        data = _cached_spy_regime()

    if data.get("error"):
        st.error(f"Could not compute regime: {data['error']}")
        return

    regime      = data["regime"]
    score       = data["score"]
    signals     = data["signals"]
    as_of       = data["as_of"]
    bull_count  = data["bull_count"]
    total_count = data["total_count"]

    _rmap = {
        "BULL":    ("var(--bull)", "FAVORABLE — momentum strategies have tailwind"),
        "BEAR":    ("var(--bear)", "RISK-OFF — reduce or hedge long exposure"),
        "NEUTRAL": ("var(--wait)", "MIXED — selective entries only, size down"),
    }
    r_color, r_desc = _rmap.get(regime, _rmap["NEUTRAL"])
    score_pct = int(score / 10 * 100)

    # ── Regime banner ─────────────────────────────────────────────────────────
    banner_s = (f"background:var(--surface-1);border:1px solid var(--border);"
                f"border-top:3px solid {r_color};border-radius:var(--radius-lg);"
                f"padding:24px 28px;margin-bottom:1rem")
    st.markdown(f"""
<div style="{banner_s}">
  <div style="display:flex;align-items:flex-start;justify-content:space-between;gap:20px;flex-wrap:wrap">
    <div>
      <div style="font-family:var(--mono);font-size:0.6rem;color:var(--muted);letter-spacing:0.1em;margin-bottom:6px">SPY COMPOSITE SIGNAL</div>
      <div style="font-family:var(--mono);font-size:2.8rem;font-weight:700;color:{r_color};letter-spacing:-0.02em;line-height:1">{regime}</div>
      <div style="font-size:0.76rem;color:var(--muted);margin-top:8px">{r_desc}</div>
    </div>
    <div style="text-align:right;flex-shrink:0">
      <div style="font-family:var(--mono);font-size:0.6rem;color:var(--muted);letter-spacing:0.1em;margin-bottom:6px">COMPOSITE SCORE</div>
      <span class="slot-score" data-target="{float(score):.1f}" data-decimals="1" style="font-size:2.2rem;font-weight:700;color:{r_color}">{score:.1f}</span>
      <span style="font-family:var(--mono);font-size:0.9rem;color:var(--muted)">/10</span>
    </div>
  </div>
  <div style="margin-top:16px">
    <div style="display:flex;justify-content:space-between;margin-bottom:5px">
      <span style="font-family:var(--mono);font-size:0.58rem;color:var(--muted);letter-spacing:0.07em">BULL SIGNALS {bull_count}/{total_count}</span>
      <span style="font-family:var(--mono);font-size:0.58rem;color:{r_color}">{score_pct}%</span>
    </div>
    <div class="fill-bar-track" style="height:5px;width:100%">
      <div class="fill-bar-fill" style="width:{score_pct}%;background:{r_color};height:5px"></div>
    </div>
  </div>
  <div style="font-family:var(--mono);font-size:0.56rem;color:var(--dim);margin-top:10px;letter-spacing:0.05em">AS OF {_html.escape(as_of)}</div>
</div>
""", unsafe_allow_html=True)

    # ── Playbook: what to actually do with this rating ────────────────────────
    _playbook = {
        "BULL": {
            "action": "HOLD & ADD",
            "detail": "Trend and breadth support staying long. Hold existing positions. New capital can be "
                      "deployed on pullbacks (don't chase extended names). Momentum/growth setups get a tailwind.",
            "horizon": "Weeks–months (swing/position). Re-check weekly — regimes like this don't flip overnight.",
        },
        "CAUTION": {
            "action": "HOLD, DON'T ADD",
            "detail": "Signals are split. Hold what you already own; new entries should be limited to names that "
                      "already clear your full checklist — avoid speculative adds. Consider trimming your weakest "
                      "or most-extended positions.",
            "horizon": "Days–weeks. Re-check every few days — this regime can tip either direction quickly.",
        },
        "BEAR": {
            "action": "REDUCE / HEDGE",
            "detail": "Trend and risk signals have broken down. Trim or hedge long exposure, raise cash, avoid new "
                      "longs. If you must hold, keep only your highest-conviction, most defensive names.",
            "horizon": "Reassess daily until the regime turns. This is risk-off, not a buy-the-dip signal.",
        },
    }
    _pb = _playbook.get(regime, _playbook["CAUTION"])
    st.markdown(f"""
<div style="background:var(--surface-1);border:1px solid var(--border);border-left:3px solid {r_color};
            border-radius:var(--radius-lg);padding:16px 20px;margin-bottom:1rem">
  <div style="font-family:var(--mono);font-size:0.6rem;color:var(--muted);letter-spacing:0.1em;margin-bottom:8px">WHAT TO DO WITH THIS</div>
  <div style="font-family:var(--mono);font-size:1.05rem;font-weight:700;color:{r_color};margin-bottom:6px">{_pb['action']}</div>
  <div style="font-size:0.8rem;color:var(--text);margin-bottom:10px;line-height:1.5">{_pb['detail']}</div>
  <div style="font-size:0.7rem;color:var(--muted)"><b>Horizon:</b> {_pb['horizon']}</div>
  <div style="font-size:0.68rem;color:var(--dim);margin-top:10px;border-top:1px solid var(--border);padding-top:8px">
    This is a market-wide (SPY) regime, not a per-ticker buy/sell call — internally it scales how many names the
    screener surfaces (full size in BULL, top-10 only in CAUTION, screen skipped in BEAR/stress). Pair it with each
    stock's own Entry Timing signal for individual decisions.
  </div>
</div>
""", unsafe_allow_html=True)

    # ── Key metrics strip ─────────────────────────────────────────────────────
    vix_sig = next((s for s in signals if s["name"] == "vix"),          None)
    yld_sig = next((s for s in signals if s["name"] == "yield_curve"),  None)
    gc_sig  = next((s for s in signals if s["name"] == "golden_cross"), None)
    rsi_sig = next((s for s in signals if s["name"] == "rsi"),          None)

    def _sv(sig: dict | None) -> str:
        return _html.escape(str(sig["value"])) if sig else "—"
    def _sc(sig: dict | None) -> str:
        if not sig: return "var(--muted)"
        return "var(--bull)" if sig["is_bull"] else "var(--bear)"
    sma_label = ("GOLDEN" if gc_sig and gc_sig["is_bull"] else
                 "DEATH"  if gc_sig and not gc_sig["is_bull"] else "—")

    st.markdown(f"""
<div class="summary-strip">
  <div class="summary-cell">
    <div class="summary-label">COMPOSITE</div>
    <div class="summary-value" style="color:{r_color}">{score:.1f}/10</div>
  </div>
  <div class="summary-cell">
    <div class="summary-label">VIX</div>
    <div class="summary-value" style="color:{_sc(vix_sig)}">{_sv(vix_sig)}</div>
  </div>
  <div class="summary-cell">
    <div class="summary-label">YIELD CURVE</div>
    <div class="summary-value" style="color:{_sc(yld_sig)}">{_sv(yld_sig)}</div>
  </div>
  <div class="summary-cell">
    <div class="summary-label">SMA CROSS</div>
    <div class="summary-value" style="color:{_sc(gc_sig)}">{sma_label}</div>
  </div>
  <div class="summary-cell">
    <div class="summary-label">RSI</div>
    <div class="summary-value" style="color:{_sc(rsi_sig)}">{_sv(rsi_sig)}</div>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── Signal breakdown table ────────────────────────────────────────────────
    st.markdown(
        '<div style="font-family:var(--mono);font-size:0.7rem;font-weight:700;'
        'letter-spacing:0.1em;color:var(--muted);margin-bottom:0.5rem">SIGNAL BREAKDOWN</div>',
        unsafe_allow_html=True,
    )
    sig_rows = []
    for i, s in enumerate(signals):
        dot_c = "var(--bull)" if s["is_bull"] else "var(--bear)"
        state = "BULL" if s["is_bull"] else "BEAR"
        sig_rows.append(
            f'<tr style="animation-delay:{i*16}ms">'
            f'<td><span style="width:6px;height:6px;border-radius:50%;background:{dot_c};'
            f'display:inline-block;margin-right:10px;flex-shrink:0"></span>'
            f'<span style="font-size:0.8rem;color:var(--text)">{_html.escape(s["label"])}</span></td>'
            f'<td style="font-family:var(--mono);font-size:0.75rem;color:var(--muted);text-align:right">{_html.escape(str(s["value"]))}</td>'
            f'<td style="text-align:right"><span style="font-family:var(--mono);font-size:0.63rem;color:{dot_c};letter-spacing:0.07em">{state}</span></td>'
            f'</tr>'
        )
    st.markdown(f"""
<div style="overflow-x:auto;border:1px solid var(--border);border-radius:var(--radius);overflow:hidden;margin-bottom:1rem">
<table class="q-table">
  <thead><tr><th>INDICATOR</th><th style="text-align:right">VALUE</th><th style="text-align:right">STATE</th></tr></thead>
  <tbody>{"".join(sig_rows)}</tbody>
</table>
</div>""", unsafe_allow_html=True)

    # ── Sector signals ────────────────────────────────────────────────────────
    try:
        import sqlite3 as _sq3, json as _json
        _conn2 = _sq3.connect("data/cache.db")
        _row2  = _conn2.execute(
            "SELECT payload FROM news_sentiment WHERE ticker='__MARKET__' ORDER BY fetched_at DESC LIMIT 1"
        ).fetchone()
        _conn2.close()
        if _row2:
            _mp         = _json.loads(_row2[0])
            _sector_sigs = _mp.get("sector_signals") or {}
            _regime_note = _mp.get("regime_note", "")
            if _sector_sigs:
                st.markdown(
                    '<div style="font-family:var(--mono);font-size:0.7rem;font-weight:700;'
                    'letter-spacing:0.1em;color:var(--muted);margin-bottom:0.5rem">SECTOR SIGNALS</div>',
                    unsafe_allow_html=True,
                )
                if _regime_note:
                    st.markdown(
                        f'<div style="font-size:0.78rem;color:var(--muted);margin-bottom:0.75rem">'
                        f'{_html.escape(_regime_note)}</div>',
                        unsafe_allow_html=True,
                    )
                chips = ""
                for _sec, _sig in _sector_sigs.items():
                    _dir  = _sig.get("direction", "")
                    _str  = _sig.get("strength", "")
                    _rsn  = _sig.get("reason", "")
                    _cc   = "var(--bear)" if _dir == "headwind" else "var(--bull)"
                    _cbg  = "var(--bear-dim)" if _dir == "headwind" else "var(--bull-dim)"
                    _arr  = "&#8595;" if _dir == "headwind" else "&#8593;"
                    _tail = f" · {_html.escape(_str)}" if _str else ""
                    chips += (
                        f'<span title="{_html.escape(_rsn)}" style="display:inline-block;'
                        f'background:{_cbg};color:{_cc};border:1px solid {_cc}30;'
                        f'padding:4px 12px;border-radius:20px;font-family:var(--mono);'
                        f'font-size:0.68rem;font-weight:600;margin:3px">'
                        f'{_arr} {_html.escape(_sec)}{_tail}</span>'
                    )
                st.markdown(f'<div style="margin-bottom:1rem">{chips}</div>', unsafe_allow_html=True)
    except Exception:
        pass

    # ── Market news ───────────────────────────────────────────────────────────
    st.markdown(
        '<div style="font-family:var(--mono);font-size:0.7rem;font-weight:700;'
        'letter-spacing:0.1em;color:var(--muted);margin:1rem 0 0.5rem">MARKET NEWS</div>',
        unsafe_allow_html=True,
    )
    with st.spinner("Loading market news…"):
        articles, news_analysis = _cached_market_news()

    _news_card(news_analysis, articles, headline_count=8)

    with st.expander("All headlines"):
        for a in articles:
            h   = a.get("headline", "")
            src = a.get("source", "")
            if h:
                st.caption(f"• {_html.escape(h)}  ({_html.escape(src)})")


# ── Open Positions ────────────────────────────────────────────────────────────

_MIN_HEALTH_WEEKS = 15  # mirrors weekly_health() in src/exit_plan.py + src/exit_alerts.py


def _plan_chip(text: str, color: str = "var(--muted)") -> str:
    return (
        f'<span style="display:inline-flex;align-items:center;gap:5px;'
        f'padding:5px 10px;border-radius:var(--radius-sm);background:var(--surface-2);'
        f'border:1px solid var(--border)">'
        f'<span style="width:7px;height:7px;border-radius:50%;background:{color};flex-shrink:0"></span>'
        f'<span style="font-family:var(--mono);font-size:0.65rem;color:{color}">{text}</span>'
        f'</span>'
    )


def _load_fidelity_data() -> tuple[list[dict], str]:
    """Load rich Fidelity positions data. Returns (positions, synced_at_str)."""
    p = Path("data/fidelity/positions_data.json")
    if not p.exists():
        return [], ""
    try:
        raw = json.loads(p.read_text())
        return raw.get("positions", []), raw.get("synced_at", "")
    except Exception:
        return [], ""


def _fmt_gl(dollar: float, pct: float) -> str:
    sign = "+" if dollar >= 0 else ""
    return f"{sign}${dollar:,.2f} ({sign}{pct:.1%})"


def _live_fid_metrics(fid: dict, live_price: float | None, prev_close: float | None) -> dict:
    """Recompute a Fidelity position's price/value/G-L from a live yfinance quote.

    Fidelity supplies qty/avg_cost (stable, only change on trades); price/value/G-L
    are recomputed live so they don't go stale between Fidelity syncs. Falls back
    to the Fidelity snapshot if the live quote fetch fails.
    """
    qty      = fid["quantity"]
    avg_cost = fid["avg_cost"]

    if live_price is None:
        return {
            "last_price": fid["last_price"],
            "current_value": fid["current_value"],
            "today_gl_d": fid["today_gl_dollar"],
            "today_gl_p": fid["today_gl_pct"],
            "total_gl_d": fid["total_gl_dollar"],
            "total_gl_p": fid["total_gl_pct"],
        }

    total_gl_d = qty * (live_price - avg_cost)
    if prev_close:
        today_gl_d = qty * (live_price - prev_close)
        today_gl_p = (live_price - prev_close) / prev_close
    else:
        today_gl_d, today_gl_p = fid["today_gl_dollar"], fid["today_gl_pct"]
    return {
        "last_price": live_price,
        "current_value": qty * live_price,
        "today_gl_d": today_gl_d,
        "today_gl_p": today_gl_p,
        "total_gl_d": total_gl_d,
        "total_gl_p": (live_price - avg_cost) / avg_cost if avg_cost else 0.0,
    }


def _render_position_card(pos: dict, fid: dict | None, cached: tuple[dict, float | None, float | None] | None = None) -> None:
    ticker  = pos["ticker"]
    _, live_price, prev_close = cached if cached is not None else ({}, None, None)

    plan          = pos.get("plan") or {}
    not_evaluated = not plan
    verdict       = plan.get("verdict", "—")
    verdict_reason = plan.get("verdict_reason")
    stop_level    = plan.get("stop_level")
    trims         = plan.get("trims_fired") or []
    health        = plan.get("health") or {}
    dte           = plan.get("days_to_earnings")
    last_eval     = plan.get("last_eval") or "never"

    entry_price = pos.get("entry_price", 0)
    entry_date  = pos.get("entry_date", "")
    try:
        held_days = (date.today() - datetime.strptime(entry_date, "%Y-%m-%d").date()).days
    except Exception:
        held_days = "?"

    if fid:
        qty            = fid["quantity"]
        avg_cost       = fid["avg_cost"]
        pct_acct       = fid["pct_of_account"]
        desc           = fid.get("description", "")

        m = _live_fid_metrics(fid, live_price, prev_close)
        last_price     = m["last_price"]
        current_value  = m["current_value"]
        today_gl_d     = m["today_gl_d"]
        today_gl_p     = m["today_gl_p"]
        total_gl_d     = m["total_gl_d"]
        total_gl_p     = m["total_gl_p"]

        price_str      = f"${last_price:,.2f}"
        total_pnl_color = "var(--bull)" if total_gl_d >= 0 else "var(--bear)"
        total_pnl_str   = f"+{total_gl_p:.1%}" if total_gl_d >= 0 else f"{total_gl_p:.1%}"
        today_color     = "var(--bull)" if today_gl_d >= 0 else "var(--bear)"
        today_str       = _fmt_gl(today_gl_d, today_gl_p)
        total_str       = _fmt_gl(total_gl_d, total_gl_p)
        meta_row = (
            f'{qty:g} shares · ${avg_cost:,.2f} avg cost · '
            f'${current_value:,.2f} value · {pct_acct:.1%} of acct'
        )
        gl_row = (
            f'<span style="color:{today_color}">Today {today_str}</span>'
            f'<span style="color:var(--muted)"> · </span>'
            f'<span style="color:{total_pnl_color}">Total {total_str}</span>'
        )
        desc_html = f'<div style="font-family:var(--mono);font-size:0.58rem;color:var(--muted);margin-top:1px">{_html.escape(desc)}</div>' if desc else ""
        price_src = "live" if live_price is not None else "fidelity snapshot"
    else:
        current_price = live_price
        price_src = "live" if current_price is not None else "—"
        price_str = f"${current_price:,.2f}" if current_price else "—"
        if current_price and entry_price:
            pnl = (current_price - entry_price) / entry_price
            total_pnl_str   = f"+{pnl:.1%}" if pnl >= 0 else f"{pnl:.1%}"
            total_pnl_color = "var(--bull)" if pnl >= 0 else "var(--bear)"
        else:
            total_pnl_str, total_pnl_color = "—", "var(--muted)"
        meta_row = f"${entry_price:,.2f} avg cost" if entry_price else ""
        gl_row   = ""
        desc_html = ""

    top_c = {"SELL": "var(--bear)", "TRIM": "var(--wait)", "HOLD": "var(--bull)"}.get(verdict, "var(--muted)")

    # Stop chip — distance is always measured off the plan's last-close
    # evaluation (plan["last_close"]), NEVER the live quote. This chip is a
    # standing-plan number, not a live one: mixing in the live price here is
    # exactly the bug this whole feature exists to remove — a mid-session dip
    # would flip this chip red/negative while the verdict beside it, which is
    # also frozen to last close, still says HOLD in green. Live price has its
    # own place (the price/P&L display above) where it belongs.
    plan_close = plan.get("last_close")
    if stop_level is not None and plan_close:
        dist = (plan_close - stop_level) / plan_close
        stop_color = "var(--bear)" if dist <= 0 else "var(--muted)"
        stop_chip = _plan_chip(f'STOP (close) ${stop_level:,.2f} ({dist:+.1%})', stop_color)
    else:
        stop_chip = _plan_chip("STOP —", "var(--muted)")

    trims_str   = "+".join(_html.escape(t) for t in trims) if trims else "none"
    trims_chip  = _plan_chip(f"TRIMS {trims_str}", "var(--wait)" if trims else "var(--muted)")

    # Health chip label comes from src.exit_plan.health_label — the single
    # source of truth also used by src/exit_alerts.py's email digest, so the
    # page and the email can never disagree on the score. (This used to
    # hardcode "?/4*" whenever a check errored, discarding the real bearish
    # count the email showed.) This block owns only the chip's color.
    h_label  = health_label(health)
    h_weeks  = health.get("weeks") or 0
    h_errors = health.get("errors") or []
    if not health or h_weeks < _MIN_HEALTH_WEEKS:
        health_chip = _plan_chip(f"HEALTH {h_label}", "var(--muted)")
    elif h_errors:
        health_chip = _plan_chip(f"HEALTH {h_label}", "var(--wait)")
    else:
        bearish = health.get("bearish", 0)
        h_color = "var(--bear)" if bearish >= 3 else ("var(--wait)" if bearish >= 1 else "var(--bull)")
        health_chip = _plan_chip(f"HEALTH {h_label}", h_color)

    # Earnings-proximity risk chip — informational, NOT part of the verdict.
    earn_chip = ""
    if isinstance(dte, (int, float)) and 0 <= dte <= 14:
        earn_chip = (
            f'<span style="display:inline-flex;align-items:center;gap:5px;'
            f'padding:5px 10px;border-radius:var(--radius-sm);'
            f'background:rgba(245,158,11,0.10);border:1px solid rgba(245,158,11,0.35)">'
            f'<span style="width:7px;height:7px;border-radius:50%;background:var(--wait);flex-shrink:0"></span>'
            f'<span style="font-family:var(--mono);font-size:0.65rem;color:var(--wait)">EARN {int(dte)}d</span>'
            f'</span>'
        )

    badges = stop_chip + " " + trims_chip + " " + health_chip + (" " + earn_chip if earn_chip else "")

    exit_html = ""
    if not_evaluated:
        exit_html = (
            f'<div style="margin-top:12px;padding:10px 14px;background:var(--surface-2);'
            f'border:1px solid var(--border);border-radius:var(--radius-sm)">'
            f'<span style="font-family:var(--mono);font-size:0.7rem;font-weight:700;'
            f'color:var(--muted);letter-spacing:0.08em">NOT YET EVALUATED — awaiting first nightly close</span>'
            f'</div>'
        )
    elif verdict == "SELL":
        # verdict_reason (persisted by evaluate_day at the moment it set this
        # verdict) is the same trigger the email carries — stop-breach,
        # floor-breach, or 50-day trend-break each read differently, and
        # without it the banner can't say which one fired (e.g. a trend-break
        # SELL pairs with a positive, reassuring stop-chip distance, so the
        # reason is the only place the user learns why this is a SELL at all).
        sell_reason_html = (
            f'<div style="font-family:var(--mono);font-size:0.62rem;color:var(--muted);'
            f'margin-top:4px">{_html.escape(verdict_reason)}</div>'
        ) if verdict_reason else ""
        exit_html = (
            f'<div style="margin-top:12px;padding:10px 14px;background:var(--bear-dim);'
            f'border:1px solid rgba(239,68,68,0.3);border-radius:var(--radius-sm);'
            f'animation:shockwave 0.55s ease-out 0.25s 1 both">'
            f'<span style="font-family:var(--mono);font-size:0.7rem;font-weight:700;'
            f'color:var(--bear);letter-spacing:0.08em">SELL — exit remaining position at next open</span>'
            f'{sell_reason_html}'
            f'</div>'
        )
    elif verdict == "TRIM":
        rungs = f" ({trims_str})" if trims else ""
        trim_reason_html = (
            f'<div style="font-family:var(--mono);font-size:0.62rem;color:var(--muted);'
            f'margin-top:4px">{_html.escape(verdict_reason)}</div>'
        ) if verdict_reason else ""
        exit_html = (
            f'<div style="margin-top:12px;padding:10px 14px;background:rgba(245,158,11,0.08);'
            f'border:1px solid rgba(245,158,11,0.3);border-radius:var(--radius-sm)">'
            f'<span style="font-family:var(--mono);font-size:0.7rem;font-weight:700;'
            f'color:var(--wait);letter-spacing:0.08em">TRIM — sell 1/3 at next open{rungs}</span>'
            f'{trim_reason_html}'
            f'</div>'
        )

    card_s = f"background:var(--surface-1);border:1px solid var(--border);border-top:2px solid {top_c};border-radius:var(--radius-lg);padding:18px;margin-bottom:10px;position:relative;overflow:hidden"

    verdict_line = (
        f'<div style="font-family:var(--mono);font-size:0.55rem;color:var(--muted);letter-spacing:0.09em;margin-bottom:8px">NOT YET EVALUATED</div>'
        if not_evaluated else
        f'<div style="font-family:var(--mono);font-size:0.55rem;color:{top_c};letter-spacing:0.09em;margin-bottom:8px">VERDICT {_html.escape(str(verdict))} · evaluated {_html.escape(str(last_eval))}</div>'
    )

    # All HTML on single lines — Streamlit's markdown parser treats 4+ leading
    # spaces as a code block, so indented multi-line HTML breaks rendering.
    card_html = (
        f'<div class="tilt-card" style="{card_s}">'
        f'<div class="card-shine"></div>'
        f'<div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:10px">'
        f'<div>'
        f'<div style="font-family:var(--mono);font-size:1.35rem;font-weight:700;color:var(--text)">{_html.escape(ticker)}</div>'
        f'{desc_html}'
        f'<div style="font-family:var(--mono);font-size:0.58rem;color:var(--muted);margin-top:4px;letter-spacing:0.04em">{_html.escape(entry_date)} · {held_days}d held</div>'
        f'</div>'
        f'<div style="text-align:right">'
        f'<div style="font-family:var(--mono);font-size:1.5rem;font-weight:700;color:{total_pnl_color}">{total_pnl_str}</div>'
        f'<div style="font-family:var(--mono);font-size:0.62rem;color:var(--muted)">{price_str} · {price_src}</div>'
        f'</div>'
        f'</div>'
        f'<div style="font-family:var(--mono);font-size:0.6rem;color:var(--muted);margin-bottom:6px">{_html.escape(meta_row)}</div>'
        f'<div style="font-family:var(--mono);font-size:0.6rem;margin-bottom:10px">{gl_row}</div>'
        f'{verdict_line}'
        f'<div style="display:flex;flex-wrap:wrap;gap:6px">{badges}</div>'
        f'{exit_html}'
        f'</div>'
    )

    st.markdown(card_html, unsafe_allow_html=True)


def _render_positions() -> None:
    _page_title("OPEN POSITIONS")

    fid_positions, synced_at = _load_fidelity_data()
    fid_by_ticker = {f["ticker"]: f for f in fid_positions}

    col_sync, col_btn = st.columns([5, 1])
    with col_sync:
        if synced_at:
            try:
                ts = datetime.fromisoformat(synced_at).strftime("%-I:%M %p")
            except Exception:
                ts = synced_at
            st.markdown(
                f'<div style="font-family:var(--mono);font-size:0.6rem;color:var(--muted);'
                f'padding-top:6px">FIDELITY SYNC · {ts}</div>',
                unsafe_allow_html=True,
            )
    with col_btn:
        if st.button("↺ LIVE REFRESH", key="positions_live_refresh", use_container_width=True):
            _batch_position_data.clear()
            st.rerun()

    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)

    positions = load_positions()
    if not positions:
        st.info("No positions synced yet — run Fidelity sync to populate.")
        return

    tickers = tuple(sorted(p["ticker"] for p in positions))
    with st.spinner(f"Fetching live data for {len(tickers)} position{'s' if len(tickers) != 1 else ''}…"):
        batch, batch_fetched_at = _batch_position_data(tickers)

    live_count = sum(1 for v in batch.values() if v[1] is not None)
    quote_ts = batch_fetched_at.strftime("%-I:%M:%S %p")
    quote_color = "var(--muted)" if live_count == len(tickers) else "var(--bear)"
    quote_note = "" if live_count == len(tickers) else " (rest showing Fidelity snapshot — yfinance fetch failed)"
    st.markdown(
        f'<div style="font-family:var(--mono);font-size:0.6rem;color:var(--muted);margin-top:-6px;margin-bottom:4px">'
        f'LIVE QUOTES · {quote_ts} · <span style="color:{quote_color}">{live_count}/{len(tickers)} live{quote_note}</span></div>',
        unsafe_allow_html=True,
    )

    # Rank by the standing plan's verdict: SELL first, then TRIM, then HOLD.
    # A position with no plan yet (never evaluated by the nightly job) gets
    # its own bucket below HOLD — it must not be conflated with an actual
    # HOLD verdict, which is a real, evaluated "stay in" decision.
    _order = {"SELL": 2, "TRIM": 1, "HOLD": 0}
    enriched = []
    for p in positions:
        plan = p.get("plan") or {}
        v = plan.get("verdict") if plan else "UNEVALUATED"
        enriched.append({**p, "_score": _order.get(v, -1), "_grade": v})
    enriched.sort(key=lambda x: x["_score"], reverse=True)

    exits = sum(1 for e in enriched if e["_grade"] == "SELL")
    exit_cls = "bear" if exits > 0 else ""

    # Portfolio totals recomputed from live quotes (falls back to Fidelity snapshot per-position)
    total_value = total_today = total_overall = 0.0
    for f in fid_positions:
        _, live_price, prev_close = batch.get(f["ticker"], ({}, None, None))
        m = _live_fid_metrics(f, live_price, prev_close)
        total_value   += m["current_value"]
        total_today   += m["today_gl_d"]
        total_overall += m["total_gl_d"]
    today_sign    = "+" if total_today >= 0 else ""
    overall_sign  = "+" if total_overall >= 0 else ""
    today_cls     = "bull" if total_today >= 0 else "bear"
    overall_cls   = "bull" if total_overall >= 0 else "bear"

    st.markdown(f"""
<div class="summary-strip">
  <div class="summary-cell">
    <div class="summary-label">POSITIONS</div>
    <div class="summary-value">{len(enriched)}</div>
  </div>
  <div class="summary-cell">
    <div class="summary-label">PORTFOLIO VALUE</div>
    <div class="summary-value">${total_value:,.0f}</div>
  </div>
  <div class="summary-cell">
    <div class="summary-label">TODAY</div>
    <div class="summary-value {today_cls}">{today_sign}${total_today:,.2f}</div>
  </div>
  <div class="summary-cell">
    <div class="summary-label">TOTAL G/L</div>
    <div class="summary-value {overall_cls}">{overall_sign}${total_overall:,.2f}</div>
  </div>
  <div class="summary-cell">
    <div class="summary-label">EXIT ALERTS</div>
    <div class="summary-value {exit_cls}">{exits}</div>
  </div>
</div>
""", unsafe_allow_html=True)

    for pos in enriched:
        _render_position_card(pos, fid_by_ticker.get(pos["ticker"]), batch.get(pos["ticker"]))


# ── Run Monitor ───────────────────────────────────────────────────────────────

def _find_todays_log() -> "Path | None":
    today = date.today().strftime("%Y-%m-%d")
    cron = Path(f"logs/run_{today}.log")
    if cron.exists():
        return cron
    manual = sorted(Path("logs").glob(f"manual_run_{today}_*.log"), reverse=True)
    return manual[0] if manual else None


def _last_run_text(text: str) -> str:
    """Return only the portion of the log file belonging to the most recent run."""
    marker = "=== Screener run started:"
    idx = text.rfind(marker)
    return text[idx:] if idx != -1 else text


def _parse_run_state(text: str) -> dict:
    s: dict = {
        "started_at": None, "finished_at": None,
        "universe": None,
        "batch_cur": None, "batch_tot": None, "liquidity_survivors": None,
        "fund_cur": None, "fund_tot": None,
        "stress_regime": None,
        "quality_in": None, "quality_out": None,
        "confirm_in": None, "confirm_out": None,
        "compose_ranked": None, "compose_top": None,
        "news_cur": None, "news_tot": None,
        "output_done": False,
    }
    for line in text.splitlines():
        if "=== Screener run started:" in line:
            m = re.search(r"started: (.+) ===", line)
            if m: s["started_at"] = m.group(1).strip()
        if "=== Screener run finished:" in line:
            m = re.search(r"finished: (.+) ===", line)
            if m: s["finished_at"] = m.group(1).strip()
        if m := re.search(r"\[universe\] loaded (\d+) tickers", line):
            s["universe"] = int(m.group(1))
        if m := re.search(r"\[prices\] batch (\d+)/(\d+)", line):
            s["batch_cur"], s["batch_tot"] = int(m.group(1)), int(m.group(2))
        if m := re.search(r"\[prices\] \d+ universe → (\d+) passed", line):
            s["liquidity_survivors"] = int(m.group(1))
        if m := re.search(r"\[fundamentals\] (\d+)/(\d+)", line):
            s["fund_cur"], s["fund_tot"] = int(m.group(1)), int(m.group(2))
        if m := re.search(r"\[stress\] regime=(\w+)", line):
            s["stress_regime"] = m.group(1)
        if m := re.search(r"\[quality_gate\] (\d+) → (\d+)", line):
            s["quality_in"], s["quality_out"] = int(m.group(1)), int(m.group(2))
        if m := re.search(r"\[confirmation_gate\] (\d+) → (\d+)", line):
            s["confirm_in"], s["confirm_out"] = int(m.group(1)), int(m.group(2))
        if m := re.search(r"\[compose\] (\d+) ranked → top (\d+)", line):
            s["compose_ranked"], s["compose_top"] = int(m.group(1)), int(m.group(2))
        if m := re.search(r"\[news\] overlay: (\d+)/(\d+)", line):
            s["news_cur"], s["news_tot"] = int(m.group(1)), int(m.group(2))
        if "[output] output/screen_" in line:
            s["output_done"] = True
    return s


def _current_stage(s: dict) -> int:
    if s["output_done"]:        return 7
    if s["news_cur"] is not None: return 6
    if s["compose_top"] is not None: return 5
    if s["stress_regime"] is not None: return 4
    if s["fund_cur"] is not None: return 3
    if s["batch_cur"] is not None: return 2
    if s["universe"] is not None: return 1
    if s["started_at"] is not None: return 0
    return -1


def _load_run_status() -> dict:
    """Last run's outcome as recorded by run_screener.sh. This is the only run
    signal that survives to the cloud deploy, where logs/ is gitignored."""
    p = Path("run_status.json")
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def _run_history(limit: int = 10) -> list[dict]:
    """Chronological summary of the most recent screener runs (cron + manual)."""
    logs_dir = Path("logs")
    if not logs_dir.exists():
        # Cloud deploy: no logs to parse, fall back to the committed status file.
        rs = _load_run_status()
        if not rs:
            return []
        secs = rs.get("duration_secs") or 0
        return [{
            "date": rs.get("date", "—"),
            "status": "COMPLETE" if rs.get("result") == "success" else "FAILED",
            "duration": f"{secs//3600:02d}:{(secs%3600)//60:02d}:{secs%60:02d}",
            "selected": rs.get("selected"),
        }]
    dates: set[str] = set()
    for f in logs_dir.glob("run_*.log"):
        m = re.match(r"run_(\d{4}-\d{2}-\d{2})\.log$", f.name)
        if m: dates.add(m.group(1))
    for f in logs_dir.glob("manual_run_*.log"):
        m = re.match(r"manual_run_(\d{4}-\d{2}-\d{2})_", f.name)
        if m: dates.add(m.group(1))

    history: list[dict] = []
    for d in sorted(dates, reverse=True)[:limit]:
        cron = logs_dir / f"run_{d}.log"
        if cron.exists():
            log_file = cron
        else:
            manual = sorted(logs_dir.glob(f"manual_run_{d}_*.log"), reverse=True)
            log_file = manual[0] if manual else None
        if log_file is None:
            continue

        text = _last_run_text(log_file.read_text(errors="replace"))
        s = _parse_run_state(text)
        if s["finished_at"]:
            status = "COMPLETE"
        elif s["started_at"]:
            status = "INCOMPLETE"
        else:
            status = "NO DATA"

        duration = "—"
        if s["started_at"] and s["finished_at"]:
            try:
                def _pt(raw: str) -> datetime:
                    cleaned = re.sub(r" [A-Z]{2,4} ", " ", raw)
                    return datetime.strptime(cleaned, "%a %b %d %H:%M:%S %Y")
                secs = int((_pt(s["finished_at"]) - _pt(s["started_at"])).total_seconds())
                duration = f"{secs//3600:02d}:{(secs%3600)//60:02d}:{secs%60:02d}"
            except Exception:
                duration = "—"

        history.append({
            "date": d, "status": status, "duration": duration,
            "selected": s["compose_top"],
        })
    return history


def _render_fidelity_and_history_status() -> None:
    """Always-visible Fidelity sync fallback + run history — independent of whether
    today's main screener run has happened, so a quiet day still surfaces state."""
    today_str = date.today().isoformat()

    # ── Fidelity sync status ──────────────────────────────────────────────────
    st.markdown(
        '<div style="font-family:var(--mono);font-size:0.65rem;font-weight:700;'
        'letter-spacing:0.12em;color:var(--muted);margin-bottom:0.6rem">FIDELITY SYNC</div>',
        unsafe_allow_html=True,
    )
    _fid_status_path = Path("logs/fidelity_sync_status.json")
    _fid: dict = {}
    if _fid_status_path.exists():
        try:
            _fid = json.loads(_fid_status_path.read_text())
        except Exception:
            _fid = {}

    _fid_date   = _fid.get("date")
    _fid_result = _fid.get("result")
    _fid_msg    = str(_fid.get("message") or "")
    _now_hour   = datetime.now().hour

    if not Path("logs").exists():
        # Cloud deploy: logs/ is gitignored, so absence proves nothing. Say that
        # instead of raising a false "hasn't launched" alarm.
        fid_html = '<span style="color:var(--muted)">— LOCAL ONLY (no log access on cloud)</span>'
    elif _fid_date == today_str and _fid_result in ("success", "no_change"):
        fid_html = (f'<span style="color:var(--bull);font-weight:700">✓ SYNCED</span> '
                    f'<span style="color:var(--dim);font-size:0.7rem">{_html.escape(_fid_msg)}</span>')
    elif _fid_date == today_str and _fid_result == "attempted":
        fid_html = ('<span style="color:var(--accent);font-weight:700;'
                    'animation:pulse 1.4s ease-in-out infinite">● WAITING FOR LOGIN</span>')
    elif _fid_date == today_str:
        fid_html = (f'<span style="color:var(--bear);font-weight:700">⚠ '
                    f'{_html.escape(str(_fid_result or "UNKNOWN").upper())}</span> '
                    f'<span style="color:var(--dim);font-size:0.7rem">{_html.escape(_fid_msg)}</span>')
    elif _now_hour < 9:
        fid_html = (f'<span style="color:var(--muted)">— NOT RUN YET TODAY</span> '
                    f'<span style="color:var(--dim);font-size:0.7rem">(last: {_html.escape(str(_fid_date or "never"))})</span>')
    else:
        fid_html = (f'<span style="color:var(--bear);font-weight:700">⚠ HASN&#39;T LAUNCHED TODAY</span> '
                    f'<span style="color:var(--dim);font-size:0.7rem">(last: {_html.escape(str(_fid_date or "never"))})</span>')

    st.markdown(
        f'<div style="font-family:var(--mono);font-size:0.8rem;padding:0.6rem 1rem;'
        f'background:var(--surface-2);border:1px solid var(--border);border-radius:var(--radius);'
        f'margin-bottom:1.2rem">{fid_html}</div>',
        unsafe_allow_html=True,
    )

    # ── Run history ────────────────────────────────────────────────────────────
    st.markdown(
        '<div style="font-family:var(--mono);font-size:0.65rem;font-weight:700;'
        'letter-spacing:0.12em;color:var(--muted);margin-bottom:0.6rem">RUN HISTORY</div>',
        unsafe_allow_html=True,
    )
    history = _run_history(10)
    if not history:
        st.markdown(
            '<div style="font-family:var(--mono);font-size:0.7rem;color:var(--muted);margin-bottom:1.2rem">'
            'No run history found.</div>',
            unsafe_allow_html=True,
        )
    else:
        _hist_color = {"COMPLETE": "var(--bull)", "INCOMPLETE": "var(--accent)",
                       "FAILED": "var(--bear)", "NO DATA": "var(--muted)"}
        hist_rows = []
        for h in history:
            color = _hist_color.get(h["status"], "var(--muted)")
            selected = h["selected"] if h["selected"] is not None else "—"
            hist_rows.append(
                f'<tr><td style="font-family:var(--mono);font-size:0.7rem;color:var(--text)">{h["date"]}</td>'
                f'<td style="font-family:var(--mono);font-size:0.7rem;color:{color};font-weight:600">{h["status"]}</td>'
                f'<td style="font-family:var(--mono);font-size:0.7rem;color:var(--muted)">{h["duration"]}</td>'
                f'<td style="font-family:var(--mono);font-size:0.7rem;color:var(--muted)">{selected}</td></tr>'
            )
        st.markdown(
            f'<div style="overflow-x:auto;border:1px solid var(--border);border-radius:var(--radius);overflow:hidden;'
            f'margin-bottom:1.2rem">'
            f'<table class="q-table"><thead><tr><th>DATE</th><th>STATUS</th><th>DURATION</th><th>SELECTED</th></tr></thead>'
            f'<tbody>{"".join(hist_rows)}</tbody></table></div>',
            unsafe_allow_html=True,
        )


@st.fragment(run_every="5s")
def _render_monitor():
    log_path = _find_todays_log()
    lock_active = Path("/tmp/screener_run.lock").exists()

    st.markdown(
        '<div style="font-family:var(--mono);font-size:0.65rem;font-weight:700;'
        'letter-spacing:0.12em;color:var(--muted);margin-bottom:1.2rem">RUN MONITOR</div>',
        unsafe_allow_html=True,
    )

    _render_fidelity_and_history_status()

    if log_path is None:
        # No live log (quiet day, or the cloud deploy where logs/ is gitignored).
        # run_status.json is the fallback signal, and the only way a failed run
        # ever shows up here rather than silently reading as "no run".
        _rs = _load_run_status()
        if _rs.get("result") == "failed":
            headline = (
                f'<div style="font-family:var(--mono);font-size:1rem;color:var(--bear);font-weight:700">'
                f'⚠ LAST RUN FAILED</div>'
                f'<div style="font-family:var(--mono);font-size:0.7rem;color:var(--dim);margin-top:0.5rem">'
                f'{_html.escape(str(_rs.get("date","—")))} · exit {_html.escape(str(_rs.get("exit_code","?")))} · '
                f'{_html.escape(str(_rs.get("error") or "no error line captured"))}</div>'
            )
        elif _rs.get("result") == "success":
            headline = (
                f'<div style="font-family:var(--mono);font-size:1rem;color:var(--bull);font-weight:700">'
                f'✓ LAST RUN OK</div>'
                f'<div style="font-family:var(--mono);font-size:0.7rem;color:var(--dim);margin-top:0.5rem">'
                f'{_html.escape(str(_rs.get("date","—")))} · '
                f'{_html.escape(str(_rs.get("selected") or "—"))} selected · '
                f'{_html.escape(str(_rs.get("duration_secs","?")))}s</div>'
            )
        else:
            headline = ('<div style="font-family:var(--mono);font-size:1rem;color:var(--muted)">'
                        '— NO RUN TODAY</div>')
        st.markdown(headline, unsafe_allow_html=True)
        _out = sorted(Path("output").glob("screen_*.csv"), reverse=True)
        if _out:
            st.markdown(
                f'<div style="font-family:var(--mono);font-size:0.7rem;color:var(--dim);margin-top:0.5rem">'
                f'Last run: {_out[0].stem.replace("screen_","")}</div>',
                unsafe_allow_html=True,
            )
        return

    text = log_path.read_text(errors="replace")
    text = _last_run_text(text)
    s = _parse_run_state(text)
    stage = _current_stage(s)
    done = s["finished_at"] is not None

    # ── Status bar ────────────────────────────────────────────────────────────
    if done:
        status_html = '<span style="color:var(--bull);font-weight:700">✓ COMPLETE</span>'
    elif lock_active or s["started_at"]:
        status_html = (
            '<span style="color:var(--accent);font-weight:700;animation:pulse 1.4s ease-in-out infinite">'
            '● LIVE</span>'
        )
    else:
        status_html = '<span style="color:var(--muted)">— STALE LOG</span>'

    elapsed_str = "—"
    if s["started_at"]:
        try:
            def _parse_ts(raw: str) -> datetime:
                cleaned = re.sub(r" [A-Z]{2,4} ", " ", raw)
                return datetime.strptime(cleaned, "%a %b %d %H:%M:%S %Y")
            t0 = _parse_ts(s["started_at"])
            t1 = _parse_ts(s["finished_at"]) if s["finished_at"] else datetime.now()
            secs = int((t1 - t0).total_seconds())
            elapsed_str = f"{secs//3600:02d}:{(secs%3600)//60:02d}:{secs%60:02d}"
        except Exception:
            elapsed_str = "—"

    st.markdown(
        f'<div style="display:flex;justify-content:space-between;align-items:center;'
        f'font-family:var(--mono);font-size:0.85rem;padding:0.75rem 1rem;'
        f'background:var(--surface-2);border:1px solid var(--border);'
        f'border-radius:var(--radius);margin-bottom:1.2rem">'
        f'<span>{status_html}</span>'
        f'<span style="color:var(--muted);font-size:0.75rem">ELAPSED {elapsed_str}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Stage pipeline ────────────────────────────────────────────────────────
    _STAGES = [
        (0, "INIT"),
        (1, "UNIVERSE"),
        (2, "PRICES"),
        (3, "FUNDS"),
        (4, "STRESS"),
        (5, "SCORING"),
        (6, "NEWS"),
        (7, "OUTPUT"),
    ]

    cells = []
    for idx, (sid, label) in enumerate(_STAGES):
        if stage > sid:
            color = "var(--bull)"
            icon = "✓"
            weight = "600"
        elif stage == sid:
            color = "var(--accent)"
            icon = "▶"
            weight = "700"
        else:
            color = "var(--muted)"
            icon = "○"
            weight = "400"
        arrow = '<span style="color:var(--border);margin:0 4px">›</span>' if idx < len(_STAGES) - 1 else ""
        cells.append(
            f'<span style="font-family:var(--mono);font-size:0.65rem;font-weight:{weight};'
            f'color:{color};white-space:nowrap">{icon} {label}</span>{arrow}'
        )

    st.markdown(
        f'<div style="display:flex;flex-wrap:wrap;align-items:center;gap:2px;'
        f'padding:0.75rem 1rem;background:var(--surface-1);border:1px solid var(--border);'
        f'border-radius:var(--radius);margin-bottom:1.2rem">'
        f'{"".join(cells)}</div>',
        unsafe_allow_html=True,
    )

    # ── Within-stage progress ─────────────────────────────────────────────────
    prog_val = 1.0
    prog_label = ""
    if stage == 2 and s["batch_cur"] and s["batch_tot"]:
        prog_val = s["batch_cur"] / s["batch_tot"]
        prog_label = f'Batch {s["batch_cur"]} / {s["batch_tot"]}  ·  {s["universe"] or 9619:,} tickers'
    elif stage == 3 and s["fund_cur"] and s["fund_tot"]:
        prog_val = s["fund_cur"] / s["fund_tot"]
        prog_label = f'Ticker {s["fund_cur"]:,} / {s["fund_tot"]:,}'
    elif stage == 6 and s["news_cur"] and s["news_tot"]:
        prog_val = s["news_cur"] / s["news_tot"]
        prog_label = f'{s["news_cur"]} / {s["news_tot"]} stocks analyzed'
    elif stage == 0:
        prog_val = 0.0
        prog_label = "Starting…"
    elif stage == -1:
        prog_val = 0.0
        prog_label = ""

    if stage >= 0:
        st.progress(min(max(prog_val, 0.0), 1.0))
        if prog_label:
            st.markdown(
                f'<div style="font-family:var(--mono);font-size:0.65rem;color:var(--muted);'
                f'margin-top:-0.4rem;margin-bottom:0.8rem">{prog_label}</div>',
                unsafe_allow_html=True,
            )

    # ── Funnel panel ─────────────────────────────────────────────────────────
    def _fv(v): return f"{v:,}" if v is not None else "—"
    universe_n  = _fv(s["universe"])
    liquidity_n = _fv(s["liquidity_survivors"])
    confirm_n   = _fv(s["confirm_out"])
    selected_n  = _fv(s["compose_top"])

    col1, col2, col3, col4 = st.columns(4)
    _funnel_style = (
        "font-family:var(--mono);text-align:center;padding:0.75rem 0.5rem;"
        "background:var(--surface-2);border:1px solid var(--border);border-radius:var(--radius)"
    )
    for col, label, val in [
        (col1, "UNIVERSE",    universe_n),
        (col2, "LIQUIDITY",   liquidity_n),
        (col3, "CONFIRM GATE", confirm_n),
        (col4, "SELECTED",    selected_n),
    ]:
        col.markdown(
            f'<div style="{_funnel_style}">'
            f'<div style="font-size:0.55rem;color:var(--muted);letter-spacing:0.1em;margin-bottom:0.3rem">{label}</div>'
            f'<div style="font-size:1.2rem;font-weight:700;color:var(--text)">{val}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Stress regime badge ───────────────────────────────────────────────────
    if s["stress_regime"]:
        regime_color = {"NORMAL": "var(--bull)", "WARNING": "var(--accent)", "STRESS": "var(--bear)"}.get(
            s["stress_regime"], "var(--muted)"
        )
        st.markdown(
            f'<div style="font-family:var(--mono);font-size:0.65rem;margin-top:0.8rem;'
            f'color:var(--muted)">MARKET REGIME  '
            f'<span style="color:{regime_color};font-weight:700">{s["stress_regime"]}</span></div>',
            unsafe_allow_html=True,
        )

    # ── Recent log lines ──────────────────────────────────────────────────────
    st.markdown('<div style="height:1.2rem"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-family:var(--mono);font-size:0.6rem;font-weight:700;'
        'letter-spacing:0.1em;color:var(--muted);margin-bottom:0.4rem">LOG TAIL</div>',
        unsafe_allow_html=True,
    )
    lines = [l for l in text.splitlines() if l.strip()][-10:]
    colored_lines = []
    for ln in lines:
        if " ERROR " in ln or " WARNING " in ln:
            c = "var(--bear)"
        elif " INFO " in ln:
            c = "var(--text)"
        else:
            c = "var(--accent)"
        escaped = _html.escape(ln)
        colored_lines.append(f'<span style="color:{c}">{escaped}</span>')

    st.markdown(
        f'<style>@keyframes log-refresh{{0%{{opacity:0.3;filter:grayscale(1)}}100%{{opacity:1;filter:none}}}}</style>'
        f'<pre style="font-family:var(--mono);font-size:0.6rem;line-height:1.6;'
        f'background:var(--surface-2);border:1px solid var(--border);border-radius:var(--radius);'
        f'padding:0.8rem;overflow-x:auto;white-space:pre-wrap;word-break:break-all;margin:0;'
        f'animation:log-refresh 0.6s ease-out">'
        f'{"<br>".join(colored_lines)}</pre>',
        unsafe_allow_html=True,
    )
