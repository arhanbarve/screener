"""Render the daily / weekly trading journal into a visual HTML email (a
"trading desk" briefing) and send it via Gmail SMTP.

Env vars (in .env):
  GMAIL_ADDRESS        sender Gmail address
  GMAIL_APP_PASSWORD   Gmail app password
  TRADER_EMAIL_TO      recipient; required to send

If credentials are absent, send_email() is a no-op that reports why, so the
trading pipeline never crashes just because email isn't configured.

CLI:
  python -m src.notify daily  [--file PATH] [--date YYYY-MM-DD] [--preview-only]
  python -m src.notify weekly [--file PATH] [--week YYYY-Www] [--preview-only]
"""
import argparse
import os
import smtplib
import sys
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path
from zoneinfo import ZoneInfo

import markdown

from src import broker

ET = ZoneInfo("America/New_York")
DEFAULT_TO = ""  # set TRADER_EMAIL_TO in .env; no address is baked in
BASELINE = 100000.0  # experiment starting equity

# Index tape: symbol -> short label
MARKET_SYMS = [("SPY", "S&P 500"), ("QQQ", "Nasdaq"), ("DIA", "Dow"),
               ("IWM", "Russell"), ("^VIX", "VIX")]

# Categorical allocation palette — deliberately avoids green/red, which are
# reserved for P&L semantics elsewhere. Leads with the site accent amber.
ALLOC_COLORS = ["#f59e0b", "#22d3ee", "#a78bfa", "#60a5fa", "#f472b6",
                "#818cf8", "#2dd4bf", "#fbbf24"]
CASH_COLOR = "rgba(100,116,139,0.10)"
CASH_BORDER = "rgba(100,116,139,0.45)"

GAIN = "#22c55e"   # --bull
LOSS = "#ef4444"   # --bear

CSS = """
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');
  :root { color-scheme: dark; }
  body { margin:0; padding:0; background:#020209;
         font-family:'IBM Plex Sans',system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
         color:#e2e8f0; }
  .mono { font-family:'IBM Plex Mono','Courier New',ui-monospace,Menlo,monospace;
          font-variant-numeric:tabular-nums; }
  .wrap { max-width:640px; margin:0 auto; padding:20px 14px 32px; }
  .card { background:#07090f; border:1px solid #161824; border-radius:14px; overflow:hidden; }
  .eyebrow { display:flex; justify-content:space-between; align-items:center;
             padding:14px 22px; border-bottom:1px solid #161824; background:#040408; }
  .brand { font-family:'IBM Plex Mono','Courier New',monospace; font-size:12px; letter-spacing:.2em;
           text-transform:uppercase; color:#f59e0b; font-weight:700; }
  .brand .dot { color:#22c55e; }
  .date { font-size:12px; letter-spacing:.06em; color:#64748b; }
  .hero { padding:26px 22px 22px; }
  .hero .k { font-size:11px; letter-spacing:.14em; text-transform:uppercase; color:#64748b; margin:0 0 6px; }
  .hero .equity { font-family:'IBM Plex Mono','Courier New',monospace; font-size:44px; font-weight:700;
                  line-height:1; margin:0; letter-spacing:-.02em; }
  .chips { margin:16px 0 0; }
  .chip { display:inline-block; padding:5px 11px; border-radius:6px; font-size:13px; font-weight:600;
          margin-right:8px; }
  .chip.pos { background:rgba(34,197,94,.10); color:#22c55e; border:1px solid rgba(34,197,94,.25); }
  .chip.neg { background:rgba(239,68,68,.10); color:#ef4444; border:1px solid rgba(239,68,68,.25); }
  .chip.flat{ background:rgba(100,116,139,.12); color:#94a3b8; border:1px solid rgba(100,116,139,.25); }
  .sec { padding:18px 22px 4px; border-top:1px solid #161824; }
  .sec h2 { font-family:'IBM Plex Mono','Courier New',monospace; font-size:11px; letter-spacing:.14em;
            text-transform:uppercase; color:#64748b; margin:0 0 14px; font-weight:600; }
  .sec h2 span { color:#2a3a54; font-weight:400; text-transform:none; letter-spacing:0; }
  .tape td { padding:2px 4px; }
  .tk { border:1px solid #161824; border-radius:6px; background:#0c0e1a; padding:9px 8px; text-align:center; }
  .tk .n { font-size:10px; letter-spacing:.07em; text-transform:uppercase; color:#64748b; margin:0 0 3px; }
  .tk .p { font-size:14px; font-weight:700; margin:0; }
  .bars td { padding:5px 0; font-size:12px; color:#94a3b8; }
  .bars .lab { width:64px; letter-spacing:.06em; text-transform:uppercase; color:#64748b; }
  .bars .val { width:72px; text-align:right; font-weight:700; }
  .track { background:#0c0e1a; border:1px solid #161824; border-radius:4px; height:16px; }
  .fill { height:16px; border-radius:3px; }
  .alloc { border-radius:6px; overflow:hidden; border:1px solid #161824; }
  .alloc td { height:30px; text-align:center; font-family:'IBM Plex Mono','Courier New',monospace;
              font-size:10px; font-weight:700; color:#020209; letter-spacing:.04em; overflow:hidden; }
  .alloc-key { margin:10px 0 0; font-size:11px; color:#64748b; }
  .alloc-key span { display:inline-block; margin:0 12px 4px 0; }
  .alloc-key i { display:inline-block; width:9px; height:9px; border-radius:2px; margin-right:5px;
                 vertical-align:baseline; }
  table.pos { width:100%; border-collapse:collapse; }
  table.pos td { padding:9px 6px; border-bottom:1px solid #111420; font-size:13px; }
  table.pos .sym { font-family:'IBM Plex Mono','Courier New',monospace; font-weight:700; color:#e2e8f0; }
  table.pos .sz { color:#64748b; font-size:11px; }
  table.pos .num { text-align:right; font-weight:700; }
  .pos { color:#22c55e; }
  .neg { color:#ef4444; }
  .dv { width:120px; }
  .dv table { width:100%; border-collapse:collapse; }
  .dv .half { width:50%; height:12px; }
  .dv .lft { text-align:right; }
  .dv .center { border-left:1px solid #242c42; }
  .empty { padding:16px 0; color:#64748b; font-style:italic; }
  .notes { padding:6px 22px 20px; border-top:1px solid #161824; }
  .notes .body { font-size:14px; line-height:1.62; color:#b4c0d4; }
  .notes .body h2 { font-family:'IBM Plex Mono','Courier New',monospace; font-size:11px; letter-spacing:.14em;
                    text-transform:uppercase; color:#64748b; margin:18px 0 8px; font-weight:600; }
  .notes .body h3 { font-family:'IBM Plex Mono','Courier New',monospace; font-size:14px; color:#e2e8f0; margin:14px 0 4px; }
  .notes .body strong { color:#e2e8f0; }
  .notes .body ul { margin:6px 0; padding-left:18px; }
  .notes .body li { margin:3px 0; }
  .notes .body table { width:100%; border-collapse:collapse; font-size:12px; margin:8px 0; }
  .notes .body th { text-align:left; background:#0c0e1a; padding:6px 8px; border:1px solid #161824; color:#94a3b8; }
  .notes .body td { padding:6px 8px; border:1px solid #111420; color:#b4c0d4; }
  .foot { text-align:center; padding:16px 8px 0; color:#2a3a54; font-size:11px; line-height:1.6; }
"""


def _f(x, default=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def fmt_money(x):
    return f"${_f(x):,.2f}"


def fmt_k(x):
    v = _f(x)
    return f"${v/1000:.1f}k" if abs(v) >= 1000 else f"${v:,.0f}"


def fmt_signed(x):
    v = _f(x)
    return f"{'+' if v >= 0 else '−'}${abs(v):,.2f}"


def fmt_pct(x):
    v = _f(x)
    return f"{'+' if v >= 0 else '−'}{abs(v):.2f}%"


def _cls(v):
    v = _f(v)
    return "pos" if v > 0 else ("neg" if v < 0 else "flat")


def _arrow(v):
    v = _f(v)
    return "▲" if v > 0 else ("▼" if v < 0 else "■")


def fetch_market():
    """Return [{sym,label,price,pct}] for the index tape. Resilient: [] on any failure."""
    try:
        import yfinance as yf
        tk = yf.Tickers(" ".join(s for s, _ in MARKET_SYMS))
        out = []
        for sym, label in MARKET_SYMS:
            try:
                fi = tk.tickers[sym].fast_info
                last, prev = float(fi.last_price), float(fi.previous_close)
                out.append({"sym": sym, "label": label, "price": last,
                            "pct": (last / prev - 1) * 100 if prev else 0.0})
            except Exception:
                continue
        return out
    except Exception:
        return []


def render_tape(market):
    if not market:
        return ""
    cells = []
    for m in market:
        c = LOSS if m["pct"] < 0 else (GAIN if m["pct"] > 0 else "#9fb1cc")
        price = f"{m['price']:,.2f}" if m["sym"] != "^VIX" else f"{m['price']:.2f}"
        cells.append(
            f'<td class="tk"><p class="n">{m["label"]}</p>'
            f'<p class="p mono" style="color:{c}">{_arrow(m["pct"])} {abs(m["pct"]):.2f}%</p>'
            f'<p class="n mono" style="color:#64748b;margin-top:2px">{price}</p></td>'
        )
    return (f'<div class="sec"><h2>Market tape <span>· today</span></h2>'
            f'<table class="tape" width="100%"><tr>{"".join(cells)}</tr></table></div>')


def _hbar(pct, color, track="#0c0e1a"):
    w = max(0, min(100, abs(pct) * (100 / _hbar.scale) if _hbar.scale else 0))
    return (f'<div class="track" style="background:{track}">'
            f'<div class="fill" style="width:{w:.0f}%;background:{color}"></div></div>')
_hbar.scale = 1.0


def render_vs_market(port_pct, spy_pct):
    scale = max(abs(port_pct), abs(spy_pct), 0.5)
    _hbar.scale = scale
    rows = [
        ("You", port_pct, "#f59e0b"),
        ("S&P 500", spy_pct, GAIN if spy_pct >= 0 else LOSS),
    ]
    trs = []
    for lab, pct, col in rows:
        trs.append(
            f'<tr><td class="lab">{lab}</td>'
            f'<td>{_hbar(pct, col)}</td>'
            f'<td class="val mono {_cls(pct)}">{fmt_pct(pct)}</td></tr>'
        )
    edge = port_pct - spy_pct
    return (f'<div class="sec"><h2>You vs S&amp;P 500 <span>· day</span></h2>'
            f'<table class="bars" width="100%">{"".join(trs)}</table>'
            f'<p style="margin:8px 0 0;font-size:12px;color:#64748b">'
            f'Relative: <span class="mono {_cls(edge)}">{fmt_pct(edge)}</span> vs benchmark today</p></div>')


def render_alloc(positions, equity, cash):
    equity = equity or 1.0
    segs = []
    keys = []
    for i, p in enumerate(positions):
        mv = _f(p.get("market_value"))
        w = mv / equity * 100
        col = ALLOC_COLORS[i % len(ALLOC_COLORS)]
        label = p.get("symbol", "") if w >= 9 else ""
        segs.append(f'<td style="width:{w:.2f}%;background:{col}">{label}</td>')
        keys.append(f'<span><i style="background:{col}"></i>{p.get("symbol","")} '
                    f'{w:.0f}%</span>')
    cw = max(0.0, cash / equity * 100)
    if cw > 0.1:
        segs.append(f'<td style="width:{cw:.2f}%;background:{CASH_COLOR};'
                    f'border-left:1px solid {CASH_BORDER};color:#64748b">'
                    f'{"CASH" if cw >= 9 else ""}</td>')
        keys.append(f'<span><i style="background:{CASH_BORDER}"></i>Cash {cw:.0f}%</span>')
    return (f'<div class="sec"><h2>Capital <span>· {fmt_k(equity-cash)} deployed / '
            f'{fmt_k(cash)} cash</span></h2>'
            f'<table class="alloc" width="100%"><tr>{"".join(segs)}</tr></table>'
            f'<p class="alloc-key">{"".join(keys)}</p></div>')


def _diverging(pl, maxabs):
    frac = min(1.0, abs(pl) / maxabs) if maxabs else 0.0
    w = frac * 100
    if pl >= 0:
        left = '<td class="half lft"></td>'
        right = f'<td class="half center"><div style="width:{w:.0f}%;height:12px;background:{GAIN};border-radius:0 4px 4px 0"></div></td>'
    else:
        left = f'<td class="half lft"><div style="width:{w:.0f}%;height:12px;background:{LOSS};border-radius:4px 0 0 4px;margin-left:auto"></div></td>'
        right = '<td class="half center"></td>'
    return f'<div class="dv"><table><tr>{left}{right}</tr></table></div>'


def render_positions_table(positions):
    if not positions:
        return '<div class="sec"><h2>Positions</h2><div class="empty">No open positions — fully in cash.</div></div>'
    maxabs = max((abs(_f(p.get("unrealized_pl"))) for p in positions), default=1.0) or 1.0
    rows = []
    for p in positions:
        pl = _f(p.get("unrealized_pl"))
        plpc = _f(p.get("unrealized_plpc")) * 100
        cls = _cls(pl)
        rows.append(
            f'<tr>'
            f'<td><span class="sym">{p.get("symbol","")}</span><br>'
            f'<span class="sz mono">{fmt_k(p.get("market_value"))}</span></td>'
            f'<td class="dv">{_diverging(pl, maxabs)}</td>'
            f'<td class="num mono {cls}">{fmt_signed(pl)}<br>'
            f'<span style="font-weight:600;font-size:11px">{fmt_pct(plpc)}</span></td>'
            f'</tr>'
        )
    return (f'<div class="sec"><h2>Positions <span>· unrealized P&amp;L</span></h2>'
            f'<table class="pos">{"".join(rows)}</table></div>')


def extract_sections(md_text, keep=None):
    """Return markdown containing only the ## sections whose header is in `keep`
    (case-insensitive substring match). keep=None returns the text unchanged."""
    if keep is None:
        return md_text
    keep_l = [k.lower() for k in keep]
    lines = md_text.splitlines()
    out, take = [], False
    for ln in lines:
        if ln.startswith("## "):
            head = ln[3:].strip().lower()
            take = any(k in head for k in keep_l)
        if take:
            out.append(ln)
    return "\n".join(out).strip()


def md_to_html(md_text):
    return markdown.markdown(md_text, extensions=["tables", "sane_lists"])


def build_email(kind, period, account, positions, body_md, market=None):
    equity = _f(account.get("equity"))
    last_equity = _f(account.get("last_equity"), equity)
    cash = _f(account.get("cash"))
    day_pl = equity - last_equity
    day_pct = (day_pl / last_equity * 100) if last_equity else 0.0
    total_pct = (equity - BASELINE) / BASELINE * 100

    if kind == "weekly":
        subject = f"🗓️ Weekly Paper Trading Digest — {period} — {fmt_money(equity)} ({fmt_pct(total_pct)} total)"
        eyebrow_date = f"Weekly digest · {period}"
        notes_md = body_md  # weekly file is already a curated digest
    else:
        subject = f"📈 Paper Trading — {period} — {fmt_money(equity)} ({fmt_pct(day_pct)})"
        eyebrow_date = period
        notes_md = extract_sections(body_md, keep=["market context", "decision"])

    hero = f"""
    <div class="hero">
      <p class="k">Account equity</p>
      <p class="equity mono">{fmt_money(equity)}</p>
      <div class="chips">
        <span class="chip {_cls(day_pl)}">{_arrow(day_pl)} {fmt_signed(day_pl)} today · {fmt_pct(day_pct)}</span>
        <span class="chip {_cls(total_pct)}">{fmt_pct(total_pct)} since start</span>
      </div>
    </div>"""

    tape = render_tape(market or [])
    spy = next((m["pct"] for m in (market or []) if m["sym"] == "SPY"), None)
    vs = render_vs_market(day_pct, spy) if spy is not None else ""
    alloc = render_alloc(positions, equity, cash)
    pos = render_positions_table(positions)
    notes = (f'<div class="notes"><div class="body">{md_to_html(notes_md)}</div></div>'
             if notes_md.strip() else "")

    html = f"""<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="dark">
<style>{CSS}</style></head>
<body><div class="wrap"><div class="card">
  <div class="eyebrow"><span class="brand"><span class="dot">●</span> Paper Desk</span>
    <span class="date mono">{eyebrow_date}</span></div>
  {hero}
  {tape}
  {vs}
  {alloc}
  {pos}
  {notes}
</div>
<div class="foot">Alpaca paper account · autonomous screener strategy<br>
Equity benchmarked to $100,000 baseline. Not investment advice.</div>
</div></body></html>"""
    return subject, html


def send_email(subject, html_body, to_addr=None):
    sender = os.environ.get("GMAIL_ADDRESS")
    password = os.environ.get("GMAIL_APP_PASSWORD")
    to_addr = to_addr or os.environ.get("TRADER_EMAIL_TO", DEFAULT_TO)
    if not sender or not password:
        return {"sent": False, "reason": "GMAIL_ADDRESS / GMAIL_APP_PASSWORD not set"}
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = to_addr
    msg.set_content("This is an HTML email — view it in an HTML-capable client.")
    msg.add_alternative(html_body, subtype="html")
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender, password)
        server.send_message(msg)
    return {"sent": True, "to": to_addr}


def _preview_path(kind, period):
    Path("logs").mkdir(exist_ok=True)
    return Path("logs") / f"email_preview_{kind}_{period}.html"


def main(argv=None):
    p = argparse.ArgumentParser(prog="notify")
    sub = p.add_subparsers(dest="kind", required=True)
    dp = sub.add_parser("daily")
    dp.add_argument("--file")
    dp.add_argument("--date")
    dp.add_argument("--preview-only", action="store_true")
    wp = sub.add_parser("weekly")
    wp.add_argument("--file")
    wp.add_argument("--week")
    wp.add_argument("--preview-only", action="store_true")
    args = p.parse_args(argv)

    now = datetime.now(ET)
    if args.kind == "daily":
        period = args.date or now.date().isoformat()
        path = Path(args.file) if args.file else Path(f"trading/journal/{period}.md")
    else:
        period = args.week or f"{now.isocalendar().year}-W{now.isocalendar().week:02d}"
        path = Path(args.file) if args.file else Path(f"trading/journal/weekly/{period}.md")

    if not path.exists():
        print(f"journal not found: {path}", file=sys.stderr)
        return 2
    body_md = path.read_text()

    try:
        account = broker.get_account()
        positions = broker.get_positions()
    except broker.BrokerError as e:
        print(f"broker error: {e}", file=sys.stderr)
        return 1

    market = fetch_market()
    subject, html = build_email(args.kind, period, account, positions, body_md, market)
    preview = _preview_path(args.kind, period)
    preview.write_text(html)

    if args.preview_only:
        print(f"preview written: {preview}")
        return 0

    result = send_email(subject, html)
    print(f"preview: {preview} | send: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
