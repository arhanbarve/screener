"""Render the daily / weekly trading journal into a clean HTML email and send
it via Gmail SMTP.

Env vars (in .env):
  GMAIL_ADDRESS        sender Gmail address
  GMAIL_APP_PASSWORD   Gmail app password (Google account -> Security -> App passwords)
  TRADER_EMAIL_TO      recipient (default arhanbarve@gmail.com)

If credentials are absent, send_email() is a no-op that reports why — so the
trading pipeline never crashes just because email isn't configured yet.

CLI:
  python -m src.notify daily  [--file PATH] [--date YYYY-MM-DD]
  python -m src.notify weekly [--file PATH] [--week YYYY-Www]
Both also write a preview to logs/email_preview_*.html for inspection.
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
DEFAULT_TO = "arhanbarve@gmail.com"
BASELINE = 100000.0  # experiment starting equity

CSS = """
  :root { color-scheme: light; }
  body { margin:0; padding:0; background:#eef1f5;
         font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;
         color:#1a2233; }
  .wrap { max-width:640px; margin:0 auto; padding:24px 16px; }
  .card { background:#ffffff; border-radius:14px; overflow:hidden;
          box-shadow:0 1px 3px rgba(16,24,40,.08),0 1px 2px rgba(16,24,40,.06); }
  .hero { padding:26px 28px 22px; background:linear-gradient(135deg,#0f172a,#1e293b); color:#fff; }
  .hero .label { font-size:12px; letter-spacing:.08em; text-transform:uppercase;
                 color:#94a3b8; margin:0 0 6px; }
  .hero .equity { font-size:38px; font-weight:700; line-height:1; margin:0; }
  .hero .sub { margin:10px 0 0; font-size:15px; color:#cbd5e1; }
  .chip { display:inline-block; padding:3px 10px; border-radius:999px; font-weight:600;
          font-size:14px; }
  .chip.pos { background:rgba(34,197,94,.18); color:#4ade80; }
  .chip.neg { background:rgba(239,68,68,.18); color:#f87171; }
  .chip.flat{ background:rgba(148,163,184,.18); color:#cbd5e1; }
  .metrics { display:flex; border-top:1px solid #eef1f5; }
  .metric { flex:1; padding:16px 18px; text-align:center; border-right:1px solid #eef1f5; }
  .metric:last-child { border-right:none; }
  .metric .k { font-size:11px; letter-spacing:.06em; text-transform:uppercase; color:#64748b; margin:0 0 4px; }
  .metric .v { font-size:18px; font-weight:700; margin:0; color:#0f172a; }
  .section { padding:8px 28px 4px; }
  .section h2 { font-size:13px; letter-spacing:.06em; text-transform:uppercase;
                color:#64748b; margin:22px 0 10px; }
  table.pos { width:100%; border-collapse:collapse; font-size:14px; }
  table.pos th { text-align:right; font-size:11px; text-transform:uppercase; letter-spacing:.04em;
                 color:#94a3b8; padding:8px 10px; border-bottom:1px solid #e5e9f0; }
  table.pos th.l, table.pos td.l { text-align:left; }
  table.pos td { padding:10px; border-bottom:1px solid #f1f4f8; }
  table.pos td.tk { font-weight:700; color:#0f172a; }
  .pos { color:#0a7d33; font-weight:600; }
  .neg { color:#c0392b; font-weight:600; }
  .empty { padding:18px 0; color:#64748b; font-style:italic; }
  .body { padding:4px 28px 24px; font-size:15px; line-height:1.6; color:#334155; }
  .body h1 { font-size:20px; color:#0f172a; margin:18px 0 8px; }
  .body h2 { font-size:13px; letter-spacing:.06em; text-transform:uppercase; color:#64748b;
             border-top:1px solid #eef1f5; padding-top:18px; margin:22px 0 10px; }
  .body h3 { font-size:15px; color:#0f172a; margin:16px 0 6px; }
  .body table { width:100%; border-collapse:collapse; font-size:13px; margin:10px 0; }
  .body th { text-align:left; background:#f8fafc; padding:7px 9px; border:1px solid #e5e9f0; color:#475569; }
  .body td { padding:7px 9px; border:1px solid #eef1f5; }
  .body strong { color:#0f172a; }
  .body ul { margin:8px 0; padding-left:20px; }
  .foot { text-align:center; padding:18px; color:#94a3b8; font-size:12px; }
  .foot a { color:#64748b; }
"""


def _f(x, default=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def fmt_money(x):
    return f"${_f(x):,.2f}"


def fmt_signed(x):
    v = _f(x)
    return f"{'+' if v >= 0 else '−'}${abs(v):,.2f}"


def fmt_pct(x):
    v = _f(x)
    return f"{'+' if v >= 0 else '−'}{abs(v):.2f}%"


def _cls(v):
    v = _f(v)
    return "pos" if v > 0 else ("neg" if v < 0 else "flat")


def render_positions_table(positions):
    if not positions:
        return '<div class="empty">No open positions — fully in cash.</div>'
    rows = []
    for p in positions:
        pl = _f(p.get("unrealized_pl"))
        plpc = _f(p.get("unrealized_plpc")) * 100
        cls = _cls(pl)
        rows.append(
            f'<tr>'
            f'<td class="l tk">{p.get("symbol","")}</td>'
            f'<td>{_f(p.get("qty")):.2f}</td>'
            f'<td>{fmt_money(p.get("avg_entry_price"))}</td>'
            f'<td>{fmt_money(p.get("current_price"))}</td>'
            f'<td>{fmt_money(p.get("market_value"))}</td>'
            f'<td class="{cls}">{fmt_signed(pl)}</td>'
            f'<td class="{cls}">{fmt_pct(plpc)}</td>'
            f'</tr>'
        )
    return (
        '<table class="pos">'
        '<tr><th class="l">Ticker</th><th>Qty</th><th>Avg</th><th>Last</th>'
        '<th>Value</th><th>Unrl P&amp;L</th><th>%</th></tr>'
        + "".join(rows) + '</table>'
    )


def md_to_html(md_text):
    return markdown.markdown(md_text, extensions=["tables", "sane_lists"])


def build_email(kind, period, account, positions, body_md):
    equity = _f(account.get("equity"))
    last_equity = _f(account.get("last_equity"), equity)
    cash = _f(account.get("cash"))
    day_pl = equity - last_equity
    day_pct = (day_pl / last_equity * 100) if last_equity else 0.0
    total_pct = (equity - BASELINE) / BASELINE * 100
    day_cls = _cls(day_pl)

    if kind == "weekly":
        subject = f"🗓️ Weekly Paper Trading Digest — {period} — {fmt_money(equity)} ({fmt_pct(total_pct)} total)"
        hero_label = f"Weekly digest · {period}"
    else:
        subject = f"📈 Paper Trading — {period} — {fmt_money(equity)} ({fmt_pct(day_pct)})"
        hero_label = f"Daily session · {period}"

    hero = f"""
    <div class="hero">
      <p class="label">{hero_label}</p>
      <p class="equity">{fmt_money(equity)}</p>
      <p class="sub">Today <span class="chip {day_cls}">{fmt_signed(day_pl)} ({fmt_pct(day_pct)})</span>
         &nbsp;·&nbsp; Since start <span class="chip {_cls(total_pct)}">{fmt_pct(total_pct)}</span></p>
    </div>
    <div class="metrics">
      <div class="metric"><p class="k">Equity</p><p class="v">{fmt_money(equity)}</p></div>
      <div class="metric"><p class="k">Cash</p><p class="v">{fmt_money(cash)}</p></div>
      <div class="metric"><p class="k">Invested</p><p class="v">{fmt_money(equity - cash)}</p></div>
      <div class="metric"><p class="k">Positions</p><p class="v">{len(positions)}</p></div>
    </div>"""

    positions_block = (
        f'<div class="section"><h2>Holdings</h2>{render_positions_table(positions)}</div>'
    )
    body_block = f'<div class="body">{md_to_html(body_md)}</div>'

    html = f"""<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>{CSS}</style></head>
<body><div class="wrap"><div class="card">
{hero}
{positions_block}
{body_block}
</div>
<div class="foot">Alpaca paper account · autonomous screener strategy<br>
Benchmark: equity vs $100,000 baseline. Not investment advice.</div>
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

    subject, html = build_email(args.kind, period, account, positions, body_md)
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
