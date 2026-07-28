"""Emails for the standing exit plan engine: action alerts + daily digest.

Builders are pure (subject, html) functions — testable without SMTP.
Sending reuses notify.send_email (Gmail SMTP, env-var credentials).
"""
import html as _html
from datetime import date

from src.notify import send_email

_VERDICT_COLOR = {"SELL": "#ef4444", "TRIM": "#f59e0b", "HOLD": "#22c55e"}
_MIN_HEALTH_WEEKS = 15  # weekly_health() in exit_plan.py treats fewer weeks as "never checked"

_CSS = """
body{background:#0b0d17;color:#e5e7eb;font-family:-apple-system,Segoe UI,sans-serif;margin:0;padding:24px}
.card{max-width:640px;margin:0 auto;background:#11142a;border:1px solid #232748;border-radius:12px;padding:24px}
h1{font-size:18px;margin:0 0 16px}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{text-align:left;padding:8px 10px;border-bottom:1px solid #232748}
th{color:#9ca3af;font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:.06em}
.mono{font-family:ui-monospace,Menlo,monospace}
.badge{display:inline-block;padding:2px 10px;border-radius:999px;font-weight:700;font-size:12px}
.reason{color:#9ca3af;font-size:12px;margin:4px 0 12px}
.instruction{background:#1a1e3a;border-left:3px solid #ef4444;padding:10px 14px;margin:8px 0 20px;font-weight:600}
.foot{color:#6b7280;font-size:11px;margin-top:16px;text-align:center}
.warn{background:#1f1a12;border:1px solid #4a3a1a;border-radius:8px;padding:10px 14px;margin:0 0 16px}
.warn h3{margin:0 0 6px;font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:#f59e0b}
.warn ul{margin:0;padding-left:18px;font-size:12px;color:#d1d5db}
.notevaled{color:#f59e0b;font-weight:700;font-size:11px;letter-spacing:.03em}
"""


def _badge(verdict: str) -> str:
    c = _VERDICT_COLOR.get(verdict, "#9ca3af")
    return (f'<span class="badge" style="color:{c};background:{c}22;'
            f'border:1px solid {c}55">{_html.escape(str(verdict))}</span>')


def _wrap(title: str, body: str) -> str:
    return (f'<!doctype html><html><head><meta charset="utf-8">'
            f'<style>{_CSS}</style></head><body><div class="card">'
            f'<h1>{_html.escape(title)}</h1>{body}'
            f'<div class="foot">Standing exit plan &middot; evaluated on closing prices &middot; '
            f'{date.today().isoformat()} &middot; Not investment advice</div>'
            f'</div></body></html>')


def _health_badge(health: dict | None) -> str:
    """Render the weekly-health bearish score. A check that threw (recorded
    in health["errors"]) must not silently read as clean, and a position with
    too little history for the weekly check (weeks < _MIN_HEALTH_WEEKS) has
    never actually been scored — showing "0/4" for either would misrepresent
    an unknown as a confident all-clear."""
    if not health:
        return '<span class="mono">?/4</span>'
    weeks = health.get("weeks") or 0
    errors = health.get("errors") or []
    if weeks < _MIN_HEALTH_WEEKS:
        return '<span class="mono" title="short history">?/4</span>'
    bearish = health.get("bearish", 0)
    if errors:
        err_str = ",".join(_html.escape(e) for e in errors)
        return (f'<span class="mono" style="color:#f59e0b" '
                f'title="checks errored: {err_str}">{bearish}/4&#42;</span>')
    return f'<span class="mono">{bearish}/4</span>'


def _plan_row(pos: dict, not_evaluated: bool = False, bar_date: str | None = None,
              error: str | None = None) -> str:
    p = pos.get("plan") or {}
    close = p.get("last_close")
    stop = p.get("stop_level")
    dist = (f"{(close - stop) / close:+.1%}" if close and stop else "—")
    trims = ",".join(p.get("trims_fired") or []) or "—"
    hstr = _health_badge(p.get("health"))
    dte = p.get("days_to_earnings")
    earn = f"{int(dte)}d" if isinstance(dte, (int, float)) else "—"
    ticker = _html.escape(str(pos.get("ticker", "?")))
    flag = ""
    if not_evaluated:
        detail = f"bar {_html.escape(bar_date)}" if bar_date else (
            f"error: {_html.escape(error)}" if error else "")
        flag = f'<br><span class="notevaled">NOT EVALUATED TODAY</span>'
        if detail:
            flag += f'<br><span class="reason" style="margin:0">{detail}</span>'
    return (f'<tr><td class="mono"><b>{ticker}</b>{flag}</td>'
            f'<td>{_badge(p.get("verdict", "?"))}</td>'
            f'<td class="mono">{close if close is not None else "—"}</td>'
            f'<td class="mono">{stop if stop is not None else "—"} ({dist})</td>'
            f'<td class="mono">{_html.escape(trims)}</td>'
            f'<td class="mono">{hstr}</td>'
            f'<td class="mono">{earn}</td></tr>')


def build_action_email(events: list[dict], positions: list[dict]) -> tuple[str, str]:
    parts = sorted({f"{e['type']} {e['ticker']}" for e in events})
    subject = f"\U0001f6a8 ACTION: {' · '.join(parts)}"
    by_ticker = {p["ticker"]: p for p in positions}
    body = ""
    for e in events:
        p = (by_ticker.get(e["ticker"], {}).get("plan")) or {}
        body += (f'<div style="margin-bottom:8px">{_badge(e["type"])} '
                 f'<span class="mono" style="font-size:16px;font-weight:700">'
                 f'{_html.escape(e["ticker"])}</span></div>'
                 f'<div class="reason">{_html.escape(e["reason"])}</div>'
                 f'<div class="instruction">{_html.escape(e["instruction"])}</div>')
        if p:
            body += (f'<div class="reason mono">floor {p.get("stop_floor")} &middot; '
                     f'stop {p.get("stop_level")} &middot; peak {p.get("peak_close")} &middot; '
                     f'trims [{_html.escape(",".join(p.get("trims_fired") or []))}]</div>')
    return subject, _wrap("Exit plan action required", body)


def _warn_block(title: str, items: list[str]) -> str:
    lis = "".join(f"<li>{i}</li>" for i in items)
    return f'<div class="warn"><h3>{_html.escape(title)}</h3><ul>{lis}</ul></div>'


def build_digest_email(positions: list[dict], skipped: list[str],
                        stale: list[dict], errored: list[dict]) -> tuple[str, str]:
    stale_tickers = {s["ticker"]: s.get("bar_date") for s in (stale or [])}
    errored_map = {e["ticker"]: e.get("error") for e in (errored or [])}
    not_eval_tickers = set(skipped or []) | set(stale_tickers) | set(errored_map)

    warnings = ""
    if skipped:
        warnings += _warn_block(
            "Data fetch skipped — plan NOT evaluated today",
            [f'<span class="mono">{_html.escape(t)}</span>' for t in skipped])
    if stale:
        warnings += _warn_block(
            "Stale bar — plan NOT evaluated today",
            [f'<span class="mono">{_html.escape(s["ticker"])}</span> '
             f'(last bar {_html.escape(s.get("bar_date", "?"))})' for s in stale])
    if errored:
        warnings += _warn_block(
            "Evaluation error — plan NOT evaluated today",
            [f'<span class="mono">{_html.escape(e["ticker"])}</span>: '
             f'{_html.escape(e.get("error", "?"))}' for e in errored])

    rows = ""
    for pos in positions:
        t = pos.get("ticker")
        if t in stale_tickers:
            rows += _plan_row(pos, not_evaluated=True, bar_date=stale_tickers[t])
        elif t in errored_map:
            rows += _plan_row(pos, not_evaluated=True, error=errored_map[t])
        else:
            rows += _plan_row(pos, not_evaluated=(t in not_eval_tickers))

    body = warnings + (
        '<table><tr><th>Ticker</th><th>Verdict</th><th>Close</th>'
        '<th>Stop (dist)</th><th>Trims</th><th>Health</th><th>Earnings</th></tr>'
        f'{rows}</table>' if positions else '<p class="reason">No open positions.</p>')

    n_flagged = len(not_eval_tickers)
    subject = f"Exit plan digest · {len(positions)} positions"
    if n_flagged:
        subject += f" · {n_flagged} not evaluated"
    return subject, _wrap("Standing exit plan — daily digest", body)


def send_action_alert(events: list[dict], positions: list[dict]) -> dict:
    subject, html = build_action_email(events, positions)
    return send_email(subject, html)


def send_daily_digest(positions: list[dict], skipped: list[str],
                       stale: list[dict] | None = None,
                       errored: list[dict] | None = None) -> dict:
    subject, html = build_digest_email(positions, skipped, stale or [], errored or [])
    return send_email(subject, html)
