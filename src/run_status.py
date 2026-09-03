"""Post-run status file + failure alert.

Called by run_screener.sh after every screener run, success or failure. Writes
run_status.json at the repo root (tracked in git) so the cloud dashboard — which
has no access to logs/ — can still show whether the last run worked. On failure
it also emails the tail of the log, so a silently-dead cron job surfaces.
"""

import argparse
import html
import os
import json
import re
from datetime import datetime, timezone
from pathlib import Path

STATUS_PATH = Path("run_status.json")
TAIL_LINES = 40
ERROR_RE = re.compile(r"^\w+(?:\.\w+)*(?:Error|Exception|Interrupt):")


def _last_run_slice(text: str) -> str:
    marker = "=== Screener run started:"
    idx = text.rfind(marker)
    return text[idx:] if idx != -1 else text


def parse_log(text: str) -> dict:
    """Pull the few facts the dashboard shows out of a run log."""
    selected = re.findall(r"\[compose\] \d+ ranked → top (\d+)", text)
    error = None
    for line in reversed(text.splitlines()):
        if ERROR_RE.match(line):
            error = line.strip()[:300]
            break
    return {
        "selected": int(selected[-1]) if selected else None,
        "error": error,
    }


def build_status(log_text: str, rc: int, started_at: str, duration_secs: int) -> dict:
    parsed = parse_log(log_text)
    return {
        "date": datetime.now().date().isoformat(),
        "started_at": started_at,
        "finished_at": datetime.now().replace(microsecond=0).isoformat(),
        "written_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "result": "success" if rc == 0 else "failed",
        "exit_code": rc,
        "duration_secs": duration_secs,
        "selected": parsed["selected"],
        "error": parsed["error"],
        # A label, not the real hostname: gethostname() leaks the network
        # and machine (e.g. a campus DHCP name) into a public repo.
        "host": os.environ.get("SCREENER_HOST_LABEL", "screener"),
    }


def build_failure_email(status: dict, log_text: str) -> tuple[str, str]:
    tail = "\n".join(log_text.splitlines()[-TAIL_LINES:])
    subject = f"⚠ Screener FAILED — {status['date']}"
    body = (
        '<div style="font-family:ui-monospace,SFMono-Regular,Menlo,monospace;'
        'background:#020209;color:#e2e8f0;padding:24px">'
        f'<div style="color:#ef4444;font-size:18px;font-weight:700;letter-spacing:0.08em">'
        f'SCREENER RUN FAILED</div>'
        f'<div style="color:#94a3b8;font-size:13px;margin-top:6px">'
        f'{html.escape(status["date"])} · exit {status["exit_code"]} · '
        f'{status["duration_secs"]}s · {html.escape(status["host"])}</div>'
        f'<div style="color:#f59e0b;font-size:14px;margin:18px 0 6px">'
        f'{html.escape(status["error"] or "no exception line found in log")}</div>'
        f'<pre style="background:#07090f;border:1px solid #1e293b;border-radius:6px;'
        f'padding:14px;font-size:11px;line-height:1.5;overflow-x:auto;color:#cbd5e1">'
        f'{html.escape(tail)}</pre>'
        f'<div style="color:#64748b;font-size:11px;margin-top:14px">'
        f'Dashboard: https://arhan-screener.streamlit.app</div>'
        '</div>'
    )
    return subject, body


def main(argv=None):
    p = argparse.ArgumentParser(prog="run_status")
    p.add_argument("--log", required=True)
    p.add_argument("--rc", type=int, required=True)
    p.add_argument("--started", required=True, help="ISO timestamp of run start")
    p.add_argument("--duration", type=int, default=0)
    p.add_argument("--no-email", action="store_true")
    args = p.parse_args(argv)

    log_path = Path(args.log)
    log_text = _last_run_slice(log_path.read_text(errors="replace")) if log_path.exists() else ""

    status = build_status(log_text, args.rc, args.started, args.duration)
    STATUS_PATH.write_text(json.dumps(status, indent=2) + "\n")
    print(f"[run_status] {status['result']} → {STATUS_PATH}")

    if args.rc != 0 and not args.no_email:
        from src.notify import send_email

        subject, body = build_failure_email(status, log_text)
        try:
            result = send_email(subject, body)
            print(f"[run_status] alert email: {result}")
        except Exception as e:
            print(f"[run_status] alert email failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
