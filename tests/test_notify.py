import json
from unittest.mock import patch, MagicMock

import pytest

from src import notify

ACCOUNT = {"equity": "99928.46", "last_equity": "100000", "cash": "72000.05"}
POSITIONS = [
    {"symbol": "TSEM", "qty": "33.58", "avg_entry_price": "238.22",
     "current_price": "238.27", "market_value": "8001.50", "unrealized_pl": "1.51",
     "unrealized_plpc": "0.00019"},
    {"symbol": "STX", "qty": "4.68", "avg_entry_price": "854.07",
     "current_price": "847.0", "market_value": "3967.42", "unrealized_pl": "-32.57",
     "unrealized_plpc": "-0.00814"},
]


def test_render_positions_table_colors_pl():
    html = notify.render_positions_table(POSITIONS)
    assert "TSEM" in html and "STX" in html
    assert 'class="pos"' in html   # positive uP&L styled green
    assert 'class="neg"' in html   # negative uP&L styled red


def test_render_positions_table_empty():
    html = notify.render_positions_table([])
    assert "No open positions" in html


def test_build_daily_email():
    subject, html = notify.build_email("daily", "2026-07-24", ACCOUNT, POSITIONS,
                                       "## Market context\n\nSPY choppy.")
    assert "2026-07-24" in subject
    assert "99,928" in subject or "99928" in subject
    assert "0.07%" in subject and "−" in subject  # day P&L pct, proper minus glyph
    assert "Equity" in html
    assert "TSEM" in html
    assert "SPY choppy" in html
    assert "<!doctype html>" in html.lower() or "<html" in html.lower()


def test_build_weekly_email():
    subject, html = notify.build_email("weekly", "2026-W30", ACCOUNT, POSITIONS,
                                       "## Best call\n\nFIX.")
    assert "Weekly" in subject
    assert "2026-W30" in subject
    assert "Best call" in html


def test_md_to_html_renders_tables():
    md = "| A | B |\n|---|---|\n| 1 | 2 |"
    html = notify.md_to_html(md)
    assert "<table" in html


def test_send_email_missing_creds_skips(monkeypatch):
    monkeypatch.delenv("GMAIL_ADDRESS", raising=False)
    monkeypatch.delenv("GMAIL_APP_PASSWORD", raising=False)
    out = notify.send_email("subj", "<p>hi</p>")
    assert out["sent"] is False
    assert "GMAIL" in out["reason"]


def test_send_email_sends(monkeypatch):
    monkeypatch.setenv("GMAIL_ADDRESS", "bot@gmail.com")
    monkeypatch.setenv("GMAIL_APP_PASSWORD", "app-pw")
    smtp = MagicMock()
    ctx = MagicMock()
    ctx.__enter__.return_value = smtp
    with patch("src.notify.smtplib.SMTP_SSL", return_value=ctx) as srv:
        out = notify.send_email("subj", "<p>hi</p>", to_addr="me@gmail.com")
    srv.assert_called_once()
    smtp.login.assert_called_once_with("bot@gmail.com", "app-pw")
    smtp.send_message.assert_called_once()
    assert out["sent"] is True


def test_main_daily_reads_journal_and_sends(tmp_path, monkeypatch):
    journal = tmp_path / "2026-07-24.md"
    journal.write_text("# Trading Journal — 2026-07-24\n\n## Market context\n\nSPY up.")
    with patch("src.notify.broker.get_account", return_value=ACCOUNT), \
         patch("src.notify.broker.get_positions", return_value=POSITIONS), \
         patch("src.notify.send_email", return_value={"sent": True}) as se:
        rc = notify.main(["daily", "--file", str(journal), "--date", "2026-07-24"])
    assert rc == 0
    se.assert_called_once()
    subject = se.call_args[0][0]
    assert "2026-07-24" in subject
