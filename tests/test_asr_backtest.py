import pandas as pd

from src.asr_backtest import confirm_hits, confirm_new_announcement_text


def test_confirms_entered_into_language():
    text = "On July 1, 2026, the Company entered into an accelerated share repurchase agreement with its bank."
    assert confirm_new_announcement_text(text) is True


def test_confirms_announce_language():
    text = "The Company announced today it will commence an accelerated share repurchase program."
    assert confirm_new_announcement_text(text) is True


def test_rejects_progress_update_language():
    text = ("During the quarter, the Company completed the previously disclosed accelerated share "
            "repurchase and retired the shares.")
    assert confirm_new_announcement_text(text) is False


def test_rejects_when_phrase_absent():
    text = "The Company entered into a new credit facility."
    assert confirm_new_announcement_text(text) is False


def test_confirm_keyword_outside_window_not_matched():
    filler = "x" * 500
    text = f"entered into something else. {filler} the accelerated share repurchase was completed as planned."
    assert confirm_new_announcement_text(text, window=50) is False


def test_confirm_hits_drops_unconfirmed_and_missing_primary_doc(monkeypatch):
    hits = pd.DataFrame([
        {"cik": "0000000001", "ticker": "AAA", "file_date": pd.Timestamp("2026-01-05"),
         "adsh": "x", "primary_doc": "doc.htm"},
        {"cik": "0000000002", "ticker": "BBB", "file_date": pd.Timestamp("2026-01-06"),
         "adsh": "y", "primary_doc": None},
    ])

    def fake_fetch(cik, adsh, primary_doc, db_path):
        return "<html>entered into an accelerated share repurchase</html>" if cik == "0000000001" else None

    monkeypatch.setattr("src.asr_backtest.fetch_filing_doc", fake_fetch)
    monkeypatch.setattr("src.asr_backtest.plain_text", lambda html: html)
    out = confirm_hits(hits)
    assert list(out["ticker"]) == ["AAA"]
