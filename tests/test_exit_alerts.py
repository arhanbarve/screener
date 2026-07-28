from src.exit_alerts import build_action_email, build_digest_email

POSITIONS = [
    {"ticker": "ANET", "entry_date": "2026-06-18", "entry_price": 100.25,
     "plan": {"initial_stop": 90.00, "risk_R": 11.47, "stop_floor": 100.25,
              "peak_close": 110.00, "trail_mult": 3.0, "stop_level": 102.0,
              "trims_fired": ["derisk"], "below_50d_streak": 0, "verdict": "SELL",
              "health": {"bearish": 2, "parts": ["macd", "rs"], "errors": [], "weeks": 60, "asof": "2026-07-24"},
              "days_to_earnings": 12, "last_eval": "2026-07-28", "last_close": 171.0}},
    {"ticker": "CRDO", "entry_date": "2026-06-18", "entry_price": 200.00,
     "plan": {"initial_stop": 250.0, "risk_R": 20.00, "stop_floor": 250.0,
              "peak_close": 300.0, "trail_mult": 3.0, "stop_level": 280.0,
              "trims_fired": [], "below_50d_streak": 0, "verdict": "HOLD",
              "health": {"bearish": 0, "parts": [], "errors": [], "weeks": 60, "asof": "2026-07-24"},
              "days_to_earnings": None, "last_eval": "2026-07-28", "last_close": 295.0}},
]

EVENTS = [{"ticker": "ANET", "type": "SELL",
           "reason": "close 171.00 below trailing stop 102.00",
           "instruction": "Sell entire remaining position at next open."}]


def test_action_email_subject_names_action_and_ticker():
    subject, html = build_action_email(EVENTS, POSITIONS)
    assert "ACTION" in subject and "SELL" in subject and "ANET" in subject
    assert "102.00" in html
    assert "Sell entire remaining position" in html


def test_digest_lists_every_position_with_verdict():
    subject, html = build_digest_email(POSITIONS, skipped=[], stale=[], errored=[])
    for t in ("ANET", "CRDO"):
        assert t in html
    assert "SELL" in html and "HOLD" in html


def test_digest_flags_skipped_tickers():
    subject, html = build_digest_email(POSITIONS, skipped=["RSI"], stale=[], errored=[])
    assert "RSI" in html and ("stale" in html.lower() or "skipped" in html.lower())


def test_digest_flags_stale_ticker_with_bar_date_and_marks_row():
    stale = [{"ticker": "ANET", "bar_date": "2026-07-20"}]
    subject, html = build_digest_email(POSITIONS, skipped=[], stale=stale, errored=[])
    assert "ANET" in html
    assert "2026-07-20" in html
    assert "NOT EVALUATED TODAY" in html


def test_digest_flags_errored_ticker_with_error_text():
    errored = [{"ticker": "CRDO", "error": "ConnectionError('timed out')"}]
    subject, html = build_digest_email(POSITIONS, skipped=[], stale=[], errored=errored)
    assert "CRDO" in html
    assert "ConnectionError" in html
    assert "NOT EVALUATED TODAY" in html


def test_digest_shows_health_errors_distinctly_from_clean_health():
    clean = [{"ticker": "CLEAN", "entry_date": "2026-06-18", "entry_price": 100.0,
              "plan": {"initial_stop": 90.0, "risk_R": 10.0, "stop_floor": 90.0,
                       "peak_close": 110.0, "trail_mult": 3.0, "stop_level": 95.0,
                       "trims_fired": [], "below_50d_streak": 0, "verdict": "HOLD",
                       "health": {"bearish": 0, "parts": [], "errors": [], "weeks": 60, "asof": "2026-07-24"},
                       "days_to_earnings": None, "last_eval": "2026-07-28", "last_close": 105.0}}]
    errored_health = [{"ticker": "ERR", "entry_date": "2026-06-18", "entry_price": 100.0,
                       "plan": {"initial_stop": 90.0, "risk_R": 10.0, "stop_floor": 90.0,
                                "peak_close": 110.0, "trail_mult": 3.0, "stop_level": 95.0,
                                "trims_fired": [], "below_50d_streak": 0, "verdict": "HOLD",
                                "health": {"bearish": 0, "parts": [], "errors": ["macd", "obv"],
                                           "weeks": 60, "asof": "2026-07-24"},
                                "days_to_earnings": None, "last_eval": "2026-07-28", "last_close": 105.0}}]
    _, clean_html = build_digest_email(clean, skipped=[], stale=[], errored=[])
    _, errored_html = build_digest_email(errored_health, skipped=[], stale=[], errored=[])
    assert "0/4" in clean_html
    # a check that threw must not silently read as a clean "0/4" bearish score:
    # the errored render must carry a distinct marker naming which checks failed
    assert "macd" in errored_html and "obv" in errored_html
    assert "checks errored" in errored_html or "&#42;" in errored_html or "*" in errored_html


def test_digest_shows_short_history_as_unknown_not_zero():
    short = [{"ticker": "IPO", "entry_date": "2026-06-18", "entry_price": 100.0,
              "plan": {"initial_stop": 90.0, "risk_R": 10.0, "stop_floor": 90.0,
                       "peak_close": 110.0, "trail_mult": 3.0, "stop_level": 95.0,
                       "trims_fired": [], "below_50d_streak": 0, "verdict": "HOLD",
                       "health": {"bearish": 0, "parts": [], "errors": [], "weeks": 5, "asof": "2026-07-24"},
                       "days_to_earnings": None, "last_eval": "2026-07-28", "last_close": 105.0}}]
    _, html = build_digest_email(short, skipped=[], stale=[], errored=[])
    assert "0/4" not in html


def test_action_email_escapes_html_metacharacters_in_reason():
    events = [{"ticker": "ANET", "type": "SELL",
               "reason": "<script>alert(1)</script>",
               "instruction": "Sell entire remaining position at next open."}]
    _, html = build_action_email(events, POSITIONS)
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_action_email_groups_multiple_events_per_ticker_dedupes_detail_block():
    # evaluate_day can fire both a derisk and a blowoff TRIM for the same
    # ticker on one bar — the floor/stop/peak/trims detail line must render
    # once per ticker, not once per event, while both instructions still show.
    events = [
        {"ticker": "ANET", "type": "TRIM", "reason": "derisk hit",
         "instruction": "Sell 1/3 of position at next open (derisk)."},
        {"ticker": "ANET", "type": "TRIM", "reason": "blowoff extension",
         "instruction": "Sell 1/3 of position at next open (blowoff)."},
    ]
    _, html = build_action_email(events, POSITIONS)
    assert html.count("floor 100.25") == 1
    assert "Sell 1/3 of position at next open (derisk)." in html
    assert "Sell 1/3 of position at next open (blowoff)." in html


def test_action_email_with_no_events_is_coherent():
    subject, html = build_action_email([], POSITIONS)
    assert subject == subject.strip()   # no trailing "ACTION: " with nothing after it
    assert "no action" in subject.lower() or "no action" in html.lower()
