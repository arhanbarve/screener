# tests/test_event_backtest.py
import pandas as pd
import pytest

from src.event_backtest import (
    _flatten_recent_dict,
    extract_category_events,
    GOING_CONCERN_RE,
    _classify_3_01_text,
    _extract_max_dollar_amount,
    _classify_1_01_materiality,
    classify_1_01_events,
    _forward_return,
    _covers_range,
    summarize,
    select_phase2_sample,
    estimate_phase2_cost,
    export_hand_label_csv,
    _section_for_category,
    MAX_PHASE2_SAMPLE,
)


# ---- flattening + item-code extraction -------------------------------------

def test_flatten_recent_dict_basic():
    d = {
        "form": ["8-K", "10-K"],
        "accessionNumber": ["0001-24-000001", "0001-24-000002"],
        "filingDate": ["2024-03-01", "2024-05-01"],
        "primaryDocument": ["a.htm", "b.htm"],
        "items": ["1.01,9.01", ""],
    }
    recs = _flatten_recent_dict(d)
    assert len(recs) == 2
    assert recs[0]["form"] == "8-K"
    assert recs[0]["items"] == "1.01,9.01"


def test_extract_category_events_no_longer_buckets_1_01_directly():
    # 1.01 is now routed through classify_1_01_events, not extract_category_events
    recs = [{"form": "8-K", "accession": "acc1", "filing_date": "2024-03-01",
             "primary_doc": "a.htm", "items": "1.01"}]
    events = extract_category_events("ABC", "0000123", recs)
    assert events == []


def test_extract_category_events_no_longer_buckets_3_01_directly():
    # 3.01 is now routed through classify_3_01_events, not extract_category_events
    recs = [{"form": "8-K", "accession": "acc1", "filing_date": "2024-03-01",
             "primary_doc": "a.htm", "items": "3.01"}]
    events = extract_category_events("ABC", "0000123", recs)
    assert events == []


def test_classify_3_01_text_regained_compliance():
    text = "The Company today announced it has regained compliance with the minimum bid price requirement."
    assert _classify_3_01_text(text) == "8k_3_01_regained"


def test_classify_3_01_text_deficiency_notice():
    text = "The Company received a notice of deficiency from the Exchange for failure to satisfy the listing rule."
    assert _classify_3_01_text(text) == "8k_3_01_notice"


def test_classify_3_01_text_unclassified_when_no_pattern_matches():
    text = "The Company filed this report in connection with an unrelated matter."
    assert _classify_3_01_text(text) == "8k_3_01_unclassified"


def test_extract_max_dollar_amount_finds_million_figure():
    text = "The Company entered into a credit agreement for $25 million with the Lender."
    assert _extract_max_dollar_amount(text) == 25_000_000


def test_extract_max_dollar_amount_picks_largest_when_multiple_present():
    text = "The Company paid a $500,000 arrangement fee in connection with the $40 million term loan."
    assert _extract_max_dollar_amount(text) == 40_000_000


def test_extract_max_dollar_amount_none_when_no_dollar_figure():
    text = "The Company entered into a material definitive agreement with a vendor."
    assert _extract_max_dollar_amount(text) is None


def test_classify_1_01_materiality_material_above_threshold():
    text = "The agreement provides for an aggregate commitment of $50 million under the facility."
    assert _classify_1_01_materiality(text) == "8k_1_01_material"


def test_classify_1_01_materiality_immaterial_below_threshold():
    text = "The Company issued a purchase order valued at $200,000 to the supplier."
    assert _classify_1_01_materiality(text) == "8k_1_01_immaterial"


def test_classify_1_01_materiality_unclassified_when_no_amount_found():
    text = "The Company entered into a strategic partnership agreement with a vendor."
    assert _classify_1_01_materiality(text) == "8k_1_01_unclassified"


def test_classify_1_01_events_ignores_dollar_amounts_outside_item_101_section(monkeypatch):
    # Multi-item 8-Ks are common — a large figure disclosed under a different
    # item (e.g. 2.03) must not leak into the 1.01 materiality call.
    html = (
        "<html><body>Item 1.01 Entry into a Material Definitive Agreement. "
        "The Company entered into a vendor services agreement for $500,000. "
        "Item 2.03 Creation of a Direct Financial Obligation. The Company also "
        "disclosed a $100,000,000 unrelated debt facility from a separate transaction."
        "</body></html>"
    )
    monkeypatch.setattr("src.event_backtest.fetch_filing_doc", lambda *a, **k: html)
    recs = [{"form": "8-K", "accession": "acc1", "filing_date": "2024-03-01",
             "primary_doc": "a.htm", "items": "1.01,2.03"}]
    events = classify_1_01_events("ABC", "0000123", recs, "unused.db")
    assert len(events) == 1
    assert events[0]["category"] == "8k_1_01_immaterial"


def test_extract_category_events_ignores_unrelated_items():
    recs = [{"form": "8-K", "accession": "acc1", "filing_date": "2024-03-01",
             "primary_doc": "a.htm", "items": "5.02,9.01"}]
    events = extract_category_events("ABC", "0000123", recs)
    assert events == []


def test_extract_category_events_odd_lot_tender():
    recs = [{"form": "SC TO-I", "accession": "acc2", "filing_date": "2024-04-01",
             "primary_doc": "t.htm", "items": ""}]
    events = extract_category_events("ABC", "0000123", recs)
    assert len(events) == 1
    assert events[0]["category"] == "odd_lot_tender"


def test_extract_category_events_delinquent_filer_regains():
    recs = [
        {"form": "NT 10-K", "accession": "nt1", "filing_date": "2024-01-15", "primary_doc": "nt.htm", "items": ""},
        {"form": "10-K", "accession": "real1", "filing_date": "2024-02-20", "primary_doc": "10k.htm", "items": ""},
        {"form": "10-K", "accession": "priorreal", "filing_date": "2023-02-01", "primary_doc": "p.htm", "items": ""},
    ]
    events = extract_category_events("ABC", "0000123", recs)
    regains = [e for e in events if e["category"] == "delinquent_filer_regains"]
    assert len(regains) == 1
    assert regains[0]["event_date"] == "2024-02-20"
    assert regains[0]["accession"] == "real1"


def test_extract_category_events_nt_with_no_followup_real_filing_produces_no_event():
    recs = [{"form": "NT 10-K", "accession": "nt1", "filing_date": "2024-01-15",
             "primary_doc": "nt.htm", "items": ""}]
    events = extract_category_events("ABC", "0000123", recs)
    assert events == []


# ---- going-concern regex ----------------------------------------------------

def test_going_concern_regex_matches_standard_phrasing():
    text = "the Company has substantial doubt about its ability to continue as a going concern"
    assert GOING_CONCERN_RE.search(text)


def test_going_concern_regex_no_false_positive_on_unrelated_text():
    text = "the Company faces intense competition and regulatory scrutiny"
    assert not GOING_CONCERN_RE.search(text)


# ---- forward return ---------------------------------------------------------

def _price_series(start="2024-01-01", n=100, start_price=10.0, daily_return=0.0):
    dates = pd.bdate_range(start=start, periods=n)
    prices = [start_price * (1 + daily_return) ** i for i in range(n)]
    return pd.DataFrame({"close": prices}, index=dates)


def test_forward_return_flat_series_is_zero():
    df = _price_series(daily_return=0.0)
    r = _forward_return(df, df.index[10].date().isoformat(), horizon=20)
    assert r == pytest.approx(0.0, abs=1e-9)


def test_forward_return_growth_is_positive():
    df = _price_series(daily_return=0.01)
    r = _forward_return(df, df.index[5].date().isoformat(), horizon=10)
    assert r > 0


def test_forward_return_none_when_insufficient_future_data():
    df = _price_series(n=15)
    r = _forward_return(df, df.index[10].date().isoformat(), horizon=20)
    assert r is None


def test_forward_return_none_on_empty_prices():
    assert _forward_return(pd.DataFrame(), "2024-01-01", 20) is None


# ---- cache freshness must cover both ends of the requested range -----------

def test_covers_range_true_when_both_ends_covered():
    df = _price_series(start="2023-01-01", n=500)
    assert _covers_range(df, "2023-06-01", "2024-01-01")


def test_covers_range_false_when_cache_does_not_reach_requested_end():
    # cache only goes through mid-2023; a later event needs forward data past that
    df = _price_series(start="2023-01-01", n=100)
    assert not _covers_range(df, "2023-01-01", "2024-06-01")


def test_covers_range_false_on_empty_or_none():
    assert not _covers_range(None, "2023-01-01", "2023-06-01")
    assert not _covers_range(pd.DataFrame(), "2023-01-01", "2023-06-01")


# ---- summarize gate ----------------------------------------------------------

def _events_df_for_category(category, abn_rets_20d, abn_rets_60d=None):
    n = len(abn_rets_20d)
    abn_rets_60d = abn_rets_60d or abn_rets_20d
    return pd.DataFrame({
        "ticker": [f"T{i}" for i in range(n)],
        "category": [category] * n,
        "event_date": pd.date_range("2024-01-01", periods=n, freq="7D").astype(str),
        "abn_ret_5d": abn_rets_20d,
        "abn_ret_20d": abn_rets_20d,
        "abn_ret_60d": abn_rets_60d,
    })


def test_summarize_marks_strong_consistent_signal_as_pass():
    df = _events_df_for_category("8k_1_01", [0.05, 0.06, 0.04, 0.05, 0.07, 0.05])
    summary = summarize(df, horizons=(20, 60), gate_horizons=(20, 60))
    row = summary[summary["category"] == "8k_1_01"].iloc[0]
    assert row["gate"] == "PASS"


def test_summarize_marks_noisy_zero_mean_signal_as_fail():
    df = _events_df_for_category("odd_lot_tender", [0.01, -0.01, 0.02, -0.02, 0.005, -0.005])
    summary = summarize(df, horizons=(20, 60), gate_horizons=(20, 60))
    row = summary[summary["category"] == "odd_lot_tender"].iloc[0]
    assert row["gate"] == "FAIL"


def test_summarize_empty_returns_empty_frame():
    assert summarize(pd.DataFrame()).empty


# ---- phase 2 sampling + cost cap --------------------------------------------

def test_select_phase2_sample_respects_hard_cap():
    n = 1000
    events_df = pd.DataFrame({
        "ticker": [f"T{i}" for i in range(n)],
        "category": ["8k_1_01"] * n,
        "abn_ret_20d": [0.01 * (i % 50) for i in range(n)],
        "accession": [f"acc{i}" for i in range(n)],
    })
    summary_df = pd.DataFrame({"category": ["8k_1_01"], "gate": ["PASS"]})
    sample = select_phase2_sample(events_df, summary_df, max_total=MAX_PHASE2_SAMPLE)
    assert len(sample) <= MAX_PHASE2_SAMPLE


def test_select_phase2_sample_excludes_fail_categories():
    events_df = pd.DataFrame({
        "ticker": ["A", "B"],
        "category": ["odd_lot_tender", "odd_lot_tender"],
        "abn_ret_20d": [0.01, -0.01],
        "accession": ["a1", "a2"],
    })
    summary_df = pd.DataFrame({"category": ["odd_lot_tender"], "gate": ["FAIL"]})
    sample = select_phase2_sample(events_df, summary_df)
    assert sample.empty


def test_estimate_phase2_cost_under_five_dollars_for_max_sample():
    sample = pd.DataFrame({"x": range(MAX_PHASE2_SAMPLE)})
    est = estimate_phase2_cost(sample)
    assert est["estimated_cost_usd"] < 5.0


# ---- hand-label export: largest/smallest apparent-return events first -------

def test_export_hand_label_csv_sorts_by_return_and_flags_extremes(tmp_path):
    n = 20
    df = pd.DataFrame({
        "ticker": [f"T{i}" for i in range(n)],
        "abn_ret_20d": [0.01 * i for i in range(n)],
    })
    out = tmp_path / "sample.csv"
    export_hand_label_csv(df, str(out), horizon=20, top_n=5)
    result = pd.read_csv(out)
    # sorted descending: biggest winner first, biggest loser last
    assert result["abn_ret_20d"].iloc[0] == pytest.approx(0.19)
    assert result["abn_ret_20d"].iloc[-1] == pytest.approx(0.0)
    # top 5 and bottom 5 flagged, middle rows not
    assert result["hand_label_priority"].iloc[:5].all()
    assert result["hand_label_priority"].iloc[-5:].all()
    assert not result["hand_label_priority"].iloc[8:12].any()


def test_export_hand_label_csv_adds_filing_url_column(tmp_path):
    df = pd.DataFrame({
        "ticker": ["ABC"],
        "cik": ["123456"],
        "accession": ["0001234567-24-000001"],
        "primary_doc": ["ex.htm"],
        "abn_ret_20d": [0.05],
    })
    out = tmp_path / "sample.csv"
    export_hand_label_csv(df, str(out), horizon=20, top_n=1)
    result = pd.read_csv(out)
    assert result["filing_url"].iloc[0] == (
        "https://www.sec.gov/Archives/edgar/data/123456/000123456724000001/ex.htm"
    )


# ---- header-anchored section extraction for Phase 2 --------------------------

def test_section_for_category_going_concern_extracts_around_phrase():
    filler = "irrelevant boilerplate. " * 200
    html = f"<html><body>{filler}the Company has substantial doubt about its ability to continue as a going concern{filler}</body></html>"
    section = _section_for_category(html, "going_concern_removed", max_chars=500)
    assert "going concern" in section.lower()
    assert len(section) <= 500


def test_section_for_category_no_anchor_falls_back_to_head_truncation():
    html = "<html><body>" + ("nothing relevant here. " * 500) + "</body></html>"
    section = _section_for_category(html, "odd_lot_tender", max_chars=200)
    assert len(section) <= 200
    assert section.strip().startswith("nothing relevant")
