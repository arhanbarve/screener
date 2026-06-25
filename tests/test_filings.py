# tests/test_filings.py
import pytest
from src.filings import (
    normalize_tokens,
    cosine_similarity,
    extract_sections,
    select_comparable,
    parse_submissions,
)


# ---- text normalization & similarity --------------------------------------

def test_normalize_tokens_lowercases_and_strips():
    toks = normalize_tokens("The Company's Revenue grew 12% in 2023!")
    assert "revenue" in toks
    assert "grew" in toks
    # punctuation / numbers / symbols dropped
    assert "12" not in toks
    assert "2023" not in toks


def test_cosine_identical_is_one():
    a = "the company faces risks from competition and regulation"
    assert abs(cosine_similarity(a, a) - 1.0) < 1e-9


def test_cosine_disjoint_is_zero():
    a = "alpha beta gamma delta"
    b = "epsilon zeta eta theta"
    assert cosine_similarity(a, b) == 0.0


def test_cosine_partial_between_zero_and_one():
    a = "competition regulation litigation supply chain risk"
    b = "competition regulation new cybersecurity risk emerged"
    sim = cosine_similarity(a, b)
    assert 0.0 < sim < 1.0


def test_cosine_empty_is_zero():
    assert cosine_similarity("", "anything") == 0.0
    assert cosine_similarity("", "") == 0.0


# ---- section extraction ----------------------------------------------------

SAMPLE_10K_HTML = """
<html><body>
<p>Item 1. Business</p>
<p>We make widgets and sell them globally.</p>
<p>Item 1A. Risk Factors</p>
<p>Our business is subject to intense competition and supply chain disruption.</p>
<p>Item 1B. Unresolved Staff Comments</p>
<p>None.</p>
<p>Item 7. Management's Discussion and Analysis of Financial Condition</p>
<p>Revenue increased due to higher widget demand and pricing.</p>
<p>Item 7A. Quantitative and Qualitative Disclosures</p>
<p>Interest rate risk discussion.</p>
<p>Item 8. Financial Statements</p>
</body></html>
"""


def test_extract_sections_finds_risk_and_mda():
    sec = extract_sections(SAMPLE_10K_HTML)
    assert "competition" in sec["risk_factors"].lower()
    assert "supply chain" in sec["risk_factors"].lower()
    assert "widget demand" in sec["mda"].lower()
    # risk factors must not bleed into MD&A
    assert "widget demand" not in sec["risk_factors"].lower()


def test_extract_sections_missing_returns_empty_strings():
    sec = extract_sections("<html><body><p>nothing useful here</p></body></html>")
    assert sec["risk_factors"] == ""
    assert sec["mda"] == ""


# ---- submissions parsing & comparable selection ---------------------------

SAMPLE_SUBMISSIONS = {
    "filings": {
        "recent": {
            "form":            ["10-Q", "10-K", "8-K", "10-K", "10-Q"],
            "accessionNumber": ["0000-24-Q2", "0000-24-K", "0000-24-8K", "0000-23-K", "0000-23-Q2"],
            "filingDate":      ["2024-08-01", "2024-02-15", "2024-01-10", "2023-02-15", "2023-08-01"],
            "reportDate":      ["2024-06-30", "2023-12-31", "2024-01-09", "2022-12-31", "2023-06-30"],
            "primaryDocument": ["q2-24.htm", "10k-24.htm", "8k.htm", "10k-23.htm", "q2-23.htm"],
        }
    }
}


def test_parse_submissions_returns_sorted_records():
    recs = parse_submissions(SAMPLE_SUBMISSIONS)
    assert len(recs) == 5
    # newest first
    assert recs[0]["filing_date"] >= recs[-1]["filing_date"]
    assert recs[0]["form"] == "10-Q"
    assert recs[0]["accession"] == "0000-24-Q2"


def test_select_comparable_10k_pairs_consecutive_annuals():
    recs = parse_submissions(SAMPLE_SUBMISSIONS)
    cur, prior = select_comparable(recs, "10-K")
    assert cur["accession"] == "0000-24-K"
    assert prior["accession"] == "0000-23-K"


def test_select_comparable_10q_matches_same_fiscal_quarter():
    recs = parse_submissions(SAMPLE_SUBMISSIONS)
    cur, prior = select_comparable(recs, "10-Q")
    # current Q2-2024 (report month 06) should pair with Q2-2023 (report month 06)
    assert cur["accession"] == "0000-24-Q2"
    assert prior["accession"] == "0000-23-Q2"


def test_select_comparable_no_prior_returns_none():
    one_filing = {
        "filings": {"recent": {
            "form": ["10-K"], "accessionNumber": ["only-one"],
            "filingDate": ["2024-02-15"], "reportDate": ["2023-12-31"],
            "primaryDocument": ["10k.htm"],
        }}
    }
    recs = parse_submissions(one_filing)
    cur, prior = select_comparable(recs, "10-K")
    assert cur is not None
    assert prior is None
