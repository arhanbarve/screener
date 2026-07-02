# tests/test_filings.py
import pytest
from src.filings import (
    normalize_tokens,
    cosine_similarity,
    bigram_jaccard,
    section_similarity,
    extract_sections,
    select_comparable,
    parse_submissions,
    _risk_metrics,
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


# ---- stopwords, bigram jaccard & blended section similarity ----------------

def test_normalize_tokens_drops_stopwords_by_default():
    toks = normalize_tokens("the company and its risks are here")
    # substantive words kept
    assert "company" in toks
    assert "risks" in toks
    # common stopwords dropped
    assert "the" not in toks
    assert "and" not in toks
    assert "are" not in toks


def test_normalize_tokens_keeps_stopwords_when_disabled():
    toks = normalize_tokens("the company and its risks", use_stopwords=False)
    assert "the" in toks
    assert "and" in toks


def test_bigram_jaccard_identical_is_one():
    a = "competition regulation litigation supply chain"
    assert abs(bigram_jaccard(a, a) - 1.0) < 1e-9


def test_bigram_jaccard_disjoint_is_zero():
    a = "alpha beta gamma delta"
    b = "epsilon zeta eta theta"
    assert bigram_jaccard(a, b) == 0.0


def test_bigram_jaccard_empty_is_zero():
    assert bigram_jaccard("", "anything else here") == 0.0
    assert bigram_jaccard("one", "two") == 0.0  # no bigrams (single tokens)


def test_section_similarity_blends_jaccard_and_cosine():
    a = "competition regulation litigation supply chain risk"
    b = "competition regulation new cybersecurity risk emerged"
    jac = bigram_jaccard(a, b)
    cos = cosine_similarity(a, b)
    blended = section_similarity(a, b, jaccard_blend=0.5)
    assert abs(blended - (0.5 * jac + 0.5 * cos)) < 1e-9
    # blend endpoints
    assert abs(section_similarity(a, b, jaccard_blend=0.0) - cos) < 1e-9
    assert abs(section_similarity(a, b, jaccard_blend=1.0) - jac) < 1e-9


# ---- risk-section growth ---------------------------------------------------

def test_risk_metrics_positive_growth():
    prior = "competition regulation"
    cur = "competition regulation litigation cybersecurity"
    risk_len, prior_risk_len, growth = _risk_metrics(cur, prior)
    assert risk_len == 4
    assert prior_risk_len == 2
    assert abs(growth - 1.0) < 1e-9


def test_risk_metrics_zero_prior_gives_zero_growth():
    risk_len, prior_risk_len, growth = _risk_metrics("competition regulation", "")
    assert risk_len == 2
    assert prior_risk_len == 0
    assert growth == 0.0


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
