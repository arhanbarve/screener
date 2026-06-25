# tests/test_filing_analysis.py
import pytest
from src.filing_analysis import _rough_diff


def test_rough_diff_identical_returns_no_diff():
    text = "The company faces intense competition in its primary markets. " \
           "Revenue growth has remained strong across all geographies."
    result = _rough_diff(text, text)
    # Identical texts produce no added/removed sentences
    assert "ADDED" not in result and "REMOVED" not in result


def test_rough_diff_detects_added_sentence():
    prior = "The company faces competition in its primary markets."
    cur   = (
        "The company faces competition in its primary markets. "
        "Additionally, the company has identified new cybersecurity risk factors "
        "that may adversely affect operations and financial results going forward."
    )
    result = _rough_diff(cur, prior)
    assert "ADDED" in result
    assert "cybersecurity" in result.lower()


def test_rough_diff_detects_removed_sentence():
    cur   = "The company faces competition in its primary markets."
    prior = (
        "The company faces competition in its primary markets. "
        "The company previously disclosed a going concern uncertainty that has since been resolved."
    )
    result = _rough_diff(cur, prior)
    assert "REMOVED" in result
    assert "going concern" in result.lower()


def test_rough_diff_respects_max_chars():
    big = " ".join([f"Sentence number {i} about various risk factors and market conditions." for i in range(500)])
    result = _rough_diff(big, "")
    assert len(result) <= 3000 + 100  # small buffer for truncation


def test_rough_diff_empty_inputs():
    result = _rough_diff("", "")
    assert result  # returns the fallback string, not empty
