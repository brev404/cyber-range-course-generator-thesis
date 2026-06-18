"""Tests for _parse_review_json — covers valid JSON, single-quoted (Gemma-style), and truncated fallback."""

from src.agents.ranking_agent import _parse_review_json


def test_valid_json_parsed():
    raw = '{"score": 8, "justification": "Good.", "improvements": []}'
    result = _parse_review_json(raw, "Technical")
    assert result is not None
    assert result["score"] == 8


def test_markdown_code_block_stripped():
    raw = '```json\n{"score": 7, "justification": "OK.", "improvements": []}\n```'
    result = _parse_review_json(raw, "Pedagogical")
    assert result is not None
    assert result["score"] == 7


def test_single_quoted_dict_gemma_style():
    """Gemma models often return Python-style single-quoted dicts."""
    raw = "{'score': 6, 'justification': 'Decent.', 'improvements': ['Add more examples.']}"
    result = _parse_review_json(raw, "Technical")
    assert result is not None
    assert result["score"] == 6
    assert result["justification"] == "Decent."


def test_single_quoted_with_dimension_scores():
    raw = (
        "{'score': 9, 'justification': 'Strong.', 'improvements': [], "
        "'dimension_scores': {'correctness': 9, 'completeness': 8}}"
    )
    result = _parse_review_json(raw, "Technical")
    assert result is not None
    assert result["score"] == 9
    assert result.get("dimension_scores", {}).get("correctness") == 9


def test_truncated_double_quoted_score_regex():
    raw = '{"score": 5, "justification": "Trun'
    result = _parse_review_json(raw, "Technical")
    assert result is not None
    assert result["score"] == 5


def test_truncated_single_quoted_score_regex():
    """Regex fallback must handle single-quoted 'score' key in truncated output."""
    raw = "{'score': 4, 'justification': 'Trun"
    result = _parse_review_json(raw, "Pedagogical")
    assert result is not None
    assert result["score"] == 4


def test_garbage_returns_none():
    raw = "This is not JSON or a dict at all."
    result = _parse_review_json(raw, "Technical")
    assert result is None


def test_score_returned_as_is_clamping_is_callers_job():
    """The parser returns the raw score value; clamping to 1-10 is the caller's responsibility.
    Only the regex truncation fallback clamps (it converts a bare int from regex)."""
    raw = "{'score': 15, 'justification': 'X', 'improvements': []}"
    result = _parse_review_json(raw, "Technical")
    assert result is not None
    assert result["score"] == 15  # not clamped by the parser

    # But the regex truncation fallback does clamp
    raw_truncated = "{'score': 15, 'justification': 'Trun"
    result2 = _parse_review_json(raw_truncated, "Technical")
    assert result2 is not None
    assert result2["score"] == 10  # regex fallback clamps
