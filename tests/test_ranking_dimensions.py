"""Tests for Rubric Anchoring in Ranking (dimension scores).

Tests verify that:
- RankingScore accepts dimension_scores dict
- JSON parsing extracts dimension_scores correctly
- Technical and pedagogical scores include dimension_scores
- Missing dimension_scores fallback to None (backward compatible)
- RankingReport export includes dimension_scores
"""

import json

from src.agents.ranking_agent import (
    _PEDAGOGICAL_DIMENSIONS,
    _TECHNICAL_DIMENSIONS,
    _build_pedagogical_score,
    _build_technical_score,
    _parse_review_json,
)
from src.models.report_models import RankingReport, RankingScore


def test_ranking_score_with_dimension_scores():
    """Test that RankingScore accepts dimension_scores dict."""
    # Technical dimensions
    tech_score = RankingScore(
        score=9,
        persona="Technical",
        justification="Test",
        improvements=[],
        dimension_scores={
            "correctness": 9,
            "completeness": 8,
            "technical_accuracy": 9,
            "code_quality": 8,
            "logical_validity": 9,
        },
    )
    assert tech_score.dimension_scores is not None
    assert tech_score.dimension_scores["correctness"] == 9
    assert tech_score.dimension_scores["completeness"] == 8

    # Pedagogical dimensions
    ped_score = RankingScore(
        score=8,
        persona="Pedagogical",
        justification="Test",
        improvements=[],
        dimension_scores={
            "sections_structure": 9,
            "cognitive_load": 8,
            "scaffolding_reproducibility": 7,
            "relevance_curriculum": 8,
            "skill_level_awareness": 9,
            "human_language_context": 8,
        },
    )
    assert ped_score.dimension_scores is not None
    assert ped_score.dimension_scores["sections_structure"] == 9
    assert ped_score.dimension_scores["cognitive_load"] == 8

    # None dimension_scores (backward compatible)
    score_none = RankingScore(
        score=7,
        persona="Technical",
        justification="Test",
        improvements=[],
        dimension_scores=None,
    )
    assert score_none.dimension_scores is None


def test_parse_review_json_extracts_dimension_scores():
    """Test that _parse_review_json extracts dimension_scores from JSON."""
    # Technical JSON with dimension_scores
    tech_json = json.dumps(
        {
            "score": 9,
            "justification": "Good technical quality",
            "improvements": ["Add comments"],
            "technical_rank": "Intermediate",
            "dimension_scores": {
                "correctness": 9,
                "completeness": 8,
                "technical_accuracy": 9,
                "code_quality": 8,
                "logical_validity": 9,
            },
        }
    )
    data = _parse_review_json(tech_json, "Technical")
    assert data is not None
    assert "dimension_scores" in data
    assert data["dimension_scores"]["correctness"] == 9
    assert data["dimension_scores"]["completeness"] == 8
    # Validate all expected dimensions are present
    for dim in _TECHNICAL_DIMENSIONS:
        assert dim in data["dimension_scores"]

    # Pedagogical JSON with dimension_scores
    ped_json = json.dumps(
        {
            "score": 8,
            "justification": "Good pedagogical structure",
            "improvements": ["Add more examples"],
            "dimension_scores": {
                "sections_structure": 9,
                "cognitive_load": 8,
                "scaffolding_reproducibility": 7,
                "relevance_curriculum": 8,
                "skill_level_awareness": 9,
                "human_language_context": 8,
            },
        }
    )
    data = _parse_review_json(ped_json, "Pedagogical")
    assert data is not None
    assert "dimension_scores" in data
    assert data["dimension_scores"]["sections_structure"] == 9
    # Validate all expected dimensions are present
    for dim in _PEDAGOGICAL_DIMENSIONS:
        assert dim in data["dimension_scores"]

    # JSON without dimension_scores (backward compatible)
    json_no_dims = json.dumps(
        {
            "score": 7,
            "justification": "OK",
            "improvements": [],
        }
    )
    data = _parse_review_json(json_no_dims, "Technical")
    assert data is not None
    assert "dimension_scores" not in data or data.get("dimension_scores") is None


def test_parse_review_json_validates_dimension_scores():
    """Test that _parse_review_json validates dimension names and scores."""
    # Invalid dimension name (should be filtered out)
    tech_json_invalid_dim = json.dumps(
        {
            "score": 9,
            "justification": "Test",
            "improvements": [],
            "dimension_scores": {
                "correctness": 9,
                "invalid_dimension": 8,  # Not in _TECHNICAL_DIMENSIONS
                "completeness": 8,
            },
        }
    )
    data = _parse_review_json(tech_json_invalid_dim, "Technical")
    assert data is not None
    assert "dimension_scores" in data
    assert "correctness" in data["dimension_scores"]
    assert "invalid_dimension" not in data["dimension_scores"]
    assert "completeness" in data["dimension_scores"]

    # Invalid score value (should be clamped to 1-10)
    tech_json_invalid_score = json.dumps(
        {
            "score": 9,
            "justification": "Test",
            "improvements": [],
            "dimension_scores": {
                "correctness": 15,  # Out of range, should be clamped to 10
                "completeness": -5,  # Out of range, should be clamped to 1
                "technical_accuracy": 9,
            },
        }
    )
    data = _parse_review_json(tech_json_invalid_score, "Technical")
    assert data is not None
    assert data["dimension_scores"]["correctness"] == 10
    assert data["dimension_scores"]["completeness"] == 1
    assert data["dimension_scores"]["technical_accuracy"] == 9


def test_build_technical_score_with_dimensions():
    """Test that _build_technical_score includes dimension_scores."""
    tech_json = json.dumps(
        {
            "score": 9,
            "justification": "Good technical quality",
            "improvements": ["Add comments"],
            "technical_rank": "Intermediate",
            "dimension_scores": {
                "correctness": 9,
                "completeness": 8,
                "technical_accuracy": 9,
                "code_quality": 8,
                "logical_validity": 9,
            },
        }
    )
    score, rank = _build_technical_score("test_challenge", tech_json)
    assert score.persona == "Technical"
    assert score.score == 9
    assert score.dimension_scores is not None
    assert score.dimension_scores["correctness"] == 9
    assert score.dimension_scores["completeness"] == 8
    assert rank == "Intermediate"

    # Without dimension_scores (backward compatible)
    tech_json_no_dims = json.dumps(
        {
            "score": 7,
            "justification": "OK",
            "improvements": [],
            "technical_rank": "Beginner",
        }
    )
    score, rank = _build_technical_score("test_challenge", tech_json_no_dims)
    assert score.dimension_scores is None


def test_build_pedagogical_score_with_dimensions():
    """Test that _build_pedagogical_score includes dimension_scores."""
    ped_json = json.dumps(
        {
            "score": 8,
            "justification": "Good pedagogical structure",
            "improvements": ["Add more examples"],
            "dimension_scores": {
                "sections_structure": 9,
                "cognitive_load": 8,
                "scaffolding_reproducibility": 7,
                "relevance_curriculum": 8,
                "skill_level_awareness": 9,
                "human_language_context": 8,
            },
        }
    )
    score = _build_pedagogical_score("test_challenge", "test_writeup", ped_json)
    assert score.persona == "Pedagogical"
    assert score.score == 8
    assert score.dimension_scores is not None
    assert score.dimension_scores["sections_structure"] == 9
    assert score.dimension_scores["cognitive_load"] == 8

    # Without dimension_scores (backward compatible)
    ped_json_no_dims = json.dumps(
        {
            "score": 7,
            "justification": "OK",
            "improvements": [],
        }
    )
    score = _build_pedagogical_score("test_challenge", "test_writeup", ped_json_no_dims)
    assert score.dimension_scores is None


def test_dimension_scores_fallback():
    """Test that missing or invalid dimension_scores fallback to None (backward compatible)."""
    # Missing dimension_scores
    json_no_dims = json.dumps(
        {
            "score": 7,
            "justification": "OK",
            "improvements": [],
        }
    )
    tech_score, _ = _build_technical_score("test", json_no_dims)
    assert tech_score.dimension_scores is None

    ped_score = _build_pedagogical_score("test", "writeup", json_no_dims)
    assert ped_score.dimension_scores is None

    # Invalid dimension_scores type (not a dict)
    json_invalid_type = json.dumps(
        {
            "score": 7,
            "justification": "OK",
            "improvements": [],
            "dimension_scores": "not a dict",
        }
    )
    tech_score, _ = _build_technical_score("test", json_invalid_type)
    assert tech_score.dimension_scores is None

    ped_score = _build_pedagogical_score("test", "writeup", json_invalid_type)
    assert ped_score.dimension_scores is None

    # Empty dimension_scores dict
    json_empty_dims = json.dumps(
        {
            "score": 7,
            "justification": "OK",
            "improvements": [],
            "dimension_scores": {},
        }
    )
    tech_score, _ = _build_technical_score("test", json_empty_dims)
    assert tech_score.dimension_scores is None

    ped_score = _build_pedagogical_score("test", "writeup", json_empty_dims)
    assert ped_score.dimension_scores is None


def test_ranking_report_export_includes_dimensions():
    """Test that RankingReport includes dimension_scores in export."""
    tech_score = RankingScore(
        score=9,
        persona="Technical",
        justification="Good",
        improvements=[],
        dimension_scores={"correctness": 9, "completeness": 8},
    )
    ped_score = RankingScore(
        score=8,
        persona="Pedagogical",
        justification="Good",
        improvements=[],
        dimension_scores={"sections_structure": 9, "cognitive_load": 8},
    )
    report = RankingReport(
        challenge_id="test_challenge",
        overall_score=8.5,
        technical_review=tech_score,
        pedagogical_review=ped_score,
        technical_rank="Intermediate",
    )

    # Verify dimension_scores are accessible
    assert report.technical_review.dimension_scores is not None
    assert report.pedagogical_review.dimension_scores is not None
    assert report.technical_review.dimension_scores["correctness"] == 9
    assert report.pedagogical_review.dimension_scores["sections_structure"] == 9

    # Test JSON serialization (as would be done in main.py export)
    report_dict = {
        "challenge_id": report.challenge_id,
        "overall_score": report.overall_score,
        "technical_score": report.technical_review.score,
        "pedagogical_score": report.pedagogical_review.score,
        "technical_rank": report.technical_rank,
    }
    if report.technical_review.dimension_scores:
        report_dict["technical_dimension_scores"] = (
            report.technical_review.dimension_scores
        )
    if report.pedagogical_review.dimension_scores:
        report_dict["pedagogical_dimension_scores"] = (
            report.pedagogical_review.dimension_scores
        )

    assert "technical_dimension_scores" in report_dict
    assert "pedagogical_dimension_scores" in report_dict
    assert report_dict["technical_dimension_scores"]["correctness"] == 9
    assert report_dict["pedagogical_dimension_scores"]["sections_structure"] == 9
