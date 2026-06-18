"""Tests for ranking_agent parse robustness (Q1 fix: robust dim_scores parsing).

Covers:
1. Well-formed judge JSON with dim_scores -> parsed correctly (regression)
2. Truncated mid-dim_scores JSON -> parser logs warning + returns persona score with empty dim_scores
3. Judge response with markdown code fence around JSON -> parsed correctly
4. Judge response with extra preamble text before JSON -> parsed correctly
5. LLM failure during retest preserves dim_scores from existing/prior pass
6. Missing dim_scores key in otherwise valid JSON -> warning logged, None returned

Test IDs: RNK-R01 through RNK-R06
"""

import json
import logging

import pytest
from loguru import logger as _loguru_logger

from src.agents.ranking_agent import (
    _evaluate_one_challenge_retest,
    _parse_review_json,
)
from src.models.report_models import RankingReport, RankingScore
from src.services.llm_service import LLMServiceError


def _make_loguru_caplog_sink(caplog_handler: logging.Handler):
    """Return a loguru sink function that forwards records into a stdlib logging handler.

    Loguru sinks receive a loguru ``Message`` (str subclass with a ``.record`` dict).
    We extract the name/level/message and re-emit as a stdlib ``LogRecord`` so pytest
    ``caplog`` (which only hooks into stdlib logging) captures the output.
    """

    def _sink(message):  # message: loguru.Message
        rec = message.record
        level_no = rec["level"].no
        # Map loguru level number -> stdlib level number (they happen to match)
        log_record = logging.LogRecord(
            name=rec["name"],
            level=level_no,
            pathname=str(rec["file"].path),
            lineno=rec["line"],
            msg=rec["message"],
            args=(),
            exc_info=None,
        )
        caplog_handler.emit(log_record)

    return _sink


@pytest.fixture(autouse=True)
def _bridge_loguru_to_caplog(caplog):
    """Sink loguru output into stdlib logging for the duration of each test.

    Without this bridge, loguru.logger.warning() calls never appear in pytest's
    caplog fixture (which only hooks into stdlib logging).
    """
    sink = _make_loguru_caplog_sink(caplog.handler)
    handler_id = _loguru_logger.add(sink, level="WARNING", format="{message}")
    with caplog.at_level(logging.WARNING):
        yield
    _loguru_logger.remove(handler_id)


# ──────────────────────────────────────────────────────────────────────────────
# RNK-R01: Well-formed JSON with dim_scores -> parsed correctly (regression)
# ──────────────────────────────────────────────────────────────────────────────


def test_r01_wellformed_technical_json_dim_scores():
    """Well-formed technical JSON with all 5 dim_scores is parsed fully."""
    raw = json.dumps(
        {
            "score": 9,
            "justification": "Technically sound.",
            "improvements": ["Add version notes."],
            "technical_rank": "Advanced",
            "dimension_scores": {
                "correctness": 9,
                "completeness": 8,
                "technical_accuracy": 10,
                "code_quality": 9,
                "logical_validity": 9,
            },
        }
    )
    result = _parse_review_json(raw, "Technical")
    assert result is not None
    assert result["score"] == 9
    dims = result.get("dimension_scores", {})
    assert dims.get("correctness") == 9
    assert dims.get("completeness") == 8
    assert dims.get("technical_accuracy") == 10


def test_r01_wellformed_pedagogical_json_dim_scores():
    """Well-formed pedagogical JSON with all 6 dim_scores is parsed fully."""
    raw = json.dumps(
        {
            "score": 8,
            "justification": "Good structure.",
            "improvements": ["Add extra resources."],
            "dimension_scores": {
                "sections_structure": 9,
                "cognitive_load": 8,
                "scaffolding_reproducibility": 8,
                "relevance_curriculum": 7,
                "skill_level_awareness": 9,
                "human_language_context": 8,
            },
        }
    )
    result = _parse_review_json(raw, "Pedagogical")
    assert result is not None
    assert result["score"] == 8
    dims = result.get("dimension_scores", {})
    assert dims.get("sections_structure") == 9
    assert dims.get("human_language_context") == 8


# ──────────────────────────────────────────────────────────────────────────────
# RNK-R02: Truncated mid-dim_scores JSON -> score extracted, dim_scores null
# ──────────────────────────────────────────────────────────────────────────────


def test_r02_truncated_mid_dim_scores_score_recovered(caplog):
    """Truncated JSON mid-dim_scores: score is recovered via regex fallback; dim_scores null."""
    # Simulate Haiku response cut off while writing dimension_scores
    raw = '{"score": 7, "justification": "Good enough.", "improvements": [], "dimension_scores": {"correctness": 7, "completeness":'
    with caplog.at_level(logging.WARNING, logger="src.agents.ranking_agent"):
        result = _parse_review_json(raw, "Technical")
    assert result is not None, "Parser must recover score from truncated JSON"
    assert result["score"] == 7
    # dim_scores should NOT be present (truncated partial dict was not fully valid)
    assert result.get("dimension_scores") is None


def test_r02_truncated_returns_no_dim_scores(caplog):
    """Truncated before dimension_scores: score recovered, no dim_scores crash."""
    raw = '{"score": 5, "justification": "Partial'
    with caplog.at_level(logging.WARNING, logger="src.agents.ranking_agent"):
        result = _parse_review_json(raw, "Pedagogical")
    assert result is not None
    assert result["score"] == 5
    assert "dimension_scores" not in (result or {})


# ──────────────────────────────────────────────────────────────────────────────
# RNK-R03: Markdown code fence -> parsed correctly
# ──────────────────────────────────────────────────────────────────────────────


def test_r03_markdown_code_fence_with_dim_scores():
    """Markdown ```json fence is stripped and dim_scores are parsed correctly."""
    payload = {
        "score": 8,
        "justification": "OK.",
        "improvements": [],
        "dimension_scores": {
            "correctness": 8,
            "completeness": 8,
            "technical_accuracy": 8,
            "code_quality": 7,
            "logical_validity": 8,
        },
    }
    raw = f"```json\n{json.dumps(payload)}\n```"
    result = _parse_review_json(raw, "Technical")
    assert result is not None
    assert result["score"] == 8
    dims = result.get("dimension_scores", {})
    assert dims.get("correctness") == 8


def test_r03_plain_code_fence_no_language_tag():
    """Plain ``` fence (no language tag) is also stripped."""
    payload = {"score": 6, "justification": "Fair.", "improvements": []}
    raw = f"```\n{json.dumps(payload)}\n```"
    result = _parse_review_json(raw, "Pedagogical")
    assert result is not None
    assert result["score"] == 6


# ──────────────────────────────────────────────────────────────────────────────
# RNK-R04: Preamble text before JSON -> parsed correctly
# ──────────────────────────────────────────────────────────────────────────────


def test_r04_preamble_before_json_technical():
    """Judge response with preamble text before JSON: JSON is extracted and parsed."""
    payload = {
        "score": 9,
        "justification": "Correct and complete.",
        "improvements": ["Minor: add version pin."],
        "technical_rank": "Advanced",
        "dimension_scores": {
            "correctness": 9,
            "completeness": 9,
            "technical_accuracy": 9,
            "code_quality": 8,
            "logical_validity": 9,
        },
    }
    raw = f"Here is my evaluation of the course:\n\n{json.dumps(payload)}"
    result = _parse_review_json(raw, "Technical")
    assert result is not None, "Preamble before JSON should be stripped"
    assert result["score"] == 9
    dims = result.get("dimension_scores", {})
    assert dims.get("correctness") == 9


def test_r04_preamble_before_json_pedagogical():
    """Pedagogical judge with preamble text: correctly extracted."""
    payload = {
        "score": 7,
        "justification": "Acceptable.",
        "improvements": ["Add conclusion."],
        "dimension_scores": {
            "sections_structure": 7,
            "cognitive_load": 6,
            "scaffolding_reproducibility": 7,
            "relevance_curriculum": 7,
            "skill_level_awareness": 7,
            "human_language_context": 8,
        },
    }
    raw = f"Based on the content I reviewed:\n{json.dumps(payload)}\n"
    result = _parse_review_json(raw, "Pedagogical")
    assert result is not None
    assert result["score"] == 7
    dims = result.get("dimension_scores", {})
    assert dims.get("sections_structure") == 7


# ──────────────────────────────────────────────────────────────────────────────
# RNK-R05: LLM failure during retest preserves dim_scores from prior pass
# ──────────────────────────────────────────────────────────────────────────────


def _make_existing_report(challenge_id: str) -> RankingReport:
    """Helper: build a RankingReport with populated dim_scores (as if from prior pass)."""
    tech = RankingScore(
        score=7,
        persona="Technical",
        justification="Prior pass result.",
        improvements=["Improve completeness."],
        dimension_scores={
            "correctness": 7,
            "completeness": 6,
            "technical_accuracy": 8,
            "code_quality": 7,
            "logical_validity": 7,
        },
    )
    ped = RankingScore(
        score=8,
        persona="Pedagogical",
        justification="Prior pass pedagogical.",
        improvements=[],
        dimension_scores={
            "sections_structure": 8,
            "cognitive_load": 8,
            "scaffolding_reproducibility": 7,
            "relevance_curriculum": 8,
            "skill_level_awareness": 8,
            "human_language_context": 9,
        },
    )
    return RankingReport(
        challenge_id=challenge_id,
        overall_score=7.5,
        pedagogical_review=ped,
        technical_review=tech,
        technical_rank="Intermediate",
    )


def test_r05_retest_technical_failure_preserves_prior_dim_scores(monkeypatch, caplog):
    """When technical retest LLM fails, existing technical dim_scores are preserved."""
    existing = _make_existing_report("test/challenge")

    def _fail(*args, **kwargs):
        raise LLMServiceError(
            "subscription usage window exhausted; reset in 9000s (>5400s max)"
        )

    monkeypatch.setattr(
        "src.agents.ranking_agent.generate_response_with_system",
        _fail,
    )

    with caplog.at_level(logging.WARNING, logger="src.agents.ranking_agent"):
        report = _evaluate_one_challenge_retest(
            challenge_id="test/challenge",
            writeup="# Course content",
            solve_script="print('flag')",
            run_technical=True,
            run_pedagogical=False,
            existing=existing,
        )

    # Technical review should have preserved dim_scores from prior pass
    assert report.technical_review is not None
    assert (
        report.technical_review.dimension_scores is not None
    ), "dim_scores must be preserved from prior pass when retest LLM fails"
    assert report.technical_review.dimension_scores.get("correctness") == 7
    assert "preserving" in caplog.text.lower() or "prior pass" in caplog.text.lower()


def test_r05_retest_pedagogical_failure_preserves_prior_dim_scores(monkeypatch, caplog):
    """When pedagogical retest LLM fails, existing pedagogical dim_scores are preserved."""
    existing = _make_existing_report("test/ped-challenge")

    def _fail(*args, **kwargs):
        raise LLMServiceError("quota exhausted")

    monkeypatch.setattr(
        "src.agents.ranking_agent.generate_response_with_system",
        _fail,
    )

    with caplog.at_level(logging.WARNING, logger="src.agents.ranking_agent"):
        report = _evaluate_one_challenge_retest(
            challenge_id="test/ped-challenge",
            writeup="# Course content",
            solve_script="",
            run_technical=False,
            run_pedagogical=True,
            existing=existing,
        )

    assert report.pedagogical_review is not None
    assert (
        report.pedagogical_review.dimension_scores is not None
    ), "Pedagogical dim_scores must be preserved from prior pass on retest failure"
    assert report.pedagogical_review.dimension_scores.get("sections_structure") == 8


# ──────────────────────────────────────────────────────────────────────────────
# RNK-R06: Different/flat schema -> handled gracefully
# ──────────────────────────────────────────────────────────────────────────────


def test_r06_unknown_dimension_keys_dropped_with_warning(caplog):
    """Unknown dimension keys are silently dropped; known keys are retained."""
    raw = json.dumps(
        {
            "score": 8,
            "justification": "Good.",
            "improvements": [],
            "dimension_scores": {
                "correctness": 8,
                "completeness": 7,
                "unknown_extra_key": 5,  # should be dropped
            },
        }
    )
    with caplog.at_level(logging.DEBUG, logger="src.agents.ranking_agent"):
        result = _parse_review_json(raw, "Technical")
    assert result is not None
    dims = result.get("dimension_scores", {})
    assert dims.get("correctness") == 8
    assert dims.get("completeness") == 7
    assert "unknown_extra_key" not in dims


def test_r06_all_unknown_keys_triggers_warning(caplog):
    """If ALL dimension keys are unknown, dim_scores is dropped and a warning is logged."""
    raw = json.dumps(
        {
            "score": 7,
            "justification": "Flat schema.",
            "improvements": [],
            "dimension_scores": {
                "flat_key_a": 7,  # not in _TECHNICAL_DIMENSIONS
                "flat_key_b": 8,
            },
        }
    )
    with caplog.at_level(logging.WARNING, logger="src.agents.ranking_agent"):
        result = _parse_review_json(raw, "Technical")
    assert result is not None
    assert result.get("dimension_scores") is None
    assert any(
        "no keys matched" in r.message.lower() or "unknown" in r.message.lower()
        for r in caplog.records
        if r.levelno >= logging.WARNING
    ), "A warning should be logged when all dimension keys are unknown"


def test_r06_dim_scores_absent_logs_warning(caplog):
    """Valid JSON without dimension_scores key should log a warning."""
    raw = json.dumps(
        {
            "score": 8,
            "justification": "Good.",
            "improvements": ["Step 3 needs more detail."],
        }
    )
    with caplog.at_level(logging.WARNING, logger="src.agents.ranking_agent"):
        result = _parse_review_json(raw, "Technical")
    assert result is not None
    assert result["score"] == 8
    assert result.get("dimension_scores") is None
    assert any(
        "dimension_scores" in r.message.lower()
        for r in caplog.records
        if r.levelno >= logging.WARNING
    ), "A warning should be logged when dimension_scores key is absent"
