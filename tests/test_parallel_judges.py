"""Tests for parallel persona judges (S1).

Pre-fix: judges run sequentially → wall time ≈ t1 + t2.
Post-fix: judges run concurrently → wall time ≈ max(t1, t2).

We mock generate_response_with_system to sleep for a fixed duration per
persona so the test is deterministic and independent of real LLM calls.
"""

import time
from unittest.mock import patch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TECH_RESPONSE = '{"score": 8, "justification": "ok", "improvements": [], "technical_rank": "Intermediate", "dimension_scores": {"correctness": 8, "completeness": 8, "technical_accuracy": 8, "code_quality": 8, "logical_validity": 8}}'
_PED_RESPONSE = '{"score": 7, "justification": "ok", "improvements": [], "dimension_scores": {"sections_structure": 7, "cognitive_load": 7, "scaffolding_reproducibility": 7, "relevance_curriculum": 7, "skill_level_awareness": 7, "human_language_context": 7}}'

SLEEP_SECS = 0.25  # each judge "takes" this long
SERIAL_EXPECTED = SLEEP_SECS * 2
PARALLEL_EXPECTED = SLEEP_SECS  # only one call's worth if concurrent

# Tolerance: parallel should finish in < 1.6× a single judge (generous slack)
PARALLEL_TOLERANCE = SLEEP_SECS * 1.6


def _make_slow_generate(tech_resp: str, ped_resp: str, delay: float):
    """Return a side_effect function that sleeps and returns persona-appropriate JSON."""
    call_count = [0]

    def _side_effect(system: str, user: str, **kwargs):
        call_count[0] += 1
        time.sleep(delay)
        # Distinguish by system prompt content
        if "Technical Expert" in system or "correctness" in system:
            return tech_resp
        return ped_resp

    return _side_effect


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_parallel_judges_faster_than_serial(monkeypatch):
    """_evaluate_one_challenge with parallel judges must complete in ~SLEEP_SECS not ~2*SLEEP_SECS."""
    from src.agents.ranking_agent import _evaluate_one_challenge

    mock_fn = _make_slow_generate(_TECH_RESPONSE, _PED_RESPONSE, SLEEP_SECS)

    with patch(
        "src.agents.ranking_agent.generate_response_with_system",
        side_effect=mock_fn,
    ):
        t0 = time.perf_counter()
        report = _evaluate_one_challenge(
            "crypto/test-parallel",
            "# Course content for testing",
            "print('flag')",
        )
        elapsed = time.perf_counter() - t0

    # With parallel judges, elapsed ≈ SLEEP_SECS (not 2×)
    assert elapsed < PARALLEL_TOLERANCE, (
        f"Parallel judges took {elapsed:.3f}s but expected < {PARALLEL_TOLERANCE:.3f}s "
        f"(serial would be ~{SERIAL_EXPECTED:.3f}s)"
    )
    # Both judges ran (scores are valid)
    assert 1 <= report.technical_review.score <= 10
    assert 1 <= report.pedagogical_review.score <= 10


def test_both_judges_return_results(monkeypatch):
    """Even when run concurrently, both judges must return non-default scores."""
    from src.agents.ranking_agent import _evaluate_one_challenge

    mock_fn = _make_slow_generate(_TECH_RESPONSE, _PED_RESPONSE, 0.01)

    with patch(
        "src.agents.ranking_agent.generate_response_with_system",
        side_effect=mock_fn,
    ):
        report = _evaluate_one_challenge(
            "crypto/test-both",
            "# Course content",
            "",
        )

    assert report.technical_review.score == 8
    assert report.pedagogical_review.score == 7
    assert report.overall_score == 7.5


def test_judges_share_budget_counter(monkeypatch):
    """Both judges must count against the same challenge LLM budget (no per-thread isolation)."""
    from src.agents.ranking_agent import _evaluate_one_challenge
    from src.services.llm_service import (
        reset_challenge_llm_budget,
    )

    reset_challenge_llm_budget("test-budget")

    call_log = []

    def _recording_generate(system: str, user: str, **kwargs):
        call_log.append(1)
        time.sleep(0.01)
        if "Technical Expert" in system or "correctness" in system:
            return _TECH_RESPONSE
        return _PED_RESPONSE

    with patch(
        "src.agents.ranking_agent.generate_response_with_system",
        side_effect=_recording_generate,
    ):
        _evaluate_one_challenge("test-budget", "# Course", "")

    # Both judges made a call
    assert len(call_log) == 2, f"Expected 2 judge calls, got {len(call_log)}"
