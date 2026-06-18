"""Tests for HITL agent: display, action parsing, and state transitions.

Covers:
- print_hitl_summary includes challenge_id, scores, routing cause
- approve_all sets hitl_approved=True (routes to END)
- approve <ids> partial removes accepted challenges from re-queue
- abort triggers SystemExit
- edit_retry injects operator hint into human_feedback
"""

from unittest.mock import patch

import pytest

from src.agents.hitl_agent import (
    _build_review_payload,
    _parse_resume_value_extended,
    print_hitl_summary,
    run_hitl_agent,
)
from src.core.state import AgentState
from src.models.report_models import RankingReport, RankingScore

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_score(score: int, persona: str) -> RankingScore:
    return RankingScore(
        score=score,
        persona=persona,
        justification="test justification",
        improvements=[],
    )


@pytest.fixture
def two_challenge_state() -> AgentState:
    """State with two challenges: one low-score, one borderline."""
    reports = [
        RankingReport(
            challenge_id="web/sqli",
            overall_score=5.5,
            technical_review=_make_score(5, "Technical"),
            pedagogical_review=_make_score(6, "Pedagogical"),
            technical_rank="Beginner",
        ),
        RankingReport(
            challenge_id="crypto/rsa",
            overall_score=7.0,
            technical_review=_make_score(7, "Technical"),
            pedagogical_review=_make_score(7, "Pedagogical"),
            technical_rank="Intermediate",
        ),
    ]
    return AgentState(
        challenge_ids=["web/sqli", "crypto/rsa"],
        ranking_reports=reports,
        iteration_count=0,
        max_hitl_iterations=3,
        refinement_count=0,
    )


@pytest.fixture
def hitl_payload(two_challenge_state: AgentState) -> dict:
    return _build_review_payload(two_challenge_state)


# ---------------------------------------------------------------------------
# Test 1: Display includes challenge_id, scores, routing cause
# ---------------------------------------------------------------------------


def test_print_hitl_summary_shows_challenge_scores_and_routing_cause(
    capsys, hitl_payload
):
    """print_hitl_summary must output challenge_id, scores, and routing cause."""
    print_hitl_summary(hitl_payload)
    out = capsys.readouterr().out

    assert "web/sqli" in out
    assert "crypto/rsa" in out
    # Technical and pedagogical scores present
    assert "5" in out  # web/sqli technical score
    assert "6" in out  # web/sqli pedagogical score
    # Routing cause line
    assert "quality failure" in out


def test_print_hitl_summary_max_rounds_cause(capsys, two_challenge_state):
    """Routing cause shows 'max refinements' when refinement_count >= MAX_REFINEMENT_ROUNDS."""
    from unittest.mock import patch as _patch

    with _patch("src.agents.hitl_agent.app_settings") as mock_settings:
        mock_settings.MAX_REFINEMENT_ROUNDS = 5
        mock_settings.RANKING_PASS_THRESHOLD = 9.0
        state = AgentState(
            challenge_ids=["web/sqli"],
            ranking_reports=[
                RankingReport(
                    challenge_id="web/sqli",
                    overall_score=7.0,
                    technical_review=_make_score(7, "Technical"),
                    pedagogical_review=_make_score(7, "Pedagogical"),
                    technical_rank="Beginner",
                )
            ],
            refinement_count=5,
            iteration_count=0,
            max_hitl_iterations=3,
        )
        payload = _build_review_payload(state)

    print_hitl_summary(payload)
    out = capsys.readouterr().out
    assert "max refinements reached" in out
    assert "round 5" in out


# ---------------------------------------------------------------------------
# Test 2: approve_all routes to END (hitl_approved=True)
# ---------------------------------------------------------------------------


def test_parse_resume_approve_all_sets_action():
    parsed = _parse_resume_value_extended({"action": "approve_all"})
    assert parsed["action"] == "approve_all"


def test_run_hitl_approve_all_sets_hitl_approved(two_challenge_state):
    """approve_all resume → hitl_approved=True so route_after_hitl goes to END."""
    with (
        patch("src.agents.hitl_agent._HAS_INTERRUPT", True),
        patch(
            "src.agents.hitl_agent.interrupt", return_value={"action": "approve_all"}
        ),
    ):
        result = run_hitl_agent(two_challenge_state)

    assert result.hitl_approved is True
    assert result.current_agent == "hitl"
    assert result.iteration_count == 1


# ---------------------------------------------------------------------------
# Test 3: approve <ids> partial — accepted challenges removed from re-queue
# ---------------------------------------------------------------------------


def test_parse_resume_approve_ids():
    parsed = _parse_resume_value_extended(
        {"action": "approve_ids", "ids": ["web/sqli"]}
    )
    assert parsed["action"] == "approve_ids"
    assert "web/sqli" in parsed["ids"]


def test_run_hitl_approve_partial_subsets_remaining(two_challenge_state):
    """Approving web/sqli only → crypto/rsa in content_generation_subset_ids, not web/sqli."""
    with (
        patch("src.agents.hitl_agent._HAS_INTERRUPT", True),
        patch(
            "src.agents.hitl_agent.interrupt",
            return_value={"action": "approve_ids", "ids": ["web/sqli"]},
        ),
    ):
        result = run_hitl_agent(two_challenge_state)

    assert result.hitl_approved is False
    subset = result.content_generation_subset_ids or []
    assert "web/sqli" not in subset
    assert "crypto/rsa" in subset
    # ranking_subset_ids mirrors content subset
    ranking_subset = result.ranking_subset_ids or []
    assert "crypto/rsa" in ranking_subset
    assert "web/sqli" not in ranking_subset


# ---------------------------------------------------------------------------
# Test 4: abort triggers clean stop (SystemExit)
# ---------------------------------------------------------------------------


def test_parse_resume_abort_action():
    parsed = _parse_resume_value_extended({"action": "abort"})
    assert parsed["action"] == "abort"


def test_run_hitl_abort_raises_system_exit(two_challenge_state):
    """abort resume → run_hitl_agent raises SystemExit(0)."""
    with (
        patch("src.agents.hitl_agent._HAS_INTERRUPT", True),
        patch(
            "src.agents.hitl_agent.interrupt",
            return_value={"action": "abort"},
        ),
    ):
        with pytest.raises(SystemExit) as exc_info:
            run_hitl_agent(two_challenge_state)

    assert exc_info.value.code == 0


# ---------------------------------------------------------------------------
# Test 5: edit_retry injects hint text into human_feedback
# ---------------------------------------------------------------------------


def test_parse_resume_edit_retry_extracts_hint():
    parsed = _parse_resume_value_extended(
        {"action": "edit_retry", "hint": "focus on XSS mitigation"}
    )
    assert parsed["action"] == "edit_retry"
    assert parsed["hint"] == "focus on XSS mitigation"


def test_run_hitl_edit_retry_injects_hint_for_low_score_challenges(two_challenge_state):
    """edit_retry → hint injected as 'Operator hint: ...' for challenges below threshold."""
    hint_text = "focus on XSS mitigation steps"
    with (
        patch("src.agents.hitl_agent._HAS_INTERRUPT", True),
        patch(
            "src.agents.hitl_agent.interrupt",
            return_value={"action": "edit_retry", "hint": hint_text},
        ),
        patch("src.agents.hitl_agent.app_settings") as mock_s,
    ):
        mock_s.MAX_REFINEMENT_ROUNDS = 5
        mock_s.RANKING_PASS_THRESHOLD = 9.0
        result = run_hitl_agent(two_challenge_state)

    assert result.hitl_approved is False
    # Both challenges score below 9.0 — both should receive the hint
    found = False
    for cid, items in result.human_feedback.items():
        for item in items:
            if hint_text in item:
                found = True
                break
    assert (
        found
    ), f"Hint '{hint_text}' not found in human_feedback: {result.human_feedback}"


def test_run_hitl_edit_retry_empty_hint_does_not_pollute_feedback(two_challenge_state):
    """edit_retry with empty hint must not write empty entries to human_feedback."""
    with (
        patch("src.agents.hitl_agent._HAS_INTERRUPT", True),
        patch(
            "src.agents.hitl_agent.interrupt",
            return_value={"action": "edit_retry", "hint": ""},
        ),
    ):
        result = run_hitl_agent(two_challenge_state)

    assert result.hitl_approved is False
    # No feedback items added for empty hint
    for items in result.human_feedback.values():
        assert not any("Operator hint" in i for i in items)


# ---------------------------------------------------------------------------
# Legacy format backward-compatibility
# ---------------------------------------------------------------------------


def test_run_hitl_legacy_approved_true(two_challenge_state):
    with (
        patch("src.agents.hitl_agent._HAS_INTERRUPT", True),
        patch(
            "src.agents.hitl_agent.interrupt",
            return_value={"approved": True},
        ),
    ):
        result = run_hitl_agent(two_challenge_state)

    assert result.hitl_approved is True


def test_run_hitl_legacy_approved_false_with_feedback(two_challenge_state):
    with (
        patch("src.agents.hitl_agent._HAS_INTERRUPT", True),
        patch(
            "src.agents.hitl_agent.interrupt",
            return_value={
                "approved": False,
                "human_feedback": {"web/sqli": ["Check for second-order injection"]},
            },
        ),
    ):
        result = run_hitl_agent(two_challenge_state)

    assert result.hitl_approved is False
    assert "Check for second-order injection" in result.human_feedback.get(
        "web/sqli", []
    )
