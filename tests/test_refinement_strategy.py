"""Tests for Strategy After N Refinements.

Verifies:
- _route_ranking_decision routes to correct destination for each strategy when
  refinement_count == MAX_REFINEMENT_ROUNDS.
- _build_review_payload includes "Max refinement rounds reached" note when routed
  to HITL because max rounds were reached.
"""

from unittest.mock import MagicMock, patch

from langgraph.graph import END

from src.agents.hitl_agent import _build_review_payload
from src.core.graph import _route_ranking_decision
from src.core.state import AgentState
from src.models.report_models import RankingReport, RankingScore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_score(score: int, persona: str = "Technical") -> RankingScore:
    return RankingScore(score=score, persona=persona, justification="test")


def _make_report(
    challenge_id: str, tech: int, ped: int, overall: float
) -> RankingReport:
    return RankingReport(
        challenge_id=challenge_id,
        overall_score=overall,
        technical_review=_make_score(tech, "Technical"),
        pedagogical_review=_make_score(ped, "Pedagogical"),
        technical_rank="Intermediate",
    )


def _state_at_max(reports: list, refinement_count: int = 5) -> AgentState:
    """State where refinement_count == MAX_REFINEMENT_ROUNDS with given reports."""
    return AgentState(
        challenge_ids=[r.challenge_id for r in reports],
        ranking_reports=reports,
        refinement_count=refinement_count,
    )


def _mock_settings(
    strategy: str, max_rounds: int = 5, threshold: float = 9.0, soft: float = 7.0
) -> MagicMock:
    m = MagicMock()
    m.RANKING_PASS_THRESHOLD = threshold
    m.RANKING_TECHNICAL_THRESHOLD = threshold  # K2: per-dim thresholds (Wave 3)
    m.RANKING_PEDAGOGICAL_THRESHOLD = threshold
    m.MAX_REFINEMENT_ROUNDS = max_rounds
    m.MAX_REFINEMENT_STRATEGY = strategy
    m.SOFT_ACCEPT_THRESHOLD = soft
    return m


# ---------------------------------------------------------------------------
# Strategy: hitl (default)
# ---------------------------------------------------------------------------


def test_hitl_strategy_routes_to_hitl_at_max_rounds():
    """When strategy=hitl and max rounds reached, route to HITL."""
    report = _make_report("chal_01", tech=7, ped=7, overall=7.0)
    state = _state_at_max([report], refinement_count=5)

    with patch("src.core.graph.app_settings", _mock_settings("hitl")):
        result = _route_ranking_decision(state)

    assert result == "hitl"


def test_hitl_strategy_still_refines_before_max_rounds():
    """When strategy=hitl but rounds remaining, still route to refinement_step."""
    report = _make_report("chal_01", tech=7, ped=7, overall=7.0)
    state = _state_at_max([report], refinement_count=2)  # only round 2 of 5

    with patch("src.core.graph.app_settings", _mock_settings("hitl")):
        result = _route_ranking_decision(state)

    assert result == "refinement_step"


def test_hitl_strategy_routes_to_end_when_all_scores_pass():
    """When all scores pass the ranking threshold, route to END regardless of strategy."""
    report = _make_report("chal_01", tech=10, ped=10, overall=10.0)
    state = _state_at_max([report], refinement_count=5)

    with patch("src.core.graph.app_settings", _mock_settings("hitl")):
        result = _route_ranking_decision(state)

    assert result == END


def test_empty_ranking_reports_routes_to_end():
    """No ranking reports → route to END."""
    state = AgentState(challenge_ids=[], ranking_reports=[], refinement_count=5)

    with patch("src.core.graph.app_settings", _mock_settings("hitl")):
        result = _route_ranking_decision(state)

    assert result == END


# ---------------------------------------------------------------------------
# Strategy: accept_best
# ---------------------------------------------------------------------------


def test_accept_best_routes_to_end_at_max_rounds():
    """When strategy=accept_best and max rounds reached, route to END."""
    report = _make_report("chal_01", tech=6, ped=6, overall=6.0)
    state = _state_at_max([report], refinement_count=5)

    with patch("src.core.graph.app_settings", _mock_settings("accept_best")):
        result = _route_ranking_decision(state)

    assert result == END


def test_accept_best_routes_to_end_even_with_very_low_score():
    """accept_best accepts any score, even overall=1.0."""
    report = _make_report("chal_01", tech=1, ped=1, overall=1.0)
    state = _state_at_max([report], refinement_count=5)

    with patch("src.core.graph.app_settings", _mock_settings("accept_best")):
        result = _route_ranking_decision(state)

    assert result == END


def test_accept_best_still_refines_before_max_rounds():
    """accept_best still routes to refinement_step when rounds remain."""
    report = _make_report("chal_01", tech=5, ped=5, overall=5.0)
    state = _state_at_max([report], refinement_count=3)

    with patch("src.core.graph.app_settings", _mock_settings("accept_best")):
        result = _route_ranking_decision(state)

    assert result == "refinement_step"


# ---------------------------------------------------------------------------
# Strategy: soft_accept
# ---------------------------------------------------------------------------


def test_soft_accept_routes_to_end_when_score_above_threshold():
    """When strategy=soft_accept and overall_score >= SOFT_ACCEPT_THRESHOLD, route to END."""
    report = _make_report("chal_01", tech=7, ped=8, overall=7.5)
    state = _state_at_max([report], refinement_count=5)

    with patch("src.core.graph.app_settings", _mock_settings("soft_accept", soft=7.0)):
        result = _route_ranking_decision(state)

    assert result == END


def test_soft_accept_routes_to_end_at_exact_threshold():
    """When overall_score == SOFT_ACCEPT_THRESHOLD, route to END (inclusive boundary)."""
    report = _make_report("chal_01", tech=7, ped=7, overall=7.0)
    state = _state_at_max([report], refinement_count=5)

    with patch("src.core.graph.app_settings", _mock_settings("soft_accept", soft=7.0)):
        result = _route_ranking_decision(state)

    assert result == END


def test_soft_accept_routes_to_hitl_when_score_below_threshold():
    """When strategy=soft_accept and overall_score < SOFT_ACCEPT_THRESHOLD, route to HITL."""
    report = _make_report("chal_01", tech=5, ped=6, overall=5.5)
    state = _state_at_max([report], refinement_count=5)

    with patch("src.core.graph.app_settings", _mock_settings("soft_accept", soft=7.0)):
        result = _route_ranking_decision(state)

    assert result == "hitl"


def test_soft_accept_any_failing_challenge_routes_to_hitl():
    """With multiple reports, one below threshold means HITL for all."""
    good = _make_report("chal_01", tech=8, ped=8, overall=8.0)
    bad = _make_report("chal_02", tech=5, ped=5, overall=5.0)
    # Put the good one first — code still finds chal_02 fails
    state = _state_at_max([good, bad], refinement_count=5)

    with patch("src.core.graph.app_settings", _mock_settings("soft_accept", soft=7.0)):
        result = _route_ranking_decision(state)

    assert result == "hitl"


def test_soft_accept_all_above_threshold_routes_to_end():
    """With multiple reports all above soft threshold, route to END."""
    r1 = _make_report("chal_01", tech=8, ped=8, overall=8.0)
    r2 = _make_report("chal_02", tech=7, ped=8, overall=7.5)
    state = _state_at_max([r1, r2], refinement_count=5)

    with patch("src.core.graph.app_settings", _mock_settings("soft_accept", soft=7.0)):
        result = _route_ranking_decision(state)

    assert result == END


# ---------------------------------------------------------------------------
# HITL payload: max_rounds_note
# ---------------------------------------------------------------------------


def test_hitl_payload_includes_max_rounds_note_at_max_rounds():
    """_build_review_payload includes max_rounds_note when refinement_count >= MAX_REFINEMENT_ROUNDS."""
    report = _make_report("chal_01", tech=7, ped=7, overall=7.0)
    state = AgentState(
        challenge_ids=["chal_01"],
        ranking_reports=[report],
        refinement_count=5,
    )

    with patch("src.agents.hitl_agent.app_settings") as mock_settings:
        mock_settings.MAX_REFINEMENT_ROUNDS = 5
        payload = _build_review_payload(state)

    assert "max_rounds_note" in payload
    note = payload["max_rounds_note"]
    assert "Max refinement rounds reached" in note
    assert "round 5" in note
    assert "Best score" in note


def test_hitl_payload_omits_max_rounds_note_before_max_rounds():
    """_build_review_payload omits max_rounds_note when refinement_count < MAX_REFINEMENT_ROUNDS."""
    report = _make_report("chal_01", tech=7, ped=7, overall=7.0)
    state = AgentState(
        challenge_ids=["chal_01"],
        ranking_reports=[report],
        refinement_count=2,
    )

    with patch("src.agents.hitl_agent.app_settings") as mock_settings:
        mock_settings.MAX_REFINEMENT_ROUNDS = 5
        payload = _build_review_payload(state)

    assert "max_rounds_note" not in payload


def test_hitl_payload_best_score_reflects_max_overall():
    """Best score in the note is the max overall_score across all ranking_reports."""
    r1 = _make_report("chal_01", tech=6, ped=7, overall=6.5)
    r2 = _make_report("chal_02", tech=8, ped=7, overall=7.5)
    state = AgentState(
        challenge_ids=["chal_01", "chal_02"],
        ranking_reports=[r1, r2],
        refinement_count=5,
    )

    with patch("src.agents.hitl_agent.app_settings") as mock_settings:
        mock_settings.MAX_REFINEMENT_ROUNDS = 5
        payload = _build_review_payload(state)

    assert "7.5" in payload["max_rounds_note"]


def test_hitl_payload_max_rounds_note_no_reports():
    """When at max rounds but no ranking_reports, best score defaults to 0.0."""
    state = AgentState(
        challenge_ids=[],
        ranking_reports=[],
        refinement_count=5,
    )

    with patch("src.agents.hitl_agent.app_settings") as mock_settings:
        mock_settings.MAX_REFINEMENT_ROUNDS = 5
        payload = _build_review_payload(state)

    assert "max_rounds_note" in payload
    assert "0.0" in payload["max_rounds_note"]
