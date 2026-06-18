"""Tests for the Granular Grading System.

Verifies:
- RankingReport.dimension_scores aggregates from both personas
- _aggregate_dimension_scores merges and averages correctly
- Weighted aggregation via DIMENSION_WEIGHTS
- AgentState.challenge_dimension_scores populated by run_ranking_agent
- Backward compatibility: None dimension_scores handled gracefully
"""

from unittest.mock import patch

from src.agents.ranking_agent import _aggregate_dimension_scores
from src.core.state import AgentState
from src.models.report_models import RankingReport, RankingScore

# ---------------------------------------------------------------------------
# _aggregate_dimension_scores unit tests
# ---------------------------------------------------------------------------


def test_aggregate_merges_non_overlapping_dims():
    tech = {
        "correctness": 9,
        "completeness": 8,
        "technical_accuracy": 9,
        "code_quality": 8,
        "logical_validity": 9,
    }
    ped = {
        "sections_structure": 8,
        "cognitive_load": 7,
        "scaffolding_reproducibility": 9,
        "relevance_curriculum": 8,
        "skill_level_awareness": 9,
        "human_language_context": 8,
    }
    result = _aggregate_dimension_scores(tech, ped)
    assert result is not None
    # All 11 dimensions present
    assert len(result) == 11
    assert result["correctness"] == 9.0
    assert result["sections_structure"] == 8.0


def test_aggregate_averages_shared_dims():
    tech = {"correctness": 8}
    ped = {"correctness": 6}
    result = _aggregate_dimension_scores(tech, ped)
    assert result is not None
    assert result["correctness"] == 7.0


def test_aggregate_returns_none_when_both_none():
    assert _aggregate_dimension_scores(None, None) is None


def test_aggregate_returns_none_when_both_empty():
    assert _aggregate_dimension_scores({}, {}) is None


def test_aggregate_one_side_none():
    tech = {"correctness": 9, "completeness": 8}
    result = _aggregate_dimension_scores(tech, None)
    assert result is not None
    assert result["correctness"] == 9.0
    assert result["completeness"] == 8.0

    result2 = _aggregate_dimension_scores(None, {"sections_structure": 7})
    assert result2 is not None
    assert result2["sections_structure"] == 7.0


def test_aggregate_with_weights():
    tech = {"correctness": 10, "completeness": 4}
    ped = {"sections_structure": 8}
    weights = {"correctness": 2.0, "completeness": 1.0}
    result = _aggregate_dimension_scores(tech, ped, weights=weights)
    assert result is not None
    # Weight doesn't change single-source dims, just confirms they're float
    assert result["correctness"] == 10.0
    assert result["completeness"] == 4.0
    assert result["sections_structure"] == 8.0


def test_aggregate_with_weights_averaging():
    # Shared dim with weights: weight applied per value, then averaged
    tech = {"shared": 10}
    ped = {"shared": 6}
    weights = {"shared": 2.0}
    result = _aggregate_dimension_scores(tech, ped, weights=weights)
    assert result is not None
    # w*10 + w*6 / (2 * w) = (10+6)/2 = 8.0
    assert result["shared"] == 8.0


# ---------------------------------------------------------------------------
# RankingReport.dimension_scores field
# ---------------------------------------------------------------------------


def _make_report(
    tech_dims=None,
    ped_dims=None,
    agg_dims=None,
    challenge_id="test_001",
    overall=8.5,
):
    tech = RankingScore(
        score=9,
        persona="Technical",
        justification="T",
        improvements=[],
        dimension_scores=tech_dims,
    )
    ped = RankingScore(
        score=8,
        persona="Pedagogical",
        justification="P",
        improvements=[],
        dimension_scores=ped_dims,
    )
    return RankingReport(
        challenge_id=challenge_id,
        overall_score=overall,
        technical_review=tech,
        pedagogical_review=ped,
        technical_rank="Intermediate",
        dimension_scores=agg_dims,
    )


def test_ranking_report_has_dimension_scores_field():
    agg = {"correctness": 9.0, "sections_structure": 8.0}
    report = _make_report(agg_dims=agg)
    assert report.dimension_scores is not None
    assert report.dimension_scores["correctness"] == 9.0
    assert report.dimension_scores["sections_structure"] == 8.0


def test_ranking_report_dimension_scores_optional_none():
    report = _make_report()
    assert report.dimension_scores is None


def test_ranking_report_dimension_scores_json_serializable():
    import json

    agg = {"correctness": 9.0, "completeness": 8.5}
    report = _make_report(agg_dims=agg)
    data = report.model_dump()
    # Should serialize without error
    json_str = json.dumps(data)
    loaded = json.loads(json_str)
    assert loaded["dimension_scores"]["correctness"] == 9.0


# ---------------------------------------------------------------------------
# AgentState.challenge_dimension_scores field
# ---------------------------------------------------------------------------


def test_agent_state_has_challenge_dimension_scores():
    state = AgentState()
    assert state.challenge_dimension_scores is None


def test_agent_state_challenge_dimension_scores_settable():
    state = AgentState()
    dims = {"challenge_001": {"correctness": 9.0, "sections_structure": 8.0}}
    state.challenge_dimension_scores = dims
    assert state.challenge_dimension_scores["challenge_001"]["correctness"] == 9.0


# ---------------------------------------------------------------------------
# run_ranking_agent populates challenge_dimension_scores
# ---------------------------------------------------------------------------


def _make_mock_report(challenge_id: str, with_dims: bool) -> RankingReport:
    tech_dims = {"correctness": 9, "completeness": 8} if with_dims else None
    ped_dims = {"sections_structure": 8, "cognitive_load": 7} if with_dims else None
    agg = _aggregate_dimension_scores(tech_dims, ped_dims)
    return RankingReport(
        challenge_id=challenge_id,
        overall_score=8.5,
        technical_review=RankingScore(
            score=9,
            persona="Technical",
            justification="T",
            improvements=[],
            dimension_scores=tech_dims,
        ),
        pedagogical_review=RankingScore(
            score=8,
            persona="Pedagogical",
            justification="P",
            improvements=[],
            dimension_scores=ped_dims,
        ),
        technical_rank="Intermediate",
        dimension_scores=agg,
    )


def test_run_ranking_agent_populates_challenge_dimension_scores():
    from src.agents.ranking_agent import run_ranking_agent

    state = AgentState(
        generated_courses={"ch_001": "some course content here"},
    )
    mock_report = _make_mock_report("ch_001", with_dims=True)

    with patch(
        "src.agents.ranking_agent._evaluate_one_challenge", return_value=mock_report
    ):
        result = run_ranking_agent(state)

    assert result.challenge_dimension_scores is not None
    assert "ch_001" in result.challenge_dimension_scores
    dims = result.challenge_dimension_scores["ch_001"]
    assert "correctness" in dims
    assert "sections_structure" in dims


def test_run_ranking_agent_no_dims_gives_none_challenge_dimension_scores():
    from src.agents.ranking_agent import run_ranking_agent

    state = AgentState(
        generated_courses={"ch_002": "some course content here"},
    )
    mock_report = _make_mock_report("ch_002", with_dims=False)

    with patch(
        "src.agents.ranking_agent._evaluate_one_challenge", return_value=mock_report
    ):
        result = run_ranking_agent(state)

    # challenge_dimension_scores should be None (no dims in any report)
    assert result.challenge_dimension_scores is None
