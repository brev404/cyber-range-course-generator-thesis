"""Tests for Wave 3 scoring improvements: K1 (weights), K2 (per-dim thresholds), K3 (policy).

Covers:
- K1: Weighted overall_score with configurable persona weights
- K2: Per-dimension pass thresholds for technical + pedagogical
- K3: SCORING_POLICY enum (mean / min / weighted) + CLI wiring
- Sensitivity analysis: overall_scores dict always emitted in every RankingReport
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from src.agents.ranking_agent import compute_overall_score
from src.config.settings import Settings
from src.core.state import AgentState
from src.models.report_models import RankingReport, RankingScore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_settings(**kwargs: Any) -> Settings:
    """Build a Settings instance with overrides from kwargs."""
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    for k, v in kwargs.items():
        object.__setattr__(s, k, v)
    return s


def _make_ranking_score(score: int, persona: str = "Technical") -> RankingScore:
    return RankingScore(
        score=score,
        persona=persona,
        justification="test",
        improvements=[],
        dimension_scores=None,
    )


def _make_report(
    challenge_id: str = "ch1",
    tech: int = 9,
    ped: int = 9,
    overall: float = 9.0,
    overall_scores: dict | None = None,
) -> RankingReport:
    return RankingReport(
        challenge_id=challenge_id,
        overall_score=overall,
        technical_review=_make_ranking_score(tech, "Technical"),
        pedagogical_review=_make_ranking_score(ped, "Pedagogical"),
        technical_rank="Intermediate",
        overall_scores=overall_scores,
    )


# ---------------------------------------------------------------------------
# K1: Weighted overall_score
# ---------------------------------------------------------------------------


def test_weighted_overall_score():
    """tech=8 ped=6 w_t=0.55 w_p=0.45 → overall=7.1"""
    cfg = _make_settings(
        SCORING_POLICY="weighted",
        SCORING_TECHNICAL_WEIGHT=0.55,
        SCORING_PEDAGOGICAL_WEIGHT=0.45,
    )
    active, _ = compute_overall_score(8.0, 6.0, cfg)
    assert active == pytest.approx(7.1, abs=0.05)


def test_default_weights_equal_mean():
    """w_t=0.5 w_p=0.5 weighted == mean."""
    cfg = _make_settings(
        SCORING_POLICY="weighted",
        SCORING_TECHNICAL_WEIGHT=0.5,
        SCORING_PEDAGOGICAL_WEIGHT=0.5,
    )
    active, _ = compute_overall_score(8.0, 6.0, cfg)
    assert active == pytest.approx((8.0 + 6.0) / 2.0, abs=0.05)


def test_weights_normalized():
    """Unnormalised weights (0.55, 0.45) sum to 1.0 automatically."""
    cfg = _make_settings(
        SCORING_POLICY="weighted",
        SCORING_TECHNICAL_WEIGHT=55.0,
        SCORING_PEDAGOGICAL_WEIGHT=45.0,
    )
    active, _ = compute_overall_score(8.0, 6.0, cfg)
    # After normalisation: w_t=0.55, w_p=0.45
    assert active == pytest.approx(8.0 * 0.55 + 6.0 * 0.45, abs=0.05)


# ---------------------------------------------------------------------------
# K2: Per-dimension thresholds
# ---------------------------------------------------------------------------


def _route(state: AgentState) -> str:
    """Import and call the routing function directly."""
    from src.core.graph import _route_ranking_decision

    return _route_ranking_decision(state)


def _state_with_reports(*reports: RankingReport) -> AgentState:
    return AgentState(
        challenges=[],
        challenge_ids=[r.challenge_id for r in reports],
        ranking_reports=list(reports),
    )


def test_per_dimension_threshold_pass():
    """tech=9 ped=9, both thresholds 9.0 → END (pass)."""
    report = _make_report(tech=9, ped=9, overall=9.0)
    state = _state_with_reports(report)
    with patch(
        "src.core.graph.app_settings",
        _make_settings(
            RANKING_PASS_THRESHOLD=9.0,
            RANKING_TECHNICAL_THRESHOLD=9.0,
            RANKING_PEDAGOGICAL_THRESHOLD=9.0,
            MAX_REFINEMENT_ROUNDS=5,
            MAX_REFINEMENT_STRATEGY="hitl",
            SOFT_ACCEPT_THRESHOLD=7.0,
        ),
    ):
        result = _route(state)
    assert result == "END" or result == "__end__"


def test_per_dimension_threshold_fail_tech():
    """tech=8 ped=9, tech_threshold=9 → fail even though ped passes."""
    # overall would be 8.5 which is < 9.0 threshold anyway in current default
    # So we lower RANKING_PASS_THRESHOLD to 8.0 to isolate K2 tech threshold effect:
    # with overall=8.5 >= 8.0 (overall passes) but tech=8 < tech_threshold=9 → should refine
    report = _make_report(tech=8, ped=9, overall=8.5)
    state = _state_with_reports(report)
    with patch(
        "src.core.graph.app_settings",
        _make_settings(
            RANKING_PASS_THRESHOLD=8.0,  # overall 8.5 would pass this
            RANKING_TECHNICAL_THRESHOLD=9.0,  # but tech=8 < 9 → must refine
            RANKING_PEDAGOGICAL_THRESHOLD=8.0,
            MAX_REFINEMENT_ROUNDS=5,
            MAX_REFINEMENT_STRATEGY="hitl",
            SOFT_ACCEPT_THRESHOLD=7.0,
        ),
    ):
        result = _route(state)
    # Because tech threshold fails, should go to refinement_step
    assert result == "refinement_step"


def test_per_dimension_threshold_disabled_when_default():
    """With default thresholds 9.0 all equal, existing-pass cases still pass."""
    report = _make_report(tech=9, ped=9, overall=9.0)
    state = _state_with_reports(report)
    with patch(
        "src.core.graph.app_settings",
        _make_settings(
            RANKING_PASS_THRESHOLD=9.0,
            RANKING_TECHNICAL_THRESHOLD=9.0,
            RANKING_PEDAGOGICAL_THRESHOLD=9.0,
            MAX_REFINEMENT_ROUNDS=5,
            MAX_REFINEMENT_STRATEGY="hitl",
            SOFT_ACCEPT_THRESHOLD=7.0,
        ),
    ):
        result = _route(state)
    assert result in ("END", "__end__")


# ---------------------------------------------------------------------------
# K3: SCORING_POLICY
# ---------------------------------------------------------------------------


def test_mean_policy():
    """mean policy returns (tech+ped)/2."""
    cfg = _make_settings(SCORING_POLICY="mean")
    active, views = compute_overall_score(8.0, 6.0, cfg)
    assert active == 7.0
    assert views["mean"] == 7.0


def test_min_policy():
    """min policy: tech=3 ped=8 → overall=3 (not 5.5)."""
    cfg = _make_settings(SCORING_POLICY="min")
    active, views = compute_overall_score(3.0, 8.0, cfg)
    assert active == 3.0
    assert views["min"] == 3.0


def test_weighted_policy():
    """weighted policy uses K1 weights."""
    cfg = _make_settings(
        SCORING_POLICY="weighted",
        SCORING_TECHNICAL_WEIGHT=0.6,
        SCORING_PEDAGOGICAL_WEIGHT=0.4,
    )
    active, views = compute_overall_score(8.0, 6.0, cfg)
    expected = round(8.0 * 0.6 + 6.0 * 0.4, 1)
    assert active == pytest.approx(expected, abs=0.05)


# ---------------------------------------------------------------------------
# Sensitivity analysis: overall_scores always emitted
# ---------------------------------------------------------------------------


def test_all_three_views_always_computed():
    """compute_overall_score always returns dict with 'mean', 'min', and a weighted key."""
    for policy in ("mean", "min", "weighted"):
        cfg = _make_settings(
            SCORING_POLICY=policy,
            SCORING_TECHNICAL_WEIGHT=0.5,
            SCORING_PEDAGOGICAL_WEIGHT=0.5,
        )
        _, views = compute_overall_score(7.0, 5.0, cfg)
        assert "mean" in views, f"policy={policy}: 'mean' missing"
        assert "min" in views, f"policy={policy}: 'min' missing"
        # There must be exactly one weighted_* key
        weighted_keys = [k for k in views if k.startswith("weighted_")]
        assert (
            len(weighted_keys) == 1
        ), f"policy={policy}: expected 1 weighted key, got {views}"


def test_active_overall_matches_policy_min():
    """When SCORING_POLICY=min, top-level overall_score equals views['min']."""
    cfg = _make_settings(SCORING_POLICY="min")
    active, views = compute_overall_score(4.0, 8.0, cfg)
    assert active == views["min"]


def test_active_overall_matches_policy_mean():
    """When SCORING_POLICY=mean, top-level overall_score equals views['mean']."""
    cfg = _make_settings(SCORING_POLICY="mean")
    active, views = compute_overall_score(4.0, 8.0, cfg)
    assert active == views["mean"]


def test_active_overall_matches_policy_weighted():
    """When SCORING_POLICY=weighted, top-level overall_score equals the weighted view."""
    cfg = _make_settings(
        SCORING_POLICY="weighted",
        SCORING_TECHNICAL_WEIGHT=0.55,
        SCORING_PEDAGOGICAL_WEIGHT=0.45,
    )
    active, views = compute_overall_score(4.0, 8.0, cfg)
    weighted_keys = [k for k in views if k.startswith("weighted_")]
    assert active == views[weighted_keys[0]]


def test_ranking_report_has_overall_scores_field():
    """RankingReport schema has optional overall_scores dict field."""
    report = _make_report(
        overall_scores={"mean": 7.0, "min": 5.0, "weighted_50_50": 7.0}
    )
    assert report.overall_scores is not None
    assert "mean" in report.overall_scores
    assert "min" in report.overall_scores


def test_ranking_report_overall_scores_none_for_legacy():
    """overall_scores=None is valid (legacy / error-fallback reports)."""
    report = _make_report(overall_scores=None)
    assert report.overall_scores is None
