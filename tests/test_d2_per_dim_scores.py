"""Tests for D2: per-dimension scores in the refinement prompt.

When a challenge scores tech=8.5 / ped=6.5, the gen prompt for the refinement round
must surface those numbers so the LLM knows which dimension failed alongside the
improvement list. Previously the LLM only saw a flat "MUST address" improvement list,
making it hard to prioritise.

State: AgentState.prior_dim_scores_per_challenge (Dict[challenge_id, Dict[dim, float]])
Plumbing: refinement_step populates it from ranking reports; content_generation_agent
reads it and renders a "Scores from previous round that need to reach 9.0:" block
inside the existing judge_feedback_block.
"""

from __future__ import annotations

from unittest.mock import patch


def test_d2_state_field_exists():
    """AgentState defaults prior_dim_scores_per_challenge to an empty dict."""
    from src.core.state import AgentState

    s = AgentState()
    assert hasattr(s, "prior_dim_scores_per_challenge")
    assert s.prior_dim_scores_per_challenge == {}


def test_d2_score_block_in_prompt_when_scores_present():
    """When prior_dim_scores is set, the user prompt contains a scores block listing
    technical and pedagogical with their targets.
    """
    captured: list[tuple[str, str]] = []

    def _fake_generate(system, user, *, temperature=0.5, max_tokens=14000, **kwargs):
        captured.append((system, user))
        return "fake course"

    with (
        patch(
            "src.agents.content_generation_agent.generate_response_with_system",
            side_effect=_fake_generate,
        ),
        patch(
            "src.agents.content_generation_agent._build_rag_context",
            return_value="",
        ),
    ):
        from src.agents.content_generation_agent import _generate_writeup_for_challenge

        _generate_writeup_for_challenge(
            challenge_id="crypto/test",
            category="crypto",
            challenge_name="test",
            description="A test challenge.",
            prior_improvements=["Improve narrative", "Add expected output"],
            prior_dim_scores={"technical": 8.5, "pedagogical": 6.5},
        )

    assert len(captured) == 1, "exactly one call expected"
    _system, user_prompt = captured[0]
    lower = user_prompt.lower()
    assert (
        "scores from previous round" in lower
    ), f"Prompt must contain 'Scores from previous round' block. Got: {user_prompt[:2000]}"
    # Both dimensions must appear with their actual scores
    assert "technical: 8.5" in lower, "Technical score must appear with value 8.5"
    assert "pedagogical: 6.5" in lower, "Pedagogical score must appear with value 6.5"
    assert "9.0" in user_prompt, "Target threshold 9.0 must be mentioned"


def test_d2_no_score_block_when_scores_absent():
    """When prior_dim_scores is None, no scores block is emitted."""
    captured: list[tuple[str, str]] = []

    def _fake_generate(system, user, *, temperature=0.5, max_tokens=14000, **kwargs):
        captured.append((system, user))
        return "fake course"

    with (
        patch(
            "src.agents.content_generation_agent.generate_response_with_system",
            side_effect=_fake_generate,
        ),
        patch(
            "src.agents.content_generation_agent._build_rag_context",
            return_value="",
        ),
    ):
        from src.agents.content_generation_agent import _generate_writeup_for_challenge

        _generate_writeup_for_challenge(
            challenge_id="crypto/test",
            category="crypto",
            challenge_name="test",
            description="A test challenge.",
            prior_improvements=["Improve narrative"],
            prior_dim_scores=None,
        )

    _system, user_prompt = captured[0]
    assert (
        "scores from previous round" not in user_prompt.lower()
    ), "Without prior_dim_scores, the scores block must NOT appear in the prompt"


def test_d2_no_score_block_when_no_improvements():
    """When there are no prior_improvements, the entire judge feedback section is
    omitted — even if prior_dim_scores is set, the block should not appear (no use
    rendering scores without a corresponding action list).
    """
    captured: list[tuple[str, str]] = []

    def _fake_generate(system, user, *, temperature=0.5, max_tokens=14000, **kwargs):
        captured.append((system, user))
        return "fake course"

    with (
        patch(
            "src.agents.content_generation_agent.generate_response_with_system",
            side_effect=_fake_generate,
        ),
        patch(
            "src.agents.content_generation_agent._build_rag_context",
            return_value="",
        ),
    ):
        from src.agents.content_generation_agent import _generate_writeup_for_challenge

        _generate_writeup_for_challenge(
            challenge_id="crypto/test",
            category="crypto",
            challenge_name="test",
            description="A test challenge.",
            prior_improvements=None,
            prior_dim_scores={"technical": 8.5, "pedagogical": 6.5},
        )

    _system, user_prompt = captured[0]
    # The entire judge_feedback_block is gated on prior_improvements; with no
    # improvements the scores block is also skipped (it lives inside the same block).
    assert (
        "scores from previous round" not in user_prompt.lower()
    ), "No prior_improvements => no judge feedback block (no scores block either)"
    assert (
        "judge feedback" not in user_prompt.lower()
    ), "No prior_improvements => no judge feedback block in prompt"


def test_d2_refinement_step_populates_state_field():
    """The _refinement_step_node_fn must populate prior_dim_scores_per_challenge
    from ranking reports so the next content_generation round sees the scores.
    """
    from dataclasses import replace

    from src.core.graph import _refinement_step_node_fn
    from src.core.state import AgentState
    from src.models.report_models import RankingReport, RankingScore

    state = AgentState()
    state = replace(
        state,
        ranking_reports=[
            RankingReport(
                challenge_id="crypto/alpha",
                overall_score=7.5,
                pedagogical_review=RankingScore(
                    score=6,
                    persona="Pedagogical",
                    justification="x",
                    improvements=["fix narrative"],
                ),
                technical_review=RankingScore(
                    score=9,
                    persona="Technical",
                    justification="x",
                    improvements=[],
                ),
                technical_rank="Intermediate",
            ),
        ],
    )

    new_state = _refinement_step_node_fn(state)

    assert "crypto/alpha" in new_state.prior_dim_scores_per_challenge
    dims = new_state.prior_dim_scores_per_challenge["crypto/alpha"]
    assert dims["technical"] == 9.0
    assert dims["pedagogical"] == 6.0
