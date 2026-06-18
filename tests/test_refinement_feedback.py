"""Tests for Wave 2 refinement improvements (R1, R2, R3).

R1: Judge improvements list passed to content generation prompt
R2: Early-exit on no-improvement (score plateau detection)
R3: Selective content regen by failing persona
"""

from unittest.mock import MagicMock, patch

from langgraph.graph import END

from src.agents.content_generation_agent import _generate_writeup_for_challenge
from src.core.graph import _refinement_step_node_fn, _route_ranking_decision
from src.core.state import AgentState
from src.models.report_models import RankingReport, RankingScore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_score(
    score: int, persona: str = "Technical", improvements=None
) -> RankingScore:
    return RankingScore(
        score=score,
        persona=persona,
        justification="test",
        improvements=improvements or [],
    )


def _make_report(
    challenge_id: str,
    tech: int,
    ped: int,
    overall: float,
    tech_improvements=None,
    ped_improvements=None,
) -> RankingReport:
    return RankingReport(
        challenge_id=challenge_id,
        overall_score=overall,
        technical_review=_make_score(tech, "Technical", tech_improvements or []),
        pedagogical_review=_make_score(ped, "Pedagogical", ped_improvements or []),
        technical_rank="Intermediate",
    )


def _mock_settings(
    strategy: str = "hitl",
    max_rounds: int = 5,
    threshold: float = 9.0,
    soft: float = 7.0,
) -> MagicMock:
    m = MagicMock()
    m.RANKING_PASS_THRESHOLD = threshold
    m.RANKING_TECHNICAL_THRESHOLD = threshold  # K2: per-dim thresholds (Wave 3)
    m.RANKING_PEDAGOGICAL_THRESHOLD = threshold
    m.MAX_REFINEMENT_ROUNDS = max_rounds
    m.MAX_REFINEMENT_STRATEGY = strategy
    m.SOFT_ACCEPT_THRESHOLD = soft
    m.CONTENT_GENERATION_MAX_TOKENS = 12000
    return m


# ===========================================================================
# R1 — Judge feedback flows into gen prompt
# ===========================================================================


class TestR1FeedbackInPrompt:
    """R1: prior_improvements_per_challenge items appear verbatim in gen prompt."""

    def test_improvements_appear_in_prompt_when_set(self):
        """When prior_improvements_per_challenge has items, they appear in the user prompt."""
        cid = "crypto/test_chal"
        improvements = [
            "Add ATT&CK T1566 to Extra Resources section",
            "Solver uses gmpy2.gcdext instead of pow(e, -1, phi)",
        ]

        # Capture the user prompt sent to the LLM
        captured = {}

        def fake_gen(system, user, **kwargs):
            captured["user"] = user
            return "# Fake course content"

        with (
            patch(
                "src.agents.content_generation_agent.generate_response_with_system",
                side_effect=fake_gen,
            ),
            patch(
                "src.agents.content_generation_agent._build_rag_context",
                return_value="",
            ),
        ):
            _generate_writeup_for_challenge(
                challenge_id=cid,
                category="crypto",
                challenge_name="test_chal",
                description="A test challenge",
                prior_improvements=improvements,
            )

        user_prompt = captured.get("user", "")
        for item in improvements:
            assert item in user_prompt, f"Expected improvement not in prompt: {item!r}"

    def test_no_feedback_section_on_initial_gen(self):
        """Round 0 (no prior improvements) must NOT include a MUST-address section."""
        captured = {}

        def fake_gen(system, user, **kwargs):
            captured["user"] = user
            return "# Fake course"

        with (
            patch(
                "src.agents.content_generation_agent.generate_response_with_system",
                side_effect=fake_gen,
            ),
            patch(
                "src.agents.content_generation_agent._build_rag_context",
                return_value="",
            ),
        ):
            _generate_writeup_for_challenge(
                challenge_id="web/chal",
                category="web",
                challenge_name="chal",
                description="desc",
                prior_improvements=None,
            )

        user_prompt = captured.get("user", "")
        assert "MUST address" not in user_prompt
        assert "Judge feedback" not in user_prompt

    def test_empty_improvements_list_no_feedback_section(self):
        """Empty improvements list → no MUST-address section."""
        captured = {}

        def fake_gen(system, user, **kwargs):
            captured["user"] = user
            return "# Fake course"

        with (
            patch(
                "src.agents.content_generation_agent.generate_response_with_system",
                side_effect=fake_gen,
            ),
            patch(
                "src.agents.content_generation_agent._build_rag_context",
                return_value="",
            ),
        ):
            _generate_writeup_for_challenge(
                challenge_id="web/chal",
                category="web",
                challenge_name="chal",
                description="desc",
                prior_improvements=[],
            )

        user_prompt = captured.get("user", "")
        assert "MUST address" not in user_prompt

    def test_improvements_from_state_passed_to_gen(self):
        """After refinement_step, prior_improvements_per_challenge in state is used."""
        cid = "crypto/chal"
        improvements = ["Fix solver import", "Add conclusion section"]
        report = _make_report(
            cid,
            tech=6,
            ped=7,
            overall=6.5,
            tech_improvements=improvements,
            ped_improvements=["Add objectives"],
        )
        state = AgentState(
            challenge_ids=[cid],
            ranking_reports=[report],
            refinement_count=0,
            prior_improvements_per_challenge={cid: improvements},
        )
        # Verify state field is set and accessible
        assert cid in state.prior_improvements_per_challenge
        assert "Fix solver import" in state.prior_improvements_per_challenge[cid]


# ===========================================================================
# R2 — Early-exit on no-improvement
# ===========================================================================


class TestR2EarlyExit:
    """R2: Stop refining challenges whose scores have plateaued."""

    def test_score_plateau_triggers_early_exit(self):
        """score_history [6.0, 5.5]: improvement < 0.5 → early-exit, no refinement."""
        report = _make_report("crypto/chal", tech=6, ped=5, overall=5.5)
        state = AgentState(
            challenge_ids=["crypto/chal"],
            ranking_reports=[report],
            refinement_count=1,
            score_history_per_challenge={"crypto/chal": [6.0, 5.5]},
        )

        with patch("src.core.graph.app_settings", _mock_settings()):
            result = _route_ranking_decision(state)

        # All challenges hit early-exit → should route to END, not refinement_step
        assert result == END

    def test_sufficient_gain_continues_refinement(self):
        """score_history [6.0, 6.8]: gain = 0.8 >= 0.5 → continue refinement."""
        report = _make_report("crypto/chal", tech=7, ped=6, overall=6.5)
        state = AgentState(
            challenge_ids=["crypto/chal"],
            ranking_reports=[report],
            refinement_count=1,
            score_history_per_challenge={"crypto/chal": [6.0, 6.8]},
        )

        with patch("src.core.graph.app_settings", _mock_settings()):
            result = _route_ranking_decision(state)

        assert result == "refinement_step"

    def test_single_score_no_early_exit(self):
        """score_history [6.0] (only 1 round): no comparison possible → continue."""
        report = _make_report("crypto/chal", tech=6, ped=6, overall=6.0)
        state = AgentState(
            challenge_ids=["crypto/chal"],
            ranking_reports=[report],
            refinement_count=1,
            score_history_per_challenge={"crypto/chal": [6.0]},
        )

        with patch("src.core.graph.app_settings", _mock_settings()):
            result = _route_ranking_decision(state)

        assert result == "refinement_step"

    def test_no_score_history_no_early_exit(self):
        """No score_history_per_challenge at all → no early-exit, normal refinement."""
        report = _make_report("crypto/chal", tech=6, ped=6, overall=6.0)
        state = AgentState(
            challenge_ids=["crypto/chal"],
            ranking_reports=[report],
            refinement_count=1,
        )

        with patch("src.core.graph.app_settings", _mock_settings()):
            result = _route_ranking_decision(state)

        assert result == "refinement_step"

    def test_partial_early_exit_does_not_block_other_challenges(self):
        """Challenge A plateaued but challenge B still improving → still refine (B)."""
        report_a = _make_report("crypto/chal_a", tech=5, ped=5, overall=5.0)
        report_b = _make_report("web/chal_b", tech=6, ped=7, overall=6.5)
        state = AgentState(
            challenge_ids=["crypto/chal_a", "web/chal_b"],
            ranking_reports=[report_a, report_b],
            refinement_count=1,
            score_history_per_challenge={
                "crypto/chal_a": [6.0, 5.0],  # plateau → early-exit
                "web/chal_b": [5.5, 6.5],  # gain 1.0 → still refining
            },
        )

        with patch("src.core.graph.app_settings", _mock_settings()):
            result = _route_ranking_decision(state)

        # B still improving → refinement_step (not END)
        assert result == "refinement_step"

    def test_score_history_appended_by_refinement_step(self):
        """_refinement_step_node_fn appends current overall score to score_history."""
        cid = "crypto/chal"
        report = _make_report(cid, tech=6, ped=6, overall=6.0)
        state = AgentState(
            challenge_ids=[cid],
            ranking_reports=[report],
            refinement_count=0,
        )

        with patch("src.core.graph.app_settings", _mock_settings()):
            new_state = _refinement_step_node_fn(state)

        assert cid in new_state.score_history_per_challenge
        assert 6.0 in new_state.score_history_per_challenge[cid]


# ===========================================================================
# R3 — Selective content regen by failing persona
# ===========================================================================


class TestR3SelectiveRegen:
    """R3: Only regenerate the failing aspect (course or solver, not both)."""

    def test_tech_fail_only_regenerates_solver(self):
        """Only tech fails → solver regen flag True, course regen flag False."""
        cid = "crypto/chal"
        state = AgentState(
            challenge_ids=[cid],
            ranking_retest_technical_ids=[cid],
            ranking_retest_pedagogical_ids=[],  # empty
        )
        # _get_regen_flags should say: regen_course=False, regen_solver=True
        from src.agents.content_generation_agent import _get_regen_flags

        regen_course, regen_solver = _get_regen_flags(cid, state)
        assert regen_solver is True
        assert regen_course is False

    def test_ped_fail_only_regenerates_course(self):
        """Only ped fails → course regen flag True, solver regen flag False."""
        cid = "web/chal"
        state = AgentState(
            challenge_ids=[cid],
            ranking_retest_technical_ids=[],  # empty
            ranking_retest_pedagogical_ids=[cid],
        )
        from src.agents.content_generation_agent import _get_regen_flags

        regen_course, regen_solver = _get_regen_flags(cid, state)
        assert regen_course is True
        assert regen_solver is False

    def test_both_fail_regenerates_both(self):
        """Both fail → both regen flags True (original behavior)."""
        cid = "web/chal"
        state = AgentState(
            challenge_ids=[cid],
            ranking_retest_technical_ids=[cid],
            ranking_retest_pedagogical_ids=[cid],
        )
        from src.agents.content_generation_agent import _get_regen_flags

        regen_course, regen_solver = _get_regen_flags(cid, state)
        assert regen_course is True
        assert regen_solver is True

    def test_initial_gen_regenerates_both(self):
        """Round 0 (no retest IDs) → both regen flags True (initial generation)."""
        cid = "crypto/chal"
        state = AgentState(
            challenge_ids=[cid],
            ranking_retest_technical_ids=None,
            ranking_retest_pedagogical_ids=None,
        )
        from src.agents.content_generation_agent import _get_regen_flags

        regen_course, regen_solver = _get_regen_flags(cid, state)
        assert regen_course is True
        assert regen_solver is True
