"""Tests for Romanian Language Output."""

from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from src.agents.content_generation_agent import _generate_writeup_for_challenge
from src.agents.ranking_agent import _evaluate_one_challenge
from src.config.settings import Settings
from src.core.state import AgentState

_TECH_JSON = (
    '{"score": 8, "justification": "ok", "improvements": [], "technical_rank": "Beginner",'
    ' "dimension_scores": {"correctness": 8, "completeness": 8, "technical_accuracy": 8,'
    ' "code_quality": 8, "logical_validity": 8}}'
)
_PED_JSON = (
    '{"score": 8, "justification": "ok", "improvements": [],'
    ' "dimension_scores": {"sections_structure": 8, "cognitive_load": 8,'
    ' "scaffolding_reproducibility": 8, "relevance_curriculum": 8,'
    ' "skill_level_awareness": 8, "human_language_context": 8}}'
)


def test_romanian_instruction_in_system_prompt():
    """output_language='ro' → content generation system prompt includes the Romanian instruction block."""
    with (
        patch(
            "src.agents.content_generation_agent.generate_response_with_system"
        ) as mock_gen,
        patch(
            "src.agents.content_generation_agent._build_rag_context", return_value=""
        ),
        patch(
            "src.agents.content_generation_agent._read_romanian_glossary",
            return_value="",
        ),
    ):
        mock_gen.return_value = "# Course"
        _generate_writeup_for_challenge(
            challenge_id="test/challenge",
            category="crypto",
            challenge_name="challenge",
            description="A crypto challenge.",
            output_language="ro",
        )
        # Arg 0 = system_prompt; arg 1 = combined_user
        system_prompt = mock_gen.call_args[0][0]
        assert "Romanian" in system_prompt


def test_glossary_in_user_message_for_romanian(tmp_path: Path):
    """output_language='ro' → glossary content is present in the user message context."""
    glossary_content = (
        "| EN | RO | Notes |\n|---|---|---|\n| cryptography | criptografie | |\n"
    )
    glossary_file = tmp_path / "romanian_glossary.md"
    glossary_file.write_text(glossary_content, encoding="utf-8")

    with (
        patch(
            "src.agents.content_generation_agent.generate_response_with_system"
        ) as mock_gen,
        patch(
            "src.agents.content_generation_agent._build_rag_context", return_value=""
        ),
        patch("src.agents.content_generation_agent.app_settings") as mock_settings,
    ):
        mock_settings.KNOWLEDGE_BASE_DIR = tmp_path
        mock_settings.CONTENT_GENERATION_MAX_TOKENS = 12000
        mock_settings.PROMPT_VARIANT = ""
        mock_gen.return_value = "# Course"
        _generate_writeup_for_challenge(
            challenge_id="test/challenge",
            category="crypto",
            challenge_name="challenge",
            description="A crypto challenge.",
            output_language="ro",
        )
        # Arg 1 = combined_user (glossary is appended here)
        combined_user = mock_gen.call_args[0][1]
        assert "criptografie" in combined_user


def test_no_romanian_instruction_for_english():
    """output_language='en' → no Romanian instruction injected (no regression)."""
    with (
        patch(
            "src.agents.content_generation_agent.generate_response_with_system"
        ) as mock_gen,
        patch(
            "src.agents.content_generation_agent._build_rag_context", return_value=""
        ),
    ):
        mock_gen.return_value = "# Course"
        _generate_writeup_for_challenge(
            challenge_id="test/challenge",
            category="crypto",
            challenge_name="challenge",
            description="A crypto challenge.",
            output_language="en",
        )
        # Arg 0 = system_prompt; Romanian instruction must be absent
        system_prompt = mock_gen.call_args[0][0]
        assert "Generate all course content in Romanian" not in system_prompt


def test_ranking_prompt_includes_romanian_note():
    """output_language='ro' → both ranking system prompts include the Romanian note."""
    with patch("src.agents.ranking_agent.generate_response_with_system") as mock_gen:
        mock_gen.side_effect = [_TECH_JSON, _PED_JSON]
        _evaluate_one_challenge(
            "test/ch",
            "## Course content in Romanian",
            "",
            output_language="ro",
        )
        assert mock_gen.call_count == 2
        for call in mock_gen.call_args_list:
            # Arg 0 = system prompt; Romanian note is prepended there
            system_prompt = call[0][0]
            assert "Romanian" in system_prompt


def test_ranking_no_romanian_note_for_english():
    """output_language='en' → Romanian note is NOT in ranking system prompts."""
    with patch("src.agents.ranking_agent.generate_response_with_system") as mock_gen:
        mock_gen.side_effect = [_TECH_JSON, _PED_JSON]
        _evaluate_one_challenge(
            "test/ch",
            "## Course content in English",
            "",
            output_language="en",
        )
        for call in mock_gen.call_args_list:
            system_prompt = call[0][0]
            assert "The course content is in Romanian" not in system_prompt


def test_output_language_state_field():
    """output_language field defaults to 'en' and can be set to 'ro' at AgentState init."""
    default_state = AgentState()
    assert default_state.output_language == "en"

    ro_state = AgentState(output_language="ro")
    assert ro_state.output_language == "ro"


def test_settings_validator_rejects_invalid_language():
    """Settings rejects output_language values outside {'en', 'ro'}."""
    with pytest.raises((ValidationError, ValueError)):
        Settings(OUTPUT_LANGUAGE="fr")  # type: ignore[arg-type]
