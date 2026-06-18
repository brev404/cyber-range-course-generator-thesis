"""Test D1: solver context bump (2000->8000) and edit-mode refinement."""

from unittest.mock import patch

import pytest


@pytest.fixture
def mock_generate_response():
    """Fixture that patches generate_response and captures calls."""
    calls = []

    def _fake_generate(prompt, *, temperature=0.7, max_tokens=2000, **kwargs):
        calls.append(prompt)
        return "# fake solver\nprint('CTF{flag}')\n"

    with patch(
        "src.agents.content_generation_agent.generate_response",
        side_effect=_fake_generate,
    ):
        yield calls


def test_solver_uses_8000_chars_of_writeup(mock_generate_response):
    """When writeup > 8000 chars, solver prompt must contain the 8000-char slice."""
    from src.agents.content_generation_agent import _generate_solve_script_for_challenge

    long_writeup = "A" * 10_000
    _generate_solve_script_for_challenge(
        challenge_id="crypto/test-challenge",
        category="crypto",
        writeup=long_writeup,
        description="Test challenge description",
    )
    assert len(mock_generate_response) == 1
    prompt = mock_generate_response[0]
    # The prompt must include 8000 'A's — proof the slice is 8000 not 2000
    assert (
        "A" * 8000 in prompt
    ), "Solver prompt must contain writeup[:8000] (D1 fix: 2000->8000)"
    # Old 2000-char behaviour: prompt would NOT contain 8000 consecutive As
    assert "A" * 2001 in prompt, "Should have more than 2000 chars of writeup context"


def test_solver_with_existing_solver_edit_mode(mock_generate_response):
    """When existing_solver is non-empty (refinement round > 0), prompt must include edit-mode wording."""
    from src.agents.content_generation_agent import _generate_solve_script_for_challenge

    existing = "def foo():\n    pass\n"
    _generate_solve_script_for_challenge(
        challenge_id="crypto/test-challenge",
        category="crypto",
        writeup="Short writeup.",
        description="Test",
        existing_solver=existing,
    )
    assert len(mock_generate_response) == 1
    prompt = mock_generate_response[0]
    assert (
        "Edit this version" in prompt
    ), "When existing_solver is set, prompt must contain 'Edit this version' (D1 edit-mode)"
    assert "def foo():" in prompt, "existing_solver code must appear in the prompt"


def test_solver_without_existing_solver_no_edit_mode(mock_generate_response):
    """When existing_solver is empty (first generation), prompt must NOT include edit-mode wording."""
    from src.agents.content_generation_agent import _generate_solve_script_for_challenge

    _generate_solve_script_for_challenge(
        challenge_id="crypto/test-challenge",
        category="crypto",
        writeup="Short writeup.",
        description="Test",
        existing_solver="",
    )
    assert len(mock_generate_response) == 1
    prompt = mock_generate_response[0]
    assert (
        "Edit this version" not in prompt
    ), "When existing_solver is empty, prompt must NOT contain 'Edit this version'"
