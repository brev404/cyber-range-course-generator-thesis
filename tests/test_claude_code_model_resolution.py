"""Regression tests for the claude-code model-resolution + quota-misdetection bugs.

Discovered during a smoke run: `.env` LLM_DEFAULT_MODEL=google/gemma-4-31b-it:free
leaked into a claude-code run → `claude --print --model google/gemma-4-31b-it:free`
→ exit 1 with the error on STDOUT → misclassified as quota exhaustion → multi-hour
false sleep.

Fix A: claude-code ignores a non-Claude LLM_DEFAULT_MODEL and uses the CLI default.
Fix B1: ClaudeCodeModel surfaces stdout in the error (claude writes errors to stdout).
Fix B2: _is_quota_signal only fires on a real usage/rate-limit signal (or a truly
        silent exit), not on any exit-1.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage

from src.services.claude_code_model import ClaudeCodeModel
from src.services.llm_service import _is_quota_signal, get_chat_model

# ---------------------------------------------------------------------------
# Fix A: claude-code must not pass a non-Claude model to `claude --print`
# ---------------------------------------------------------------------------


@patch("src.services.llm_service.get_available_providers", return_value=["claude-code"])
@patch("src.services.llm_service.settings")
def test_claude_code_ignores_non_claude_default_model(mock_settings, _avail):
    """A non-Claude LLM_DEFAULT_MODEL (e.g. an OpenRouter/Gemma name) is dropped."""
    mock_settings.LLM_DEFAULT_PROVIDER = "claude-code"
    mock_settings.LLM_DEFAULT_MODEL = "google/gemma-4-31b-it:free"
    mock_settings.LLM_TEMPERATURE = 0.0
    mock_settings.LLM_MAX_TOKENS = 4000
    mock_settings.LLM_TIMEOUT = 300

    m = get_chat_model(provider="claude-code")
    assert isinstance(m, ClaudeCodeModel)
    assert (
        m.model_name == "claude-code"
    ), f"non-Claude default leaked into claude-code: model_name={m.model_name!r}"


@patch("src.services.llm_service.get_available_providers", return_value=["claude-code"])
@patch("src.services.llm_service.settings")
def test_claude_code_preserves_explicit_claude_model(mock_settings, _avail):
    """An explicit claude-* model is preserved (not dropped)."""
    mock_settings.LLM_DEFAULT_PROVIDER = "claude-code"
    mock_settings.LLM_DEFAULT_MODEL = None
    mock_settings.LLM_TEMPERATURE = 0.0
    mock_settings.LLM_MAX_TOKENS = 4000
    mock_settings.LLM_TIMEOUT = 300

    m = get_chat_model(provider="claude-code", model="claude-haiku-4-5")
    assert m.model_name == "claude-haiku-4-5"


# ---------------------------------------------------------------------------
# Fix B1: ClaudeCodeModel surfaces stdout (claude writes errors there)
# ---------------------------------------------------------------------------


@patch("src.services.claude_code_model.subprocess.run")
@patch("src.services.claude_code_model._is_available", return_value=True)
def test_error_includes_stdout_when_stderr_empty(_avail, mock_subproc):
    """An exit-1 with the error on stdout must surface that text (not '(no stderr)')."""
    mock_subproc.return_value = MagicMock(
        returncode=1,
        stdout="There's an issue with the selected model (google/gemma-4-31b-it:free). "
        "It may not exist or you may not have access to it.",
        stderr="",
    )
    model = ClaudeCodeModel(model_name="google/gemma-4-31b-it:free")
    with pytest.raises(ValueError) as exc:
        model.invoke([HumanMessage(content="hi")])
    msg = str(exc.value)
    assert "selected model" in msg, f"stdout error not surfaced: {msg!r}"
    assert "(no stderr)" not in msg


@patch("src.services.claude_code_model.subprocess.run")
@patch("src.services.claude_code_model._is_available", return_value=True)
def test_truly_silent_failure_keeps_no_stderr_marker(_avail, mock_subproc):
    """A genuinely silent exit-1 (no stdout AND no stderr) keeps the quota marker."""
    mock_subproc.return_value = MagicMock(returncode=1, stdout="", stderr="")
    model = ClaudeCodeModel()
    with pytest.raises(ValueError) as exc:
        model.invoke([HumanMessage(content="hi")])
    assert "(no stderr)" in str(exc.value)


# ---------------------------------------------------------------------------
# Fix B2: _is_quota_signal — real quota only, not any exit-1
# ---------------------------------------------------------------------------


def test_unknown_model_error_is_not_quota():
    err = ValueError(
        "claude --print exited with code 1: There's an issue with the selected "
        "model (google/gemma-4-31b-it:free). It may not exist or you may not have "
        "access to it."
    )
    assert _is_quota_signal(err) is False


def test_http_400_invalid_request_is_not_quota():
    err = ValueError(
        "claude --print exited with code 1: API Error: 400 "
        '{"type":"error","error":{"type":"invalid_request_error"}}'
    )
    assert _is_quota_signal(err) is False


def test_silent_exit_is_quota_legacy():
    err = ValueError("claude --print exited with code 1: (no stderr)")
    assert _is_quota_signal(err) is True


@pytest.mark.parametrize(
    "text",
    [
        "claude --print exited with code 1: API Error: 429 rate limit exceeded",
        "claude --print exited with code 1: usage limit reached for this window",
        "claude --print exited with code 1: Too Many Requests",
    ],
)
def test_real_quota_keywords_are_quota(text):
    assert _is_quota_signal(ValueError(text)) is True
