"""Tests for src/services/claude_code_model.py — LangChain callback wiring."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from langchain_core.messages import HumanMessage

from src.services.claude_code_model import ClaudeCodeModel


class TestCallbackWiring:
    """Verify on_llm_new_token is fired when run_manager is provided."""

    @patch("src.services.claude_code_model.subprocess.run")
    @patch("src.services.claude_code_model._is_available", return_value=True)
    def test_generate_fires_on_llm_new_token(self, _mock_avail, mock_subproc):
        mock_subproc.return_value = MagicMock(
            returncode=0,
            stdout="Hello from Claude CLI",
            stderr="",
        )

        fake_manager = MagicMock()
        model = ClaudeCodeModel()
        result = model._generate(
            messages=[HumanMessage(content="test")],
            run_manager=fake_manager,
        )

        fake_manager.on_llm_new_token.assert_called_once_with("Hello from Claude CLI")
        assert result.generations[0].message.content == "Hello from Claude CLI"

    @patch("src.services.claude_code_model.subprocess.run")
    @patch("src.services.claude_code_model._is_available", return_value=True)
    def test_generate_without_run_manager_no_error(self, _mock_avail, mock_subproc):
        mock_subproc.return_value = MagicMock(
            returncode=0,
            stdout="Response text",
            stderr="",
        )

        model = ClaudeCodeModel()
        result = model._generate(
            messages=[HumanMessage(content="test")],
            run_manager=None,
        )

        assert result.generations[0].message.content == "Response text"

    @patch("src.services.claude_code_model.subprocess.run")
    @patch("src.services.claude_code_model._is_available", return_value=True)
    def test_stream_fires_on_llm_new_token(self, _mock_avail, mock_subproc):
        mock_subproc.return_value = MagicMock(
            returncode=0,
            stdout="Streamed response",
            stderr="",
        )

        fake_manager = MagicMock()
        model = ClaudeCodeModel()
        chunks = list(
            model._stream(
                messages=[HumanMessage(content="test")],
                run_manager=fake_manager,
            )
        )

        assert len(chunks) == 1
        assert chunks[0].message.content == "Streamed response"
        # _stream calls _generate which fires once, then _stream fires again
        assert fake_manager.on_llm_new_token.call_count == 2
        fake_manager.on_llm_new_token.assert_called_with("Streamed response")
