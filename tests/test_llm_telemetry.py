"""Tests for per-LLM-call telemetry (G).

Each call to ClaudeCodeModel._generate must append a JSON line to
output/experiments/<exp_id>/llm_calls.jsonl with all expected fields.

Pre-fix: tests fail because telemetry writing does not exist.
Post-fix: all tests pass.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch


class TestLLMCallsTelemetry:
    """ClaudeCodeModel._generate writes a jsonl entry per call."""

    @patch("src.services.claude_code_model.subprocess.run")
    @patch("src.services.claude_code_model._is_available", return_value=True)
    def test_jsonl_entry_written(self, _mock_avail, mock_subproc, tmp_path):
        """After one _generate call a jsonl entry appears with all required fields."""
        from langchain_core.messages import HumanMessage

        from src.services.claude_code_model import ClaudeCodeModel, set_telemetry_dir

        mock_subproc.return_value = MagicMock(
            returncode=0,
            stdout="test response",
            stderr="",
        )

        telemetry_dir = tmp_path / "EXP-TEST"
        telemetry_dir.mkdir()
        set_telemetry_dir(telemetry_dir)

        model = ClaudeCodeModel(model_name="claude-haiku-4-5")
        model._generate(messages=[HumanMessage(content="hello world")])

        jsonl_path = telemetry_dir / "llm_calls.jsonl"
        assert jsonl_path.exists(), "llm_calls.jsonl must be created"

        lines = [ln for ln in jsonl_path.read_text().splitlines() if ln.strip()]
        assert len(lines) == 1

        entry = json.loads(lines[0])
        assert "ts" in entry
        assert "prompt_chars" in entry
        assert "response_chars" in entry
        assert "duration_ms" in entry
        assert "model" in entry
        assert entry["model"] == "claude-haiku-4-5"
        assert entry["prompt_chars"] > 0
        assert entry["response_chars"] > 0
        assert entry["duration_ms"] >= 0

    @patch("src.services.claude_code_model.subprocess.run")
    @patch("src.services.claude_code_model._is_available", return_value=True)
    def test_jsonl_role_field(self, _mock_avail, mock_subproc, tmp_path):
        """Entry must include a role field (may be 'other' if no ContextVar set)."""
        from langchain_core.messages import HumanMessage

        from src.services.claude_code_model import (
            ClaudeCodeModel,
            set_llm_call_role,
            set_telemetry_dir,
        )

        mock_subproc.return_value = MagicMock(returncode=0, stdout="resp", stderr="")

        telemetry_dir = tmp_path / "EXP-ROLE"
        telemetry_dir.mkdir()
        set_telemetry_dir(telemetry_dir)
        set_llm_call_role("judge-tech")

        model = ClaudeCodeModel()
        model._generate(messages=[HumanMessage(content="judge this")])

        jsonl_path = telemetry_dir / "llm_calls.jsonl"
        entry = json.loads(jsonl_path.read_text().strip())
        assert entry["role"] == "judge-tech"

    @patch("src.services.claude_code_model.subprocess.run")
    @patch("src.services.claude_code_model._is_available", return_value=True)
    def test_jsonl_challenge_id_field(self, _mock_avail, mock_subproc, tmp_path):
        """Entry must include challenge_id when set via ContextVar."""
        from langchain_core.messages import HumanMessage

        from src.services.claude_code_model import (
            ClaudeCodeModel,
            set_telemetry_challenge_id,
            set_telemetry_dir,
        )

        mock_subproc.return_value = MagicMock(returncode=0, stdout="resp", stderr="")

        telemetry_dir = tmp_path / "EXP-CID"
        telemetry_dir.mkdir()
        set_telemetry_dir(telemetry_dir)
        set_telemetry_challenge_id("crypto/test-challenge")

        model = ClaudeCodeModel()
        model._generate(messages=[HumanMessage(content="hello")])

        jsonl_path = telemetry_dir / "llm_calls.jsonl"
        entry = json.loads(jsonl_path.read_text().strip())
        assert entry["challenge_id"] == "crypto/test-challenge"

    @patch("src.services.claude_code_model.subprocess.run")
    @patch("src.services.claude_code_model._is_available", return_value=True)
    def test_jsonl_appends_multiple_calls(self, _mock_avail, mock_subproc, tmp_path):
        """Multiple calls produce multiple lines (append mode)."""
        from langchain_core.messages import HumanMessage

        from src.services.claude_code_model import ClaudeCodeModel, set_telemetry_dir

        mock_subproc.return_value = MagicMock(returncode=0, stdout="r", stderr="")

        telemetry_dir = tmp_path / "EXP-MULTI"
        telemetry_dir.mkdir()
        set_telemetry_dir(telemetry_dir)

        model = ClaudeCodeModel()
        model._generate(messages=[HumanMessage(content="call 1")])
        model._generate(messages=[HumanMessage(content="call 2")])

        jsonl_path = telemetry_dir / "llm_calls.jsonl"
        lines = [ln for ln in jsonl_path.read_text().splitlines() if ln.strip()]
        assert len(lines) == 2

    @patch("src.services.claude_code_model.subprocess.run")
    @patch("src.services.claude_code_model._is_available", return_value=True)
    def test_no_telemetry_dir_does_not_crash(self, _mock_avail, mock_subproc):
        """When no telemetry_dir is set, _generate must not raise."""
        from langchain_core.messages import HumanMessage

        from src.services.claude_code_model import ClaudeCodeModel, set_telemetry_dir

        mock_subproc.return_value = MagicMock(returncode=0, stdout="ok", stderr="")

        set_telemetry_dir(None)  # explicitly clear

        model = ClaudeCodeModel()
        result = model._generate(messages=[HumanMessage(content="hi")])
        assert result.generations[0].message.content == "ok"
