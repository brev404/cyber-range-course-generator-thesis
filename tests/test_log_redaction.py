"""Tests for log redaction utility."""

from __future__ import annotations

from src.utils.log_redaction import redact_long_text, redact_sensitive


def test_redact_sensitive_empty() -> None:
    assert redact_sensitive("") == "(empty)"
    assert redact_sensitive(None) == "None"
    assert redact_sensitive("   ") == "(empty)"


def test_redact_sensitive_short_safe() -> None:
    """Short, non-sensitive text passes through."""
    assert redact_sensitive("challenge_id: crypto-001") == "challenge_id: crypto-001"
    assert redact_sensitive("Parse failed") == "Parse failed"
    assert redact_sensitive("score: 8") == "score: 8"


def test_redact_sensitive_long_text_redacted() -> None:
    """Long text (> 100 chars) is redacted."""
    long_text = "x" * 150
    assert redact_sensitive(long_text) == "[REDACTED: 150 chars]"


def test_redact_sensitive_openai_key() -> None:
    """OpenAI-style API keys are redacted."""
    text = "Error: invalid key sk-proj-abc123def456ghi789jkl012mno345pqr"
    result = redact_sensitive(text)
    assert "sk-proj" not in result
    assert "[REDACTED:OPENAI]" in result


def test_redact_sensitive_langsmith_key() -> None:
    """LangSmith API keys are redacted."""
    text = "Trace failed: lsv2_pt_abcdef1234567890xyz123456"
    result = redact_sensitive(text)
    assert "lsv2_pt_" not in result
    assert "[REDACTED:LANGSMITH]" in result


def test_redact_long_text_custom_max() -> None:
    """redact_long_text respects max_chars."""
    text = "a" * 80
    assert redact_long_text(text, max_chars=100) == text
    assert redact_long_text(text, max_chars=50) == "[REDACTED: 80 chars]"


def test_redact_sensitive_non_string() -> None:
    """Non-string input is converted and redacted."""
    assert redact_sensitive(12345) == "12345"
    assert redact_sensitive(True) == "True"
