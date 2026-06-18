"""Log redaction utilities for sanitizing sensitive content before logging.

Redacts API keys, long text blocks (course content, writeups, PDF text),
and other patterns that should not appear in logs. Use at call sites
before passing content to logger.debug(), logger.info(), etc.

Example:
    from src.utils.log_redaction import redact_sensitive

    logger.debug("Parse failed: {}", redact_sensitive(raw_llm_response))
"""

from __future__ import annotations

import re
from typing import Any

# Patterns for API key-like strings (common prefixes)
_API_KEY_PATTERNS = [
    (re.compile(r"sk-[a-zA-Z0-9_-]{20,}"), "[REDACTED:OPENAI]"),
    (re.compile(r"lsv2_pt_[a-zA-Z0-9_-]{20,}"), "[REDACTED:LANGSMITH]"),
    (re.compile(r"xai-[a-zA-Z0-9_-]{20,}"), "[REDACTED:XAI]"),
]

# Max chars to show before redacting long text
_MAX_SAFE_LENGTH = 100


def redact_sensitive(text: Any) -> str:
    """Redact sensitive content from text before logging.

    - API key-like patterns (sk-..., lsv2_pt_..., etc.) → [REDACTED:...]
    - Long text blocks (> 100 chars) → [REDACTED: N chars]
    - Short text, IDs, counts → returned as-is (up to max length)

    Args:
        text: Content to redact. Converted to str if not already.

    Returns:
        Sanitized string safe for logging.
    """
    if text is None:
        return "None"
    s = str(text).strip()
    if not s:
        return "(empty)"

    # 1. Redact API keys
    for pattern, replacement in _API_KEY_PATTERNS:
        s = pattern.sub(replacement, s)

    # 2. Redact long content (course text, writeups, PDF snippets)
    if len(s) > _MAX_SAFE_LENGTH:
        return f"[REDACTED: {len(s)} chars]"

    return s


def redact_long_text(text: Any, max_chars: int = _MAX_SAFE_LENGTH) -> str:
    """Redact or truncate long text; always apply API key redaction.

    Use when you need to log a short prefix (e.g. first 50 chars) or
    metadata (length) instead of full content.

    Args:
        text: Content to redact.
        max_chars: Max characters to allow before redacting. Default 100.

    Returns:
        Sanitized string.
    """
    if text is None:
        return "None"
    s = str(text).strip()
    if not s:
        return "(empty)"

    for pattern, replacement in _API_KEY_PATTERNS:
        s = pattern.sub(replacement, s)

    if len(s) > max_chars:
        return f"[REDACTED: {len(s)} chars]"
    return s
