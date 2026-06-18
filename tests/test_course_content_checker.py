"""Tests for the course content checker (no-access framing detection)."""

import pytest

from src.models.report_models import IssueSeverity
from src.utils.course_content_checker import check_no_access_framing


def test_check_no_access_framing_detects_patterns() -> None:
    """Detects all pattern types."""
    patterns_and_texts = [
        ("you don't have access to", "You don't have access to the source code."),
        ("you cannot access", "You cannot access the server internals."),
        ("you do not have", "You do not have the flag in advance."),
        ("the user cannot see", "The user cannot see the internal state."),
        ("not available to you", "The source is not available to you."),
        ("you cannot see", "You cannot see the backend code."),
    ]
    for label, text in patterns_and_texts:
        issues = check_no_access_framing(text, challenge_id="test")
        assert len(issues) >= 1, f"Expected detection for pattern: {label}"
        assert all(i.code == "NO_ACCESS_FRAMING" for i in issues)
        assert all(i.severity == IssueSeverity.MEDIUM for i in issues)


def test_check_no_access_framing_generates_suggestions() -> None:
    """Suggestions include rewrite examples."""
    text = "You don't have access to the source code."
    issues = check_no_access_framing(text, challenge_id="web/xss-01")
    assert len(issues) == 1
    assert issues[0].suggestion is not None
    assert "what the student has" in issues[0].suggestion.lower()
    assert "Instead of" in issues[0].suggestion
    assert (
        "You have access to" in issues[0].suggestion
        or "challenge description" in issues[0].suggestion.lower()
    )


def test_check_no_access_framing_no_false_positives() -> None:
    """'You have access to...' does not trigger."""
    good_texts = [
        "You have access to the challenge description and public files.",
        "You have access to app.py and config.json.",
        "The student has access to the deployment.",
    ]
    for text in good_texts:
        issues = check_no_access_framing(text, challenge_id="test")
        assert len(issues) == 0, f"False positive for: {text!r}"


def test_check_no_access_framing_empty_input() -> None:
    """Empty or whitespace-only input returns no issues."""
    assert check_no_access_framing("") == []
    assert check_no_access_framing("   \n\t  ") == []
    assert check_no_access_framing("", challenge_id="x") == []


def test_check_no_access_framing_case_insensitive() -> None:
    """Patterns match case-insensitively."""
    text = "YOU CANNOT ACCESS the server."
    issues = check_no_access_framing(text)
    assert len(issues) >= 1
    assert any("cannot access" in i.message.lower() for i in issues)


def test_content_generation_prompt_includes_framing_guidance() -> None:
    """Content generation prompt contains no-access avoidance instruction."""
    from src.agents.content_generation_agent import _WRITEUP_SYSTEM

    assert (
        "no access" in _WRITEUP_SYSTEM.lower()
        or "don't have access" in _WRITEUP_SYSTEM.lower()
    )
    assert "what the student has" in _WRITEUP_SYSTEM.lower()
    assert "positive framing" in _WRITEUP_SYSTEM.lower() or "Good:" in _WRITEUP_SYSTEM


def test_graph_integration_no_access_issues_in_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Issues appear in state when detected (via course_terminology_checker)."""
    from src.agents.course_terminology_agent import run_course_terminology_checker
    from src.config.settings import settings as app_settings
    from src.core.state import AgentState

    monkeypatch.setattr(app_settings, "TERMINOLOGY_CHECK_MODE", "warn")
    state = AgentState(
        generated_courses={
            "web/xss-01": "You don't have access to the source code. Use XSS.",
        },
    )
    result = run_course_terminology_checker(state)
    assert result.course_terminology_issues
    assert "web/xss-01" in result.course_terminology_issues
    issues = result.course_terminology_issues["web/xss-01"]
    no_access_issues = [i for i in issues if i.code == "NO_ACCESS_FRAMING"]
    assert len(no_access_issues) >= 1


# ---------------------------------------------------------------------------
# v4.2.1: numbered-heading section detection (fix for false-missing-section flags)
# ---------------------------------------------------------------------------


def test_check_section_presence_matches_numbered_headings() -> None:
    """v4.2.1: `## 1. Title`, `## 2. Abstract / TL;DR`, etc. must be detected.

    Pre-fix: regexes required `## Abstract` directly — `## 2. Abstract` was missed
    so every v4 numbered course produced false-missing-section flags fed to the judge.
    """
    from src.utils.course_content_checker import check_section_presence

    numbered_course = (
        "## 1. Title and Context\n"
        "Some text.\n\n"
        "## 2. Abstract / TL;DR\n"
        "Brief.\n\n"
        "## 3. Objectives\n"
        "Goals.\n\n"
        "## 4. Technical Skills\n"
        "Skills.\n\n"
        "## 5. Definitions and Concepts\n"
        "Defs.\n\n"
        "## 6. Reproducibility (Step 0)\n"
        "Step zero.\n\n"
        "## 7. Thought Process / Narrative\n"
        "Narrative.\n\n"
        "## 8. Step-by-step Resolution\n"
        "Steps.\n\n"
        "## 9. Solution Script\n"
        "```python\nprint('hi')\n```\n\n"
        "## 10. Conclusion\n"
        "Summary.\n\n"
        "## 11. Extra Resources\n"
        "Refs.\n"
    )
    issues = check_section_presence(numbered_course)
    assert (
        issues == []
    ), f"Numbered-heading course must produce zero missing-section issues, got: {[i.message for i in issues]}"


def test_check_section_presence_still_matches_plain_headings() -> None:
    """Backward-compat: plain `## Abstract` headings (no number prefix) still match."""
    from src.utils.course_content_checker import check_section_presence

    plain_course = (
        "## Title\n"
        "## Abstract\n"
        "## Objectives\n"
        "## Technical Skills\n"
        "## Definitions\n"
        "## Reproducibility\n"
        "## Thought Process\n"
        "## Step-by-Step\n"
        "## Solution Script\n"
        "## Conclusion\n"
        "## Extra Resources\n"
    )
    issues = check_section_presence(plain_course)
    assert (
        issues == []
    ), f"Plain-heading course must still match, got: {[i.message for i in issues]}"


def test_check_section_presence_still_flags_genuinely_missing_sections() -> None:
    """Sanity: if a section IS truly absent, the checker still flags it."""
    from src.utils.course_content_checker import check_section_presence

    incomplete = (
        "## 1. Title\nSome text.\n\n"
        "## 2. Abstract\nBrief.\n\n"
        # No Objectives, no Conclusion, etc.
    )
    issues = check_section_presence(incomplete)
    assert len(issues) > 0, "Course missing most sections must still produce issues"
    messages = " ".join(i.message for i in issues)
    assert "Objectives" in messages or "Conclusion" in messages
