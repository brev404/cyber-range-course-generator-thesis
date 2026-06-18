"""Test _truncate_for_prompt cap (v4.2 truncation-bug fix).

Discovered 2026-05-30 during smoke #5 analysis: this function was capping at 12000 chars
and appending `[... truncated for length ...]` to every course longer than that. Judges
across 5 smokes plus all v1/v2/v3 baselines were reading framework-inserted truncation
markers and scoring courses on partial content. Courses 19-26k bytes (typical) had
40-60% of content hidden from the judge.

Fix: bumped default 12000 → 100000 for course, solver-call 6000 → 50000. Claude models
support 200k+ context; the old cap was wildly conservative.
"""

from src.agents.ranking_agent import _truncate_for_prompt


class TestTruncationCapBumpToProductionScale:
    """The cap must accommodate every course we've observed (max ~26631 bytes)."""

    def test_default_cap_at_least_100k(self):
        """Default cap must allow ≥100k chars (covers all observed courses)."""
        text = "a" * 95000
        result = _truncate_for_prompt(text)
        assert (
            result == text
        ), "95k-char text must pass through unchanged at default cap"

    def test_typical_course_size_26k_passes_through(self):
        """A 26631-byte course (atentie smoke #5 size) must not be truncated."""
        text = "a" * 26631
        result = _truncate_for_prompt(text)
        assert result == text, "26k-byte typical course must pass through"
        assert "[... truncated for length ...]" not in result

    def test_solver_50k_cap_accommodates_typical_solver(self):
        """Solver calls now use max_chars=50000. A 10k-line solver must pass."""
        text = "x = 1\n" * 5000  # ~30000 chars
        result = _truncate_for_prompt(text, max_chars=50000)
        assert result == text


class TestTruncationStillFiresOnOversizedInput:
    """When input genuinely exceeds the cap, truncation still fires (cap is a real cap)."""

    def test_oversized_text_gets_truncated(self):
        text = "a" * 150000  # 150k > 100k default
        result = _truncate_for_prompt(text)
        assert "[... truncated for length ...]" in result
        assert len(result) < len(text)

    def test_oversized_explicit_cap(self):
        text = "x" * 60000
        result = _truncate_for_prompt(text, max_chars=50000)
        assert "[... truncated for length ...]" in result


class TestEdgeCases:
    def test_empty_returns_empty(self):
        assert _truncate_for_prompt("") == ""
        assert _truncate_for_prompt(None) == ""

    def test_unchanged_when_below_cap(self):
        text = "short course content"
        assert _truncate_for_prompt(text) == text
