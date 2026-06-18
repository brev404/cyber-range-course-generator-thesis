"""Tests for F3 (v4.2.1) category-specific guidance blocks.

F3 injects per-category requirement blocks (tool patterns, expected ATT&CK / CWE / OWASP
references, common failure modes) into the course-gen USER prompt — not the system prompt,
to preserve prompt cache. Gated behind `settings.F3_CATEGORY_GUIDANCE_ENABLED` so an
isolating ablation EXP can disable just this lever.

Covered behaviours:
  - The `_CATEGORY_GUIDANCE` dict contains every category we observe in the dataset.
  - When the flag is True (default), the block appears in the user prompt with the canonical
    `## Category-specific requirements ({category})` heading.
  - When the flag is False, no block is injected (ablation parity with v3 user prompt).
  - Case-insensitive category lookup (`crypto`, `Crypto`, `CRYPTO` all match the same block).
  - Unknown categories silently produce no block (no error, just a debug log).
  - Every category block sanity-checks that at least one of ATT&CK / CWE / OWASP appears,
    so F8's reference-marker validation sees a guidance signal.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

_EXPECTED_CATEGORIES = [
    "pwn",
    "crypto",
    "web",
    "forensics",
    "rev",
    "osint",
    "misc",
    "mobile",
    "electron",
]


# ---------------------------------------------------------------------------
# Dict contents
# ---------------------------------------------------------------------------


def test_category_guidance_dict_has_all_categories():
    """_CATEGORY_GUIDANCE must contain every category observed in the dataset.

    Categories: pwn, crypto, web, forensics, rev, osint, misc, mobile, electron.
    """
    from src.agents.content_generation_agent import _CATEGORY_GUIDANCE

    for cat in _EXPECTED_CATEGORIES:
        assert (
            cat in _CATEGORY_GUIDANCE
        ), f"_CATEGORY_GUIDANCE missing category: {cat!r}"
    # Sanity: no duplicate keys, all lower-case
    assert all(
        k == k.lower() for k in _CATEGORY_GUIDANCE
    ), "_CATEGORY_GUIDANCE keys must be lower-case for case-insensitive lookup"


@pytest.mark.parametrize("category", _EXPECTED_CATEGORIES)
def test_category_guidance_block_references_attck_cwe_or_owasp(category):
    """Every category block must mention at least one of ATT&CK / CWE / OWASP.

    These are the reference markers the F8 structural validator expects in the generated
    course (Extra Resources section). Guidance blocks that don't surface those markers
    can't shape the LLM's reference citations.
    """
    from src.agents.content_generation_agent import _CATEGORY_GUIDANCE

    block = _CATEGORY_GUIDANCE[category]
    lower = block.lower()
    has_attck = "att&ck" in lower or "attck" in lower or "att ck" in lower
    has_cwe = "cwe-" in lower or "cwe " in lower
    has_owasp = "owasp" in lower
    assert has_attck or has_cwe or has_owasp, (
        f"Category {category!r} block must reference at least one of "
        f"ATT&CK / CWE / OWASP. Block:\n{block!r}"
    )


@pytest.mark.parametrize("category", _EXPECTED_CATEGORIES)
def test_category_guidance_block_non_trivial_length(category):
    """Each block must be substantive (>=150 chars) — short blocks won't actually guide Haiku."""
    from src.agents.content_generation_agent import _CATEGORY_GUIDANCE

    block = _CATEGORY_GUIDANCE[category]
    assert len(block) >= 150, (
        f"Category {category!r} block too short ({len(block)} chars); "
        f"target is ~150-300+ chars to be useful guidance."
    )


# ---------------------------------------------------------------------------
# _build_category_guidance_block helper
# ---------------------------------------------------------------------------


def test_build_block_returns_block_when_enabled(monkeypatch):
    """When the flag is True and category is known, the helper returns a non-empty
    block with the canonical heading.

    Note: F3 default flipped to False post-smoke-#7 (regressed -0.83 mean vs smoke #6).
    Test now opt-in via monkeypatch. F3 remains available as an ablation flag.
    """
    from src.agents.content_generation_agent import _build_category_guidance_block
    from src.config.settings import settings

    monkeypatch.setattr(settings, "F3_CATEGORY_GUIDANCE_ENABLED", True)
    block = _build_category_guidance_block("crypto")
    assert block, "Expected non-empty guidance block for 'crypto' when flag is enabled"
    assert (
        "## Category-specific requirements (crypto)" in block
    ), f"Block must use canonical heading; got:\n{block!r}"


def test_build_block_returns_empty_when_flag_disabled(monkeypatch):
    """F3_CATEGORY_GUIDANCE_ENABLED=False → empty string for any category (ablation parity)."""
    from src.agents.content_generation_agent import _build_category_guidance_block
    from src.config.settings import settings

    monkeypatch.setattr(settings, "F3_CATEGORY_GUIDANCE_ENABLED", False)
    for cat in _EXPECTED_CATEGORIES:
        assert (
            _build_category_guidance_block(cat) == ""
        ), f"With F3 disabled, block for {cat!r} must be empty"


def test_build_block_unknown_category_returns_empty():
    """An unknown category (not in _CATEGORY_GUIDANCE) silently returns empty — no error."""
    from src.agents.content_generation_agent import _build_category_guidance_block

    # Should not raise, just return ""
    assert _build_category_guidance_block("not-a-real-category") == ""
    assert _build_category_guidance_block("zzz_unseen") == ""


def test_build_block_none_or_empty_category_returns_empty():
    """category=None or "" silently returns empty."""
    from src.agents.content_generation_agent import _build_category_guidance_block

    assert _build_category_guidance_block(None) == ""
    assert _build_category_guidance_block("") == ""


def test_build_block_case_insensitive(monkeypatch):
    """`crypto`, `Crypto`, `CRYPTO`, ` crypto ` all return the same block (flag must be on)."""
    from src.agents.content_generation_agent import _build_category_guidance_block
    from src.config.settings import settings

    monkeypatch.setattr(settings, "F3_CATEGORY_GUIDANCE_ENABLED", True)
    canonical = _build_category_guidance_block("crypto")
    assert canonical, "Baseline lookup failed"
    for variant in ("Crypto", "CRYPTO", " crypto ", "cRyPtO"):
        assert (
            _build_category_guidance_block(variant) == canonical
        ), f"Case/whitespace variant {variant!r} must match canonical lookup"


# ---------------------------------------------------------------------------
# Integration: block in user prompt
# ---------------------------------------------------------------------------


def test_category_guidance_block_in_user_prompt_when_enabled(monkeypatch):
    """When F3 is enabled, the user prompt passed to the LLM contains the category block."""
    from src.config.settings import settings

    monkeypatch.setattr(settings, "F3_CATEGORY_GUIDANCE_ENABLED", True)
    captured: list[tuple[str, str]] = []

    def _fake_generate(system, user, *, temperature=0.5, max_tokens=14000, **kwargs):
        captured.append((system, user))
        return "## 1. Title\nx\n"

    with (
        patch(
            "src.agents.content_generation_agent.generate_response_with_system",
            side_effect=_fake_generate,
        ),
        patch(
            "src.agents.content_generation_agent._build_rag_context",
            return_value="",
        ),
    ):
        from src.agents.content_generation_agent import _generate_writeup_for_challenge

        _generate_writeup_for_challenge(
            challenge_id="crypto/test",
            category="crypto",
            challenge_name="test",
            description="A test crypto challenge.",
        )

    assert len(captured) == 1
    system_prompt, user_prompt = captured[0]
    assert (
        "## Category-specific requirements (crypto)" in user_prompt
    ), "F3 enabled: user prompt must contain the category requirements heading"
    # And the block must NOT have been smuggled into the system prompt (cache preservation).
    assert "## Category-specific requirements" not in system_prompt, (
        "Category guidance must live in the USER prompt only — putting it in the system "
        "prompt invalidates the prompt cache."
    )


def test_no_category_block_when_flag_disabled(monkeypatch):
    """With F3_CATEGORY_GUIDANCE_ENABLED=False, the user prompt has no category block."""
    from src.config.settings import settings

    monkeypatch.setattr(settings, "F3_CATEGORY_GUIDANCE_ENABLED", False)

    captured: list[tuple[str, str]] = []

    def _fake_generate(system, user, *, temperature=0.5, max_tokens=14000, **kwargs):
        captured.append((system, user))
        return "## 1. Title\nx\n"

    with (
        patch(
            "src.agents.content_generation_agent.generate_response_with_system",
            side_effect=_fake_generate,
        ),
        patch(
            "src.agents.content_generation_agent._build_rag_context",
            return_value="",
        ),
    ):
        from src.agents.content_generation_agent import _generate_writeup_for_challenge

        _generate_writeup_for_challenge(
            challenge_id="crypto/test",
            category="crypto",
            challenge_name="test",
            description="A test crypto challenge.",
        )

    assert len(captured) == 1
    _system_prompt, user_prompt = captured[0]
    assert (
        "## Category-specific requirements" not in user_prompt
    ), "With F3 disabled, no category block should appear in the user prompt"


def test_unknown_category_no_block():
    """When the challenge category is not in the dict, the user prompt has no block (no error)."""
    captured: list[tuple[str, str]] = []

    def _fake_generate(system, user, *, temperature=0.5, max_tokens=14000, **kwargs):
        captured.append((system, user))
        return "## 1. Title\nx\n"

    with (
        patch(
            "src.agents.content_generation_agent.generate_response_with_system",
            side_effect=_fake_generate,
        ),
        patch(
            "src.agents.content_generation_agent._build_rag_context",
            return_value="",
        ),
    ):
        from src.agents.content_generation_agent import _generate_writeup_for_challenge

        _generate_writeup_for_challenge(
            challenge_id="unknown/test",
            category="not-a-real-category",
            challenge_name="test",
            description="A test challenge.",
        )

    assert len(captured) == 1
    _system_prompt, user_prompt = captured[0]
    assert (
        "## Category-specific requirements" not in user_prompt
    ), "Unknown category must not inject a block — silent skip is required"


def test_category_guidance_block_case_insensitive_in_prompt(monkeypatch):
    """A challenge with category='Crypto' (mixed case) gets the same block as 'crypto'."""
    from src.config.settings import settings

    monkeypatch.setattr(settings, "F3_CATEGORY_GUIDANCE_ENABLED", True)
    captured: list[tuple[str, str]] = []

    def _fake_generate(system, user, *, temperature=0.5, max_tokens=14000, **kwargs):
        captured.append((system, user))
        return "## 1. Title\nx\n"

    with (
        patch(
            "src.agents.content_generation_agent.generate_response_with_system",
            side_effect=_fake_generate,
        ),
        patch(
            "src.agents.content_generation_agent._build_rag_context",
            return_value="",
        ),
    ):
        from src.agents.content_generation_agent import _generate_writeup_for_challenge

        _generate_writeup_for_challenge(
            challenge_id="crypto/test",
            category="CRYPTO",  # upper-case input
            challenge_name="test",
            description="A test challenge.",
        )

    _system_prompt, user_prompt = captured[0]
    # The heading lower-cases the category (canonical form).
    assert (
        "## Category-specific requirements (crypto)" in user_prompt
    ), "Case-insensitive category lookup must normalise to the canonical lower-case key"


# ---------------------------------------------------------------------------
# Reproducibility export
# ---------------------------------------------------------------------------


def test_f3_flag_in_reproducibility_export():
    """F3_CATEGORY_GUIDANCE_ENABLED must appear in export_for_reproducibility so the
    ablation EXP can be disambiguated in reproducibility.json (METHODOLOGY_VERSION stays v4.2).
    """
    from src.config.settings import settings

    data = settings.export_for_reproducibility()
    nested = data.get("settings", {})
    assert (
        "F3_CATEGORY_GUIDANCE_ENABLED" in nested
    ), "F3_CATEGORY_GUIDANCE_ENABLED must be in reproducibility export for ablation tracking"


def test_methodology_version_unchanged_at_v4_2():
    """F3 is a content-quality intervention, not a methodology change.
    METHODOLOGY_VERSION must stay at v4.2."""
    from src.config.settings import settings

    assert (
        settings.METHODOLOGY_VERSION == "v4.2"
    ), f"F3 must NOT bump METHODOLOGY_VERSION; expected 'v4.2', got {settings.METHODOLOGY_VERSION!r}"
