"""Tests for Bug 1 (always-attempt) and Bug 2 (challenge-ids loader path).

Bug 1 — defensive refusal pattern:
  content_generation_agent should ALWAYS attempt to generate a course,
  even when challenge metadata is sparse (no description, no public files,
  no deployment). The LLM prompt must instruct best-effort generation.

Bug 2 — --challenge-ids loader path gap:
  _load_processed must return challenge_dir (not cyberedu_dir) as e.path
  so that _get_challenge_paths derives the correct category and challenge_name,
  and downstream file lookups (description.md, writeup.md, solve.py) resolve
  to the correct paths.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Bug 1 — prompt must instruct best-effort generation, never refuse
# ---------------------------------------------------------------------------


class TestBug1AlwaysAttempt:
    """The content_generation system prompt must never instruct the LLM to refuse."""

    def test_writeup_system_prompt_forbids_refusal(self):
        """_WRITEUP_SYSTEM must contain an explicit instruction to always attempt."""
        from src.agents.content_generation_agent import _WRITEUP_SYSTEM

        prompt_lower = _WRITEUP_SYSTEM.lower()
        # Must tell the model to always attempt, not to refuse
        always_attempt_phrase = (
            "always attempt" in prompt_lower
            or "always generate" in prompt_lower
            or "never refuse" in prompt_lower
            or "never output a refusal" in prompt_lower
            or "best-effort" in prompt_lower
            or "best effort" in prompt_lower
        )
        assert always_attempt_phrase, (
            "_WRITEUP_SYSTEM must contain an instruction to always attempt generation "
            "(e.g. 'always attempt', 'never refuse', 'best-effort'). "
            f"Current prompt excerpt: {_WRITEUP_SYSTEM[:200]!r}"
        )

    def test_writeup_system_prompt_does_not_instruct_refusal_on_sparse(self):
        """_WRITEUP_SYSTEM must not tell the LLM to refuse when data is sparse."""
        from src.agents.content_generation_agent import _WRITEUP_SYSTEM

        # These phrases would instruct the LLM to produce a refusal.
        # Note: "refuse" alone is acceptable in a "never refuse" instruction;
        # only active refusal directives are disallowed.
        refusal_triggers = [
            "if you cannot",
            "say so explicitly",
            "decline to generate",
            "do not generate if",
            "cannot produce a course",
            "output a refusal",  # acceptable only in "do not output a refusal" context
        ]
        prompt_lower = _WRITEUP_SYSTEM.lower()
        for trigger in refusal_triggers:
            # Allow "never refuse" / "do not output a refusal" phrases (they forbid refusal)
            # but disallow affirmative refusal instructions like "output a refusal if..."
            idx = prompt_lower.find(trigger.lower())
            if idx == -1:
                continue
            # Check surrounding context: if preceded by "never", "do not", "don't", it's OK
            context = prompt_lower[max(0, idx - 20) : idx]
            negation_words = ("never", "do not", "don't", "not ")
            if any(neg in context for neg in negation_words):
                continue  # e.g. "never output a refusal" is fine
            assert False, (
                f"_WRITEUP_SYSTEM contains refusal trigger {trigger!r} at index {idx} "
                f"(context: {_WRITEUP_SYSTEM[max(0,idx-20):idx+len(trigger)+20]!r}). "
                "Remove affirmative refusal instructions."
            )

    def test_generate_writeup_for_challenge_sparse_uses_llm(self):
        """_generate_writeup_for_challenge calls LLM even with empty description/path."""
        from src.agents.content_generation_agent import _generate_writeup_for_challenge

        called = []

        def fake_generate(system, user, temperature=0.5, max_tokens=None):
            called.append((system, user))
            return "# Sparse Challenge Course\n\n## Section 1\nBest-effort content."

        with patch(
            "src.agents.content_generation_agent.generate_response_with_system",
            side_effect=fake_generate,
        ):
            result = _generate_writeup_for_challenge(
                challenge_id="crypto/sparse_test",
                category="crypto",
                challenge_name="sparse_test",
                description="",  # no description
                challenge_path=None,  # no path → no public files, no deployment
                skill_level=None,
                human_feedback_items=None,
                author_writeup="",
                author_solver="",
            )

        assert (
            len(called) == 1
        ), "LLM must be called exactly once even with sparse input"
        assert (
            result == "# Sparse Challenge Course\n\n## Section 1\nBest-effort content."
        )

    def test_generate_writeup_task_section_does_not_condone_refusal(self):
        """The task section of the user prompt should not include refusal fallback text."""
        from src.agents.content_generation_agent import _generate_writeup_for_challenge

        captured_user_prompt = []

        def fake_generate(system, user, temperature=0.5, max_tokens=None):
            captured_user_prompt.append(user)
            return "# Best-effort Course\n\nContent."

        with patch(
            "src.agents.content_generation_agent.generate_response_with_system",
            side_effect=fake_generate,
        ):
            _generate_writeup_for_challenge(
                challenge_id="misc/0_solves",
                category="misc",
                challenge_name="0_solves",
                description="",
                challenge_path=None,
            )

        assert len(captured_user_prompt) == 1
        user_prompt = captured_user_prompt[0].lower()
        # The prompt must NOT tell the model to refuse when data is missing
        refusal_fallback_phrases = [
            "if data is insufficient",
            "if you cannot generate",
            "state that you cannot",
        ]
        for phrase in refusal_fallback_phrases:
            assert (
                phrase not in user_prompt
            ), f"User prompt contains refusal fallback phrase {phrase!r}"

    def test_generate_writeup_with_sparse_data_produces_non_refusal_output(
        self, tmp_path
    ):
        """Full integration: sparse challenge path should produce non-refusal course.

        The LLM is mocked to return a realistic course fragment. The test verifies
        that the pipeline does NOT intercept the call with a refusal before reaching
        the LLM.
        """
        from src.agents.content_generation_agent import _generate_writeup_for_challenge

        # Create a minimal challenge directory: cyberedu/ dir exists but no description.md
        challenge_path = tmp_path / "crypto" / "0_solves"
        challenge_path.mkdir(parents=True)
        (challenge_path / "cyberedu").mkdir()
        (challenge_path / "cyberedu" / "write-up").mkdir(parents=True)
        # No description.md, no writeup.md, no solver

        def fake_generate(system, user, temperature=0.5, max_tokens=None):
            # Return a minimal but non-refusal course
            return (
                "# 0_solves — CTF Challenge\n\n"
                "## 1. Title and Context\n**Category:** crypto\n\n"
                "## 2. Abstract\nBest-effort course with limited metadata.\n\n"
                "## 3. Objectives\nLearn CTF methodology.\n"
            )

        with patch(
            "src.agents.content_generation_agent.generate_response_with_system",
            side_effect=fake_generate,
        ):
            result = _generate_writeup_for_challenge(
                challenge_id="crypto/0_solves",
                category="crypto",
                challenge_name="0_solves",
                description="",
                challenge_path=challenge_path,
            )

        assert result, "Result must not be empty"
        assert (
            "Insufficient challenge data" not in result
        ), "Course must NOT be a refusal even with sparse metadata"
        assert (
            "no description, public files, or deployment endpoint were provided"
            not in result
        ), "Course must NOT contain the known refusal boilerplate"


# ---------------------------------------------------------------------------
# Bug 2 — _load_processed must return challenge_dir, not cyberedu_dir
# ---------------------------------------------------------------------------


class TestBug2LoaderPath:
    """_load_processed must return challenge_dir (not cyberedu_dir) as e.path."""

    @pytest.fixture
    def processed_root(self, tmp_path) -> Path:
        """Create a minimal processed-source directory layout.

        Layout mirrors data/processed/raw_challenges/:
          tmp/raw_challenges/crypto/rsb/cyberedu/write-up/description.md
          tmp/raw_challenges/crypto/rsb/public/
        """
        root = tmp_path / "raw_challenges"
        rsb_dir = root / "crypto" / "rsb"
        (rsb_dir / "cyberedu" / "write-up").mkdir(parents=True)
        (rsb_dir / "cyberedu" / "write-up" / "description.md").write_text(
            "RSB: RSA Broadcast Attack challenge.", encoding="utf-8"
        )
        (rsb_dir / "public").mkdir(parents=True)
        (rsb_dir / "public" / "public.zip").write_text("fake", encoding="utf-8")
        return root

    def test_load_processed_returns_challenge_dir_not_cyberedu(
        self, processed_root, monkeypatch
    ):
        """e.path must be the challenge_dir (rsb/), not challenge_dir/cyberedu."""
        from src.tui import challenge_loader as cl

        monkeypatch.setattr(cl, "_PROCESSED_LOCAL_DIR", processed_root)

        entries = cl.load_challenges("processed", ["crypto"])
        assert len(entries) == 1, f"Expected 1 entry, got {len(entries)}"
        e = entries[0]

        assert e.challenge_id == "crypto/rsb", f"Wrong challenge_id: {e.challenge_id}"
        assert e.category == "crypto", f"Wrong category: {e.category}"

        # THE CORE BUG 2 ASSERTION:
        # e.path must be .../rsb NOT .../rsb/cyberedu
        assert e.path.name == "rsb", (
            f"e.path.name must be 'rsb' (challenge dir), got {e.path.name!r}. "
            "Bug 2: _load_processed returns cyberedu_dir as path instead of challenge_dir."
        )
        assert (
            e.path / "cyberedu"
        ).exists(), "challenge_dir must contain cyberedu/ subdirectory"

    def test_load_processed_path_enables_correct_description_lookup(
        self, processed_root, monkeypatch
    ):
        """With e.path = challenge_dir, reading description.md works correctly."""
        from src.agents.content_generation_agent import _read_challenge_description
        from src.tui import challenge_loader as cl

        monkeypatch.setattr(cl, "_PROCESSED_LOCAL_DIR", processed_root)

        entries = cl.load_challenges("processed", ["crypto"])
        assert len(entries) == 1
        e = entries[0]

        description = _read_challenge_description(e.path)
        assert description == "RSB: RSA Broadcast Attack challenge.", (
            f"Description lookup failed. e.path={e.path}; "
            f"looked in {e.path / 'cyberedu' / 'write-up' / 'description.md'}"
        )

    def test_load_processed_path_enables_correct_category_in_get_challenge_paths(
        self, processed_root, monkeypatch
    ):
        """_get_challenge_paths derives correct (path, category) from loader entries."""
        from src.agents.validation_agent import _get_challenge_paths
        from src.core.state import AgentState
        from src.tui import challenge_loader as cl

        monkeypatch.setattr(cl, "_PROCESSED_LOCAL_DIR", processed_root)

        entries = cl.load_challenges("processed", ["crypto"])
        assert len(entries) == 1

        organized_paths = [e.path for e in entries]
        state = AgentState(
            organized_challenges=organized_paths,
            challenge_ids=["crypto/rsb"],
        )

        from src.config.settings import settings as app_settings

        monkeypatch.setattr(app_settings, "PROCESSED_DIR", processed_root.parent)

        paths_with_category = _get_challenge_paths(state)
        assert (
            len(paths_with_category) == 1
        ), f"Expected 1 challenge path, got {len(paths_with_category)}"
        challenge_path, category = paths_with_category[0]

        assert category == "crypto", (
            f"Category must be 'crypto', got {category!r}. "
            "Bug 2: _get_challenge_paths derives category from wrong path component."
        )
        assert (
            challenge_path.name == "rsb"
        ), f"Challenge path name must be 'rsb', got {challenge_path.name!r}."

    def test_challenge_ids_filter_same_result_as_all_categories(
        self, processed_root, monkeypatch
    ):
        """--challenge-ids and --all-categories produce the same ChallengeEntry.path."""
        from src.tui import challenge_loader as cl

        monkeypatch.setattr(cl, "_PROCESSED_LOCAL_DIR", processed_root)

        # Add a second challenge to test filtering
        other_dir = processed_root / "crypto" / "other_challenge"
        (other_dir / "cyberedu" / "write-up").mkdir(parents=True)
        (other_dir / "cyberedu" / "write-up" / "description.md").write_text(
            "Other challenge.", encoding="utf-8"
        )

        all_entries = cl.load_challenges("processed", ["crypto"])
        assert len(all_entries) == 2

        # Simulate --challenge-ids filtering (same as _resolve_challenges in the runner)
        id_set = {"crypto/rsb"}
        filtered_entries = [e for e in all_entries if e.challenge_id in id_set]
        assert len(filtered_entries) == 1

        e = filtered_entries[0]
        assert (
            e.path.name == "rsb"
        ), f"Filtered entry path must be challenge_dir 'rsb', got {e.path.name!r}"
        assert (e.path / "cyberedu").exists()

    def test_load_processed_skips_challenges_without_cyberedu(
        self, processed_root, monkeypatch
    ):
        """Challenges without cyberedu/ subdir are still skipped (regression guard)."""
        from src.tui import challenge_loader as cl

        monkeypatch.setattr(cl, "_PROCESSED_LOCAL_DIR", processed_root)

        # Add a bare challenge directory without cyberedu/
        bare_dir = processed_root / "crypto" / "bare_challenge"
        bare_dir.mkdir(parents=True)
        # No cyberedu/ subdirectory → should be excluded

        entries = cl.load_challenges("processed", ["crypto"])
        entry_names = [e.path.name for e in entries]
        assert (
            "bare_challenge" not in entry_names
        ), "Challenges without cyberedu/ must be excluded"
        assert "rsb" in entry_names, "Valid challenge rsb must be included"
