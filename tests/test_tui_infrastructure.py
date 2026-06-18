"""TUI infrastructure tests — tested without UI.

Verifies:
1. PromptCaptureHandler queues prompt on on_llm_start call.
2. ArtifactWriter.start_run writes manifest.json with status="running".
3. ArtifactWriter.finish_run updates manifest status to "complete".
4. ArtifactWriter.write_course writes correct file path.
5. ArtifactWriter.append_llm_call appends lines to llm_calls.jsonl.
6. challenge_loader.get_available_categories returns a list without raising.
7. challenge_loader.load_challenges returns a list without raising.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# PromptCaptureHandler
# ---------------------------------------------------------------------------


class TestPromptCaptureHandler:
    """Tests for PromptCaptureHandler."""

    def test_on_llm_start_puts_prompt_in_queue(self):
        """on_llm_start with a prompt list should put the first prompt in the queue."""
        from src.tui.callback import PromptCaptureHandler

        queue: asyncio.Queue = asyncio.Queue()
        handler = PromptCaptureHandler(queue)

        asyncio.run(
            handler.on_llm_start(
                serialized={},
                prompts=["You are an expert CTF educator. Generate a course."],
                run_id=uuid.uuid4(),
            )
        )

        assert not queue.empty()
        item = queue.get_nowait()
        assert item["prompt"] == "You are an expert CTF educator. Generate a course."

    def test_on_llm_start_empty_prompts_puts_empty_string(self):
        """on_llm_start with empty prompts list should put empty string."""
        from src.tui.callback import PromptCaptureHandler

        queue: asyncio.Queue = asyncio.Queue()
        handler = PromptCaptureHandler(queue)

        asyncio.run(
            handler.on_llm_start(
                serialized={},
                prompts=[],
                run_id=uuid.uuid4(),
            )
        )

        assert not queue.empty()
        item = queue.get_nowait()
        assert item["prompt"] == ""

    def test_multiple_calls_queue_multiple_prompts(self):
        """Multiple on_llm_start calls should each put one item in the queue."""
        from src.tui.callback import PromptCaptureHandler

        queue: asyncio.Queue = asyncio.Queue()
        handler = PromptCaptureHandler(queue)

        async def _run():
            await handler.on_llm_start(
                serialized={}, prompts=["first"], run_id=uuid.uuid4()
            )
            await handler.on_llm_start(
                serialized={}, prompts=["second"], run_id=uuid.uuid4()
            )

        asyncio.run(_run())

        assert queue.qsize() == 2
        assert queue.get_nowait()["prompt"] == "first"
        assert queue.get_nowait()["prompt"] == "second"


# ---------------------------------------------------------------------------
# ArtifactWriter
# ---------------------------------------------------------------------------


def _make_run_config(exp_id: str = "TEST-001"):
    from src.tui.run_config import RunConfig

    return RunConfig(
        exp_id=exp_id,
        provider="openai",
        model="gpt-4o",
        temperature=0.7,
        threshold=9.0,
        challenge_ids=["rsb", "careflow"],
        categories=["crypto", "web"],
        source="local",
        max_refinements=3,
        skip_ranking=False,
    )


class TestArtifactWriter:
    """Tests for ArtifactWriter (uses tmp_path — no real filesystem side effects)."""

    def test_start_run_creates_manifest_with_status_running(self, tmp_path: Path):
        """start_run should write manifest.json with status='running'."""
        from src.tui.artifact_writer import ArtifactWriter

        cfg = _make_run_config("EXP-TEST-001")
        writer = ArtifactWriter(base_dir=tmp_path, cfg=cfg)
        writer.start_run(cfg)

        manifest_path = tmp_path / "EXP-TEST-001" / "manifest.json"
        assert manifest_path.exists(), "manifest.json was not created"

        data = json.loads(manifest_path.read_text())
        assert data["status"] == "running"
        assert data["exp_id"] == "EXP-TEST-001"
        assert data["finished_at"] is None

    def test_start_run_creates_run_config_json(self, tmp_path: Path):
        """start_run should also write run_config.json."""
        from src.tui.artifact_writer import ArtifactWriter

        cfg = _make_run_config("EXP-TEST-002")
        writer = ArtifactWriter(base_dir=tmp_path, cfg=cfg)
        writer.start_run(cfg)

        run_config_path = tmp_path / "EXP-TEST-002" / "run_config.json"
        assert run_config_path.exists(), "run_config.json was not created"

        data = json.loads(run_config_path.read_text())
        assert data["exp_id"] == "EXP-TEST-002"
        assert data["provider"] == "openai"

    def test_finish_run_updates_status_to_complete(self, tmp_path: Path):
        """finish_run(success=True) should update manifest status to 'complete'."""
        from src.tui.artifact_writer import ArtifactWriter

        cfg = _make_run_config("EXP-TEST-003")
        writer = ArtifactWriter(base_dir=tmp_path, cfg=cfg)
        writer.start_run(cfg)
        writer.finish_run(
            success=True, node_timings={"coordinator": 0.5, "ranking": 2.1}
        )

        manifest_path = tmp_path / "EXP-TEST-003" / "manifest.json"
        data = json.loads(manifest_path.read_text())
        assert data["status"] == "complete"
        assert data["finished_at"] is not None
        assert data["node_timings"] == {"coordinator": 0.5, "ranking": 2.1}

    def test_finish_run_failed_sets_status_failed(self, tmp_path: Path):
        """finish_run(success=False) should update manifest status to 'failed'."""
        from src.tui.artifact_writer import ArtifactWriter

        cfg = _make_run_config("EXP-TEST-004")
        writer = ArtifactWriter(base_dir=tmp_path, cfg=cfg)
        writer.start_run(cfg)
        writer.finish_run(success=False, node_timings={})

        manifest_path = tmp_path / "EXP-TEST-004" / "manifest.json"
        data = json.loads(manifest_path.read_text())
        assert data["status"] == "failed"

    def test_write_course_with_category_slash_name(self, tmp_path: Path):
        """write_course('crypto/rsb', ...) should write to courses/crypto/rsb/course.md."""
        from src.tui.artifact_writer import ArtifactWriter

        cfg = _make_run_config("EXP-TEST-005")
        writer = ArtifactWriter(base_dir=tmp_path, cfg=cfg)
        writer.start_run(cfg)
        writer.write_course("crypto/rsb", "# RSB Course\n\nContent here.")

        course_path = (
            tmp_path / "EXP-TEST-005" / "courses" / "crypto" / "rsb" / "course.md"
        )
        assert course_path.exists(), f"course.md not found at {course_path}"
        assert "RSB Course" in course_path.read_text()

    def test_write_course_without_slash_uses_misc_category(self, tmp_path: Path):
        """write_course('rsb', ...) without slash should use 'misc' as default category."""
        from src.tui.artifact_writer import ArtifactWriter

        cfg = _make_run_config("EXP-TEST-006")
        writer = ArtifactWriter(base_dir=tmp_path, cfg=cfg)
        writer.start_run(cfg)
        writer.write_course("rsb", "# RSB content")

        course_path = (
            tmp_path / "EXP-TEST-006" / "courses" / "misc" / "rsb" / "course.md"
        )
        assert course_path.exists(), f"course.md not found at {course_path}"

    def test_append_llm_call_single_line(self, tmp_path: Path):
        """append_llm_call should create llm_calls.jsonl with one line."""
        from src.tui.artifact_writer import ArtifactWriter

        cfg = _make_run_config("EXP-TEST-007")
        writer = ArtifactWriter(base_dir=tmp_path, cfg=cfg)
        writer.start_run(cfg)
        writer.append_llm_call(
            node="content_generation",
            prompt="You are an expert CTF educator.",
            tokens=["Here", " is", " the", " course"],
            duration_s=1.23,
        )

        jsonl_path = tmp_path / "EXP-TEST-007" / "llm_calls.jsonl"
        assert jsonl_path.exists(), "llm_calls.jsonl not created"
        lines = jsonl_path.read_text().strip().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["node"] == "content_generation"
        assert record["prompt"] == "You are an expert CTF educator."
        assert record["response_tokens"] == ["Here", " is", " the", " course"]
        assert record["duration_s"] == pytest.approx(1.23)

    def test_append_llm_call_two_calls_two_lines(self, tmp_path: Path):
        """Two append_llm_call calls should produce two lines in llm_calls.jsonl."""
        from src.tui.artifact_writer import ArtifactWriter

        cfg = _make_run_config("EXP-TEST-008")
        writer = ArtifactWriter(base_dir=tmp_path, cfg=cfg)
        writer.start_run(cfg)
        writer.append_llm_call("ranking", "Prompt 1", ["tok1"], 0.5)
        writer.append_llm_call("content_generation", "Prompt 2", ["tok2", "tok3"], 1.0)

        jsonl_path = tmp_path / "EXP-TEST-008" / "llm_calls.jsonl"
        lines = jsonl_path.read_text().strip().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["node"] == "ranking"
        assert json.loads(lines[1])["node"] == "content_generation"

    def test_write_ranking_creates_file(self, tmp_path: Path):
        """write_ranking should write ranking_reports.json."""
        from src.tui.artifact_writer import ArtifactWriter

        cfg = _make_run_config("EXP-TEST-009")
        writer = ArtifactWriter(base_dir=tmp_path, cfg=cfg)
        writer.start_run(cfg)

        reports = [
            {
                "challenge_id": "rsb",
                "overall_score": 9.5,
                "technical_score": 9.8,
                "pedagogical_score": 9.2,
            }
        ]
        writer.write_ranking(reports)

        ranking_path = tmp_path / "EXP-TEST-009" / "ranking_reports.json"
        assert ranking_path.exists(), "ranking_reports.json not created"
        data = json.loads(ranking_path.read_text())
        assert isinstance(data, list)
        assert data[0]["challenge_id"] == "rsb"

    def test_write_ranking_normalises_name_source_to_category_name(
        self, tmp_path: Path
    ):
        """D11 regression: ranking_reports IDs in name/source format must be normalised
        to category/name format matching manifest.challenge_ids."""
        from src.tui.artifact_writer import ArtifactWriter
        from src.tui.run_config import RunConfig

        cfg = RunConfig(
            exp_id="EXP-D11-TEST",
            provider="anthropic",
            model="claude-sonnet-4-6",
            temperature=0.0,
            threshold=9.0,
            # challenge_ids in cfg will be later overridden to category/name via update_challenge_ids
            challenge_ids=[],
            categories=["crypto"],
            source="processed",
            max_refinements=5,
            skip_ranking=False,
        )
        writer = ArtifactWriter(base_dir=tmp_path, cfg=cfg)
        writer.start_run(cfg)
        # Simulate update_challenge_ids with canonical category/name IDs
        writer.update_challenge_ids(["crypto/0_solves", "crypto/frigography"])

        # Ranking agent produces name/source IDs (the pre-fix format)
        reports = [
            {
                "challenge_id": "0_solves/cyberedu",
                "overall_score": 9.5,
                "technical_review": {"score": 9},
                "pedagogical_review": {"score": 10},
            },
            {
                "challenge_id": "frigography/cyberedu",
                "overall_score": 8.0,
                "technical_review": {"score": 8},
                "pedagogical_review": {"score": 8},
            },
        ]
        writer.write_ranking(reports)

        ranking_path = tmp_path / "EXP-D11-TEST" / "ranking_reports.json"
        data = json.loads(ranking_path.read_text())
        assert isinstance(data, list)
        ids = [r["challenge_id"] for r in data]
        # Must match manifest format exactly
        assert ids == [
            "crypto/0_solves",
            "crypto/frigography",
        ], f"Expected canonical IDs but got: {ids}"

    def test_write_ranking_preserves_already_canonical_ids(self, tmp_path: Path):
        """D11: IDs already in category/name format must not be modified."""
        from src.tui.artifact_writer import ArtifactWriter
        from src.tui.run_config import RunConfig

        cfg = RunConfig(
            exp_id="EXP-D11-CANONICAL",
            provider="openai",
            model="gpt-4o",
            temperature=0.0,
            threshold=9.0,
            challenge_ids=[],
            categories=["web"],
            source="processed",
            max_refinements=3,
            skip_ranking=False,
        )
        writer = ArtifactWriter(base_dir=tmp_path, cfg=cfg)
        writer.start_run(cfg)
        writer.update_challenge_ids(["web/sqli", "web/xss"])

        reports = [
            {"challenge_id": "web/sqli", "overall_score": 9.0},
            {"challenge_id": "web/xss", "overall_score": 8.5},
        ]
        writer.write_ranking(reports)

        ranking_path = tmp_path / "EXP-D11-CANONICAL" / "ranking_reports.json"
        data = json.loads(ranking_path.read_text())
        ids = [r["challenge_id"] for r in data]
        assert ids == ["web/sqli", "web/xss"]


# ---------------------------------------------------------------------------
# challenge_loader
# ---------------------------------------------------------------------------


class TestChallengeLoader:
    """Tests for challenge_loader — non-raising behavior and basic contracts."""

    def test_get_available_categories_local_returns_list(self):
        """get_available_categories for local source returns a list (may be empty)."""
        from src.tui.challenge_loader import get_available_categories

        result = get_available_categories("local")
        assert isinstance(result, list), "Expected a list"
        # Each item should be a string
        for item in result:
            assert isinstance(item, str)

    def test_get_available_categories_processed_returns_list(self):
        """get_available_categories for processed returns a list (may be empty)."""
        from src.tui.challenge_loader import get_available_categories

        result = get_available_categories("processed")
        assert isinstance(result, list)

    def test_get_available_categories_unknown_source_returns_empty(self):
        """get_available_categories for unknown source returns empty list without raising."""
        from src.tui.challenge_loader import get_available_categories

        result = get_available_categories("nonexistent-source")
        assert result == []

    def test_load_challenges_local_returns_list(self):
        """load_challenges for local source returns a list without raising."""
        from src.tui.challenge_loader import load_challenges

        result = load_challenges("local")
        assert isinstance(result, list)

    def test_load_challenges_filtered_by_category(self):
        """load_challenges with categories filter returns only matching entries."""
        from src.tui.challenge_loader import ChallengeEntry, load_challenges

        result = load_challenges("local", categories=["crypto"])
        assert isinstance(result, list)
        for entry in result:
            assert entry.category == "crypto"
            assert isinstance(entry, ChallengeEntry)

    def test_load_challenges_processed_returns_list(self):
        """load_challenges for processed source returns a list without raising."""
        from src.tui.challenge_loader import load_challenges

        result = load_challenges("processed", categories=["crypto"])
        assert isinstance(result, list)

    def test_load_challenges_missing_dir_returns_empty(
        self, tmp_path: Path, monkeypatch
    ):
        """load_challenges with non-existent root returns empty list."""
        import src.tui.challenge_loader as cl

        monkeypatch.setattr(cl, "_LOCAL_DIR", tmp_path / "nonexistent")
        result = cl.load_challenges("local")
        assert result == []

    def test_challenge_entry_fields(self):
        """ChallengeEntry dataclass has all required fields."""
        from src.tui.challenge_loader import ChallengeEntry

        entry = ChallengeEntry(
            challenge_id="rsb",
            category="crypto",
            source="local",
            path=Path("/some/path"),
        )
        assert entry.challenge_id == "rsb"
        assert entry.category == "crypto"
        assert entry.source == "local"
        assert entry.path == Path("/some/path")

    def test_local_entries_have_correct_source(self):
        """All entries from local source have source='local'."""
        from src.tui.challenge_loader import load_challenges

        result = load_challenges("local")
        for entry in result:
            assert entry.source == "local"

    def test_processed_entries_have_correct_source(self):
        """All entries from processed source have source='processed'."""
        from src.tui.challenge_loader import load_challenges

        result = load_challenges("processed")
        for entry in result:
            assert entry.source == "processed"
