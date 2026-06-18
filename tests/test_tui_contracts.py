"""TUI contract tests — data types only.

Verifies:
- All 8 event Message subclasses instantiate and expose expected fields.
- RunConfig instantiates and exposes all fields.
- ManifestData instantiates and exposes all fields.
- ArtifactWriter class is importable and instantiable.
- No circular imports between tui modules.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Circular-import guard — import all tui modules at top level
# ---------------------------------------------------------------------------
import src.tui  # noqa: F401
import src.tui.artifact_writer  # noqa: F401
import src.tui.events  # noqa: F401
import src.tui.run_config  # noqa: F401

# ---------------------------------------------------------------------------
# RunConfig
# ---------------------------------------------------------------------------


class TestRunConfig:
    def _make(self, **overrides):
        from src.tui.run_config import RunConfig

        defaults = dict(
            exp_id="run-001",
            provider="openai",
            model="gpt-4o",
            temperature=0.0,
            threshold=9.0,
            challenge_ids=["crypto/rsb"],
            categories=["crypto"],
            source="local",
            max_refinements=5,
            skip_ranking=False,
        )
        defaults.update(overrides)
        return RunConfig(**defaults)

    def test_all_fields(self):
        cfg = self._make()
        assert cfg.exp_id == "run-001"
        assert cfg.provider == "openai"
        assert cfg.model == "gpt-4o"
        assert cfg.temperature == 0.0
        assert cfg.threshold == 9.0
        assert cfg.challenge_ids == ["crypto/rsb"]
        assert cfg.categories == ["crypto"]
        assert cfg.source == "local"
        assert cfg.max_refinements == 5
        assert cfg.skip_ranking is False

    def test_processed_source(self):
        cfg = self._make(source="processed")
        assert cfg.source == "processed"

    def test_skip_ranking_true(self):
        cfg = self._make(skip_ranking=True)
        assert cfg.skip_ranking is True

    def test_defaults(self):
        from src.tui.run_config import RunConfig

        cfg = RunConfig(
            exp_id="run-002",
            provider="anthropic",
            model="claude-3-5-sonnet-20241022",
            temperature=0.0,
            threshold=9.0,
        )
        assert cfg.challenge_ids == []
        assert cfg.categories == []
        assert cfg.source == "local"
        assert cfg.max_refinements == 5
        assert cfg.skip_ranking is False


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


class TestNodeStarted:
    def test_fields(self):
        from src.tui.events import NodeStarted

        ev = NodeStarted(run_id="run-001", node_name="coordinator")
        assert ev.run_id == "run-001"
        assert ev.node_name == "coordinator"


class TestNodeFinished:
    def test_fields_no_error(self):
        from src.tui.events import NodeFinished

        ev = NodeFinished(run_id="run-001", node_name="ranking", duration_s=3.14)
        assert ev.run_id == "run-001"
        assert ev.node_name == "ranking"
        assert ev.duration_s == pytest.approx(3.14)
        assert ev.error is None

    def test_fields_with_error(self):
        from src.tui.events import NodeFinished

        ev = NodeFinished(
            run_id="run-001", node_name="ranking", duration_s=0.5, error="timeout"
        )
        assert ev.error == "timeout"


class TestLLMToken:
    def test_fields(self):
        from src.tui.events import LLMToken

        ev = LLMToken(run_id="run-001", node_name="content_generation", token="Hello")
        assert ev.run_id == "run-001"
        assert ev.node_name == "content_generation"
        assert ev.token == "Hello"


class TestPromptCaptured:
    def test_fields(self):
        from src.tui.events import PromptCaptured

        ev = PromptCaptured(run_id="run-001", prompt="You are an expert CTF educator.")
        assert ev.run_id == "run-001"
        assert ev.prompt == "You are an expert CTF educator."


class TestChallengeScored:
    def test_fields(self):
        from src.tui.events import ChallengeScored

        ev = ChallengeScored(
            run_id="run-001",
            challenge_id="crypto/rsb",
            overall=8.5,
            technical=9.0,
            pedagogical=8.0,
        )
        assert ev.run_id == "run-001"
        assert ev.challenge_id == "crypto/rsb"
        assert ev.overall == pytest.approx(8.5)
        assert ev.technical == pytest.approx(9.0)
        assert ev.pedagogical == pytest.approx(8.0)


class TestRunStarted:
    def test_fields(self):
        from src.tui.events import RunStarted

        ev = RunStarted(run_id="run-001", challenge_count=3)
        assert ev.run_id == "run-001"
        assert ev.challenge_count == 3


class TestRunFinished:
    def test_success(self):
        from src.tui.events import RunFinished

        ev = RunFinished(run_id="run-001", success=True, elapsed_s=120.0)
        assert ev.run_id == "run-001"
        assert ev.success is True
        assert ev.elapsed_s == pytest.approx(120.0)
        assert ev.error is None

    def test_failure(self):
        from src.tui.events import RunFinished

        ev = RunFinished(
            run_id="run-001", success=False, elapsed_s=5.0, error="LLM error"
        )
        assert ev.success is False
        assert ev.error == "LLM error"


class TestHITLPaused:
    def test_fields(self):
        from src.tui.events import HITLPaused

        ev = HITLPaused(run_id="run-001")
        assert ev.run_id == "run-001"


# ---------------------------------------------------------------------------
# ManifestData
# ---------------------------------------------------------------------------


class TestManifestData:
    def test_all_fields(self):
        from src.tui.artifact_writer import ManifestData

        md = ManifestData(
            exp_id="run-001",
            status="running",
            started_at="2026-05-11T10:00:00Z",
            finished_at=None,
            challenge_ids=["crypto/rsb", "web/careflow"],
            node_timings={"coordinator": 0.5, "ranking": 3.0},
            pass_count=1,
            mean_overall_score=8.75,
            settings_snapshot={"model": "gpt-4o", "threshold": 9.0},
        )
        assert md.exp_id == "run-001"
        assert md.status == "running"
        assert md.started_at == "2026-05-11T10:00:00Z"
        assert md.finished_at is None
        assert md.challenge_ids == ["crypto/rsb", "web/careflow"]
        assert md.node_timings == {"coordinator": 0.5, "ranking": 3.0}
        assert md.pass_count == 1
        assert md.mean_overall_score == pytest.approx(8.75)
        assert md.settings_snapshot == {"model": "gpt-4o", "threshold": 9.0}

    def test_defaults(self):
        from src.tui.artifact_writer import ManifestData

        md = ManifestData(
            exp_id="run-002",
            status="complete",
            started_at="2026-05-11T11:00:00Z",
            finished_at="2026-05-11T11:30:00Z",
        )
        assert md.challenge_ids == []
        assert md.node_timings == {}
        assert md.pass_count == 0
        assert md.mean_overall_score == pytest.approx(0.0)
        assert md.settings_snapshot == {}


# ---------------------------------------------------------------------------
# ArtifactWriter — class-level instantiation only (stubs, no I/O)
# ---------------------------------------------------------------------------


class TestArtifactWriter:
    def test_instantiable(self):
        from src.tui.artifact_writer import ArtifactWriter
        from src.tui.run_config import RunConfig

        cfg = RunConfig(
            exp_id="EXP-TEST",
            provider="openai",
            model="gpt-4o",
            temperature=0.0,
            threshold=9.0,
        )
        # Stubs have `...` bodies — instantiation must not raise
        writer = ArtifactWriter(base_dir=Path("/tmp/test_artifacts"), cfg=cfg)
        assert writer is not None

    def test_stub_methods_exist(self):
        """Confirm all method names are present (to be implemented)."""
        from src.tui.artifact_writer import ArtifactWriter

        for method in (
            "start_run",
            "write_course",
            "append_llm_call",
            "write_ranking",
            "finish_run",
        ):
            assert hasattr(ArtifactWriter, method), f"Missing method: {method}"
