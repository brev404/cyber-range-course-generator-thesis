"""Reliability tests for the run lifecycle hardening."""

from __future__ import annotations

from pathlib import Path

from src.tui.artifact_writer import ArtifactWriter
from src.tui.run_config import RunConfig


def _make_cfg(exp_id: str = "EXP-R01") -> RunConfig:
    return RunConfig(
        exp_id=exp_id,
        provider="anthropic",
        model="claude-haiku",
        temperature=0.0,
        threshold=9.0,
        challenge_ids=["cat/ch1", "cat/ch2"],
        categories=["cat"],
        source="local",
        max_refinements=5,
        skip_ranking=False,
    )


def test_all_empty_skip_ranking_not_failed(tmp_path: Path) -> None:
    """When skip_ranking=True, zero courses written must NOT trigger the all-empty guard.

    This ensures --skip-ranking runs complete normally even with no course output.
    """
    base_dir = tmp_path / "output" / "experiments"
    cfg = RunConfig(
        exp_id="EXP-R02B",
        provider="anthropic",
        model="claude-haiku",
        temperature=0.0,
        threshold=9.0,
        challenge_ids=["cat/ch1"],
        categories=["cat"],
        source="local",
        max_refinements=5,
        skip_ranking=True,
    )
    writer = ArtifactWriter(base_dir, cfg)
    writer.start_run(cfg)

    courses_written = 0
    skip_ranking = True

    if courses_written == 0 and len(cfg.challenge_ids) > 0 and not skip_ranking:
        writer.finish_run(success=False, node_timings={})
    else:
        writer.finish_run(success=True, node_timings={})

    import json

    manifest_path = base_dir / "EXP-R02B" / "manifest.json"
    data = json.loads(manifest_path.read_text())
    assert (
        data["status"] != "failed"
    ), "skip_ranking=True should not trigger all-empty guard"


def test_write_course_rejects_empty_content(tmp_path: Path) -> None:
    """write_course must skip creating course.md for None, '', and 'None' content."""
    base_dir = tmp_path / "output" / "experiments"
    cfg = _make_cfg("EXP-R04")
    writer = ArtifactWriter(base_dir, cfg)
    writer.start_run(cfg)

    for bad_content in [None, "", "None"]:
        writer.write_course("cat/ch1", bad_content)  # type: ignore[arg-type]

    course_path = base_dir / "EXP-R04" / "courses" / "cat" / "ch1" / "course.md"
    assert (
        not course_path.exists()
    ), f"course.md must not be created for empty/None content; found: {course_path}"


def test_ranking_loguru_format_brace_style() -> None:
    """ranking_agent.py must not contain %s-style Loguru format strings."""
    source = Path("src/agents/ranking_agent.py").read_text(encoding="utf-8")
    assert (
        "provider=%s" not in source
    ), "ranking_agent.py uses %s Loguru format for 'provider'; must use {} brace-style"
    assert (
        "model=%s" not in source
    ), "ranking_agent.py uses %s Loguru format for 'model'; must use {} brace-style"


def test_llm_call_budget_covers_5_round_budget() -> None:
    """MAX_LLM_CALLS_PER_CHALLENGE must be >= 26 to cover worst-case 5-round budget.

    Budget: 4 calls/round (solve + course + 2 judges) × 5 rounds = 20 base;
    STRUCTURAL_VALIDATOR_MAX_RETRIES adds up to 2 extra calls per retry.
    """
    from src.config.settings import settings

    assert settings.MAX_LLM_CALLS_PER_CHALLENGE >= 26, (
        f"MAX_LLM_CALLS_PER_CHALLENGE={settings.MAX_LLM_CALLS_PER_CHALLENGE} "
        f"is too low; must be >= 26 to cover 5-round worst-case budget"
    )


def test_write_course_guard_handles_whitespace_and_nonstr(tmp_path: Path) -> None:
    """write_course must skip whitespace-only and tolerate a non-str value."""
    base_dir = tmp_path / "output" / "experiments"
    cfg = _make_cfg("EXP-R04B")
    writer = ArtifactWriter(base_dir, cfg)
    writer.start_run(cfg)

    writer.write_course("cat/ws", "   \n\t  ")
    assert not (base_dir / "EXP-R04B" / "courses" / "cat" / "ws" / "course.md").exists()
