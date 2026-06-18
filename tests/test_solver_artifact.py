"""Tests for solve_generated.py artifact landing on disk.

M3 regression: when EXPERIMENT_ID is set, solve_generated.py must land
in the same experiment directory as course.md, not silently to PROCESSED_DIR.

Pre-fix: these tests FAIL because _write_generated_to_disk always writes
solve_generated.py to wu_dir (PROCESSED_DIR path) regardless of EXPERIMENT_ID.

Post-fix: solve_generated.py follows course.md into the exp output dir.
"""

from src.agents.content_generation_agent import _write_generated_to_disk
from src.config.settings import settings


def test_solve_script_written_to_exp_dir_when_experiment_id_set(tmp_path, monkeypatch):
    """When EXPERIMENT_ID is set, solve_generated.py must land in the exp output dir."""
    monkeypatch.setattr(settings, "EXPERIMENT_ID", "EXP-TEST")
    monkeypatch.setattr(settings, "OUTPUT_DIR", tmp_path)

    challenge_dir = tmp_path / "crypto" / "test-challenge"
    challenge_dir.mkdir(parents=True)

    paths_with_category = [(challenge_dir, "crypto")]
    courses = {"crypto/test-challenge": "# Course content"}
    scripts = {"crypto/test-challenge": "#!/usr/bin/env python3\nprint('flag')"}

    _write_generated_to_disk(paths_with_category, courses, scripts)

    # solve_generated.py must appear next to course.md in the experiment dir
    expected_script = (
        tmp_path / "EXP-TEST" / "crypto" / "test-challenge" / "solve_generated.py"
    )
    assert (
        expected_script.exists()
    ), f"Expected solve_generated.py at {expected_script} but it was not written"
    assert (
        expected_script.read_text().strip() != ""
    ), "solve_generated.py must not be empty"

    # The file must NOT be written to the PROCESSED_DIR path
    wrong_path = challenge_dir / "cyberedu" / "write-up" / "solve_generated.py"
    assert (
        not wrong_path.exists()
    ), "solve_generated.py must not be written to PROCESSED_DIR path when EXPERIMENT_ID is set"


def test_solve_script_written_to_default_path_when_no_experiment_id(
    tmp_path, monkeypatch
):
    """When EXPERIMENT_ID is empty, solve_generated.py goes to the default challenge dir."""
    monkeypatch.setattr(settings, "EXPERIMENT_ID", "")
    monkeypatch.setattr(settings, "OUTPUT_DIR", tmp_path / "outputs")

    challenge_dir = tmp_path / "crypto" / "test-challenge"
    challenge_dir.mkdir(parents=True)

    paths_with_category = [(challenge_dir, "crypto")]
    courses = {"crypto/test-challenge": "# Course content"}
    scripts = {"crypto/test-challenge": "#!/usr/bin/env python3\nprint('flag')"}

    _write_generated_to_disk(paths_with_category, courses, scripts)

    expected_script = challenge_dir / "cyberedu" / "write-up" / "solve_generated.py"
    assert (
        expected_script.exists()
    ), f"Expected solve_generated.py at {expected_script} but it was not written"


def test_solve_script_non_empty_after_write(tmp_path, monkeypatch):
    """After _write_generated_to_disk, the script file must be non-empty."""
    monkeypatch.setattr(settings, "EXPERIMENT_ID", "")
    monkeypatch.setattr(settings, "OUTPUT_DIR", tmp_path / "outputs")

    challenge_dir = tmp_path / "crypto" / "test2"
    challenge_dir.mkdir(parents=True)

    script_content = "# solve.py\nimport socket\nprint('hello')"
    _write_generated_to_disk(
        [(challenge_dir, "crypto")],
        {"crypto/test2": "# Course"},
        {"crypto/test2": script_content},
    )

    p = challenge_dir / "cyberedu" / "write-up" / "solve_generated.py"
    assert p.exists()
    assert p.read_text(encoding="utf-8") == script_content


def test_token_constants_at_v2_values():
    """Regression: token limits must be at their v2 bumped values."""
    assert (
        settings.CONTENT_GENERATION_MAX_TOKENS >= 14000
    ), f"CONTENT_GENERATION_MAX_TOKENS={settings.CONTENT_GENERATION_MAX_TOKENS} < 14000"
    assert (
        settings.CONTENT_GENERATION_SOLVE_MAX_TOKENS >= 6000
    ), f"CONTENT_GENERATION_SOLVE_MAX_TOKENS={settings.CONTENT_GENERATION_SOLVE_MAX_TOKENS} < 6000"
