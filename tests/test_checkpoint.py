"""Tests for per-challenge checkpoint + resume (Feature 2).

Tests verify:
1. save_atomic writes valid JSON and the tmp file is gone after.
2. load returns None when checkpoint file is absent.
3. record_challenge_done appends to completed_challenges and persists.
4. Atomic write: if tmp file exists and rename has not happened, original is unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_exp_dir(tmp_path: Path) -> Path:
    """Create a temporary experiment directory."""
    exp_dir = tmp_path / "EXP-TEST"
    exp_dir.mkdir(parents=True)
    return exp_dir


# ---------------------------------------------------------------------------
# Test 1: save_atomic writes valid JSON; tmp file is gone
# ---------------------------------------------------------------------------


def test_save_atomic_writes_valid_json(tmp_exp_dir: Path):
    """save_atomic should produce a valid checkpoint.json file."""
    from src.core.checkpoint import Checkpoint, save_atomic

    cp = Checkpoint(
        exp_id="EXP-TEST",
        completed_challenges=["crypto/challenge-a"],
        in_progress_challenge="crypto/challenge-b",
    )
    save_atomic(tmp_exp_dir, cp)

    ckpt_file = tmp_exp_dir / ".checkpoint.json"
    assert ckpt_file.exists(), "checkpoint.json was not created"

    data = json.loads(ckpt_file.read_text())
    assert data["exp_id"] == "EXP-TEST"
    assert "crypto/challenge-a" in data["completed_challenges"]
    assert data["checkpoint_version"] == 1


def test_save_atomic_no_tmp_file_after(tmp_exp_dir: Path):
    """No .checkpoint.json.tmp should remain after save_atomic."""
    from src.core.checkpoint import Checkpoint, save_atomic

    cp = Checkpoint(exp_id="EXP-TEST", completed_challenges=[])
    save_atomic(tmp_exp_dir, cp)

    tmp_file = tmp_exp_dir / ".checkpoint.json.tmp"
    assert not tmp_file.exists(), ".tmp file still present after save_atomic"


# ---------------------------------------------------------------------------
# Test 2: load returns None when checkpoint is absent
# ---------------------------------------------------------------------------


def test_load_returns_none_when_absent(tmp_exp_dir: Path):
    """load() should return None when no checkpoint file exists."""
    from src.core.checkpoint import load

    result = load(tmp_exp_dir)
    assert result is None


def test_load_returns_checkpoint_when_present(tmp_exp_dir: Path):
    """load() should return a Checkpoint when the file exists."""
    from src.core.checkpoint import Checkpoint, load, save_atomic

    cp = Checkpoint(
        exp_id="EXP-TEST",
        completed_challenges=["web/xss-basic"],
    )
    save_atomic(tmp_exp_dir, cp)

    loaded = load(tmp_exp_dir)
    assert loaded is not None
    assert loaded.exp_id == "EXP-TEST"
    assert "web/xss-basic" in loaded.completed_challenges


# ---------------------------------------------------------------------------
# Test 3: record_challenge_done appends and persists
# ---------------------------------------------------------------------------


def test_record_challenge_done_appends(tmp_exp_dir: Path):
    """record_challenge_done should add the challenge_id to completed_challenges."""
    from src.core.checkpoint import Checkpoint, load, record_challenge_done, save_atomic

    cp = Checkpoint(exp_id="EXP-TEST", completed_challenges=["crypto/a"])
    save_atomic(tmp_exp_dir, cp)

    record_challenge_done(tmp_exp_dir, "crypto/b")

    loaded = load(tmp_exp_dir)
    assert loaded is not None
    assert "crypto/a" in loaded.completed_challenges
    assert "crypto/b" in loaded.completed_challenges


def test_record_challenge_done_creates_checkpoint_if_absent(tmp_exp_dir: Path):
    """record_challenge_done should work even if no checkpoint file exists yet."""
    from src.core.checkpoint import load, record_challenge_done

    record_challenge_done(tmp_exp_dir, "misc/first")

    loaded = load(tmp_exp_dir)
    assert loaded is not None
    assert "misc/first" in loaded.completed_challenges


def test_record_challenge_done_no_duplicates(tmp_exp_dir: Path):
    """Calling record_challenge_done twice with the same id should not duplicate."""
    from src.core.checkpoint import load, record_challenge_done

    record_challenge_done(tmp_exp_dir, "crypto/dupe")
    record_challenge_done(tmp_exp_dir, "crypto/dupe")

    loaded = load(tmp_exp_dir)
    assert loaded is not None
    assert loaded.completed_challenges.count("crypto/dupe") == 1


# ---------------------------------------------------------------------------
# Test 4: Atomic write — original file unchanged if only .tmp exists
# ---------------------------------------------------------------------------


def test_atomic_write_original_unchanged_if_tmp_only(tmp_exp_dir: Path):
    """If we write the .tmp file manually but don't rename it,
    the original checkpoint.json should be unchanged."""
    from src.core.checkpoint import Checkpoint, save_atomic

    # Write an initial checkpoint
    original_cp = Checkpoint(exp_id="EXP-TEST", completed_challenges=["original"])
    save_atomic(tmp_exp_dir, original_cp)

    ckpt_file = tmp_exp_dir / ".checkpoint.json"
    original_content = ckpt_file.read_bytes()

    # Simulate a crash by writing a .tmp file directly (without rename)
    tmp_file = tmp_exp_dir / ".checkpoint.json.tmp"
    tmp_file.write_text(
        json.dumps({"exp_id": "EXP-TEST", "completed_challenges": ["tampered"]}),
        encoding="utf-8",
    )

    # The original file should still be intact
    assert (
        ckpt_file.read_bytes() == original_content
    ), "Original checkpoint.json was modified even though .tmp file was not renamed"

    # Cleanup: load still reads from .checkpoint.json (original)
    from src.core.checkpoint import load

    loaded = load(tmp_exp_dir)
    assert loaded is not None
    assert "original" in loaded.completed_challenges
    assert "tampered" not in loaded.completed_challenges
