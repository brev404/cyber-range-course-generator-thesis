"""Tests for atomic write helpers (M4).

Pre-fix: atomic_write module does not exist → ImportError.
Post-fix: module exists and behaves correctly under normal and crash conditions.
"""

import json
from unittest.mock import patch

import pytest


def test_atomic_write_text_creates_file(tmp_path):
    """atomic_write_text must create the file with the correct content."""
    from src.services.atomic_write import atomic_write_text

    target = tmp_path / "test.md"
    atomic_write_text(target, "# Hello\nworld")
    assert target.exists()
    assert target.read_text(encoding="utf-8") == "# Hello\nworld"


def test_atomic_write_json_creates_file(tmp_path):
    """atomic_write_json must create valid JSON."""
    from src.services.atomic_write import atomic_write_json

    target = tmp_path / "data.json"
    data = {"key": "value", "n": 42}
    atomic_write_json(target, data)
    assert target.exists()
    parsed = json.loads(target.read_text(encoding="utf-8"))
    assert parsed == data


def test_atomic_write_text_no_tmp_left_on_success(tmp_path):
    """After a successful write, no .tmp file must remain."""
    from src.services.atomic_write import atomic_write_text

    target = tmp_path / "out.md"
    atomic_write_text(target, "content")
    tmp_candidates = list(tmp_path.glob("*.tmp"))
    assert tmp_candidates == [], f"Leftover .tmp files: {tmp_candidates}"


def test_atomic_write_text_original_not_corrupted_on_crash(tmp_path):
    """If os.rename raises mid-write, the original file (if any) is preserved."""
    from src.services.atomic_write import atomic_write_text

    target = tmp_path / "existing.md"
    target.write_text("original content", encoding="utf-8")

    with patch("os.rename", side_effect=OSError("rename failed")):
        with pytest.raises(OSError):
            atomic_write_text(target, "new content")

    # Original must be intact
    assert target.read_text(encoding="utf-8") == "original content"


def test_atomic_write_creates_parent_dirs(tmp_path):
    """atomic_write_text must create intermediate directories."""
    from src.services.atomic_write import atomic_write_text

    target = tmp_path / "a" / "b" / "c" / "file.md"
    atomic_write_text(target, "deep content")
    assert target.exists()
    assert target.read_text(encoding="utf-8") == "deep content"
