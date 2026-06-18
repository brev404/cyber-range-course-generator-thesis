"""Regression test: analyze_challenges reads from PROCESSED_DIR, not the raw source."""

from pathlib import Path

import src.pipeline.analyze_challenges as analyze_module
from src.config.settings import settings


def test_challenge_folders_root_is_under_processed_dir():
    """CHALLENGE_FOLDERS_ROOT must be under PROCESSED_DIR, not RAW_CHALLENGES_SOURCE."""
    root = analyze_module.CHALLENGE_FOLDERS_ROOT
    assert isinstance(root, Path)
    assert root.is_relative_to(
        settings.PROCESSED_DIR
    ), f"CHALLENGE_FOLDERS_ROOT {root!r} is not under PROCESSED_DIR {settings.PROCESSED_DIR!r}"
    assert (
        root != settings.RAW_CHALLENGES_SOURCE
    ), "CHALLENGE_FOLDERS_ROOT must not equal RAW_CHALLENGES_SOURCE"
    assert root.name == settings.RAW_CHALLENGES_SOURCE.name
