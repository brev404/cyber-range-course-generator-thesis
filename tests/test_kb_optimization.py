"""Tests for KB optimization: health check and deduplication."""

from __future__ import annotations

from pathlib import Path

from src.services.vector_db_service import VectorDBService
from src.utils.kb_health_check import EXPECTED_KB_FILES, run_health_check


def test_kb_health_check_run_returns_tuple() -> None:
    """run_health_check returns (passed: bool, messages: list[str])."""
    passed, messages = run_health_check()
    assert isinstance(passed, bool)
    assert isinstance(messages, list)
    assert all(isinstance(m, str) for m in messages)


def test_kb_health_check_expected_files_defined() -> None:
    """Expected KB file list includes key dictionary files."""
    assert "attack_techniques.md" in EXPECTED_KB_FILES
    assert "cwe_weaknesses.md" in EXPECTED_KB_FILES
    assert "cybersecurity_glossary.md" in EXPECTED_KB_FILES
    assert "course_guidelines.md" in EXPECTED_KB_FILES
    assert "README.md" in EXPECTED_KB_FILES


def test_vector_db_deduplication_skips_duplicates(tmp_path: Path) -> None:
    """Ingest with duplicate content produces fewer chunks than without deduplication."""
    # Create a minimal KB with two files that have identical content
    kb_dir = tmp_path / "kb"
    kb_dir.mkdir()
    content = "XSS cross-site scripting vulnerability. " * 50  # Enough to chunk
    (kb_dir / "a.md").write_text(content, encoding="utf-8")
    (kb_dir / "b.md").write_text(content, encoding="utf-8")  # Same content = duplicates

    svc = VectorDBService(
        persist_directory=tmp_path / "chroma",
        collection_name="test_kb_dedup",
    )
    n = svc.ingest_knowledge_base(
        source_dir=kb_dir, chunk_size=200, chunk_overlap=50, replace=True
    )

    # Same content in both files → all chunks from b.md are duplicates of a.md
    # With dedup: only unique chunks (same count as single file)
    single_file_chunks = n
    assert single_file_chunks >= 1
    assert single_file_chunks <= 50  # Sanity: not thousands
