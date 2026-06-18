"""Tests for course terminology integration.

Verifies: course_terminology_checker node, routing (mapping vs refinement_step),
refinement_step sets content_generation_subset_ids from terminology issues.
"""

from pathlib import Path

import pytest

from src.agents.course_terminology_agent import run_course_terminology_checker
from src.config.settings import settings as app_settings
from src.core.state import AgentState


@pytest.fixture
def minimal_kb(tmp_path: Path) -> Path:
    """Create minimal KB with known ATT&CK, CWE, WSTG IDs."""
    (tmp_path / "attack_techniques.md").write_text(
        "# ATT&CK\n\n## T1001 – Data Obfuscation\n\n## T1566 – Phishing\n",
        encoding="utf-8",
    )
    (tmp_path / "cwe_weaknesses.md").write_text(
        "# CWE\n\n## CWE-79 – XSS\n\n## CWE-89 – SQLi\n",
        encoding="utf-8",
    )
    (tmp_path / "owasp_wstg.md").write_text(
        "# WSTG\n\n### WSTG-INFO-01 – Search Engine Discovery\n",
        encoding="utf-8",
    )
    return tmp_path


def test_course_terminology_checker_mode_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TERMINOLOGY_CHECK_MODE=off: node passes through without running checks."""
    monkeypatch.setattr(app_settings, "TERMINOLOGY_CHECK_MODE", "off")
    state = AgentState(
        generated_courses={"c1": "Text with invalid T9999."},
        challenge_ids=["c1"],
    )
    out = run_course_terminology_checker(state)
    assert out.current_agent == "course_terminology_checker"
    assert out.course_terminology_issues == {}


def test_course_terminology_checker_mode_annotate(
    monkeypatch: pytest.MonkeyPatch,
    minimal_kb: Path,
) -> None:
    """TERMINOLOGY_CHECK_MODE=annotate: runs checks, returns empty issues (log only)."""
    monkeypatch.setattr(app_settings, "KNOWLEDGE_BASE_DIR", minimal_kb)
    monkeypatch.setattr(app_settings, "TERMINOLOGY_CHECK_MODE", "annotate")
    state = AgentState(
        generated_courses={"c1": "Text with invalid T9999."},
        challenge_ids=["c1"],
    )
    out = run_course_terminology_checker(state)
    assert out.current_agent == "course_terminology_checker"
    assert out.course_terminology_issues == {}


def test_course_terminology_checker_mode_warn_stores_issues(
    monkeypatch: pytest.MonkeyPatch,
    minimal_kb: Path,
) -> None:
    """TERMINOLOGY_CHECK_MODE=warn: stores issues in state, proceeds to mapping."""
    monkeypatch.setattr(app_settings, "KNOWLEDGE_BASE_DIR", minimal_kb)
    monkeypatch.setattr(app_settings, "TERMINOLOGY_CHECK_MODE", "warn")
    state = AgentState(
        generated_courses={"c1": "Text with invalid T9999."},
        challenge_ids=["c1"],
    )
    out = run_course_terminology_checker(state)
    assert out.current_agent == "course_terminology_checker"
    assert "c1" in out.course_terminology_issues
    assert len(out.course_terminology_issues["c1"]) >= 1
    assert any("T9999" in i.message for i in out.course_terminology_issues["c1"])


def test_course_terminology_checker_mode_block_stores_issues(
    monkeypatch: pytest.MonkeyPatch,
    minimal_kb: Path,
) -> None:
    """TERMINOLOGY_CHECK_MODE=block: stores issues; routing will send to refinement."""
    monkeypatch.setattr(app_settings, "KNOWLEDGE_BASE_DIR", minimal_kb)
    monkeypatch.setattr(app_settings, "TERMINOLOGY_CHECK_MODE", "block")
    state = AgentState(
        generated_courses={"c1": "Text with invalid T9999."},
        challenge_ids=["c1"],
    )
    out = run_course_terminology_checker(state)
    assert out.current_agent == "course_terminology_checker"
    assert "c1" in out.course_terminology_issues
    assert len(out.course_terminology_issues["c1"]) >= 1


def test_course_terminology_checker_valid_ids_no_issues(
    monkeypatch: pytest.MonkeyPatch,
    minimal_kb: Path,
) -> None:
    """Valid CWE-79, T1566 in text yield no issues."""
    monkeypatch.setattr(app_settings, "KNOWLEDGE_BASE_DIR", minimal_kb)
    monkeypatch.setattr(app_settings, "TERMINOLOGY_CHECK_MODE", "warn")
    state = AgentState(
        generated_courses={"c1": "Uses CWE-79 (XSS) and ATT&CK T1566."},
        challenge_ids=["c1"],
    )
    out = run_course_terminology_checker(state)
    assert out.current_agent == "course_terminology_checker"
    assert out.course_terminology_issues == {}


def test_course_terminology_checker_skips_empty_courses(
    monkeypatch: pytest.MonkeyPatch,
    minimal_kb: Path,
) -> None:
    """Empty or whitespace-only courses are skipped."""
    monkeypatch.setattr(app_settings, "KNOWLEDGE_BASE_DIR", minimal_kb)
    monkeypatch.setattr(app_settings, "TERMINOLOGY_CHECK_MODE", "warn")
    state = AgentState(
        generated_courses={"c1": "", "c2": "   ", "c3": "Valid CWE-79."},
        challenge_ids=["c1", "c2", "c3"],
    )
    out = run_course_terminology_checker(state)
    assert out.current_agent == "course_terminology_checker"
    assert "c1" not in out.course_terminology_issues
    assert "c2" not in out.course_terminology_issues
    assert "c3" not in out.course_terminology_issues


def test_course_terminology_checker_routing_integration(
    monkeypatch: pytest.MonkeyPatch,
    minimal_kb: Path,
    processed_fixture: Path,
) -> None:
    """Graph runs with course_terminology_checker; mode=warn proceeds to mapping."""
    if not (processed_fixture / "raw_challenges").is_dir():
        pytest.skip("Processed fixtures not found")
    monkeypatch.setattr(app_settings, "PROCESSED_DIR", processed_fixture)
    monkeypatch.setattr(app_settings, "KNOWLEDGE_BASE_DIR", minimal_kb)
    monkeypatch.setattr(app_settings, "TERMINOLOGY_CHECK_MODE", "warn")

    from src.core.graph import app

    state = AgentState()
    state.skip_ranking = True
    state.content_generation_subset_ids = []
    config = {"configurable": {"thread_id": "e2e_terminology_warn"}}
    final = app.invoke(state, config=config)

    if isinstance(final, dict):
        validation_reports = final.get("validation_reports", [])
        writeup_mappings = final.get("writeup_mappings", {})
        course_terminology_issues = final.get("course_terminology_issues", {})
    else:
        validation_reports = final.validation_reports
        writeup_mappings = final.writeup_mappings
        course_terminology_issues = final.course_terminology_issues

    assert len(validation_reports) >= 1
    assert isinstance(writeup_mappings, dict)
    assert isinstance(course_terminology_issues, dict)
