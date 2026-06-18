"""Tests for the terminology checker.

Verifies: ID loading from KB, invalid ID detection, valid IDs pass,
malformed CWE detection, TERMINOLOGY_CHECK_MODE behavior.
"""

from pathlib import Path

import pytest

from src.config.settings import settings
from src.models.report_models import IssueSeverity
from src.validators.terminology_checker import (
    _load_attack_ids,
    _load_cwe_ids,
    _load_owasp_wstg_ids,
    check_terminology,
)


@pytest.fixture
def minimal_kb(tmp_path: Path) -> Path:
    """Create minimal KB with known ATT&CK, CWE, WSTG IDs for deterministic tests."""
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


def test_terminology_checker_loads_ids(minimal_kb: Path) -> None:
    """KB files load; ATT&CK/CWE/OWASP sets non-empty."""
    attack = _load_attack_ids(minimal_kb)
    cwe = _load_cwe_ids(minimal_kb)
    wstg = _load_owasp_wstg_ids(minimal_kb)
    assert "T1001" in attack
    assert "T1566" in attack
    assert "CWE-79" in cwe
    assert "CWE-89" in cwe
    assert "WSTG-INFO-01" in wstg


def test_terminology_checker_loads_ids_empty_if_missing(tmp_path: Path) -> None:
    """Empty KB dir yields empty sets."""
    assert _load_attack_ids(tmp_path) == frozenset()
    assert _load_cwe_ids(tmp_path) == frozenset()
    assert _load_owasp_wstg_ids(tmp_path) == frozenset()


def test_terminology_checker_invalid_attack_id(
    monkeypatch: pytest.MonkeyPatch,
    minimal_kb: Path,
) -> None:
    """Text with T9999 (invalid) yields ValidationIssue."""
    monkeypatch.setattr(settings, "KNOWLEDGE_BASE_DIR", minimal_kb)
    monkeypatch.setattr(settings, "TERMINOLOGY_CHECK_MODE", "warn")
    issues = check_terminology("The technique T9999 is used.", challenge_id="test")
    attack_issues = [i for i in issues if i.code == "TERM_INVALID_ATTACK_ID"]
    assert len(attack_issues) >= 1
    assert any("T9999" in i.message for i in attack_issues)
    assert attack_issues[0].severity == IssueSeverity.MEDIUM


def test_terminology_checker_invalid_cwe_id(
    monkeypatch: pytest.MonkeyPatch,
    minimal_kb: Path,
) -> None:
    """Text with CWE-99999 (invalid) yields ValidationIssue."""
    monkeypatch.setattr(settings, "KNOWLEDGE_BASE_DIR", minimal_kb)
    monkeypatch.setattr(settings, "TERMINOLOGY_CHECK_MODE", "warn")
    issues = check_terminology("Related to CWE-99999.", challenge_id="test")
    cwe_issues = [i for i in issues if i.code == "TERM_INVALID_CWE_ID"]
    assert len(cwe_issues) >= 1
    assert any("CWE-99999" in i.message for i in cwe_issues)
    assert cwe_issues[0].severity == IssueSeverity.MEDIUM


def test_terminology_checker_valid_ids_pass(
    monkeypatch: pytest.MonkeyPatch,
    minimal_kb: Path,
) -> None:
    """Valid CWE-79, T1566 in text with IDs in KB yield no invalid-ID issues."""
    monkeypatch.setattr(settings, "KNOWLEDGE_BASE_DIR", minimal_kb)
    monkeypatch.setattr(settings, "TERMINOLOGY_CHECK_MODE", "warn")
    text = "Uses CWE-79 (XSS) and ATT&CK T1566."
    issues = check_terminology(text, challenge_id="test")
    invalid_issues = [
        i
        for i in issues
        if i.code
        in ("TERM_INVALID_ATTACK_ID", "TERM_INVALID_CWE_ID", "TERM_INVALID_WSTG_ID")
    ]
    assert len(invalid_issues) == 0


def test_terminology_checker_malformed_cwe(
    monkeypatch: pytest.MonkeyPatch,
    minimal_kb: Path,
) -> None:
    """CWE79 (no hyphen) yields LOW severity issue."""
    monkeypatch.setattr(settings, "KNOWLEDGE_BASE_DIR", minimal_kb)
    monkeypatch.setattr(settings, "TERMINOLOGY_CHECK_MODE", "warn")
    issues = check_terminology("CWE79 is a common weakness.", challenge_id="test")
    malformed = [i for i in issues if i.code == "TERM_MALFORMED_CWE"]
    assert len(malformed) >= 1
    assert malformed[0].severity == IssueSeverity.LOW
    assert "CWE79" in malformed[0].message
    assert "CWE-79" in malformed[0].suggestion or "CWE-79" in malformed[0].message


def test_terminology_checker_mode_annotate(
    monkeypatch: pytest.MonkeyPatch,
    minimal_kb: Path,
) -> None:
    """TERMINOLOGY_CHECK_MODE=annotate returns no ValidationIssues."""
    monkeypatch.setattr(settings, "KNOWLEDGE_BASE_DIR", minimal_kb)
    monkeypatch.setattr(settings, "TERMINOLOGY_CHECK_MODE", "annotate")
    issues = check_terminology("Invalid T9999 and CWE-99999.", challenge_id="test")
    assert issues == []


def test_terminology_checker_mode_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TERMINOLOGY_CHECK_MODE=off returns no ValidationIssues."""
    monkeypatch.setattr(settings, "TERMINOLOGY_CHECK_MODE", "off")
    issues = check_terminology("Invalid T9999.", challenge_id="test")
    assert issues == []

    # Integration with validation_agent is intentionally not tested here:
    # terminology checks are applied only to generated courses, not author writeups.
