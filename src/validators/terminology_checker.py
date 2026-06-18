"""Rule-based terminology validation.

Loads term sets from the knowledge base (ATT&CK, CWE, OWASP WSTG sources)
and validates IDs in course/writeup text. Detects invalid IDs (not in KB),
malformed IDs (e.g. CWE79 without hyphen), and raises ValidationIssue with
configurable severity. Reduces LLM load and improves consistency.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import FrozenSet, List, Optional, Set

from loguru import logger

from src.config.settings import settings
from src.models.report_models import IssueSeverity, ValidationIssue

# Regex to extract taxonomy IDs from text (align with mapping_agent)
_RE_ATTACK = re.compile(r"\b(T\d{4}(?:\.\d{3})?)\b")
_RE_CWE = re.compile(r"\b(CWE-\d+)\b")
_RE_WSTG = re.compile(r"\b(WSTG-(?:v\d+-)?[A-Z]+-\d+)\b")
# Malformed CWE: CWE79 without hyphen
_RE_CWE_MALFORMED = re.compile(r"\b(CWE\d+)\b")

# Regex to parse section headers in knowledge base files
_RE_ATTACK_HEADER = re.compile(r"^##\s+(T\d{4}(?:\.\d{3})?)\s+[–\-]")
_RE_CWE_HEADER = re.compile(r"^##\s+(CWE-\d+)\s+[–\-]")
_RE_WSTG_HEADER = re.compile(r"^###\s+(WSTG-[A-Z]+-\d+)\s+[–\-]")


@lru_cache(maxsize=None)
def _load_attack_ids(kb_dir: Path) -> FrozenSet[str]:
    """Load valid MITRE ATT&CK technique IDs from attack_techniques.md."""
    path = kb_dir / "attack_techniques.md"
    if not path.exists():
        logger.warning(
            "attack_techniques.md not found; no ATT&CK validation", path=str(path)
        )
        return frozenset()
    ids: Set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = _RE_ATTACK_HEADER.match(line.strip())
            if m:
                ids.add(m.group(1))
    logger.debug("Loaded {} ATT&CK technique IDs from {}", len(ids), path.name)
    return frozenset(ids)


@lru_cache(maxsize=None)
def _load_cwe_ids(kb_dir: Path) -> FrozenSet[str]:
    """Load valid CWE IDs from cwe_weaknesses.md."""
    path = kb_dir / "cwe_weaknesses.md"
    if not path.exists():
        logger.warning("cwe_weaknesses.md not found; no CWE validation", path=str(path))
        return frozenset()
    ids: Set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = _RE_CWE_HEADER.match(line.strip())
            if m:
                ids.add(m.group(1))
    logger.debug("Loaded {} CWE IDs from {}", len(ids), path.name)
    return frozenset(ids)


@lru_cache(maxsize=None)
def _load_owasp_wstg_ids(kb_dir: Path) -> FrozenSet[str]:
    """Load valid OWASP WSTG scenario IDs from owasp_wstg.md (non-versioned form)."""
    path = kb_dir / "owasp_wstg.md"
    if not path.exists():
        logger.warning("owasp_wstg.md not found; no WSTG validation", path=str(path))
        return frozenset()
    ids: Set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = _RE_WSTG_HEADER.match(line.strip())
            if m:
                ids.add(m.group(1))
    logger.debug("Loaded {} OWASP WSTG IDs from {}", len(ids), path.name)
    return frozenset(ids)


def _normalize_wstg_id(raw: str) -> str:
    """Normalize WSTG ID: strip version prefix (e.g. WSTG-v42-INFO-02 -> WSTG-INFO-02)."""
    if raw.startswith("WSTG-v") and "-" in raw[6:]:
        rest = raw[6:]
        idx = rest.find("-")
        if idx != -1:
            return "WSTG-" + rest[idx + 1 :]
    return raw


def check_terminology(
    text: str,
    challenge_id: str = "",
    file_path: Optional[str] = None,
) -> List[ValidationIssue]:
    """Check text for terminology inconsistencies (invalid/malformed IDs).

    Loads valid IDs from KB (attack_techniques.md, cwe_weaknesses.md, owasp_wstg.md)
    and detects:
    - Invalid IDs: IDs in text but not in KB (MEDIUM)
    - Malformed CWE: CWE79 instead of CWE-79 (LOW)

    Respects TERMINOLOGY_CHECK_MODE: "annotate" returns [] (log only);
    "warn"/"block" return issues; "off" should not call this (caller skips).

    Args:
        text: Course or writeup text to check.
        challenge_id: Challenge identifier for logging.
        file_path: Relative path to file (e.g. cyberedu/write-up/writeup.md).

    Returns:
        List of ValidationIssue for terminology concerns. Empty if mode is
        "annotate" or no issues found.
    """
    mode = getattr(settings, "TERMINOLOGY_CHECK_MODE", "warn")
    if mode == "annotate":
        # Run checks but return no issues; log only
        _run_checks_and_log(text, challenge_id, file_path)
        return []
    if mode == "off":
        return []

    kb_dir: Path = settings.KNOWLEDGE_BASE_DIR
    valid_attack = _load_attack_ids(kb_dir)
    valid_cwe = _load_cwe_ids(kb_dir)
    valid_wstg = _load_owasp_wstg_ids(kb_dir)

    issues: List[ValidationIssue] = []

    # 1. Invalid ATT&CK IDs
    for m in _RE_ATTACK.finditer(text):
        tid = m.group(1)
        if valid_attack and tid not in valid_attack:
            issues.append(
                ValidationIssue(
                    code="TERM_INVALID_ATTACK_ID",
                    message=f"ATT&CK ID '{tid}' is not in knowledge base (attack_techniques.md).",
                    severity=IssueSeverity.MEDIUM,
                    file_path=file_path,
                    suggestion="Use a valid MITRE ATT&CK technique ID from attack_techniques.md.",
                )
            )

    # 2. Invalid CWE IDs
    for m in _RE_CWE.finditer(text):
        cid = m.group(1)
        if valid_cwe and cid not in valid_cwe:
            issues.append(
                ValidationIssue(
                    code="TERM_INVALID_CWE_ID",
                    message=f"CWE ID '{cid}' is not in knowledge base (cwe_weaknesses.md).",
                    severity=IssueSeverity.MEDIUM,
                    file_path=file_path,
                    suggestion="Use a valid CWE ID from cwe_weaknesses.md.",
                )
            )

    # 3. Invalid WSTG IDs
    for m in _RE_WSTG.finditer(text):
        raw = m.group(1)
        norm = _normalize_wstg_id(raw)
        if valid_wstg and norm not in valid_wstg:
            issues.append(
                ValidationIssue(
                    code="TERM_INVALID_WSTG_ID",
                    message=f"OWASP WSTG ID '{raw}' is not in knowledge base (owasp_wstg.md).",
                    severity=IssueSeverity.MEDIUM,
                    file_path=file_path,
                    suggestion="Use a valid WSTG scenario ID from owasp_wstg.md.",
                )
            )

    # 4. Malformed CWE (CWE79 instead of CWE-79)
    for m in _RE_CWE_MALFORMED.finditer(text):
        malformed = m.group(1)
        # Avoid double-reporting: if CWE-79 is also in text, we already checked it
        corrected = "CWE-" + malformed[3:]  # CWE79 -> CWE-79
        issues.append(
            ValidationIssue(
                code="TERM_MALFORMED_CWE",
                message=f"Malformed CWE ID '{malformed}' (use hyphen: '{corrected}').",
                severity=IssueSeverity.LOW,
                file_path=file_path,
                suggestion=f"Use standard format: {corrected}",
            )
        )

    if issues and challenge_id:
        logger.debug("Terminology issues for {}: {}", challenge_id, len(issues))

    return issues


def _run_checks_and_log(
    text: str,
    challenge_id: str,
    file_path: Optional[str],
) -> None:
    """Run checks and log findings only (annotate mode)."""
    kb_dir: Path = settings.KNOWLEDGE_BASE_DIR
    valid_attack = _load_attack_ids(kb_dir)
    valid_cwe = _load_cwe_ids(kb_dir)
    valid_wstg = _load_owasp_wstg_ids(kb_dir)

    invalid_attack: List[str] = []
    invalid_cwe: List[str] = []
    invalid_wstg: List[str] = []
    malformed_cwe: List[str] = []

    for m in _RE_ATTACK.finditer(text):
        tid = m.group(1)
        if valid_attack and tid not in valid_attack and tid not in invalid_attack:
            invalid_attack.append(tid)
    for m in _RE_CWE.finditer(text):
        cid = m.group(1)
        if valid_cwe and cid not in valid_cwe and cid not in invalid_cwe:
            invalid_cwe.append(cid)
    for m in _RE_WSTG.finditer(text):
        raw = m.group(1)
        norm = _normalize_wstg_id(raw)
        if valid_wstg and norm not in valid_wstg and raw not in invalid_wstg:
            invalid_wstg.append(raw)
    for m in _RE_CWE_MALFORMED.finditer(text):
        mal = m.group(1)
        if mal not in malformed_cwe:
            malformed_cwe.append(mal)

    if invalid_attack or invalid_cwe or invalid_wstg or malformed_cwe:
        logger.info(
            "Terminology (annotate) for {}: invalid ATT&CK={}, CWE={}, WSTG={}; malformed CWE={}",
            challenge_id or "unknown",
            invalid_attack,
            invalid_cwe,
            invalid_wstg,
            malformed_cwe,
        )
