"""Pre-ranking structural validation of generated course + solver.

Catches structurally-broken generation BEFORE invoking the (expensive) judge LLM:
- All 10 required course sections present (no missing Conclusion / Extra Resources / etc)
- Solver parses as Python via ast.parse (no syntax errors)
- Extra Resources section contains real references (ATT&CK / CWE / OWASP / RFC)
- No truncation markers ([... truncated], [... continued], etc.)

When validation fails, the content_generation agent retries (up to
settings.STRUCTURAL_VALIDATOR_MAX_RETRIES) with the issues injected as feedback
in the next gen prompt.

Designed to address C1 (truncation), C2 (missing references), C4 (missing Conclusion)
failure clusters identified during internal evaluation.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import List

from loguru import logger


@dataclass
class StructuralReport:
    """Result of structural validation. `is_valid=True` iff `issues` is empty."""

    is_valid: bool
    issues: List[str] = field(default_factory=list)


# Required course sections (lower-cased substring match against course.md headers).
# Tolerant — matches "## Conclusion", "10. Conclusion", "**Conclusion**", etc.
_REQUIRED_SECTIONS = [
    "title",
    "abstract",
    "objectives",
    "technical skills",
    "definitions",
    "reproducibility",
    "thought process",
    "step-by-step",
    "solution script",
    "conclusion",
    "extra resources",
]

# Markers the LLM emits when truncating output (forbidden per F1 rule 7).
# v4.1.2: expanded list after smoke #3 evidence — every judge cited truncation
# but only the first 6 patterns matched. Added deferral phrases ("left as exercise"),
# generic placeholders ("complete the rest"), and lazy section-skip phrases.
_TRUNCATION_MARKERS = [
    "[... truncated",
    "[... continued",
    "[continues]",
    "[code continues]",
    "...truncated...",
    "...continued...",
    "complete the rest",
    "left as an exercise",
    "left as exercise",
    "implementation left to",
    "left to the reader",
    "rest is omitted",
    "rest of the implementation",
    "[remainder omitted",
    "[abbreviated",
    "[shortened",
    "(continued)",
    "(continues)",
]

# Reference patterns that count as "real" references in the Extra Resources section.
_REFERENCE_MARKERS = [
    "att&ck",
    "attack",
    "cwe-",
    "owasp",
    "rfc",
    "nist",
    "cve-",
]

# MITRE ATT&CK technique IDs: T1234 or T1234.001 (sub-technique).
_MITRE_TECHNIQUE_PATTERN = re.compile(r"\bT\d{4}(?:\.\d{3})?\b")

# v4.1: numbered section heading at start of a line — captures the leading digit(s).
# Tolerant: matches "## 1.", "## 11.", "##  3." (extra space); requires the dot.
_NUMBERED_SECTION_PATTERN = re.compile(r"^##\s+(\d+)\.", re.MULTILINE)

# Required section numbering for an assembled v4 course: 1..11, exactly once each, in order.
_REQUIRED_SECTION_NUMBERS = list(range(1, 12))


def validate_course_structure(course_md: str) -> StructuralReport:
    """Check that all required sections are present (case-insensitive header match)
    and that no truncation markers appear anywhere in the course.

    Args:
        course_md: full course.md text.

    Returns:
        StructuralReport with `is_valid=True` iff every required section is
        present and no truncation marker is detected.
    """
    issues: List[str] = []
    if not course_md or not course_md.strip():
        return StructuralReport(is_valid=False, issues=["course.md is empty"])

    lower = course_md.lower()
    for section in _REQUIRED_SECTIONS:
        if section not in lower:
            issues.append(f"missing section: {section}")

    for marker in _TRUNCATION_MARKERS:
        if marker.lower() in lower:
            issues.append(f"truncation marker found: {marker}")

    return StructuralReport(is_valid=not issues, issues=issues)


def validate_solver_syntax(solver_code: str) -> StructuralReport:
    """Check the solver parses as Python (via ast.parse).

    Empty solver is treated as invalid — a course without a runnable solver
    cannot be judged for technical correctness.

    Args:
        solver_code: contents of solve_generated.py.

    Returns:
        StructuralReport with `is_valid=True` iff the code parses cleanly.
    """
    if not solver_code or not solver_code.strip():
        return StructuralReport(is_valid=False, issues=["solver is empty"])
    try:
        ast.parse(solver_code)
        return StructuralReport(is_valid=True)
    except SyntaxError as e:
        return StructuralReport(is_valid=False, issues=[f"solver syntax error: {e}"])


def validate_extra_resources(course_md: str) -> StructuralReport:
    """Check the Extra Resources section contains at least one real reference
    (ATT&CK / CWE / OWASP / RFC / NIST / CVE marker).

    Args:
        course_md: full course.md text.

    Returns:
        StructuralReport with `is_valid=True` iff Extra Resources section exists
        AND contains at least one reference marker within ~1500 chars of the heading.
    """
    if not course_md:
        return StructuralReport(
            is_valid=False, issues=["course.md is empty (cannot check Extra Resources)"]
        )
    lower = course_md.lower()
    idx = lower.find("extra resources")
    if idx < 0:
        return StructuralReport(
            is_valid=False, issues=["Extra Resources section not found"]
        )
    # Scan a window after the heading for any reference marker
    sample = lower[idx : idx + 1500]
    if any(marker in sample for marker in _REFERENCE_MARKERS):
        return StructuralReport(is_valid=True)
    # Also accept bare MITRE ATT&CK technique IDs (T1234, T1234.001) — these are
    # the canonical reference form even when the literal word "ATT&CK" is absent.
    original_sample = course_md[idx : idx + 1500]
    if _MITRE_TECHNIQUE_PATTERN.search(original_sample):
        return StructuralReport(is_valid=True)
    return StructuralReport(
        is_valid=False,
        issues=["Extra Resources section has no ATT&CK / CWE / OWASP / RFC references"],
    )


def validate_section_numbering(course_md: str) -> StructuralReport:
    """Check the course has sections numbered ## 1. through ## 11. in sequence (no missing, no duplicate).

    Added in v4.1 to catch LLM numbering drift introduced by the v4 prompt's
    "OMIT Section 9" instruction (observed in EXP-SMOKE-V4):
      - duplicate ## 9. headings (LLM added an off-spec section then assembly inserted another)
      - missing numbers (LLM dropped numbers entirely, writing ## Title and Context, etc.)
      - off-sequence numbering (e.g. ## 1., ## 2., ## 4.)

    Args:
        course_md: full course.md text.

    Returns:
        StructuralReport with `is_valid=True` iff the course contains exactly one
        heading for each of `## 1.` through `## 11.` and they appear in ascending order.
    """
    if not course_md or not course_md.strip():
        return StructuralReport(
            is_valid=False,
            issues=["course.md is empty (cannot check section numbering)"],
        )

    matches = _NUMBERED_SECTION_PATTERN.findall(course_md)
    if not matches:
        return StructuralReport(
            is_valid=False,
            issues=[
                "no numbered section headings found (expected ## 1. through ## 11.)"
            ],
        )

    numbers = [int(n) for n in matches]
    issues: List[str] = []

    # Duplicates
    seen: set[int] = set()
    duplicates: List[int] = []
    for n in numbers:
        if n in seen and n not in duplicates:
            duplicates.append(n)
        seen.add(n)
    if duplicates:
        issues.append(
            "duplicate section number(s): " + ", ".join(f"## {d}." for d in duplicates)
        )

    # Missing
    missing = [n for n in _REQUIRED_SECTION_NUMBERS if n not in seen]
    if missing:
        issues.append(
            "missing section number(s): " + ", ".join(f"## {n}." for n in missing)
        )

    # Off-sequence (filter to required range so duplicates noted separately don't double-report).
    # Compare the de-duplicated, required-range subset of the actual order to the expected sequence.
    seen_in_order: List[int] = []
    seen_set: set[int] = set()
    for n in numbers:
        if n in _REQUIRED_SECTION_NUMBERS and n not in seen_set:
            seen_in_order.append(n)
            seen_set.add(n)
    if seen_in_order and seen_in_order != sorted(seen_in_order):
        issues.append(
            f"section numbers out of order: got {seen_in_order}, expected ascending"
        )

    return StructuralReport(is_valid=not issues, issues=issues)


def validate_all(course_md: str, solver_code: str) -> StructuralReport:
    """Run all checks and merge issues into a single report.

    Args:
        course_md: full course.md text.
        solver_code: solver source code.

    Returns:
        Combined StructuralReport. `is_valid=True` iff all sub-checks pass.
    """
    all_issues: List[str] = []
    for fn, arg in [
        (validate_course_structure, course_md),
        (validate_solver_syntax, solver_code),
        (validate_extra_resources, course_md),
        (validate_section_numbering, course_md),
    ]:
        rep = fn(arg)
        all_issues.extend(rep.issues)
    rep = StructuralReport(is_valid=not all_issues, issues=all_issues)
    if not rep.is_valid:
        logger.debug(
            "Structural validator FAILED with {} issue(s): {}",
            len(all_issues),
            all_issues,
        )
    return rep


def format_feedback_for_prompt(report: StructuralReport) -> str:
    """Render a validator report as a feedback block suitable for the next gen prompt.

    Empty when `is_valid=True`. Non-empty content goes into the LLM prompt under a
    "Structural validator feedback — MUST address" header.
    """
    if report.is_valid or not report.issues:
        return ""
    lines = [
        "Structural validator feedback — MUST address in this revision:",
    ]
    for issue in report.issues:
        lines.append(f"  - {issue}")
    return "\n".join(lines)
