"""Consistency checker for generated courses.

Compares section ordering, glossary terms, and style markers across all
courses generated in a single contest run. Pure structural analysis — no LLM calls.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Set

from loguru import logger
from pydantic import BaseModel, Field

from src.config.settings import settings

# ---------------------------------------------------------------------------
# Pydantic V2 models
# ---------------------------------------------------------------------------


class ChallengeConsistencyMetrics(BaseModel):
    """Per-challenge structural metrics extracted from course text."""

    challenge_id: str
    section_order: List[str] = Field(default_factory=list)
    terms: List[str] = Field(default_factory=list)
    line_count: int = 0
    has_code_blocks: bool = False
    has_numbered_steps: bool = False


class ConsistencyDeviation(BaseModel):
    """A single detected inconsistency across courses in a run."""

    challenge_id: str
    deviation_type: str  # "section_order" | "missing_term"
    detail: str


class ConsistencyReport(BaseModel):
    """Consistency analysis result across all courses in a single contest run."""

    run_id: str
    challenge_count: int
    metrics: List[ChallengeConsistencyMetrics] = Field(default_factory=list)
    deviations: List[ConsistencyDeviation] = Field(default_factory=list)
    expected_sections: List[str] = Field(default_factory=list)
    expected_terms: List[str] = Field(default_factory=list)
    summary: str = ""


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

_H2_H3_RE = re.compile(r"^#{2,3}\s+(.+)$", re.MULTILINE)
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")
_CODE_BLOCK_RE = re.compile(r"```")
_NUMBERED_STEP_RE = re.compile(r"^\d+\.\s+\S", re.MULTILINE)


def _extract_headers(text: str) -> List[str]:
    """Return H2/H3 heading text in document order, lowercased and stripped."""
    return [m.group(1).strip().lower() for m in _H2_H3_RE.finditer(text)]


def _extract_terms(text: str) -> Set[str]:
    """Return all bold (**term**) and inline-code (`term`) tokens, lowercased."""
    bold = {m.group(1).strip().lower() for m in _BOLD_RE.finditer(text)}
    code = {m.group(1).strip().lower() for m in _INLINE_CODE_RE.finditer(text)}
    return bold | code


def _build_metrics(challenge_id: str, course_text: str) -> ChallengeConsistencyMetrics:
    headers = _extract_headers(course_text)
    terms = sorted(_extract_terms(course_text))
    return ChallengeConsistencyMetrics(
        challenge_id=challenge_id,
        section_order=headers,
        terms=terms,
        line_count=len(course_text.splitlines()),
        has_code_blocks=bool(_CODE_BLOCK_RE.search(course_text)),
        has_numbered_steps=bool(_NUMBERED_STEP_RE.search(course_text)),
    )


# ---------------------------------------------------------------------------
# Consistency checks
# ---------------------------------------------------------------------------


def _check_section_order(
    metrics_list: List[ChallengeConsistencyMetrics],
) -> tuple[List[str], List[ConsistencyDeviation]]:
    """Compare each course's section list against the first course (reference).

    Returns (expected_sections, deviations).  Courses sorted by challenge_id
    before this is called, so the reference is deterministic.
    """
    if not metrics_list:
        return [], []

    reference = metrics_list[0].section_order
    deviations: List[ConsistencyDeviation] = []

    for m in metrics_list[1:]:
        if m.section_order != reference:
            deviations.append(
                ConsistencyDeviation(
                    challenge_id=m.challenge_id,
                    deviation_type="section_order",
                    detail=f"Expected {reference!r}, got {m.section_order!r}",
                )
            )

    return reference, deviations


def _check_term_consistency(
    metrics_list: List[ChallengeConsistencyMetrics],
    threshold: float,
) -> tuple[List[str], List[ConsistencyDeviation]]:
    """Flag terms that appear in >threshold fraction of courses but are absent in some.

    Returns (expected_terms, deviations).
    """
    if not metrics_list:
        return [], []

    total = len(metrics_list)
    all_terms: Set[str] = set()
    for m in metrics_list:
        all_terms.update(m.terms)

    expected_terms = sorted(
        term
        for term in all_terms
        if sum(1 for m in metrics_list if term in m.terms) / total > threshold
    )

    deviations: List[ConsistencyDeviation] = []
    for m in metrics_list:
        missing = [t for t in expected_terms if t not in m.terms]
        if missing:
            deviations.append(
                ConsistencyDeviation(
                    challenge_id=m.challenge_id,
                    deviation_type="missing_term",
                    detail=f"Missing expected terms: {missing!r}",
                )
            )

    return expected_terms, deviations


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def check_consistency(
    courses: Dict[str, str],
    run_id: str,
    *,
    term_threshold: Optional[float] = None,
) -> ConsistencyReport:
    """Analyse structural consistency across all courses in a contest run.

    Args:
        courses: mapping of challenge_id → course markdown text.
        run_id: identifier for this pipeline run (used in the report header).
        term_threshold: fraction threshold for "expected" terms. Defaults to
            settings.CONSISTENCY_TERM_THRESHOLD.

    Returns:
        ConsistencyReport with per-challenge metrics and detected deviations.
    """
    if term_threshold is None:
        term_threshold = float(getattr(settings, "CONSISTENCY_TERM_THRESHOLD", 0.5))

    logger.info(
        "Consistency check: %d course(s), run_id=%s, term_threshold=%.2f",
        len(courses),
        run_id,
        term_threshold,
    )

    if not courses:
        logger.warning("Consistency check: no courses provided — skipping")
        return ConsistencyReport(
            run_id=run_id,
            challenge_count=0,
            summary="No courses to check.",
        )

    # Sort by challenge_id for deterministic reference selection
    metrics_list = [_build_metrics(cid, text) for cid, text in sorted(courses.items())]

    expected_sections, section_devs = _check_section_order(metrics_list)
    expected_terms, term_devs = _check_term_consistency(metrics_list, term_threshold)

    all_devs = section_devs + term_devs
    n_courses = len(metrics_list)
    summary = (
        f"Checked {n_courses} course(s). "
        f"{len(section_devs)} section-order deviation(s). "
        f"{len(expected_terms)} expected term(s); "
        f"{len(term_devs)} course(s) missing at least one."
    )

    logger.info("Consistency check complete: %d total deviation(s)", len(all_devs))

    return ConsistencyReport(
        run_id=run_id,
        challenge_count=n_courses,
        metrics=metrics_list,
        deviations=all_devs,
        expected_sections=expected_sections,
        expected_terms=expected_terms,
        summary=summary,
    )


def write_consistency_report(report: ConsistencyReport, output_dir: Path) -> Path:
    """Write consistency_report.md under output_dir / run_id /.

    Creates parent directories as needed. Returns path to the written file.
    """
    run_dir = output_dir / report.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    report_path = run_dir / "consistency_report.md"

    lines: List[str] = [
        f"# Consistency Report — {report.run_id}",
        "",
        f"**Run ID:** {report.run_id}  ",
        f"**Challenges analysed:** {report.challenge_count}  ",
        "",
        "## Summary",
        "",
        report.summary,
        "",
        "## Metrics Table",
        "",
        "| Challenge | Sections | Terms | Lines | Code Blocks | Numbered Steps |",
        "|-----------|----------|-------|-------|-------------|----------------|",
    ]
    for m in report.metrics:
        lines.append(
            f"| {m.challenge_id} "
            f"| {len(m.section_order)} "
            f"| {len(m.terms)} "
            f"| {m.line_count} "
            f"| {'yes' if m.has_code_blocks else 'no'} "
            f"| {'yes' if m.has_numbered_steps else 'no'} |"
        )

    lines += ["", "## Deviations", ""]
    if not report.deviations:
        lines.append("No deviations detected.")
    else:
        lines += [
            "| Challenge | Type | Detail |",
            "|-----------|------|--------|",
        ]
        for d in report.deviations:
            detail_safe = d.detail.replace("|", "\\|")
            lines.append(f"| {d.challenge_id} | {d.deviation_type} | {detail_safe} |")

    lines += [
        "",
        "## Expected Sections",
        "",
        (", ".join(f"`{s}`" for s in report.expected_sections) or "_none_"),
        "",
        "## Expected Terms",
        "",
        (", ".join(f"`{t}`" for t in report.expected_terms) or "_none_"),
        "",
    ]

    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Consistency report written to %s", report_path)
    return report_path
