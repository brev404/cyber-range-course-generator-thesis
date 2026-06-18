"""Validation Agent: structural and research-based (Step 0) checks.

This agent validates that challenges meet the CyberEdu structure requirements
and research-based reproducibility/metadata checks (Step 0: challenge metadata,
description, resources). It orchestrates the pipeline validator and adds
Step 0 issues so Content Generation or humans can complete missing info.

Structural checks: required files, hierarchy (delegated to
src.pipeline.validate_challenge_structure.validate_challenge).
Research-based: reproducibility and metadata checks (Step 0).

Flow optimization: This agent is Python-only; it does not call the LLM.
Validation uses only validate_challenge (pipeline) and rule-based Step 0 checks.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import List

from loguru import logger

from src.config.settings import settings
from src.core.state import AgentState
from src.models.report_models import (
    IssueSeverity,
    ValidationIssue,
    ValidationReport,
)

# Minimum description length to consider "Step 0" description present (chars)
_MIN_DESCRIPTION_LENGTH = 80
# Minimum content length to consider writeup mentioning resources (chars)
_MIN_WRITEUP_FOR_RESOURCES = 100
# Keywords that suggest resource availability is documented
_RESOURCE_KEYWORDS = (
    "binary",
    "pcap",
    "source code",
    "attachment",
    "public/",
    "download",
    "file",
    "resource",
)


def _check_step0_reproducibility(
    challenge_path: Path,
    category: str,
    challenge_name: str,
    challenge_id: str,
) -> List[ValidationIssue]:
    """Check Step 0: challenge metadata, description, and resource availability.

    Research-based: flag missing reproducibility and metadata so
    Content Generation or humans can complete them.

    Args:
        challenge_path: Root path of the challenge (category/challenge_name).
        category: Category name (parent directory).
        challenge_name: Challenge directory name.
        challenge_id: Identifier for logging (e.g. category/challenge_name).

    Returns:
        List of validation issues for missing Step 0 elements (MEDIUM/LOW).
    """
    issues: List[ValidationIssue] = []
    wu_dir = challenge_path / "cyberedu" / "write-up"
    description_path = wu_dir / "description.md"
    writeup_path = wu_dir / "writeup.md"

    # 1. Description content (metadata / problem statement)
    if description_path.is_file():
        try:
            text = description_path.read_text(
                encoding="utf-8", errors="replace"
            ).strip()
            if len(text) < _MIN_DESCRIPTION_LENGTH:
                issues.append(
                    ValidationIssue(
                        code="STEP0_DESCRIPTION_TOO_SHORT",
                        message=(
                            "Step 0 (reproducibility): description.md is very short. "
                            "Add challenge name, category, difficulty, and problem statement."
                        ),
                        severity=IssueSeverity.MEDIUM,
                        file_path="cyberedu/write-up/description.md",
                        suggestion="Expand description.md with challenge metadata and problem statement for reproducibility.",
                    )
                )
        except OSError as e:
            logger.debug("Could not read description.md for Step 0 check: {}", e)
    else:
        # Structural validator already flags missing description.md
        pass

    # 2. Resource availability (binary, PCAP, source) when writeup exists
    if writeup_path.is_file():
        try:
            text = writeup_path.read_text(encoding="utf-8", errors="replace").strip()
            if len(text) >= _MIN_WRITEUP_FOR_RESOURCES:
                has_resource_mention = any(
                    kw in text.lower() for kw in _RESOURCE_KEYWORDS
                )
                if not has_resource_mention:
                    issues.append(
                        ValidationIssue(
                            code="STEP0_RESOURCES_NOT_DOCUMENTED",
                            message=(
                                "Step 0 (reproducibility): writeup does not clearly mention "
                                "resource availability (binary, PCAP, source code) when applicable."
                            ),
                            severity=IssueSeverity.LOW,
                            file_path="cyberedu/write-up/writeup.md",
                            suggestion="Document which resources are provided (binary, PCAP, source) so the writeup remains reproducible.",
                        )
                    )
        except OSError as e:
            logger.debug("Could not read writeup.md for Step 0 check: {}", e)

    return issues


def _get_challenge_paths(state: AgentState) -> List[tuple[Path, str]]:
    """Return list of (challenge_path, category) for validation.

    Uses state.organized_challenges if set; otherwise scans PROCESSED_DIR
    for the default organized root (raw_challenges).
    """
    paths_with_category: List[tuple[Path, str]] = []
    organized_root = getattr(settings, "PROCESSED_DIR", None)
    if not organized_root or not organized_root.is_dir():
        return paths_with_category

    # Prefer paths from state if provided (each path = .../category/challenge_name)
    if state.organized_challenges:
        for path in state.organized_challenges:
            path = Path(path)
            if path.is_dir():
                # Assume path is .../category/challenge_name
                parent = path.parent
                if parent.name and path.name:
                    paths_with_category.append((path, parent.name))
        return paths_with_category

    # Fallback: scan PROCESSED_DIR for category/challenge_name
    default_root = organized_root / settings.RAW_CHALLENGES_SOURCE.name
    if not default_root.is_dir():
        return paths_with_category
    for category_dir in default_root.iterdir():
        if not category_dir.is_dir():
            continue
        for challenge_dir in category_dir.iterdir():
            if challenge_dir.is_dir():
                paths_with_category.append((challenge_dir, category_dir.name))
    return paths_with_category


def run_validation_agent(state: AgentState) -> AgentState:
    """Run structural and Step 0 validation for all challenges in state.

    Calls the pipeline validator per challenge, then adds research-based
    Step 0 (reproducibility/metadata) checks and appends reports to state.

    Args:
        state: Current agent state (organized_challenges or PROCESSED_DIR used).

    Returns:
        Updated state with validation_reports populated.
    """
    from src.pipeline.validate_challenge_structure import validate_challenge

    paths_with_category = _get_challenge_paths(state)
    if not paths_with_category:
        logger.warning(
            "Validation agent: no challenge paths to validate (organized_challenges empty or PROCESSED_DIR missing)"
        )
        return state

    reports: List[ValidationReport] = []
    for challenge_path, category in paths_with_category:
        challenge_name = challenge_path.name
        challenge_id = f"{category}/{challenge_name}"

        try:
            report = validate_challenge(challenge_path, category)
            step0_issues = _check_step0_reproducibility(
                challenge_path, category, challenge_name, challenge_id
            )
            if step0_issues:
                report = report.model_copy(
                    update={"issues": list(report.issues) + step0_issues}
                )
                logger.debug(
                    "Step 0 issues for {}: {}", challenge_id, len(step0_issues)
                )
            reports.append(report)
            status = "PASS" if report.is_valid else "FAIL"
            logger.info(
                "Validation {} {} (score={:.2f})",
                status,
                challenge_id,
                report.structure_score,
            )
        except Exception as e:
            logger.exception("Validation failed for {}: {}", challenge_id, e)
            state.add_error("validation_agent", challenge_id, str(e))
            reports.append(
                ValidationReport(
                    challenge_id=challenge_id,
                    is_valid=False,
                    issues=[
                        ValidationIssue(
                            code="VALIDATION_ERROR",
                            message=str(e),
                            severity=IssueSeverity.CRITICAL,
                        )
                    ],
                    structure_score=0.0,
                )
            )

    return replace(
        state,
        validation_reports=reports,
        current_agent="validation_agent",
    )


async def validation_agent(state: AgentState) -> AgentState:
    """Validate challenge structural completeness and Step 0 (reproducibility).

    Structural checks: required files, directory organization (via pipeline
    validate_challenge). Research-based: Step 0 metadata and resource
    availability (flag missing so Content Generation or humans can complete).

    Args:
        state: Pipeline state with challenges or organized_challenges.

    Returns:
        Updated state with validation_reports populated.

    Example:
        >>> state = AgentState(challenges=[...])
        >>> state = await validation_agent(state)
        >>> for r in state.validation_reports:
        ...     print(r.challenge_id, r.is_valid, len(r.issues))
    """
    return run_validation_agent(state)
