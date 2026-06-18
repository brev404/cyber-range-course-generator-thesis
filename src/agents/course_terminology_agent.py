"""Course terminology checker + content checks.

Python-first gate on generated courses before mapping/ranking. Validates
ATT&CK/CWE/OWASP IDs in course text and checks for "no access" framing.
Respects TERMINOLOGY_CHECK_MODE: off, annotate, warn, block.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, List

from loguru import logger

from src.config.settings import settings as app_settings
from src.core.state import AgentState
from src.models.report_models import ValidationIssue
from src.utils.course_content_checker import check_no_access_framing
from src.validators.terminology_checker import check_terminology


def run_course_terminology_checker(state: AgentState) -> AgentState:
    """Check generated courses for terminology issues (invalid ATT&CK/CWE/OWASP IDs).

    Respects TERMINOLOGY_CHECK_MODE:
    - off: Skip checks, pass-through to mapping.
    - annotate: Run checks, log only, proceed (check_terminology returns []).
    - warn: Run checks, store issues in state, proceed.
    - block: Run checks, store issues; routing will send to refinement_step if any.

    Args:
        state: AgentState with generated_courses populated.

    Returns:
        Updated state with course_terminology_issues and current_agent set.
    """
    mode = getattr(app_settings, "TERMINOLOGY_CHECK_MODE", "warn")
    if mode == "off":
        return replace(
            state,
            current_agent="course_terminology_checker",
        )

    issues_by_id: Dict[str, List[ValidationIssue]] = {}
    for cid, course_text in (state.generated_courses or {}).items():
        if not course_text or not course_text.strip():
            continue
        issues: List[ValidationIssue] = []
        # Terminology: ATT&CK/CWE/OWASP IDs
        issues.extend(
            check_terminology(
                course_text,
                challenge_id=cid,
                file_path="cyberedu/write-up/course.md",
            )
        )
        # Content: "no access" framing
        issues.extend(
            check_no_access_framing(
                course_text,
                challenge_id=cid,
                file_path="cyberedu/write-up/course.md",
            )
        )
        if issues:
            issues_by_id[cid] = issues
            if mode == "warn":
                logger.warning(
                    "Course terminology (warn): %s has %s issue(s)",
                    cid,
                    len(issues),
                )
            elif mode == "block":
                logger.info(
                    "Course terminology (block): %s has %s issue(s); will route to refinement",
                    cid,
                    len(issues),
                )

    return replace(
        state,
        current_agent="course_terminology_checker",
        course_terminology_issues=issues_by_id,
    )
