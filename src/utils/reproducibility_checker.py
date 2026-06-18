"""Reproducibility checker for generated courses.

Validates that course.md content satisfies reproducibility requirements:
- Step 0 / Reproducibility section present
- Explicit "what the student has" (description, public files, deployment)
- Documented resources (binary, PCAP, source, etc.)
- Metadata completeness (challenge name, category, difficulty)

See the course writeup guidelines §6 for reproducibility requirements.
"""

from __future__ import annotations

import re
from typing import List

# Step 0 section heading (aligned with evaluation_service._RUBRIC_SECTIONS)
# Allows numbered headings like "### 6. Reproducibility (Step 0)"
_STEP0_HEADING_PATTERN = re.compile(
    r"(?:^#+|\n#+)\s*[^\n]*(?:reproducibility|step\s*0)\b",
    re.IGNORECASE,
)

# "What the student has" indicators (at least one required in Step 0 region)
_WHAT_STUDENT_HAS_PATTERNS = [
    r"you will have access to",
    r"what you have",
    r"public files",
    r"challenge description",
    r"deployment",
    r"student has",
    r"to follow along",
]

# Resource indicators (at least one required in Step 0 region)
_RESOURCE_PATTERNS = [
    r"public files",
    r"binary",
    r"pcap",
    r"source",
    r"deployment",
    r"description",
]

# Metadata indicators (challenge name, category, difficulty) - can be in Title/Context or Step 0
_METADATA_PATTERNS = [
    r"challenge name",
    r"category",
    r"difficulty",
]


def _extract_step0_region(text: str) -> str | None:
    """Extract text after Reproducibility/Step 0 heading until next ## heading.

    Args:
        text: Full course markdown.

    Returns:
        Step 0 region text, or None if section not found.
    """
    match = re.search(
        r"(?:^#+|\n#+)\s*[^\n]*(?:reproducibility|step\s*0)\b[^\n]*\n+(.+?)(?=\n#+|\Z)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return None
    return match.group(1).strip()


def _region_contains_any(text: str, patterns: List[str]) -> bool:
    """Check if text contains any of the given patterns (case-insensitive)."""
    lower = text.lower()
    for pat in patterns:
        if re.search(pat, lower):
            return True
    return False


def check_course_reproducibility(course_text: str, challenge_id: str = "") -> List[str]:
    """Check generated course for reproducibility requirements.

    Args:
        course_text: Full course markdown content.
        challenge_id: Optional challenge identifier for error messages.

    Returns:
        List of error messages; empty if all checks pass.
    """
    errors: List[str] = []
    prefix = f"[{challenge_id}] " if challenge_id else ""

    if not (course_text or "").strip():
        errors.append(f"{prefix}Empty course; cannot check reproducibility.")
        return errors

    text = course_text.strip()

    # 1. Step 0 section present
    if not _STEP0_HEADING_PATTERN.search(text):
        errors.append(f"{prefix}Missing Reproducibility (Step 0) section heading.")
        return errors  # Cannot check further without Step 0

    step0_region = _extract_step0_region(text)
    if not step0_region:
        errors.append(f"{prefix}Reproducibility section heading found but no content.")
        return errors

    # 2. "What the student has" - at least one indicator
    if not _region_contains_any(step0_region, _WHAT_STUDENT_HAS_PATTERNS):
        errors.append(
            f"{prefix}Step 0 missing explicit 'what the student has' "
            "(e.g. 'you will have access to', 'public files', 'challenge description', 'deployment')."
        )

    # 3. Documented resources - at least one mention
    if not _region_contains_any(step0_region, _RESOURCE_PATTERNS):
        errors.append(
            f"{prefix}Step 0 missing documented resources "
            "(e.g. public files, binary, PCAP, source, deployment, description)."
        )

    # 4. Metadata completeness - check full text (Title/Context or Step 0)
    if not _region_contains_any(text, _METADATA_PATTERNS):
        errors.append(
            f"{prefix}Course missing metadata (challenge name, category, difficulty) "
            "in Title/Context or Step 0."
        )

    return errors
