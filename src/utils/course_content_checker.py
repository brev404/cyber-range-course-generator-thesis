"""Post-generation content checks.

Detects and flags phrases that imply "you don't have access" and suggests
rewording to "what the student has" framing. Used to prevent negative framing
in generated courses and catch any that slip through during generation.

Additional checks:
  - dimension_hint on every ValidationIssue (exact key from RankingReport.dimension_scores)
  - check_section_presence: detects missing required sections
  - build_ranking_hint_string: formats issues into a compact LLM context hint

CHECK-TO-DIMENSION MAPPING
==========================
Each check maps to the dimension (exact key in RankingReport.dimension_scores) most
affected by its failure. Dimensions come from the two ranking personas:

  Technical:   correctness | completeness | technical_accuracy | code_quality | logical_validity
  Pedagogical: sections_structure | cognitive_load | scaffolding_reproducibility |
               relevance_curriculum | skill_level_awareness | human_language_context

Mapping:
  check_no_access_framing     → human_language_context
    Rationale: negative/restriction framing degrades course language quality and
    audience appropriateness, directly penalised by the human_language_context dimension.

  check_section_presence
    → sections_structure         for most missing required sections
      (Abstract, Objectives, Skills, Definitions, Thought Process,
       Step-by-step, Solution Script, Conclusion)
    → scaffolding_reproducibility for missing Step 0 / Reproducibility section
      Rationale: Step 0 is the primary reproducibility anchor; its absence maps to
      both sections_structure and scaffolding_reproducibility; we emit two issues so
      both dimensions are flagged in the hint string.

Hints are advisory only — the LLM reviewer may disagree; scores are never hard-coded.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from src.models.report_models import IssueSeverity, ValidationIssue

# ---------------------------------------------------------------------------
# No-access framing patterns
# ---------------------------------------------------------------------------

# Each tuple: (compiled pattern, short label).  Case-insensitive matching.
_NO_ACCESS_PATTERNS: List[Tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"you\s+don'?t\s+have\s+access\s+to", re.I),
        "you don't have access to",
    ),
    (re.compile(r"you\s+cannot\s+access", re.I), "you cannot access"),
    (
        re.compile(r"you\s+do\s+not\s+have\s+(?:access\s+to\s+)?(?:the\s+)?", re.I),
        "you do not have",
    ),
    (re.compile(r"the\s+user\s+cannot\s+see", re.I), "the user cannot see"),
    (re.compile(r"not\s+available\s+to\s+you", re.I), "not available to you"),
    (re.compile(r"you\s+cannot\s+see", re.I), "you cannot see"),
]

# ---------------------------------------------------------------------------
# Required section definitions for section-presence check
# ---------------------------------------------------------------------------

# Each entry: (section_name, [heading regex alternations], dimension_hint)
# Patterns match markdown headings (# or ##) with flexible wording.
# v4.2.1: now matches numbered headings `## N. <name>` as well as plain `## <name>`.
# Without this, v4 numbered courses (## 1. Title, ## 2. Abstract, ...) were
# triggering false-missing-section flags in every judge check_hints, anchoring scores down.
_NUM_PREFIX = (
    r"(?:\d+\.\s+)?"  # optional "N. " prefix between heading hashes and the name
)
_REQUIRED_SECTIONS: List[Tuple[str, re.Pattern[str], str]] = [
    (
        "Abstract/TL;DR",
        re.compile(
            rf"^#{{1,3}}\s+{_NUM_PREFIX}(?:abstract|tl[;:]?dr|tldr|overview|summary)",
            re.I | re.M,
        ),
        "sections_structure",
    ),
    (
        "Objectives",
        re.compile(
            rf"^#{{1,3}}\s+{_NUM_PREFIX}(?:objective|learning\s+goal|goal|purpose)",
            re.I | re.M,
        ),
        "sections_structure",
    ),
    (
        "Technical Skills",
        re.compile(
            rf"^#{{1,3}}\s+{_NUM_PREFIX}(?:technical\s+skill|skill|prerequisites?|what\s+you.ll\s+learn)",
            re.I | re.M,
        ),
        "sections_structure",
    ),
    (
        "Definitions/Concepts",
        re.compile(
            rf"^#{{1,3}}\s+{_NUM_PREFIX}(?:definition|concept|terminolog|glossar|background|key\s+concept)",
            re.I | re.M,
        ),
        "sections_structure",
    ),
    (
        "Thought Process",
        re.compile(
            rf"^#{{1,3}}\s+{_NUM_PREFIX}(?:thought\s+process|approach|analysis|reasoning|narrative|methodology)",
            re.I | re.M,
        ),
        "sections_structure",
    ),
    (
        "Step-by-Step",
        re.compile(
            rf"^#{{1,3}}\s+{_NUM_PREFIX}(?:step[- ]by[- ]step|steps?|walkthrough|exploitation|solution\s+steps?|procedure)",
            re.I | re.M,
        ),
        "sections_structure",
    ),
    (
        "Solution Script",
        re.compile(
            rf"^#{{1,3}}\s+{_NUM_PREFIX}(?:solution\s+script|solve\s+script|exploit|code|script|solver)",
            re.I | re.M,
        ),
        "sections_structure",
    ),
    (
        "Conclusion",
        re.compile(
            rf"^#{{1,3}}\s+{_NUM_PREFIX}(?:conclusion|summary|wrap.?up|final|takeaway|lesson)",
            re.I | re.M,
        ),
        "sections_structure",
    ),
]

# Step 0 / Reproducibility section — dual-dimension (sections_structure + scaffolding_reproducibility)
_STEP0_PATTERN = re.compile(
    rf"^#{{1,3}}\s+{_NUM_PREFIX}(?:step\s+0|step\s+zero|reproducib|prerequisite|setup|environment|resource|getting\s+started)",
    re.I | re.M,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _suggest_rewrite(matched_phrase: str) -> str:
    """Generate a rewrite suggestion using 'what the student has' framing."""
    lower = matched_phrase.lower()
    if "source" in lower or "code" in lower:
        return (
            "You have access to the challenge description and public files; "
            "the server source code is not provided."
        )
    if "server" in lower or "internals" in lower:
        return (
            "You can interact with the deployed service via HTTP; "
            "the server internals are not exposed."
        )
    if "flag" in lower:
        return (
            "The flag is revealed after solving the challenge; "
            "it is not given in advance."
        )
    if "writeup" in lower or "solution" in lower:
        return (
            "You have access to the challenge description and public files; "
            "the author writeup is not provided."
        )
    return (
        "Use 'what the student has' framing: state what the student has access to "
        "(challenge description, public files, deployment), then optionally note "
        "what is not provided as a fact, not a restriction."
    )


# ---------------------------------------------------------------------------
# Public check functions
# ---------------------------------------------------------------------------


def check_no_access_framing(
    course_text: str,
    challenge_id: str = "",
    file_path: Optional[str] = None,
) -> List[ValidationIssue]:
    """Detect phrases that imply 'you don't have access' and suggest rewording.

    Scans course text for patterns like "you don't have access to", "you cannot
    see", "not available to you", etc. For each match, creates a ValidationIssue
    with a rewrite suggestion using "what the student has" framing.

    dimension_hint: "human_language_context" — negative/restriction framing
    degrades course language quality and audience appropriateness.

    Args:
        course_text: The generated course text to scan.
        challenge_id: Optional challenge ID for context in messages.
        file_path: Optional file path (e.g. "cyberedu/write-up/course.md").

    Returns:
        List of ValidationIssue for each detected pattern. Empty if none found.

    Example:
        >>> issues = check_no_access_framing(
        ...     "You don't have access to the source code.",
        ...     challenge_id="web/xss-01",
        ... )
        >>> len(issues)
        1
        >>> issues[0].code
        'NO_ACCESS_FRAMING'
        >>> issues[0].dimension_hint
        'human_language_context'
    """
    if not course_text or not course_text.strip():
        return []

    issues: List[ValidationIssue] = []
    seen_spans: set[Tuple[int, int]] = set()

    for pattern, label in _NO_ACCESS_PATTERNS:
        for match in pattern.finditer(course_text):
            span = match.span()
            if span in seen_spans:
                continue
            seen_spans.add(span)
            matched = match.group(0).strip()
            rewrite_example = _suggest_rewrite(matched)
            message = f"Detected 'no access' framing: '{matched}'"
            full_suggestion = (
                f"Instead of '{matched}', use 'what the student has' framing: "
                f"'{rewrite_example}'"
            )
            issues.append(
                ValidationIssue(
                    code="NO_ACCESS_FRAMING",
                    message=message,
                    severity=IssueSeverity.MEDIUM,
                    file_path=file_path,
                    suggestion=full_suggestion,
                    dimension_hint="human_language_context",
                )
            )

    return issues


def check_section_presence(
    course_text: str,
    challenge_id: str = "",
    file_path: Optional[str] = None,
) -> List[ValidationIssue]:
    """Detect missing required sections in a generated course.

    Checks for 8 required sections plus the Step 0 / Reproducibility section.
    Missing standard sections map to dimension_hint="sections_structure".
    Missing Step 0 emits two issues: one for sections_structure and one for
    scaffolding_reproducibility, since it is the primary reproducibility anchor.

    Args:
        course_text: The generated course text to scan.
        challenge_id: Optional challenge ID for context in messages.
        file_path: Optional file path.

    Returns:
        List of ValidationIssue for each missing section. Empty if all present.

    Example:
        >>> issues = check_section_presence(
        ...     "## Objectives\\n...\\n## Step-by-Step\\n...",
        ...     challenge_id="web/xss-01",
        ... )
        >>> # Missing many sections → multiple issues
        >>> all(i.code == 'MISSING_SECTION' for i in issues)
        True
        >>> all(i.dimension_hint in ('sections_structure', 'scaffolding_reproducibility')
        ...     for i in issues)
        True
    """
    if not course_text or not course_text.strip():
        return []

    issues: List[ValidationIssue] = []

    for section_name, pattern, dim_hint in _REQUIRED_SECTIONS:
        if not pattern.search(course_text):
            issues.append(
                ValidationIssue(
                    code="MISSING_SECTION",
                    message=f"Missing required section: '{section_name}'",
                    severity=IssueSeverity.MEDIUM,
                    file_path=file_path,
                    suggestion=f"Add a '## {section_name}' heading with appropriate content.",
                    dimension_hint=dim_hint,
                )
            )

    # Step 0 / Reproducibility: dual-dimension
    if not _STEP0_PATTERN.search(course_text):
        issues.append(
            ValidationIssue(
                code="MISSING_SECTION",
                message="Missing required section: 'Step 0 / Reproducibility'",
                severity=IssueSeverity.MEDIUM,
                file_path=file_path,
                suggestion=(
                    "Add a '## Step 0' or '## Reproducibility' section covering "
                    "metadata, required tools, and environment setup."
                ),
                dimension_hint="sections_structure",
            )
        )
        issues.append(
            ValidationIssue(
                code="MISSING_STEP0",
                message="Step 0 / Reproducibility section absent — scaffolding anchor missing",
                severity=IssueSeverity.MEDIUM,
                file_path=file_path,
                suggestion=(
                    "Add a reproducibility section (Step 0) with challenge metadata, "
                    "tool list, and environment prerequisites so students can follow along."
                ),
                dimension_hint="scaffolding_reproducibility",
            )
        )

    return issues


def build_ranking_hint_string(issues: List[ValidationIssue]) -> str:
    """Format a list of ValidationIssue objects into a compact LLM context hint.

    Groups issues by their dimension_hint and produces a one-line advisory string
    suitable for injection into a ranking prompt. Issues without a dimension_hint
    are silently ignored.

    Hints are advisory only — the LLM reviewer may disagree; no scores are
    hard-coded.

    Args:
        issues: List of ValidationIssue (typically from check_no_access_framing
            and check_section_presence).

    Returns:
        A non-empty string like
            "Automated checks flagged: human_language_context (no-access phrasing, 2 instances),
             sections_structure (missing sections: Abstract/TL;DR, Objectives),
             scaffolding_reproducibility (Step 0 absent)"
        or an empty string if no dimension-hinted issues exist.

    Example:
        >>> from src.models.report_models import ValidationIssue, IssueSeverity
        >>> issues = [
        ...     ValidationIssue(code='NO_ACCESS_FRAMING', message='m', severity=IssueSeverity.MEDIUM,
        ...                     dimension_hint='human_language_context'),
        ...     ValidationIssue(code='MISSING_SECTION', message="Missing: 'Objectives'",
        ...                     severity=IssueSeverity.MEDIUM, dimension_hint='sections_structure'),
        ... ]
        >>> s = build_ranking_hint_string(issues)
        >>> 'human_language_context' in s
        True
        >>> 'sections_structure' in s
        True
    """
    if not issues:
        return ""

    # Group by dimension_hint
    by_dim: Dict[str, List[ValidationIssue]] = defaultdict(list)
    for issue in issues:
        if issue.dimension_hint:
            by_dim[issue.dimension_hint].append(issue)

    if not by_dim:
        return ""

    parts: List[str] = []

    # human_language_context: count no-access instances
    if "human_language_context" in by_dim:
        count = len(by_dim["human_language_context"])
        noun = "instance" if count == 1 else "instances"
        parts.append(f"human_language_context (no-access phrasing, {count} {noun})")

    # sections_structure: list missing section names
    if "sections_structure" in by_dim:
        names: List[str] = []
        for iss in by_dim["sections_structure"]:
            # Extract section name from message: "Missing required section: 'X'"
            m = re.search(r"['\"]([^'\"]+)['\"]", iss.message)
            if m:
                names.append(m.group(1))
        detail = (
            ", ".join(names)
            if names
            else f"{len(by_dim['sections_structure'])} sections"
        )
        parts.append(f"sections_structure (missing: {detail})")

    # scaffolding_reproducibility
    if "scaffolding_reproducibility" in by_dim:
        parts.append(
            "scaffolding_reproducibility (Step 0 / reproducibility section absent)"
        )

    # Any other dimension hints not handled above
    handled = {
        "human_language_context",
        "sections_structure",
        "scaffolding_reproducibility",
    }
    for dim, dim_issues in by_dim.items():
        if dim not in handled:
            parts.append(f"{dim} ({len(dim_issues)} issue(s))")

    if not parts:
        return ""

    return "Automated checks flagged: " + "; ".join(parts)
