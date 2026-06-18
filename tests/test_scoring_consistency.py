"""Tests for Post-Generation Checks and Scoring Consistency.

Verifies:
- check_no_access_framing produces issues with dimension_hint='human_language_context'
- check_section_presence detects missing sections with correct dimension hints
- build_ranking_hint_string formats issues into an advisory string
- run_ranking_agent passes hint strings to _evaluate_one_challenge
- ValidationIssue.dimension_hint field is optional and backward-compatible
"""

from unittest.mock import patch

from src.agents.ranking_agent import _aggregate_dimension_scores, run_ranking_agent
from src.core.state import AgentState
from src.models.report_models import (
    IssueSeverity,
    RankingReport,
    RankingScore,
    ValidationIssue,
)
from src.utils.course_content_checker import (
    build_ranking_hint_string,
    check_no_access_framing,
    check_section_presence,
)

# ---------------------------------------------------------------------------
# ValidationIssue.dimension_hint — backward compatibility
# ---------------------------------------------------------------------------


def test_validation_issue_dimension_hint_defaults_none() -> None:
    """Existing code creating ValidationIssue without dimension_hint still works."""
    issue = ValidationIssue(
        code="MISSING_FILE",
        message="Missing solver script",
        severity=IssueSeverity.CRITICAL,
    )
    assert issue.dimension_hint is None


def test_validation_issue_dimension_hint_stored() -> None:
    """dimension_hint is stored and retrieved correctly."""
    issue = ValidationIssue(
        code="NO_ACCESS_FRAMING",
        message="Detected 'no access' framing",
        severity=IssueSeverity.MEDIUM,
        dimension_hint="human_language_context",
    )
    assert issue.dimension_hint == "human_language_context"


def test_validation_issue_dimension_hint_serializable() -> None:
    """dimension_hint survives model_dump / JSON round-trip."""
    import json

    issue = ValidationIssue(
        code="MISSING_SECTION",
        message="Missing: 'Objectives'",
        severity=IssueSeverity.MEDIUM,
        dimension_hint="sections_structure",
    )
    data = issue.model_dump()
    serialized = json.dumps(data)
    loaded = json.loads(serialized)
    assert loaded["dimension_hint"] == "sections_structure"


# ---------------------------------------------------------------------------
# check_no_access_framing — dimension_hint
# ---------------------------------------------------------------------------


def test_no_access_framing_dimension_hint() -> None:
    """Every issue from check_no_access_framing has dimension_hint='human_language_context'."""
    issues = check_no_access_framing(
        "You don't have access to the source code.", challenge_id="web/xss-01"
    )
    assert len(issues) >= 1
    for issue in issues:
        assert issue.dimension_hint == "human_language_context"


def test_no_access_framing_all_patterns_have_hint() -> None:
    """All pattern matches produce the correct dimension hint."""
    texts = [
        "You don't have access to the server.",
        "You cannot access the backend.",
        "You do not have the flag.",
        "The user cannot see the state.",
        "This feature is not available to you.",
        "You cannot see the source code.",
    ]
    for text in texts:
        issues = check_no_access_framing(text)
        assert all(
            i.dimension_hint == "human_language_context" for i in issues
        ), f"Wrong hint for: {text!r}"


def test_no_access_framing_no_hint_on_empty() -> None:
    """Empty input returns no issues (no hint needed)."""
    assert check_no_access_framing("") == []


# ---------------------------------------------------------------------------
# check_section_presence — dimension hints
# ---------------------------------------------------------------------------


_FULL_COURSE = """\
## Abstract
A short summary.

## Objectives
Learn XSS.

## Technical Skills
HTML, JavaScript.

## Definitions and Concepts
XSS is a vulnerability.

## Step 0
Install tools.

## Thought Process
I noticed the input was reflected.

## Step-by-Step
1. Inject payload.

## Solution Script
```python
print("flag")
```

## Conclusion
We learned XSS.
"""


def test_section_presence_no_issues_on_full_course() -> None:
    """A course with all required sections produces no missing-section issues."""
    issues = check_section_presence(_FULL_COURSE, challenge_id="test")
    section_issues = [
        i for i in issues if i.code in ("MISSING_SECTION", "MISSING_STEP0")
    ]
    assert len(section_issues) == 0


def test_section_presence_detects_missing_objectives() -> None:
    """Missing Objectives heading is flagged with sections_structure hint."""
    text = "## Abstract\nSome text.\n## Step-by-Step\nDo stuff."
    issues = check_section_presence(text, challenge_id="test")
    obj_issues = [i for i in issues if "Objectives" in i.message]
    assert len(obj_issues) >= 1
    for issue in obj_issues:
        assert issue.dimension_hint == "sections_structure"


def test_section_presence_all_standard_dims_are_sections_structure() -> None:
    """All non-Step0 missing sections map to sections_structure."""
    issues = check_section_presence("No sections here at all.", challenge_id="test")
    for issue in issues:
        if issue.code == "MISSING_SECTION":
            assert (
                issue.dimension_hint == "sections_structure"
            ), f"Expected sections_structure, got {issue.dimension_hint!r} for: {issue.message}"


def test_section_presence_step0_dual_dimension() -> None:
    """Missing Step 0 produces one sections_structure and one scaffolding_reproducibility issue."""
    text = "## Abstract\nHello.\n## Objectives\nLearn."
    issues = check_section_presence(text, challenge_id="test")
    step0_issues = [
        i for i in issues if "Step 0" in i.message or i.code == "MISSING_STEP0"
    ]
    dims = {i.dimension_hint for i in step0_issues}
    assert (
        "sections_structure" in dims
    ), "sections_structure not flagged for missing Step 0"
    assert (
        "scaffolding_reproducibility" in dims
    ), "scaffolding_reproducibility not flagged for missing Step 0"


def test_section_presence_empty_input() -> None:
    """Empty input returns no issues."""
    assert check_section_presence("") == []
    assert check_section_presence("   ") == []


# ---------------------------------------------------------------------------
# build_ranking_hint_string
# ---------------------------------------------------------------------------


def test_build_hint_empty_on_no_issues() -> None:
    assert build_ranking_hint_string([]) == ""


def test_build_hint_empty_on_issues_without_hints() -> None:
    """Issues with no dimension_hint are ignored."""
    issues = [ValidationIssue(code="FOO", message="bar", severity=IssueSeverity.LOW)]
    assert build_ranking_hint_string(issues) == ""


def test_build_hint_includes_human_language_context() -> None:
    issues = check_no_access_framing(
        "You cannot access the server.", challenge_id="test"
    )
    hint = build_ranking_hint_string(issues)
    assert "human_language_context" in hint
    assert "no-access phrasing" in hint


def test_build_hint_includes_sections_structure() -> None:
    issues = check_section_presence("No sections here.", challenge_id="test")
    hint = build_ranking_hint_string(issues)
    assert "sections_structure" in hint


def test_build_hint_includes_scaffolding_reproducibility() -> None:
    issues = check_section_presence("No sections here.", challenge_id="test")
    hint = build_ranking_hint_string(issues)
    assert "scaffolding_reproducibility" in hint


def test_build_hint_starts_with_automated_checks() -> None:
    """Hint string starts with the expected prefix."""
    issues = check_no_access_framing("You cannot see the backend.", challenge_id="x")
    hint = build_ranking_hint_string(issues)
    assert hint.startswith("Automated checks flagged:")


def test_build_hint_counts_multiple_no_access() -> None:
    """Multiple no-access matches produce a count in the hint."""
    text = "You cannot access the server. Also you cannot see the source code."
    issues = check_no_access_framing(text)
    hint = build_ranking_hint_string(issues)
    assert "2 instances" in hint or "instance" in hint


def test_build_hint_combined_checks() -> None:
    """Combined no-access + missing sections produces both dimensions in hint."""
    na_issues = check_no_access_framing("You cannot access the server.")
    sec_issues = check_section_presence("No sections here.")
    all_issues = na_issues + sec_issues
    hint = build_ranking_hint_string(all_issues)
    assert "human_language_context" in hint
    assert "sections_structure" in hint
    assert "scaffolding_reproducibility" in hint


# ---------------------------------------------------------------------------
# run_ranking_agent — hints passed to _evaluate_one_challenge
# ---------------------------------------------------------------------------


def _make_mock_report(challenge_id: str) -> RankingReport:
    tech_dims = {
        "correctness": 9,
        "completeness": 8,
        "technical_accuracy": 9,
        "code_quality": 8,
        "logical_validity": 9,
    }
    ped_dims = {
        "sections_structure": 8,
        "cognitive_load": 7,
        "scaffolding_reproducibility": 9,
        "relevance_curriculum": 8,
        "skill_level_awareness": 9,
        "human_language_context": 8,
    }
    agg = _aggregate_dimension_scores(tech_dims, ped_dims)
    return RankingReport(
        challenge_id=challenge_id,
        overall_score=8.5,
        technical_review=RankingScore(
            score=9,
            persona="Technical",
            justification="T",
            improvements=[],
            dimension_scores=tech_dims,
        ),
        pedagogical_review=RankingScore(
            score=8,
            persona="Pedagogical",
            justification="P",
            improvements=[],
            dimension_scores=ped_dims,
        ),
        technical_rank="Intermediate",
        dimension_scores=agg,
    )


def test_run_ranking_agent_passes_hint_when_no_access_found() -> None:
    """When course text has no-access phrasing, run_ranking_agent calls
    _evaluate_one_challenge with a non-empty check_hints string."""
    course = "You cannot access the server internals. Here is the exploit."

    captured: dict = {}

    def fake_evaluate(challenge_id, writeup, solve_script, check_hints=None, **kwargs):
        captured["check_hints"] = check_hints
        return _make_mock_report(challenge_id)

    state = AgentState(generated_courses={"ch_hint": course})
    with patch(
        "src.agents.ranking_agent._evaluate_one_challenge", side_effect=fake_evaluate
    ):
        run_ranking_agent(state)

    assert "check_hints" in captured, "_evaluate_one_challenge was not called"
    assert (
        captured["check_hints"] is not None and captured["check_hints"] != ""
    ), "Expected a non-empty hint string for course with no-access phrasing"
    assert "human_language_context" in captured["check_hints"]


def test_run_ranking_agent_no_hint_for_clean_course() -> None:
    """A well-structured course with no issues produces an empty/None hint."""
    captured: dict = {}

    def fake_evaluate(challenge_id, writeup, solve_script, check_hints=None, **kwargs):
        captured["check_hints"] = check_hints
        return _make_mock_report(challenge_id)

    state = AgentState(generated_courses={"ch_clean": _FULL_COURSE})
    with patch(
        "src.agents.ranking_agent._evaluate_one_challenge", side_effect=fake_evaluate
    ):
        run_ranking_agent(state)

    assert "check_hints" in captured
    # A full course may still have some edge-case misses but should not have
    # no-access or major section issues → hint should be empty or None
    hint = captured["check_hints"] or ""
    assert "human_language_context" not in hint


def test_run_ranking_agent_incorporates_state_terminology_issues() -> None:
    """Issues already in state.course_terminology_issues are merged into hints."""
    course = "## Abstract\nsome content without no-access phrasing"
    term_issue = ValidationIssue(
        code="NO_ACCESS_FRAMING",
        message="Detected 'no access' framing: 'you cannot see'",
        severity=IssueSeverity.MEDIUM,
        dimension_hint="human_language_context",
    )

    captured: dict = {}

    def fake_evaluate(challenge_id, writeup, solve_script, check_hints=None, **kwargs):
        captured["check_hints"] = check_hints
        return _make_mock_report(challenge_id)

    state = AgentState(
        generated_courses={"ch_term": course},
        course_terminology_issues={"ch_term": [term_issue]},
    )
    with patch(
        "src.agents.ranking_agent._evaluate_one_challenge", side_effect=fake_evaluate
    ):
        run_ranking_agent(state)

    assert "check_hints" in captured
    hint = captured["check_hints"] or ""
    assert "human_language_context" in hint
