"""Test F8: structural_validator service — validate_course_structure, validate_solver_syntax,
validate_extra_resources, and validate_all."""


def _make_full_course(include_conclusion=True, include_extra=True, extra_ref="T1566"):
    """Build a minimal course.md with all 11 sections."""
    sections = [
        "## 1. Title and Context\nSome challenge.",
        "## 2. Abstract / TL;DR\nBrief abstract.",
        "## 3. Objectives\nLearner goals.",
        "## 4. Technical Skills\nPython, crypto.",
        "## 5. Definitions and Concepts\nXSS: cross-site scripting.",
        "## 6. Reproducibility (Step 0)\nYou have the description.",
        "## 7. Thought Process / Narrative\nWe observe the service accepts input.",
        "## 8. Step-by-Step Resolution\n**Step 1**: run script. Expected output: `CTF{...}`",
        "## 9. Solution Script\n```python\nprint('CTF{flag}')\n```",
    ]
    if include_conclusion:
        sections.append("## 10. Conclusion\nWe exploited the weakness.")
    if include_extra:
        ref_text = (
            f"| {extra_ref} | Relevant technique |"
            if extra_ref
            else "No references here."
        )
        sections.append(f"## 11. Extra Resources\n{ref_text}")
    return "\n\n".join(sections)


VALID_SOLVER = "def solve():\n    print('CTF{flag}')\n\nsolve()\n"
EMPTY_SOLVER = ""
SYNTAX_ERROR_SOLVER = "def solve(\n    print('broken')\n"


# ── validate_course_structure ────────────────────────────────────────────────


def test_valid_10_section_course_passes():
    from src.services.structural_validator import validate_course_structure

    course = _make_full_course()
    report = validate_course_structure(course)
    assert report.is_valid is True
    assert len(report.issues) == 0


def test_missing_conclusion_detected():
    from src.services.structural_validator import validate_course_structure

    course = _make_full_course(include_conclusion=False)
    report = validate_course_structure(course)
    assert report.is_valid is False
    assert any("conclusion" in issue.lower() for issue in report.issues)


def test_truncation_marker_detected():
    from src.services.structural_validator import validate_course_structure

    course = _make_full_course() + "\n\n[... truncated for length ...]"
    report = validate_course_structure(course)
    assert report.is_valid is False
    assert any("truncat" in issue.lower() for issue in report.issues)


def test_truncation_deferral_phrases_detected():
    """v4.1.2: F8 must catch deferral / exercise-left / abbreviated phrases that
    smoke #3 saw judges complain about but the original marker list missed."""
    from src.services.structural_validator import validate_course_structure

    phrases = [
        "complete the rest using the same approach",
        "left as an exercise to the reader",
        "implementation left to the student",
        "rest is omitted for brevity",
        "[remainder omitted]",
        "[abbreviated]",
        "(continued)",
    ]
    for phrase in phrases:
        course = _make_full_course() + f"\n\nSome step text. {phrase}"
        report = validate_course_structure(course)
        assert report.is_valid is False, f"Expected F8 to catch phrase: {phrase!r}"
        assert any(
            "truncation marker found" in issue.lower() for issue in report.issues
        ), f"For phrase {phrase!r}, expected truncation issue, got: {report.issues}"


# ── validate_solver_syntax ───────────────────────────────────────────────────


def test_empty_solver_not_valid():
    from src.services.structural_validator import validate_solver_syntax

    report = validate_solver_syntax(EMPTY_SOLVER)
    assert report.is_valid is False
    assert len(report.issues) > 0


def test_valid_solver_passes():
    from src.services.structural_validator import validate_solver_syntax

    report = validate_solver_syntax(VALID_SOLVER)
    assert report.is_valid is True


def test_syntax_error_captured():
    from src.services.structural_validator import validate_solver_syntax

    report = validate_solver_syntax(SYNTAX_ERROR_SOLVER)
    assert report.is_valid is False
    # Issue should mention the parser error
    assert any(
        "syntax" in issue.lower() or "SyntaxError" in issue for issue in report.issues
    )


# ── validate_extra_resources ─────────────────────────────────────────────────


def test_extra_resources_with_attack_ref_passes():
    from src.services.structural_validator import validate_extra_resources

    course = _make_full_course(extra_ref="T1566")
    report = validate_extra_resources(course)
    assert report.is_valid is True


def test_extra_resources_with_cwe_ref_passes():
    from src.services.structural_validator import validate_extra_resources

    course = _make_full_course(extra_ref="CWE-79")
    report = validate_extra_resources(course)
    assert report.is_valid is True


def test_extra_resources_with_owasp_ref_passes():
    from src.services.structural_validator import validate_extra_resources

    course = _make_full_course(extra_ref="OWASP")
    report = validate_extra_resources(course)
    assert report.is_valid is True


def test_extra_resources_no_refs_not_valid():
    from src.services.structural_validator import validate_extra_resources

    course = _make_full_course(extra_ref="")
    report = validate_extra_resources(course)
    assert report.is_valid is False
    assert len(report.issues) > 0


# ── validate_section_numbering (v4.1) ────────────────────────────────────────


def test_validate_section_numbering_valid_1_through_11_passes():
    from src.services.structural_validator import validate_section_numbering

    course = _make_full_course()
    report = validate_section_numbering(course)
    assert report.is_valid is True, f"Expected pass, got issues: {report.issues}"


def test_validate_section_numbering_duplicate_9_fails():
    """The 0_solves smoke-trial failure: LLM wrote `## 9. Step-by-Step Resolution` AND
    assembly inserted `## 9. Solution Script` → two `## 9.` headings."""
    from src.services.structural_validator import validate_section_numbering

    course = _make_full_course() + "\n\n## 9. Bogus Duplicate\nExtra content."
    report = validate_section_numbering(course)
    assert report.is_valid is False
    assert any(
        "duplicate" in issue.lower() and "9" in issue for issue in report.issues
    ), f"Expected duplicate-9 message, got: {report.issues}"


def test_validate_section_numbering_missing_numbers_fails():
    """Course missing Section 2 and Section 5 (regression on legacy generators)."""
    from src.services.structural_validator import validate_section_numbering

    course = (
        "## 1. Title\nx\n\n"
        "## 3. Objectives\ny\n\n"
        "## 4. Skills\nz\n\n"
        "## 6. Repro\nq\n\n"
        "## 7. Thought\nr\n\n"
        "## 8. Steps\ns\n\n"
        "## 9. Script\nt\n\n"
        "## 10. Conclusion\nu\n\n"
        "## 11. Resources\nv\n"
    )
    report = validate_section_numbering(course)
    assert report.is_valid is False
    joined = " ".join(report.issues).lower()
    assert "missing" in joined
    assert "2" in joined and "5" in joined


def test_validate_section_numbering_no_numbers_fails():
    """The atentie-la-transport smoke-trial failure: LLM dropped all section numbers
    (## Title and Context, ## Abstract, ## Prerequisites, ...) so no `## N.` headings
    are present at all."""
    from src.services.structural_validator import validate_section_numbering

    course = (
        "## Title and Context\nx\n\n"
        "## Abstract (TL;DR)\ny\n\n"
        "## Prerequisites\nz\n\n"
        "## Objectives\nq\n"
    )
    report = validate_section_numbering(course)
    assert report.is_valid is False
    assert any(
        "no numbered" in issue.lower() or "expected" in issue.lower()
        for issue in report.issues
    ), f"Expected 'no numbered headings' message, got: {report.issues}"


def test_validate_section_numbering_out_of_order_fails():
    """Sections present but not in ascending order."""
    from src.services.structural_validator import validate_section_numbering

    course = (
        "## 1. Title\nx\n\n"
        "## 3. Objectives\ny\n\n"
        "## 2. Abstract\nz\n\n"
        "## 4. Skills\nq\n\n"
        "## 5. Defs\nr\n\n"
        "## 6. Repro\ns\n\n"
        "## 7. Thought\nt\n\n"
        "## 8. Steps\nu\n\n"
        "## 9. Script\nv\n\n"
        "## 10. Conclusion\nw\n\n"
        "## 11. Resources\nz\n"
    )
    report = validate_section_numbering(course)
    assert report.is_valid is False
    assert any(
        "out of order" in issue.lower() or "ascending" in issue.lower()
        for issue in report.issues
    ), f"Expected out-of-order message, got: {report.issues}"


def test_validate_section_numbering_empty_course_fails():
    from src.services.structural_validator import validate_section_numbering

    report = validate_section_numbering("")
    assert report.is_valid is False
    assert len(report.issues) > 0


# ── validate_all ─────────────────────────────────────────────────────────────


def test_validate_all_fully_valid():
    from src.services.structural_validator import validate_all

    course = _make_full_course()
    report = validate_all(course, VALID_SOLVER)
    assert report.is_valid is True
    assert len(report.issues) == 0
