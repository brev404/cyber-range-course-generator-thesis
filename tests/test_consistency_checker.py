"""Tests for the consistency checker."""

from src.utils.consistency_checker import (
    ConsistencyReport,
    check_consistency,
    write_consistency_report,
)

# ---------------------------------------------------------------------------
# Fixture course texts
# ---------------------------------------------------------------------------

# Normal course: has Abstract, Objectives, Solution; uses **SQL injection** and `payload`
_COURSE_NORMAL = """\
## Abstract

This challenge is about SQL injection.

## Objectives

Learn SQL injection techniques to bypass authentication.

**SQL injection** is a technique where you inject `payload` into a query.

## Solution

1. Identify the injection point
2. Craft the payload

```bash
sqlmap -u http://target/?id=1
```
"""

# Missing section: no Objectives heading; otherwise identical terms
_COURSE_MISSING_SECTION = """\
## Abstract

A basic web challenge.

**SQL injection** is used with `payload` to exploit the backend.

## Solution

1. Fuzz the inputs
2. Exploit

```bash
curl http://target/?id=1'
```
"""

# Different term: has Objectives, but uses **cross-site scripting** and `script` instead
_COURSE_DIFFERENT_TERM = """\
## Abstract

This challenge is about XSS.

## Objectives

Learn cross-site scripting techniques.

**cross-site scripting** is a vulnerability where `script` tags are injected.

## Solution

1. Inject the payload
2. Steal cookies

```bash
<script>alert(document.cookie)</script>
```
"""


# ---------------------------------------------------------------------------
# Tests: section-order deviation
# ---------------------------------------------------------------------------


def test_section_order_deviation_detected() -> None:
    """Course missing 'objectives' section is flagged as section_order deviation."""
    courses = {
        "crypto/normal": _COURSE_NORMAL,
        "web/missing": _COURSE_MISSING_SECTION,
    }
    report = check_consistency(courses, run_id="test_run")

    assert report.challenge_count == 2
    section_devs = [d for d in report.deviations if d.deviation_type == "section_order"]
    assert len(section_devs) == 1
    assert section_devs[0].challenge_id == "web/missing"
    assert "objectives" in section_devs[0].detail


def test_no_deviation_when_all_sections_match() -> None:
    """Identical section orders produce no section_order deviations."""
    courses = {
        "a/chall": _COURSE_NORMAL,
        "b/chall": _COURSE_NORMAL,
    }
    report = check_consistency(courses, run_id="test_run")
    section_devs = [d for d in report.deviations if d.deviation_type == "section_order"]
    assert section_devs == []


# ---------------------------------------------------------------------------
# Tests: term consistency deviation
# ---------------------------------------------------------------------------


def test_missing_term_detected_in_different_term_course() -> None:
    """Course with different terms is flagged for missing expected terms."""
    # sql injection and payload appear in 2/3 > 0.5 → expected
    # cross-site scripting and script appear in 1/3 < 0.5 → not expected
    # _COURSE_DIFFERENT_TERM is missing "sql injection" and "payload"
    courses = {
        "a/normal": _COURSE_NORMAL,
        "b/missing_section": _COURSE_MISSING_SECTION,
        "c/different": _COURSE_DIFFERENT_TERM,
    }
    report = check_consistency(courses, run_id="test_run", term_threshold=0.5)

    term_devs = [d for d in report.deviations if d.deviation_type == "missing_term"]
    assert len(term_devs) >= 1

    dev_ids = {d.challenge_id for d in term_devs}
    assert "c/different" in dev_ids

    # "sql injection" and "payload" must be in expected_terms
    assert "sql injection" in report.expected_terms
    assert "payload" in report.expected_terms

    # Verify the deviation detail mentions the missing term
    c_dev = next(d for d in term_devs if d.challenge_id == "c/different")
    assert "sql injection" in c_dev.detail or "payload" in c_dev.detail


def test_no_term_deviation_when_terms_match() -> None:
    """Identical courses produce no term deviations."""
    courses = {
        "x/chall": _COURSE_NORMAL,
        "y/chall": _COURSE_NORMAL,
    }
    report = check_consistency(courses, run_id="test_run")
    term_devs = [d for d in report.deviations if d.deviation_type == "missing_term"]
    assert term_devs == []


# ---------------------------------------------------------------------------
# Tests: report file generation
# ---------------------------------------------------------------------------


def test_write_consistency_report_creates_file(tmp_path) -> None:
    """write_consistency_report creates consistency_report.md under run_id dir."""
    courses = {
        "a/normal": _COURSE_NORMAL,
        "b/missing": _COURSE_MISSING_SECTION,
    }
    report = check_consistency(courses, run_id="myrun123")
    report_path = write_consistency_report(report, tmp_path)

    assert report_path.exists()
    assert report_path.name == "consistency_report.md"
    assert report_path.parent.name == "myrun123"


def test_report_file_contains_expected_content(tmp_path) -> None:
    """Report markdown contains challenge IDs, deviation type, and summary."""
    courses = {
        "a/normal": _COURSE_NORMAL,
        "b/missing": _COURSE_MISSING_SECTION,
        "c/different": _COURSE_DIFFERENT_TERM,
    }
    report = check_consistency(courses, run_id="testrun")
    report_path = write_consistency_report(report, tmp_path)

    content = report_path.read_text(encoding="utf-8")
    assert "testrun" in content
    assert "section_order" in content or "missing_term" in content
    assert "a/normal" in content
    assert "Checked 3 course(s)" in content


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------


def test_empty_courses_returns_zero_count_report() -> None:
    """Empty courses dict returns a valid ConsistencyReport with challenge_count=0."""
    report = check_consistency({}, run_id="empty_run")
    assert isinstance(report, ConsistencyReport)
    assert report.challenge_count == 0
    assert report.deviations == []
    assert "No courses to check" in report.summary


def test_single_course_no_deviations() -> None:
    """Single course cannot deviate from itself."""
    report = check_consistency({"solo/chall": _COURSE_NORMAL}, run_id="solo")
    assert report.challenge_count == 1
    assert report.deviations == []


def test_style_markers_extracted_correctly() -> None:
    """Code blocks and numbered steps are correctly detected."""
    report = check_consistency({"a/ch": _COURSE_NORMAL}, run_id="style_test")
    m = report.metrics[0]
    assert m.has_code_blocks is True
    assert m.has_numbered_steps is True
    assert m.line_count > 0
