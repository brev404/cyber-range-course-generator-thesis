"""Tests for output verification and reproducibility.

Verifies that verify_outputs finds expected files and JSON shape, and reports
errors when validation summary is missing or malformed. Also tests
reproducibility checks on generated courses.
"""

import pytest

from src.utils.reproducibility_checker import check_course_reproducibility
from src.utils.verify_outputs import (
    VALIDATION_REPORT_KEYS,
    WRITEUP_REL_PATH,
    verify_course_reproducibility,
    verify_outputs,
    verify_validation_summary,
    verify_writeup_paths,
)


@pytest.fixture
def output_dir_with_valid_summary(tmp_path):
    """Create output_dir/validation_reports/validation_summary.json with valid shape."""
    reports_dir = tmp_path / "validation_reports"
    reports_dir.mkdir(parents=True)
    summary_path = reports_dir / "validation_summary.json"
    data = [
        {
            "challenge_id": "crypto/test_crypto_01",
            "is_valid": True,
            "issues": [],
            "structure_score": 0.95,
        },
        {
            "challenge_id": "web/test_web_01",
            "is_valid": False,
            "issues": [
                {"code": "MISSING_FILE", "message": "Missing x", "severity": "high"}
            ],
            "structure_score": 0.7,
        },
    ]
    summary_path.write_text(__import__("json").dumps(data), encoding="utf-8")
    return tmp_path


@pytest.fixture
def output_dir_empty(tmp_path):
    """Empty output dir (no validation_reports)."""
    return tmp_path


def test_verify_validation_summary_valid(output_dir_with_valid_summary):
    """Valid validation_summary.json passes."""
    path = (
        output_dir_with_valid_summary / "validation_reports" / "validation_summary.json"
    )
    errors = verify_validation_summary(path)
    assert errors == []


def test_verify_validation_summary_missing(tmp_path):
    """Missing file yields error."""
    path = tmp_path / "validation_summary.json"
    errors = verify_validation_summary(path)
    assert len(errors) == 1
    assert "Missing" in errors[0]


def test_verify_validation_summary_invalid_json(tmp_path):
    """Invalid JSON yields error."""
    path = tmp_path / "validation_summary.json"
    path.write_text("not json", encoding="utf-8")
    errors = verify_validation_summary(path)
    assert len(errors) == 1
    assert "JSON" in errors[0] or "json" in errors[0].lower()


def test_verify_validation_summary_missing_keys(tmp_path):
    """Item missing required keys yields error."""
    path = tmp_path / "validation_summary.json"
    path.write_text(
        '[{"challenge_id": "x"}]', encoding="utf-8"
    )  # missing is_valid, issues, structure_score
    errors = verify_validation_summary(path)
    assert len(errors) == 1
    assert "missing keys" in errors[0].lower()


def test_verify_writeup_paths_fixture(processed_fixture):
    """Processed fixture has writeups; none missing."""
    missing, errs = verify_writeup_paths(processed_fixture, "raw_challenges")
    assert errs == []
    assert missing == []


def test_verify_writeup_paths_missing(tmp_path):
    """Archive with category/challenge but no writeup reports missing."""
    archive = tmp_path / "raw_challenges" / "web" / "no_writeup_challenge"
    (archive / "cyberedu" / "write-up").mkdir(parents=True)
    # no writeup.md
    missing, errs = verify_writeup_paths(tmp_path)
    assert errs == []
    assert len(missing) == 1
    assert "writeup.md" in str(missing[0])


def test_verify_outputs_passes_with_fixtures(
    processed_fixture, output_dir_with_valid_summary
):
    """Full verify_outputs passes when validation summary and processed dir with writeups exist."""
    errors = verify_outputs(
        output_dir=output_dir_with_valid_summary,
        processed_dir=processed_fixture,
        check_writeups=True,
        require_writeups=False,
    )
    assert errors == []


def test_verify_outputs_fails_when_summary_missing(processed_fixture, output_dir_empty):
    """verify_outputs fails when validation_summary.json is missing."""
    errors = verify_outputs(
        output_dir=output_dir_empty,
        processed_dir=processed_fixture,
        check_writeups=False,
    )
    assert len(errors) >= 1
    assert any("validation" in e.lower() or "Missing" in e for e in errors)


def test_validation_report_keys_defined():
    """Required keys for ValidationReport are as expected."""
    assert "challenge_id" in VALIDATION_REPORT_KEYS
    assert "is_valid" in VALIDATION_REPORT_KEYS
    assert "issues" in VALIDATION_REPORT_KEYS
    assert "structure_score" in VALIDATION_REPORT_KEYS
    assert WRITEUP_REL_PATH.name == "writeup.md"


# --- Reproducibility checker tests ---

COURSE_WITH_FULL_STEP0 = """
## Title
**Challenge Name:** test-chall
**Category:** Crypto
**Difficulty:** Medium

### Reproducibility (Step 0)
To follow along with this course, you will have access to:
* **Challenge description:** The challenge text.
* **Public files:** chall.py, secret.txt
* **Deployment:** A remote server (nc host port).

**You do NOT have access to:** The server source code or the flag.
"""


def test_reproducibility_checker_passes():
    """Course with full Step 0 (what student has, resources, metadata) passes."""
    errors = check_course_reproducibility(COURSE_WITH_FULL_STEP0)
    assert errors == []


def test_reproducibility_checker_missing_step0():
    """Course without Reproducibility/Step 0 heading yields error."""
    course = "## Title\n\nSome content. No Step 0."
    errors = check_course_reproducibility(course)
    assert len(errors) >= 1
    assert any("Reproducibility" in e or "Step 0" in e for e in errors)


def test_reproducibility_checker_missing_what_student_has():
    """Step 0 present but no 'you will have access to' / 'public files' etc. yields error."""
    course = """
## Title
**Challenge Name:** x
**Category:** Web
**Difficulty:** Easy

### Reproducibility (Step 0)
This section exists but does not state what the learner receives.
Nothing about availability or access.
"""
    errors = check_course_reproducibility(course)
    assert len(errors) >= 1
    assert any("what the student has" in e.lower() for e in errors)


def test_reproducibility_checker_missing_resources():
    """Step 0 present but no resource mention yields error."""
    course = """
## Title
**Challenge Name:** x
**Category:** Crypto
**Difficulty:** Hard

### Reproducibility (Step 0)
You will have access to the challenge. That is all.
"""
    errors = check_course_reproducibility(course)
    # "you will have access to" satisfies what-student-has; but we need resource mention
    # "challenge" might match "description" in _RESOURCE_PATTERNS? No - "description" is the pattern.
    # "challenge" is not in _RESOURCE_PATTERNS. So we should get "missing documented resources"
    assert len(errors) >= 1
    assert any("resource" in e.lower() for e in errors)


def test_reproducibility_checker_empty_course():
    """Empty course yields error."""
    errors = check_course_reproducibility("")
    assert len(errors) == 1
    assert "Empty" in errors[0]


def test_verify_outputs_includes_reproducibility(
    tmp_path, output_dir_with_valid_summary
):
    """When check_generated_courses=True, reproducibility errors are included."""
    # Create archive with a course that fails reproducibility (no Step 0)
    archive = tmp_path / "raw_challenges" / "web" / "bad_course"
    writeup_dir = archive / "cyberedu" / "write-up"
    writeup_dir.mkdir(parents=True)
    course_path = writeup_dir / "course.md"
    course_path.write_text("## Title\n\nNo Step 0 section.", encoding="utf-8")

    errors = verify_outputs(
        output_dir=output_dir_with_valid_summary,
        processed_dir=tmp_path,
        check_writeups=False,
        check_generated_courses=True,
    )
    assert any("Reproducibility" in e or "Step 0" in e for e in errors)


def test_verify_course_reproducibility_passes_with_valid_course(tmp_path):
    """verify_course_reproducibility passes when course has full Step 0."""
    archive = tmp_path / "raw_challenges" / "crypto" / "good_chall"
    writeup_dir = archive / "cyberedu" / "write-up"
    writeup_dir.mkdir(parents=True)
    (writeup_dir / "course.md").write_text(COURSE_WITH_FULL_STEP0, encoding="utf-8")

    errors = verify_course_reproducibility(tmp_path)
    assert errors == []
