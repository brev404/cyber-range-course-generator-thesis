"""Tests for repair_challenge_structure.

All tests use tmp_path fixtures — no real challenge data is read or written.
"""

from pathlib import Path
from typing import List

from src.models.report_models import (
    ChallengeChecklist,
    IssueSeverity,
    ValidationIssue,
    ValidationReport,
)
from src.pipeline.repair_challenge_structure import (
    REPAIRABLE_CODES,
    RepairAction,
    has_repairable_issues,
    repair_challenge,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_challenge(tmp_path: Path, name: str = "my-challenge") -> Path:
    """Return an empty challenge root directory."""
    p = tmp_path / "crypto" / name
    p.mkdir(parents=True)
    return p


def _make_full_challenge(tmp_path: Path, name: str = "full-challenge") -> Path:
    """Return a fully-structured challenge directory."""
    root = _make_challenge(tmp_path, name)
    wu = root / "cyberedu" / "write-up"
    wu.mkdir(parents=True)
    (root / "cyberedu").mkdir(exist_ok=True)
    (root / "public").mkdir()
    (wu / "writeup.md").write_text("# Writeup\n")
    (wu / "description.md").write_text("Description content.\n")
    (wu / "solve.py").write_text("print('solve')\n")
    (wu / "challenge-flags.txt").write_text("CTF{flag}\n")
    return root


def _report(issues: List[ValidationIssue]) -> ValidationReport:
    return ValidationReport(
        challenge_id="test/challenge",
        is_valid=False,
        checklist=ChallengeChecklist(),
        issues=issues,
        structure_score=0.0,
        has_writeup=False,
        has_solve_script=False,
        files_found={},
    )


def _issue(code: str, severity: IssueSeverity = IssueSeverity.HIGH) -> ValidationIssue:
    return ValidationIssue(code=code, message=code, severity=severity)


# ---------------------------------------------------------------------------
# RepairAction model
# ---------------------------------------------------------------------------


def test_repair_action_model() -> None:
    a = RepairAction(
        action_type="create_dir", target="cyberedu", reason="MISSING_CYBEREDU"
    )
    assert a.success is False
    assert a.source is None


def test_repair_action_success() -> None:
    a = RepairAction(
        action_type="rename",
        source="cyberedu/write-up/write-up.md",
        target="cyberedu/write-up/writeup.md",
        reason="MISSING_WRITEUP_MD",
        success=True,
        message="Renamed",
    )
    assert a.success is True


# ---------------------------------------------------------------------------
# has_repairable_issues
# ---------------------------------------------------------------------------


def test_has_repairable_issues_true() -> None:
    report = _report([_issue("MISSING_CYBEREDU")])
    assert has_repairable_issues([report]) is True


def test_has_repairable_issues_false_for_non_repairable() -> None:
    report = _report([_issue("STEP0_DESCRIPTION_TOO_SHORT", IssueSeverity.MEDIUM)])
    assert has_repairable_issues([report]) is False


def test_has_repairable_issues_empty() -> None:
    assert has_repairable_issues([]) is False


def test_has_repairable_issues_mixed() -> None:
    report = _report(
        [
            _issue("STEP0_RESOURCES_NOT_DOCUMENTED", IssueSeverity.LOW),
            _issue("MISSING_PUBLIC_DIR"),
        ]
    )
    assert has_repairable_issues([report]) is True


# ---------------------------------------------------------------------------
# repair_challenge — directory creation
# ---------------------------------------------------------------------------


def test_repair_creates_cyberedu_dir(tmp_path: Path) -> None:
    root = _make_challenge(tmp_path)
    assert not (root / "cyberedu").exists()

    actions = repair_challenge(root)

    assert (root / "cyberedu").is_dir()
    create_actions = [
        a for a in actions if a.action_type == "create_dir" and a.target == "cyberedu"
    ]
    assert len(create_actions) == 1
    assert create_actions[0].success is True


def test_repair_creates_writeup_dir(tmp_path: Path) -> None:
    root = _make_challenge(tmp_path)
    (root / "cyberedu").mkdir()

    actions = repair_challenge(root)

    assert (root / "cyberedu" / "write-up").is_dir()
    wu_actions = [a for a in actions if "write-up" in a.target]
    assert any(a.success for a in wu_actions)


def test_repair_creates_public_dir(tmp_path: Path) -> None:
    root = _make_challenge(tmp_path)
    (root / "cyberedu").mkdir()
    (root / "cyberedu" / "write-up").mkdir(parents=True)

    actions = repair_challenge(root)

    assert (root / "public").is_dir()
    pub_actions = [a for a in actions if a.target == "public"]
    assert len(pub_actions) == 1
    assert pub_actions[0].success is True


def test_repair_creates_all_missing_dirs(tmp_path: Path) -> None:
    root = _make_challenge(tmp_path)

    actions = repair_challenge(root)

    assert (root / "cyberedu").is_dir()
    assert (root / "cyberedu" / "write-up").is_dir()
    assert (root / "public").is_dir()
    assert all(a.success for a in actions if a.action_type == "create_dir")


# ---------------------------------------------------------------------------
# repair_challenge — no-op when fully structured
# ---------------------------------------------------------------------------


def test_repair_noop_on_full_challenge(tmp_path: Path) -> None:
    root = _make_full_challenge(tmp_path)
    actions = repair_challenge(root)
    assert actions == []


# ---------------------------------------------------------------------------
# repair_challenge — writeup.md rename variants
# ---------------------------------------------------------------------------


def test_repair_renames_hyphenated_writeup(tmp_path: Path) -> None:
    root = _make_challenge(tmp_path)
    wu = root / "cyberedu" / "write-up"
    wu.mkdir(parents=True)
    (root / "public").mkdir()
    (wu / "write-up.md").write_text("# Write-up\n")

    actions = repair_challenge(root)

    assert (wu / "writeup.md").exists()
    assert not (wu / "write-up.md").exists()
    rename_actions = [
        a for a in actions if a.action_type == "rename" and "writeup.md" in a.target
    ]
    assert len(rename_actions) == 1
    assert rename_actions[0].success is True


def test_repair_renames_write_underscore_up(tmp_path: Path) -> None:
    root = _make_challenge(tmp_path)
    wu = root / "cyberedu" / "write-up"
    wu.mkdir(parents=True)
    (root / "public").mkdir()
    (wu / "write_up.md").write_text("# Write up\n")

    repair_challenge(root)

    assert (wu / "writeup.md").exists()


def test_repair_does_not_overwrite_existing_writeup(tmp_path: Path) -> None:
    root = _make_challenge(tmp_path)
    wu = root / "cyberedu" / "write-up"
    wu.mkdir(parents=True)
    (root / "public").mkdir()
    (wu / "writeup.md").write_text("# Existing\n")
    (wu / "write-up.md").write_text("# Variant\n")

    repair_challenge(root)

    assert (wu / "writeup.md").read_text() == "# Existing\n"
    # write-up.md stays untouched (target exists)
    assert (wu / "write-up.md").exists()


# ---------------------------------------------------------------------------
# repair_challenge — move writeup.md from wrong location
# ---------------------------------------------------------------------------


def test_repair_moves_writeup_from_wrong_location(tmp_path: Path) -> None:
    root = _make_challenge(tmp_path)
    cyberedu = root / "cyberedu"
    wu = cyberedu / "write-up"
    wu.mkdir(parents=True)
    (root / "public").mkdir()
    # writeup.md in cyberedu/ instead of cyberedu/write-up/
    (cyberedu / "writeup.md").write_text("# Misplaced\n")

    actions = repair_challenge(root)

    assert (wu / "writeup.md").exists()
    assert not (cyberedu / "writeup.md").exists()
    rename_actions = [
        a for a in actions if a.action_type == "rename" and "writeup.md" in a.target
    ]
    assert any(a.success for a in rename_actions)


# ---------------------------------------------------------------------------
# repair_challenge — description.md variants
# ---------------------------------------------------------------------------


def test_repair_renames_desc_md(tmp_path: Path) -> None:
    root = _make_challenge(tmp_path)
    wu = root / "cyberedu" / "write-up"
    wu.mkdir(parents=True)
    (root / "public").mkdir()
    (wu / "writeup.md").write_text("# Writeup\n")
    (wu / "desc.md").write_text("Description.\n")

    repair_challenge(root)

    assert (wu / "description.md").exists()
    assert not (wu / "desc.md").exists()


# ---------------------------------------------------------------------------
# repair_challenge — flags.txt → challenge-flags.txt
# ---------------------------------------------------------------------------


def test_repair_renames_flags_txt(tmp_path: Path) -> None:
    root = _make_challenge(tmp_path)
    wu = root / "cyberedu" / "write-up"
    wu.mkdir(parents=True)
    (root / "public").mkdir()
    (wu / "writeup.md").write_text("# Writeup\n")
    (wu / "flags.txt").write_text("CTF{flag}\n")

    repair_challenge(root)

    assert (wu / "challenge-flags.txt").exists()
    assert not (wu / "flags.txt").exists()


# ---------------------------------------------------------------------------
# repair_challenge — solver move
# ---------------------------------------------------------------------------


def test_repair_moves_solver_from_cyberedu(tmp_path: Path) -> None:
    root = _make_challenge(tmp_path)
    cyberedu = root / "cyberedu"
    wu = cyberedu / "write-up"
    wu.mkdir(parents=True)
    (root / "public").mkdir()
    (wu / "writeup.md").write_text("# Writeup\n")
    # solve.py in cyberedu/ not in write-up/
    (cyberedu / "solve.py").write_text("print('solve')\n")

    repair_challenge(root)

    assert (wu / "solve.py").exists()
    assert not (cyberedu / "solve.py").exists()


# ---------------------------------------------------------------------------
# repair_challenge — idempotency
# ---------------------------------------------------------------------------


def test_repair_is_idempotent(tmp_path: Path) -> None:
    root = _make_challenge(tmp_path)

    actions1 = repair_challenge(root)
    actions2 = repair_challenge(root)

    # Second run: no new repairs (dirs exist, files in place)
    assert actions2 == []
    # First run succeeded
    assert any(a.success for a in actions1)


# ---------------------------------------------------------------------------
# repair_challenge — skip if target already exists
# ---------------------------------------------------------------------------


def test_repair_skips_when_target_exists(tmp_path: Path) -> None:
    root = _make_challenge(tmp_path)
    wu = root / "cyberedu" / "write-up"
    wu.mkdir(parents=True)
    (root / "public").mkdir()
    # Both variants exist
    (wu / "writeup.md").write_text("# Existing\n")
    (wu / "write-up.md").write_text("# Variant\n")

    actions = repair_challenge(root)

    # No rename should have succeeded (target already existed)
    rename_actions = [a for a in actions if a.action_type == "rename"]
    assert all(not a.success for a in rename_actions)
    # Original files unchanged
    assert (wu / "writeup.md").read_text() == "# Existing\n"


# ---------------------------------------------------------------------------
# REPAIRABLE_CODES constant
# ---------------------------------------------------------------------------


def test_repairable_codes_contains_expected() -> None:
    expected = {
        "MISSING_CYBEREDU",
        "MISSING_WRITEUP_DIR",
        "MISSING_WRITEUP_MD",
        "MISSING_DESCRIPTION_MD",
        "MISSING_SOLVER",
        "MISSING_FLAGS_TXT",
        "MISSING_PUBLIC_DIR",
    }
    assert expected <= REPAIRABLE_CODES
