"""Structural validation for challenge folders (CHALLENGE_STRUCTURE.md).

Validates category/challenge_name/ layout: cyberedu/ (write-up/, optional src/,
deploy/), writeup.md, description.md, solver (solve.py/.sage/.sh),
challenge-flags.txt, public/. Step 0 reproducibility checks (description length,
resource documentation) are performed by the Validation Agent in
src/agents/validation_agent.py.
"""

import json
import sys
from pathlib import Path

from loguru import logger

# Add project root to Python path for imports when run as script
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config.settings import settings  # noqa: E402
from src.models.report_models import (  # noqa: E402
    ChallengeChecklist,
    IssueSeverity,
    ValidationIssue,
    ValidationReport,
)
from src.utils.logging_config import setup_logging  # noqa: E402

ORGANIZED_CHALLENGES_ROOT = settings.PROCESSED_DIR / settings.RAW_CHALLENGES_SOURCE.name


def validate_challenge(challenge_path: Path, category: str) -> ValidationReport:
    """Validate structure and key files for one challenge (CHALLENGE_STRUCTURE.md).

    Checks: cyberedu/, write-up/, description.md, writeup.md, solver,
    challenge-flags.txt, public/. Step 0 checks are done by the Validation Agent.
    """
    challenge_name = challenge_path.name
    challenge_id = f"{category}/{challenge_name}"
    logger.info("Validating: {}", challenge_id)

    checklist = ChallengeChecklist()
    issues = []
    files_found = {
        "cyberedu/src": [],
        "cyberedu/write-up": [],
        "cyberedu/deploy": [],
        "cyberedu/official-docs": [],
        "public": [],
    }

    try:
        # 1. Check for expected subdirectories
        cyberedu_dir = challenge_path / "cyberedu"
        src_dir = cyberedu_dir / "src"
        wu_dir = cyberedu_dir / "write-up"
        deploy_dir = cyberedu_dir / "deploy"
        off_docs_dir = cyberedu_dir / "official-docs"
        pub_dir = challenge_path / "public"

        checklist.has_cyberedu_dir = cyberedu_dir.is_dir()
        checklist.has_src_dir = src_dir.is_dir()
        checklist.has_writeup_dir = wu_dir.is_dir()

        try:
            checklist.has_deploy_dir = deploy_dir.is_dir() and any(deploy_dir.iterdir())
        except (OSError, PermissionError) as e:
            logger.warning("Cannot read deploy_dir for {}: {}", challenge_id, e)
            checklist.has_deploy_dir = False

        try:
            checklist.has_official_docs = off_docs_dir.is_dir() and any(
                off_docs_dir.iterdir()
            )
        except (OSError, PermissionError) as e:
            logger.warning("Cannot read official_docs_dir for {}: {}", challenge_id, e)
            checklist.has_official_docs = False

        checklist.has_public_dir = pub_dir.is_dir()

        # 2. Inventory and check content in cyberedu/write-up/
        if checklist.has_writeup_dir:
            try:
                files_found["cyberedu/write-up"] = [
                    f.name for f in wu_dir.iterdir() if f.is_file()
                ]
                wu_files = [f.lower() for f in files_found["cyberedu/write-up"]]

                checklist.has_writeup_md = "writeup.md" in wu_files
                # Accept .py, .sage, or .sh as valid solvers
                checklist.has_solver = any(
                    f in wu_files for f in ["solve.py", "solve.sage", "solve.sh"]
                )
                checklist.has_description_md = "description.md" in wu_files
                checklist.has_flags_txt = "challenge-flags.txt" in wu_files

                logger.debug(
                    "Checklist for {}: WU_MD={}, SOLVER={}, DESC={}, FLAGS={}",
                    challenge_id,
                    checklist.has_writeup_md,
                    checklist.has_solver,
                    checklist.has_description_md,
                    checklist.has_flags_txt,
                )
            except (OSError, PermissionError) as e:
                logger.error(
                    "Error reading write-up directory for {}: {}", challenge_id, e
                )
                issues.append(
                    ValidationIssue(
                        code="READ_ERROR_WRITEUP",
                        message=f"Cannot read write-up directory: {e}",
                        severity=IssueSeverity.CRITICAL,
                        file_path="cyberedu/write-up",
                        suggestion="Fix permissions or path so write-up directory is readable.",
                    )
                )

        # 3. Inventory other directories with error handling
        if checklist.has_src_dir:
            try:
                files_found["cyberedu/src"] = [
                    f.name for f in src_dir.iterdir() if f.is_file()
                ]
            except (OSError, PermissionError) as e:
                logger.warning("Cannot read src directory for {}: {}", challenge_id, e)

        if checklist.has_deploy_dir:
            try:
                files_found["cyberedu/deploy"] = [
                    f.name for f in deploy_dir.iterdir() if f.is_file()
                ]
            except (OSError, PermissionError) as e:
                logger.warning(
                    "Cannot read deploy directory for {}: {}", challenge_id, e
                )

        if checklist.has_official_docs:
            try:
                files_found["cyberedu/official-docs"] = [
                    f.name for f in off_docs_dir.iterdir() if f.is_file()
                ]
            except (OSError, PermissionError) as e:
                logger.warning(
                    "Cannot read official-docs directory for {}: {}", challenge_id, e
                )

        if checklist.has_public_dir:
            try:
                files_found["public"] = [
                    f.name for f in pub_dir.iterdir() if f.is_file()
                ]
                checklist.has_attachments = len(files_found["public"]) > 0
            except (OSError, PermissionError) as e:
                logger.warning(
                    "Cannot read public directory for {}: {}", challenge_id, e
                )

        # 4. Generate Issues (file_path and suggestion per CHALLENGE_STRUCTURE)
        if not checklist.has_cyberedu_dir:
            issues.append(
                ValidationIssue(
                    code="MISSING_CYBEREDU",
                    message="Missing 'cyberedu' directory",
                    severity=IssueSeverity.CRITICAL,
                    file_path="cyberedu",
                    suggestion="Create cyberedu/ with write-up/, optional src/ and deploy/.",
                )
            )
        if checklist.has_writeup_dir:
            if not checklist.has_writeup_md:
                issues.append(
                    ValidationIssue(
                        code="MISSING_WRITEUP_MD",
                        message="Missing 'writeup.md' in cyberedu/write-up/",
                        severity=IssueSeverity.HIGH,
                        file_path="cyberedu/write-up/writeup.md",
                        suggestion="Add writeup.md with full solution documentation.",
                    )
                )
            if not checklist.has_solver:
                issues.append(
                    ValidationIssue(
                        code="MISSING_SOLVER",
                        message="Missing solver script (solve.py/sage/sh) in cyberedu/write-up/",
                        severity=IssueSeverity.MEDIUM,
                        file_path="cyberedu/write-up/solve.py",
                        suggestion="Add solve.py, solve.sage, or solve.sh as automated solver.",
                    )
                )
            if not checklist.has_description_md:
                issues.append(
                    ValidationIssue(
                        code="MISSING_DESCRIPTION_MD",
                        message="Missing 'description.md' in cyberedu/write-up/",
                        severity=IssueSeverity.HIGH,
                        file_path="cyberedu/write-up/description.md",
                        suggestion="Add description.md with problem statement and metadata.",
                    )
                )
            if not checklist.has_flags_txt:
                issues.append(
                    ValidationIssue(
                        code="MISSING_FLAGS_TXT",
                        message="Missing 'challenge-flags.txt' in cyberedu/write-up/",
                        severity=IssueSeverity.HIGH,
                        file_path="cyberedu/write-up/challenge-flags.txt",
                        suggestion="Add challenge-flags.txt with flags and questions.",
                    )
                )
        else:
            issues.append(
                ValidationIssue(
                    code="MISSING_WRITEUP_DIR",
                    message="Missing 'cyberedu/write-up' directory",
                    severity=IssueSeverity.CRITICAL,
                    file_path="cyberedu/write-up",
                    suggestion="Create cyberedu/write-up/ with writeup.md, description.md, solver, challenge-flags.txt.",
                )
            )
        if not checklist.has_public_dir:
            issues.append(
                ValidationIssue(
                    code="MISSING_PUBLIC_DIR",
                    message="Missing 'public' directory",
                    severity=IssueSeverity.HIGH,
                    file_path="public",
                    suggestion="Create public/ for files distributed to participants.",
                )
            )
        if not checklist.has_official_docs:
            issues.append(
                ValidationIssue(
                    code="MISSING_OFFICIAL_DOCS",
                    message="No official PDF documentation mapped for this challenge",
                    severity=IssueSeverity.LOW,
                    file_path="cyberedu/official-docs",
                    suggestion="Optional: map official PDF docs to cyberedu/official-docs/.",
                )
            )

        # Calculate structure score (7 core items)
        core_items = [
            checklist.has_cyberedu_dir,
            checklist.has_writeup_dir,
            checklist.has_writeup_md,
            checklist.has_solver,
            checklist.has_description_md,
            checklist.has_flags_txt,
            checklist.has_public_dir,
        ]
        structure_score = sum(core_items) / len(core_items)

    except (OSError, PermissionError) as e:
        logger.error("Error validating challenge {}: {}", challenge_id, e)
        issues.append(
            ValidationIssue(
                code="VALIDATION_ERROR",
                message=f"Validation error: {e}",
                severity=IssueSeverity.CRITICAL,
                suggestion="Check path and permissions; retry validation.",
            )
        )
        return ValidationReport(
            challenge_id=challenge_id,
            is_valid=False,
            checklist=checklist,
            issues=issues,
            structure_score=0.0,
            has_writeup=False,
            has_solve_script=False,
            files_found=files_found,
        )

    return ValidationReport(
        challenge_id=challenge_id,
        is_valid=checklist.is_complete(),
        checklist=checklist,
        issues=issues,
        structure_score=structure_score,
        has_writeup=checklist.has_writeup_md,
        has_solve_script=checklist.has_solver,
        files_found=files_found,
    )


def validate_all_challenges():
    """
    Validates all challenge folders within the organized root directory.
    """
    if not ORGANIZED_CHALLENGES_ROOT.exists():
        logger.error(
            "Organized challenges root does not exist: {}", ORGANIZED_CHALLENGES_ROOT
        )
        return

    logger.info(
        "Starting validation of organized challenges in: {}",
        ORGANIZED_CHALLENGES_ROOT,
    )

    all_reports = []
    for category_dir in ORGANIZED_CHALLENGES_ROOT.iterdir():
        if not category_dir.is_dir():
            continue

        for challenge_dir in category_dir.iterdir():
            if not challenge_dir.is_dir():
                continue

            report = validate_challenge(challenge_dir, category_dir.name)
            all_reports.append(report)

            status = "PASS" if report.is_valid else "FAIL"
            logger.info(
                "[{}] {} - Score: {:.2f}",
                status,
                report.challenge_id,
                report.structure_score,
            )
            if not report.is_valid:
                for issue in report.issues:
                    logger.warning("  [{}] - {}", report.challenge_id, issue.message)

    output_dir = settings.OUTPUT_DIR / "validation_reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "validation_summary.json"
    try:
        summary_path.write_text(
            json.dumps([r.model_dump() for r in all_reports], indent=2),
            encoding="utf-8",
        )
        logger.info("Validation summary saved to {}", summary_path)
    except OSError as e:
        logger.error("Failed to save validation summary: {}", e)


def run_validator():
    """Main function to run the challenge validation."""
    validate_all_challenges()


if __name__ == "__main__":
    setup_logging("validate_challenges")
    run_validator()
