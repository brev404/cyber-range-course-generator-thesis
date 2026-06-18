"""Lightweight output verification for pipeline and graph outputs.

Checks presence of expected files (e.g. validation_summary.json, per-challenge
writeup paths) and basic JSON shape so CI or a post-run step can catch obvious
breakage. When --check-courses is used, also runs reproducibility checks on
generated courses. Full quality checks remain in the evaluation
service or manual review.

Usage:
    python -m src.utils.verify_outputs [--output-dir PATH] [--processed-dir PATH]
    python -m src.utils.verify_outputs --check-courses  # include reproducibility checks
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

from loguru import logger

from src.config.settings import settings
from src.utils.reproducibility_checker import check_course_reproducibility

# Required keys per ValidationReport (see src/models/report_models.py)
VALIDATION_REPORT_KEYS = frozenset(
    {"challenge_id", "is_valid", "issues", "structure_score"}
)

# Optional: keys that may appear in RankingReport if ranking summary is written to disk
RANKING_REPORT_KEYS = frozenset(
    {
        "challenge_id",
        "overall_score",
        "pedagogical_review",
        "technical_review",
        "technical_rank",
    }
)

ARCHIVE_SUBDIR = settings.RAW_CHALLENGES_SOURCE.name
VALIDATION_SUMMARY_FILENAME = "validation_summary.json"
WRITEUP_REL_PATH = (
    Path("cyberedu") / "write-up" / "writeup.md"
)  # author's writeup (structure)
COURSE_REL_PATH = (
    Path("cyberedu") / "write-up" / "course.md"
)  # generated course (graph output)


def verify_validation_summary(path: Path) -> List[str]:
    """Verify validation_summary.json exists and each item has expected keys.

    Args:
        path: Path to validation_summary.json.

    Returns:
        List of error messages; empty if all checks pass.
    """
    errors: List[str] = []
    if not path.exists():
        errors.append(f"Missing: {path}")
        return errors
    if not path.is_file():
        errors.append(f"Not a file: {path}")
        return errors
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except (OSError, json.JSONDecodeError) as e:
        errors.append(f"Invalid or unreadable JSON at {path}: {e}")
        return errors
    if not isinstance(data, list):
        errors.append(f"Expected JSON array at {path}, got {type(data).__name__}")
        return errors
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            errors.append(f"Item {i} is not an object: {type(item).__name__}")
            continue
        missing = VALIDATION_REPORT_KEYS - set(item.keys())
        if missing:
            errors.append(
                f"Item {i} (challenge_id={item.get('challenge_id', '?')}) missing keys: {sorted(missing)}"
            )
    return errors


def _verify_challenge_file_paths(
    processed_root: Path,
    rel_path: Path,
    archive_subdir: str,
    file_label: str,
) -> Tuple[List[Path], List[str]]:
    """Check per-challenge file paths under processed_root/archive_subdir."""
    archive = processed_root / archive_subdir
    missing: List[Path] = []
    errors: List[str] = []
    if not archive.exists() or not archive.is_dir():
        errors.append(f"Processed archive dir not found: {archive}")
        return missing, errors
    for category_dir in sorted(archive.iterdir()):
        if not category_dir.is_dir():
            continue
        for challenge_dir in sorted(category_dir.iterdir()):
            if not challenge_dir.is_dir():
                continue
            file_path = challenge_dir / rel_path
            if not file_path.exists() or not file_path.is_file():
                missing.append(file_path)
            elif file_path.stat().st_size == 0:
                errors.append(f"Empty {file_label}: {file_path}")
    return missing, errors


def verify_writeup_paths(
    processed_root: Path, archive_subdir: str = ARCHIVE_SUBDIR
) -> Tuple[List[Path], List[str]]:
    """Check per-challenge author writeup (writeup.md) paths."""
    return _verify_challenge_file_paths(
        processed_root, WRITEUP_REL_PATH, archive_subdir, "writeup"
    )


def verify_course_paths(
    processed_root: Path, archive_subdir: str = ARCHIVE_SUBDIR
) -> Tuple[List[Path], List[str]]:
    """Check per-challenge generated course (course.md) paths."""
    return _verify_challenge_file_paths(
        processed_root, COURSE_REL_PATH, archive_subdir, "course"
    )


def verify_course_reproducibility(
    processed_root: Path, archive_subdir: str = ARCHIVE_SUBDIR
) -> List[str]:
    """Run reproducibility checks on existing course.md files.

    Args:
        processed_root: Root of processed data (e.g. data/processed).
        archive_subdir: Archive subdirectory (e.g. raw_challenges).

    Returns:
        List of reproducibility error messages; empty if all pass.
    """
    errors: List[str] = []
    archive = processed_root / archive_subdir
    if not archive.exists() or not archive.is_dir():
        return errors
    for category_dir in sorted(archive.iterdir()):
        if not category_dir.is_dir():
            continue
        for challenge_dir in sorted(category_dir.iterdir()):
            if not challenge_dir.is_dir():
                continue
            course_path = challenge_dir / COURSE_REL_PATH
            if not course_path.exists() or not course_path.is_file():
                continue
            challenge_id = f"{category_dir.name}/{challenge_dir.name}"
            try:
                content = course_path.read_text(encoding="utf-8", errors="replace")
            except OSError as e:
                errors.append(f"[{challenge_id}] Could not read course: {e}")
                continue
            errors.extend(check_course_reproducibility(content, challenge_id))
    return errors


def verify_outputs(
    output_dir: Optional[Path] = None,
    processed_dir: Optional[Path] = None,
    check_writeups: bool = True,
    require_writeups: bool = False,
    check_generated_courses: bool = False,
) -> List[str]:
    """Run lightweight verification on output and processed directories.

    Args:
        output_dir: Directory containing validation_reports/ (default: settings.OUTPUT_DIR).
        processed_dir: Directory containing the challenge set (default: settings.PROCESSED_DIR).
        check_writeups: If True, check presence of author writeup (writeup.md) under each challenge.
        require_writeups: If True, report missing writeups as errors; else as warnings only.
        check_generated_courses: If True, check presence of generated course (course.md) under each challenge.

    Returns:
        List of error messages; empty if all checks pass.
    """
    if output_dir is None or processed_dir is None:
        from src.config.settings import settings

        output_dir = output_dir or settings.OUTPUT_DIR
        processed_dir = processed_dir or settings.PROCESSED_DIR

    errors: List[str] = []
    # Validation summary
    summary_path = output_dir / "validation_reports" / VALIDATION_SUMMARY_FILENAME
    errors.extend(verify_validation_summary(summary_path))

    # Per-challenge author writeups (optional)
    if check_writeups:
        missing_writeups, writeup_errors = verify_writeup_paths(processed_dir)
        errors.extend(writeup_errors)
        if require_writeups and missing_writeups:
            for p in missing_writeups[:10]:  # cap to avoid huge output
                errors.append(f"Missing writeup: {p}")
            if len(missing_writeups) > 10:
                errors.append(
                    f"... and {len(missing_writeups) - 10} more missing writeups"
                )
        elif missing_writeups and not require_writeups:
            logger.debug(
                "Missing writeups (not required): {} (count={})",
                [str(p) for p in missing_writeups[:5]],
                len(missing_writeups),
            )

    # Per-challenge generated courses (optional; use after graph run)
    if check_generated_courses:
        missing_courses, course_errors = verify_course_paths(processed_dir)
        errors.extend(course_errors)
        for p in missing_courses[:10]:
            errors.append(f"Missing course: {p}")
        if len(missing_courses) > 10:
            errors.append(f"... and {len(missing_courses) - 10} more missing courses")
        # Reproducibility checks on existing courses
        errors.extend(verify_course_reproducibility(processed_dir))

    return errors


def main() -> int:
    """CLI entry point. Returns 0 if verification passes, 1 otherwise."""
    parser = argparse.ArgumentParser(
        description="Verify pipeline/graph outputs: validation summary JSON and optional writeup paths."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: settings.OUTPUT_DIR).",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=None,
        help="Processed directory (default: settings.PROCESSED_DIR).",
    )
    parser.add_argument(
        "--no-writeups",
        action="store_true",
        help="Skip checking per-challenge writeup paths.",
    )
    parser.add_argument(
        "--require-writeups",
        action="store_true",
        help="Treat missing writeups as errors (default: only report validation summary).",
    )
    parser.add_argument(
        "--check-courses",
        action="store_true",
        help="Check presence and reproducibility of generated courses (course.md); includes Step 0 checks.",
    )
    args = parser.parse_args()

    errors = verify_outputs(
        output_dir=args.output_dir,
        processed_dir=args.processed_dir,
        check_writeups=not args.no_writeups,
        require_writeups=args.require_writeups,
        check_generated_courses=args.check_courses,
    )
    if errors:
        for err in errors:
            logger.error("{}", err)
        return 1
    logger.info("Output verification passed.")
    return 0


if __name__ == "__main__":
    from src.utils.logging_config import setup_logging

    setup_logging("verify_outputs")
    raise SystemExit(main())
