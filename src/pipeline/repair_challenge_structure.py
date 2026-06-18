"""Structural repair for challenge folders.

Attempts file-level repairs: rename/move files to expected locations,
create missing directories. No content changes. Safe to re-run; skips
if target already exists.

Repairable issues:
- MISSING_CYBEREDU        → create cyberedu/
- MISSING_WRITEUP_DIR     → create cyberedu/write-up/
- MISSING_PUBLIC_DIR      → create public/
- MISSING_WRITEUP_MD      → rename write-up.md / write_up.md → writeup.md,
                             or move from elsewhere under challenge_path
- MISSING_DESCRIPTION_MD  → rename desc.md / move from elsewhere
- MISSING_SOLVER          → move solve.py / solve.sage / solve.sh from elsewhere
- MISSING_FLAGS_TXT       → rename flags.txt / challenge_flags.txt → challenge-flags.txt

Non-repairable (silently skipped): READ_ERROR_*, VALIDATION_ERROR, STEP0_*.
"""

from pathlib import Path
from typing import List, Optional

from loguru import logger
from pydantic import BaseModel

# Issue codes this module can repair
REPAIRABLE_CODES = frozenset(
    {
        "MISSING_CYBEREDU",
        "MISSING_WRITEUP_DIR",
        "MISSING_WRITEUP_MD",
        "MISSING_DESCRIPTION_MD",
        "MISSING_SOLVER",
        "MISSING_FLAGS_TXT",
        "MISSING_PUBLIC_DIR",
    }
)

# Variant filenames → canonical name for files that belong in cyberedu/write-up/
_WRITEUP_VARIANTS: dict[str, str] = {
    "write-up.md": "writeup.md",
    "write_up.md": "writeup.md",
    "writeup.md": "writeup.md",
}
_DESCRIPTION_VARIANTS: dict[str, str] = {
    "description.md": "description.md",
    "desc.md": "description.md",
}
_FLAGS_VARIANTS: dict[str, str] = {
    "challenge-flags.txt": "challenge-flags.txt",
    "challenge_flags.txt": "challenge-flags.txt",
    "flags.txt": "challenge-flags.txt",
}
_SOLVER_NAMES: frozenset[str] = frozenset({"solve.py", "solve.sage", "solve.sh"})


class RepairAction(BaseModel):
    """Record of one repair attempt for a challenge."""

    action_type: str
    """Type of repair: 'create_dir' or 'rename'."""

    source: Optional[str] = None
    """Relative path of source file (None for create_dir)."""

    target: str
    """Relative path of target file or directory."""

    reason: str
    """Validation issue code or short description motivating this repair."""

    success: bool = False
    """True if the repair completed without error."""

    message: str = ""
    """Human-readable outcome (what happened or why it was skipped)."""


def has_repairable_issues(validation_reports: list) -> bool:
    """Return True if any report in the list contains a repairable issue code."""
    for report in validation_reports:
        for issue in getattr(report, "issues", []):
            if getattr(issue, "code", None) in REPAIRABLE_CODES:
                return True
    return False


def repair_challenge(challenge_path: Path) -> List[RepairAction]:
    """Inspect and repair one challenge directory. No content changes.

    Moves/renames files to expected locations and creates missing directories.
    Skips any action whose target already exists (logs a warning).

    Args:
        challenge_path: Root directory of the challenge
            (e.g. data/processed/raw_challenges/crypto/my-challenge/).

    Returns:
        List of RepairAction describing every attempted repair.
    """
    challenge_name = challenge_path.name
    actions: List[RepairAction] = []

    cyberedu = challenge_path / "cyberedu"
    writeup_dir = cyberedu / "write-up"
    pub_dir = challenge_path / "public"

    # 1. Create missing directories
    _ensure_dir(cyberedu, "cyberedu", "MISSING_CYBEREDU", challenge_name, actions)
    _ensure_dir(
        writeup_dir, "cyberedu/write-up", "MISSING_WRITEUP_DIR", challenge_name, actions
    )
    _ensure_dir(pub_dir, "public", "MISSING_PUBLIC_DIR", challenge_name, actions)

    # 2. Repair file naming / location issues — only if write-up dir exists now
    if writeup_dir.is_dir():
        _repair_file(
            writeup_dir=writeup_dir,
            challenge_path=challenge_path,
            canonical_name="writeup.md",
            name_variants=_WRITEUP_VARIANTS,
            issue_code="MISSING_WRITEUP_MD",
            challenge_name=challenge_name,
            actions=actions,
        )
        _repair_file(
            writeup_dir=writeup_dir,
            challenge_path=challenge_path,
            canonical_name="description.md",
            name_variants=_DESCRIPTION_VARIANTS,
            issue_code="MISSING_DESCRIPTION_MD",
            challenge_name=challenge_name,
            actions=actions,
        )
        _repair_solver(writeup_dir, challenge_path, challenge_name, actions)
        _repair_file(
            writeup_dir=writeup_dir,
            challenge_path=challenge_path,
            canonical_name="challenge-flags.txt",
            name_variants=_FLAGS_VARIANTS,
            issue_code="MISSING_FLAGS_TXT",
            challenge_name=challenge_name,
            actions=actions,
        )

    succeeded = sum(1 for a in actions if a.success)
    if actions:
        logger.info(
            "Repair [{}]: {} action(s) attempted, {} succeeded",
            challenge_name,
            len(actions),
            succeeded,
        )
    else:
        logger.debug("Repair [{}]: nothing to repair", challenge_name)

    return actions


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ensure_dir(
    directory: Path,
    rel_path: str,
    issue_code: str,
    challenge_name: str,
    actions: List[RepairAction],
) -> None:
    """Create directory if absent; append RepairAction."""
    if directory.is_dir():
        return
    action = RepairAction(
        action_type="create_dir",
        source=None,
        target=rel_path,
        reason=issue_code,
    )
    try:
        directory.mkdir(parents=True, exist_ok=True)
        action.success = True
        action.message = f"Created {rel_path}/"
        logger.info("Repair [{}]: created {}/", challenge_name, rel_path)
    except OSError as exc:
        action.message = f"Failed to create {rel_path}/: {exc}"
        logger.warning(
            "Repair [{}]: failed to create {}/: {}", challenge_name, rel_path, exc
        )
    actions.append(action)


def _repair_file(
    *,
    writeup_dir: Path,
    challenge_path: Path,
    canonical_name: str,
    name_variants: dict[str, str],
    issue_code: str,
    challenge_name: str,
    actions: List[RepairAction],
) -> None:
    """Repair a single expected file in writeup_dir.

    Strategy (in order):
    1. If canonical target already exists → nothing to do.
    2. Rename a variant that exists IN writeup_dir → canonical_name.
    3. Move the first variant found ANYWHERE under challenge_path → writeup_dir/canonical_name.

    Content is never modified. Overwriting an existing target is prevented by
    _do_rename's target.exists() check.
    """
    target = writeup_dir / canonical_name
    if target.exists():
        return

    # Step 2: rename variant in the same directory
    for variant_lower, canonical in name_variants.items():
        if canonical != canonical_name:
            continue
        variant_path = writeup_dir / variant_lower
        if variant_path.exists() and variant_path != target:
            actions.append(_do_rename(variant_path, target, issue_code, challenge_path))
            return

    # Step 3: find any variant elsewhere under challenge_path
    found = _find_variant(challenge_path, name_variants, exclude_dir=writeup_dir)
    if found is None:
        # Try exact canonical name in other locations
        found = _find_by_name(challenge_path, canonical_name, exclude_dir=writeup_dir)
    if found is not None:
        actions.append(_do_rename(found, target, issue_code, challenge_path))


def _repair_solver(
    writeup_dir: Path,
    challenge_path: Path,
    challenge_name: str,
    actions: List[RepairAction],
) -> None:
    """Repair missing solver: move first solver found elsewhere to writeup_dir."""
    if any((writeup_dir / s).exists() for s in _SOLVER_NAMES):
        return
    for solver_name in sorted(_SOLVER_NAMES):  # deterministic order
        found = _find_by_name(challenge_path, solver_name, exclude_dir=writeup_dir)
        if found is not None:
            target = writeup_dir / solver_name
            actions.append(_do_rename(found, target, "MISSING_SOLVER", challenge_path))
            return


def _do_rename(
    source: Path, target: Path, reason: str, challenge_path: Path
) -> RepairAction:
    """Move source to target; skip if target already exists."""
    rel_source = str(source.relative_to(challenge_path))
    rel_target = str(target.relative_to(challenge_path))
    action = RepairAction(
        action_type="rename",
        source=rel_source,
        target=rel_target,
        reason=reason,
    )
    if target.exists():
        action.message = f"Skipped: {rel_target} already exists"
        logger.warning(
            "Repair: skipped {} → {} (target exists)", rel_source, rel_target
        )
        return action
    try:
        source.rename(target)
        action.success = True
        action.message = f"Moved {rel_source} → {rel_target}"
        logger.info("Repair: moved {} → {}", rel_source, rel_target)
    except OSError as exc:
        action.message = f"Failed to move {rel_source} → {rel_target}: {exc}"
        logger.warning(
            "Repair: failed to move {} → {}: {}", rel_source, rel_target, exc
        )
    return action


def _find_variant(
    root: Path, variants: dict[str, str], exclude_dir: Path
) -> Optional[Path]:
    """Return first file under root whose lowercase name is a key in variants.

    Excludes files already in exclude_dir.
    """
    for path in root.rglob("*"):
        if path.is_file() and path.parent != exclude_dir:
            if path.name.lower() in variants:
                return path
    return None


def _find_by_name(root: Path, name: str, exclude_dir: Path) -> Optional[Path]:
    """Return first file under root with the exact name, excluding exclude_dir."""
    for path in root.rglob(name):
        if path.is_file() and path.parent != exclude_dir:
            return path
    return None
