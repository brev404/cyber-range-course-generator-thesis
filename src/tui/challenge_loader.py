"""challenge_loader — discovers challenge entries from available sources.

Two sources are supported:
  "local"     — project root raw_challenges/{category}/{challenge_name}/
  "processed" — data/processed/raw_challenges/{category}/{challenge_name}/cyberedu/

Both sources handle gracefully when directories don't exist (return empty lists, never raise).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from loguru import logger

# Project root — two levels up from src/tui/
_PROJECT_ROOT = Path(__file__).parent.parent.parent

# Known category names for local source (used to filter noise at root level)
_KNOWN_CATEGORIES = {
    "pwn",
    "web",
    "crypto",
    "forensics",
    "rev",
    "osint",
    "misc",
    "mobile",
    "electron",
}

# Paths
_LOCAL_DIR = _PROJECT_ROOT / "raw_challenges"
_PROCESSED_LOCAL_DIR = _PROJECT_ROOT / "data" / "processed" / "raw_challenges"
# Staged 5-competition corpus (scripts/stage_and_extract.py + build_manifests.py):
# data/processed/datasets/<competition>/<category>/<challenge>/manifest.json
_DATASETS_DIR = _PROJECT_ROOT / "data" / "processed" / "datasets"


@dataclass
class ChallengeEntry:
    """A single discovered challenge."""

    challenge_id: str  # e.g. "rsb" or "crypto/rsb" for processed source
    category: str  # e.g. "crypto"
    source: str  # "processed" or "local"
    path: Path  # directory path containing challenge files


def load_challenges(
    source: str,
    categories: list[str] | None = None,
) -> list[ChallengeEntry]:
    """Discover challenges from the given source, optionally filtered by categories.

    Args:
        source: "local" or "processed"
        categories: If given, only return challenges from these categories.
                    None means all categories.

    Returns:
        List of ChallengeEntry. Empty if source directory doesn't exist.
    """
    if source == "local":
        return _load_local(categories)
    elif source == "processed":
        return _load_processed(categories)
    elif source == "datasets":
        return _load_datasets(categories)
    else:
        logger.warning(
            f"Unknown source: {source!r} — expected 'local', 'processed', or 'datasets'"
        )
        return []


def get_available_categories(source: str) -> list[str]:
    """Return category names that have at least one challenge for the given source.

    Args:
        source: "local" or "processed"

    Returns:
        Sorted list of category names. Empty if source directory doesn't exist.
    """
    if source == "local":
        return _get_local_categories()
    elif source == "processed":
        return _get_processed_categories()
    elif source == "datasets":
        return sorted({e.category for e in _load_datasets(None)})
    else:
        logger.warning(f"Unknown source: {source!r}")
        return []


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_local(categories: list[str] | None) -> list[ChallengeEntry]:
    """Scan raw_challenges/{category}/ for challenge subdirectories."""
    root = _LOCAL_DIR
    if not root.exists():
        logger.debug(f"raw_challenges dir not found at {root} — returning empty")
        return []

    entries: list[ChallengeEntry] = []
    for cat_dir in sorted(root.iterdir()):
        if not cat_dir.is_dir():
            continue
        category = cat_dir.name
        if category not in _KNOWN_CATEGORIES:
            continue
        if categories is not None and category not in categories:
            continue
        for challenge_dir in sorted(cat_dir.iterdir()):
            if not challenge_dir.is_dir():
                continue
            entries.append(
                ChallengeEntry(
                    challenge_id=challenge_dir.name,
                    category=category,
                    source="local",
                    path=challenge_dir,
                )
            )
    logger.debug(f"Loaded {len(entries)} challenges from raw_challenges")
    return entries


def _load_datasets(categories: list[str] | None) -> list[ChallengeEntry]:
    """Scan data/processed/datasets/<comp>/<category>/<challenge>/ for manifest-bearing
    challenges (the staged 5-competition corpus). challenge_id is <comp>/<category>/<name>
    so it is unique across competitions; e.path is the challenge dir (which holds manifest.json
    that the manifest-grounded generator consumes)."""
    root = _DATASETS_DIR
    if not root.exists():
        logger.debug(f"datasets dir not found at {root} — returning empty")
        return []
    entries: list[ChallengeEntry] = []
    for comp_dir in sorted(root.iterdir()):
        if not comp_dir.is_dir():
            continue
        for cat_dir in sorted(comp_dir.iterdir()):
            if not cat_dir.is_dir():
                continue
            category = cat_dir.name
            if categories is not None and category not in categories:
                continue
            for challenge_dir in sorted(cat_dir.iterdir()):
                if not challenge_dir.is_dir():
                    continue
                if not (challenge_dir / "manifest.json").is_file():
                    continue  # only staged, manifest-bearing challenges
                entries.append(
                    ChallengeEntry(
                        challenge_id=f"{comp_dir.name}/{category}/{challenge_dir.name}",
                        category=category,
                        source="datasets",
                        path=challenge_dir,
                    )
                )
    logger.debug(f"Loaded {len(entries)} challenges from datasets")
    return entries


def _load_processed(categories: list[str] | None) -> list[ChallengeEntry]:
    """Scan data/processed/raw_challenges/{category}/{name}/ for challenges.

    A challenge is included only when a cyberedu/ subdirectory exists (structural
    requirement), but e.path is set to challenge_dir (not cyberedu_dir) so that
    downstream agents can resolve the correct sub-paths (e.g.
    challenge_dir/cyberedu/write-up/description.md) without double-nesting.
    This aligns with coordinator_agent._prepare_organized_challenges which also
    returns challenge_dir paths.
    """
    root = _PROCESSED_LOCAL_DIR
    if not root.exists():
        logger.debug(f"processed dir not found at {root} — returning empty")
        return []

    entries: list[ChallengeEntry] = []
    for cat_dir in sorted(root.iterdir()):
        if not cat_dir.is_dir():
            continue
        category = cat_dir.name
        if categories is not None and category not in categories:
            continue
        for challenge_dir in sorted(cat_dir.iterdir()):
            if not challenge_dir.is_dir():
                continue
            cyberedu_dir = challenge_dir / "cyberedu"
            if not cyberedu_dir.exists():
                continue
            entries.append(
                ChallengeEntry(
                    challenge_id=f"{category}/{challenge_dir.name}",
                    category=category,
                    source="processed",
                    path=challenge_dir,  # challenge_dir, NOT cyberedu_dir
                )
            )
    logger.debug(f"Loaded {len(entries)} challenges from processed")
    return entries


def _get_local_categories() -> list[str]:
    """Return sorted list of non-empty category dirs inside raw_challenges/."""
    root = _LOCAL_DIR
    if not root.exists():
        return []
    cats: list[str] = []
    for cat_dir in sorted(root.iterdir()):
        if not cat_dir.is_dir():
            continue
        if cat_dir.name not in _KNOWN_CATEGORIES:
            continue
        # A category is available if it contains at least one subdirectory
        if any(p.is_dir() for p in cat_dir.iterdir()):
            cats.append(cat_dir.name)
    return sorted(cats)


def _get_processed_categories() -> list[str]:
    """Return sorted list of non-empty category dirs inside processed/raw_challenges/."""
    root = _PROCESSED_LOCAL_DIR
    if not root.exists():
        return []
    cats: list[str] = []
    for cat_dir in sorted(root.iterdir()):
        if not cat_dir.is_dir():
            continue
        # A category is available if it has at least one challenge with cyberedu/ dir
        has_challenge = any(
            (ch / "cyberedu").exists() for ch in cat_dir.iterdir() if ch.is_dir()
        )
        if has_challenge:
            cats.append(cat_dir.name)
    return sorted(cats)
