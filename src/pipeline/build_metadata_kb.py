"""Build structured metadata knowledge base from processed challenge directory.

Walks PROCESSED_DIR ({contest}/{category}/{challenge}/) and writes:
  - contests.json     — one entry per top-level directory
  - categories.json   — one entry per unique category
  - challenges_index.json — one entry per challenge with contest/category linkage

YAML frontmatter from description.md or writeup.md is parsed when present
to extract author info. Falls back gracefully when absent.

Usage:
    uv run python src/pipeline/build_metadata_kb.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

# Allow running as script without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config.settings import settings  # noqa: E402


def _parse_yaml_frontmatter(path: Path) -> Dict[str, Any]:
    """Extract YAML frontmatter from a markdown file. Returns {} if absent or malformed."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    if not text.startswith("---"):
        return {}
    end = text.find("\n---", 3)
    if end == -1:
        return {}
    block = text[3:end].strip()
    result: Dict[str, Any] = {}
    for line in block.splitlines():
        m = re.match(r"^(\w[\w\s-]*):\s*(.+)$", line.strip())
        if m:
            result[m.group(1).strip().lower()] = m.group(2).strip()
    return result


def _find_author(challenge_dir: Path) -> Optional[str]:
    """Try to extract author from description.md or writeup.md frontmatter."""
    for name in ("description.md", "writeup.md"):
        candidate = challenge_dir.rglob(name)
        for p in candidate:
            fm = _parse_yaml_frontmatter(p)
            author = fm.get("author") or fm.get("authors")
            if author:
                return str(author)
    return None


def build_metadata_kb(
    processed_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, int]:
    """Walk processed_dir and write contests/categories/challenges JSON files.

    Args:
        processed_dir: Root of organized challenges. Defaults to settings.PROCESSED_DIR.
        output_dir: Where to write JSON files. Defaults to settings.KB_METADATA_DIR.

    Returns:
        Dict with counts: contests, categories, challenges.
    """
    src = Path(processed_dir or settings.PROCESSED_DIR)
    dst = Path(output_dir or settings.KB_METADATA_DIR)
    dst.mkdir(parents=True, exist_ok=True)

    if not src.is_dir():
        logger.warning("PROCESSED_DIR does not exist: {}", src)
        return {"contests": 0, "categories": 0, "challenges": 0}

    contests: Dict[str, Dict[str, Any]] = {}
    categories: Dict[str, Dict[str, Any]] = {}
    challenges: List[Dict[str, Any]] = []

    for contest_dir in sorted(src.iterdir()):
        if not contest_dir.is_dir() or contest_dir.name.startswith("."):
            continue
        contest_id = contest_dir.name
        if contest_id not in contests:
            contests[contest_id] = {
                "id": contest_id,
                "name": contest_id,
                "type": "ctf",
                "date": "",
            }
        logger.debug("Contest: {}", contest_id)

        for cat_dir in sorted(contest_dir.iterdir()):
            if not cat_dir.is_dir() or cat_dir.name.startswith("."):
                continue
            cat_id = cat_dir.name
            cat_key = cat_id
            if cat_key not in categories:
                categories[cat_key] = {
                    "id": cat_id,
                    "name": cat_id,
                    "parent": None,
                }
            logger.debug("  Category: {}", cat_id)

            for chall_dir in sorted(cat_dir.iterdir()):
                if not chall_dir.is_dir() or chall_dir.name.startswith("."):
                    continue
                challenge_id = chall_dir.name
                author = _find_author(chall_dir)
                entry: Dict[str, Any] = {
                    "challenge_id": challenge_id,
                    "contest_id": contest_id,
                    "category": cat_id,
                }
                if author:
                    entry["author"] = author
                challenges.append(entry)
                logger.debug("    Challenge: {}", challenge_id)

    # Write output files
    (dst / "contests.json").write_text(
        json.dumps(list(contests.values()), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (dst / "categories.json").write_text(
        json.dumps(list(categories.values()), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (dst / "challenges_index.json").write_text(
        json.dumps(challenges, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    counts = {
        "contests": len(contests),
        "categories": len(categories),
        "challenges": len(challenges),
    }
    logger.info(
        "Metadata KB built: {} contests, {} categories, {} challenges → {}",
        counts["contests"],
        counts["categories"],
        counts["challenges"],
        dst,
    )
    return counts


if __name__ == "__main__":
    from loguru import logger as _logger

    _logger.remove()
    _logger.add(sys.stderr, level="INFO")
    result = build_metadata_kb()
    print(json.dumps(result, indent=2))
