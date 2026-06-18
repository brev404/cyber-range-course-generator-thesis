"""Analytics helper for the metadata knowledge base."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def summarise_kb(kb_dir: Path) -> Dict[str, Any]:
    """Read challenges_index.json and return aggregate counts.

    Args:
        kb_dir: Directory containing challenges_index.json (KB_METADATA_DIR).

    Returns:
        Dict with keys: total_challenges (int), by_category (dict), by_contest (dict).
        Returns zero counts if the file is missing or malformed.
    """
    index_path = Path(kb_dir) / "challenges_index.json"
    if not index_path.is_file():
        return {"total_challenges": 0, "by_category": {}, "by_contest": {}}

    try:
        data = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"total_challenges": 0, "by_category": {}, "by_contest": {}}

    by_category: Dict[str, int] = {}
    by_contest: Dict[str, int] = {}

    for entry in data:
        cat = entry.get("category", "unknown")
        contest = entry.get("contest_id", "unknown")
        by_category[cat] = by_category.get(cat, 0) + 1
        by_contest[contest] = by_contest.get(contest, 0) + 1

    return {
        "total_challenges": len(data),
        "by_category": by_category,
        "by_contest": by_contest,
    }
