"""Atomic REVIEW_QUEUE.md updater.

Appends or updates experiment review entries in the REVIEW_QUEUE.md file
under the "Experiment runs" section. Uses atomic write (POSIX rename) to
prevent partial writes.
"""

from __future__ import annotations

import fcntl
import re
from pathlib import Path

from loguru import logger

_DEFAULT_QUEUE_PATH = Path("output/review_queue.md")
_SECTION_HEADER = "## \U0001f4ca Experiment runs"
_PLACEHOLDER = "_None._"


def append_to_review_queue(
    exp_id: str,
    review_path: Path,
    summary: str,
    queue_path: Path | None = None,
) -> None:
    """Append or update an experiment entry in REVIEW_QUEUE.md.

    Idempotent: if exp_id already exists, the line is replaced in-place.
    Atomic: writes to a .tmp file then renames (POSIX atomic).
    """
    qp = queue_path or _DEFAULT_QUEUE_PATH
    if not qp.exists():
        logger.warning(f"REVIEW_QUEUE.md not found at {qp}")
        return

    try:
        rel = review_path.relative_to(qp.parent)
    except ValueError:
        rel = review_path

    entry_line = f"- **{exp_id}**: {summary} -> [{review_path.name}]({rel})"

    with open(qp, "r+", encoding="utf-8") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            content = fh.read()
            new_content = _update_content(content, exp_id, entry_line)
            tmp = qp.with_suffix(".md.tmp")
            tmp.write_text(new_content, encoding="utf-8")
            tmp.replace(qp)
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)

    logger.info(f"REVIEW_QUEUE.md updated with {exp_id}")


def _update_content(content: str, exp_id: str, entry_line: str) -> str:
    lines = content.split("\n")
    section_idx = None
    separator_idx = None
    existing_idx = None

    exp_pattern = re.compile(re.escape(f"**{exp_id}**"))

    for i, line in enumerate(lines):
        if line.startswith(_SECTION_HEADER):
            section_idx = i
        if section_idx is not None and i > section_idx:
            if exp_pattern.search(line):
                existing_idx = i
            if line.strip() == "---" and separator_idx is None:
                separator_idx = i

    if section_idx is None:
        lines.append("")
        lines.append(f"{_SECTION_HEADER}")
        lines.append("")
        lines.append(entry_line)
        return "\n".join(lines)

    if existing_idx is not None:
        lines[existing_idx] = entry_line
        return "\n".join(lines)

    placeholder_idx = None
    for i in range(section_idx + 1, separator_idx or len(lines)):
        if _PLACEHOLDER in lines[i]:
            placeholder_idx = i
            break

    if placeholder_idx is not None:
        lines[placeholder_idx] = entry_line
    elif separator_idx is not None:
        lines.insert(separator_idx, entry_line)
    else:
        lines.append(entry_line)

    return "\n".join(lines)
