"""Atomic file write helpers — tmp + fsync + rename pattern.

Prevents partial/stub files on crash. Follows the same pattern already used
for manifest.json (artifact_writer._write_manifest_atomic) and checkpoint.py.

Usage:
    from src.services.atomic_write import atomic_write_text, atomic_write_json

    atomic_write_text(Path("/out/course.md"), content)
    atomic_write_json(Path("/out/ranking_reports.json"), data)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def atomic_write_text(path: Path, content: str, encoding: str = "utf-8") -> None:
    """Write *content* to *path* atomically via .tmp + fsync + rename.

    Steps:
      1. Write to <path>.tmp
      2. fsync the file descriptor (flush kernel buffer to disk)
      3. Close the fd
      4. os.rename(<path>.tmp → <path>) — atomic on POSIX
      5. Cleanup: .tmp file is removed by rename on success; on failure the
         caller's exception propagates and the original file is untouched.

    Args:
        path: Destination path. Parent directory must exist (created if missing).
        content: Text to write.
        encoding: Encoding for text (default utf-8).

    Raises:
        OSError: If the write, fsync, or rename fails.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    payload = content.encode(encoding)
    fd = os.open(str(tmp_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        os.write(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)
    # Atomic rename — on POSIX this replaces the target atomically.
    os.rename(str(tmp_path), str(path))


def atomic_write_json(path: Path, data: Any, indent: int = 2) -> None:
    """Serialize *data* as JSON and write *path* atomically.

    Args:
        path: Destination path.
        data: Any JSON-serializable object.
        indent: JSON indent level (default 2).

    Raises:
        OSError: If the write fails.
        TypeError: If *data* is not JSON-serializable.
    """
    atomic_write_text(path, json.dumps(data, indent=indent))
