"""Per-challenge checkpoint for experiment resume.

Writes a .checkpoint.json file inside the experiment directory after each
challenge completes so that interrupted experiments can be resumed with
``--resume`` without regenerating already-completed challenges.

Schema (checkpoint_version=1):
    {
        "exp_id": "EXP-EXAMPLE",
        "completed_challenges": ["crypto/0_solves", "crypto/atentie-la-transport"],
        "in_progress_challenge": "crypto/frigography",
        "last_update": "2026-05-26T14:30:00+03:00",
        "checkpoint_version": 1
    }

Atomic write contract:
    1. Serialise to JSON.
    2. Write to <exp_dir>/.checkpoint.json.tmp.
    3. fsync the file descriptor.
    4. Rename .tmp → .checkpoint.json  (atomic on POSIX).
    The .tmp file is never left behind after a successful write.
    If the process crashes after step 2 but before step 4, the original
    .checkpoint.json is unchanged — callers can detect a stale .tmp if needed
    but we don't require that.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field

_CHECKPOINT_FILENAME = ".checkpoint.json"
_CHECKPOINT_VERSION = 1


class Checkpoint(BaseModel):
    """Schema for the experiment checkpoint file."""

    exp_id: str = ""
    completed_challenges: List[str] = Field(default_factory=list)
    in_progress_challenge: Optional[str] = None
    last_update: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    checkpoint_version: int = _CHECKPOINT_VERSION


def load(exp_dir: Path) -> Optional[Checkpoint]:
    """Load checkpoint from exp_dir/.checkpoint.json.

    Returns:
        A Checkpoint instance, or None if the file does not exist or is
        corrupt (corrupt files are logged and ignored — don't crash on resume).
    """
    ckpt_file = exp_dir / _CHECKPOINT_FILENAME
    if not ckpt_file.exists():
        return None
    try:
        return Checkpoint.model_validate_json(ckpt_file.read_text(encoding="utf-8"))
    except Exception as exc:
        from loguru import logger

        logger.warning(
            "Checkpoint file corrupt or unreadable ({}); ignoring: {}", ckpt_file, exc
        )
        return None


def save_atomic(exp_dir: Path, checkpoint: Checkpoint) -> None:
    """Write checkpoint atomically to exp_dir/.checkpoint.json.

    Steps: write to .tmp, fsync, rename to final name.
    """
    exp_dir.mkdir(parents=True, exist_ok=True)
    checkpoint.last_update = datetime.now(timezone.utc).isoformat()

    tmp_file = exp_dir / (_CHECKPOINT_FILENAME + ".tmp")
    final_file = exp_dir / _CHECKPOINT_FILENAME

    data = checkpoint.model_dump_json(indent=2)
    # Write and fsync tmp
    fd = os.open(str(tmp_file), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        os.write(fd, data.encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)

    # Atomic rename
    os.rename(str(tmp_file), str(final_file))


def record_challenge_done(exp_dir: Path, challenge_id: str) -> None:
    """Append challenge_id to completed_challenges and atomically save.

    Idempotent: if challenge_id is already in completed_challenges, this is a no-op.
    If no checkpoint file exists yet, creates a minimal one.

    Args:
        exp_dir: The experiment output directory.
        challenge_id: The challenge_id string to mark as done.
    """
    ckpt = load(exp_dir)
    if ckpt is None:
        ckpt = Checkpoint(exp_id=exp_dir.name)
    if challenge_id not in ckpt.completed_challenges:
        ckpt.completed_challenges.append(challenge_id)
    ckpt.in_progress_challenge = None
    save_atomic(exp_dir, ckpt)
