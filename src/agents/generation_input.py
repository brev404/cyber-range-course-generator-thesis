"""G1 — manifest -> generation-input loader.

Resolves a challenge's ``manifest.json`` (produced by scripts/build_manifests.py) into the
clean contract the generators consume:

* **course-gen** reads ``student_prompt`` + ``student_files`` ONLY — never the author writeup
  or solver. This is the data-layer fix for the context leak (a course can no longer absorb
  the author's solution because it is not in its context).
* **solver-gen** additionally reads ``author_writeup`` / ``author_solver`` (correctness).

Dual-role descriptions (cloud/web challenges whose ``description.md`` embeds a ``## Solution``)
are split so the student prompt stops before the solution; the solution portion is exposed
only as ``author_writeup``. Binary student files are flagged, not inlined.

Returns ``None`` for a challenge without a manifest (legacy path / reproducible baselines).
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from loguru import logger

_MAX_STUDENT_FILE_CHARS = 8000
_MAX_AUTHOR_CHARS = 12000
_MAX_PROMPT_CHARS = 20000
_MAX_BINARY_META_CHARS = 2000  # metadata snapshot cap per binary file


@dataclass
class StudentFile:
    """A file the student is given. Binary files get metadata snapshot, never inlined."""

    path: str
    kind: str  # "text" | "binary"
    content: Optional[str] = None  # text content (capped) or binary metadata snapshot


@dataclass
class GenerationInput:
    """The resolved generation contract for one challenge."""

    challenge_id: str
    tier: str
    eligible: bool
    student_prompt: (
        str  # course-gen input — prompt portion only (dual-role split applied)
    )
    student_files: list[StudentFile]  # course-gen input
    author_writeup: Optional[str]  # solver-gen ONLY — resolved text (never course-gen)
    author_solver: Optional[str]  # solver-gen ONLY — resolved text
    flag_status: str
    notes: list[str] = field(default_factory=list)


def _read_text(p: Path, cap: int) -> Optional[str]:
    try:
        return p.read_text(encoding="utf-8", errors="ignore")[:cap]
    except OSError:
        return None


def _binary_metadata_snapshot(p: Path) -> str:
    """Build a text metadata snapshot for a binary student file.

    Uses system ``file`` for type detection, extracts printable strings,
    and reports size. Capped at ``_MAX_BINARY_META_CHARS``.
    """
    parts: list[str] = []
    try:
        size = p.stat().st_size
        parts.append(f"size: {size:,} bytes ({size / 1024:.1f} KB)")
    except OSError:
        parts.append("size: unknown")

    # file type
    try:
        file_out = subprocess.check_output(
            ["file", "-b", str(p)], stderr=subprocess.DEVNULL, text=True, timeout=5
        ).strip()
        parts.append(f"type: {file_out}")
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        subprocess.TimeoutExpired,
    ):
        pass

    # printable strings — first 200 chars, useful for hints about binary content
    try:
        strings_out = subprocess.check_output(
            ["strings", "-n", "4", str(p)],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10,
        )
        # Deduplicate and take first meaningful lines
        seen: set[str] = set()
        unique_lines: list[str] = []
        for line in strings_out.splitlines():
            stripped = line.strip()
            if stripped and stripped not in seen:
                seen.add(stripped)
                unique_lines.append(stripped)
            if len(unique_lines) >= 15:
                break
        if unique_lines:
            parts.append("extracted strings:\n" + "\n".join(unique_lines))
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        subprocess.TimeoutExpired,
    ):
        pass

    raw = "\n".join(parts)
    if len(raw) > _MAX_BINARY_META_CHARS:
        raw = raw[:_MAX_BINARY_META_CHARS] + "\n... (truncated)"
    return raw


def _is_text_file(p: Path) -> bool:
    """Sniff: a file is text if its first 1KB decodes as UTF-8 and has no NUL bytes."""
    try:
        head = p.read_bytes()[:1024]
    except OSError:
        return False
    if b"\x00" in head:
        return False
    try:
        head.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def _resolve_student_prompt(ch: Path, manifest: dict) -> str:
    desc_rel = manifest.get("description")
    if not desc_rel:
        return ""
    full = _read_text(ch / desc_rel, _MAX_PROMPT_CHARS) or ""
    split = manifest.get("description_split")
    if split and split.get("solution_starts_at_char") is not None:
        # dual-role: student sees only the prompt, never the embedded '## Solution'
        return full[: split["solution_starts_at_char"]].strip()
    return full.strip()


def _resolve_author_writeup(
    ch: Path, manifest: dict, wu_rel: Optional[str]
) -> Optional[str]:
    if not wu_rel:
        return None
    if wu_rel.endswith("#solution"):
        # dual-role: the writeup IS the solution portion of the description file
        desc_rel = manifest.get("description")
        split = manifest.get("description_split") or {}
        start = split.get("solution_starts_at_char")
        if desc_rel and start is not None:
            full = (
                _read_text(ch / desc_rel, _MAX_PROMPT_CHARS + _MAX_AUTHOR_CHARS) or ""
            )
            return full[start:].strip()[:_MAX_AUTHOR_CHARS]
        return None
    return _read_text(ch / wu_rel, _MAX_AUTHOR_CHARS)


def load_generation_input(challenge_path: Path) -> Optional[GenerationInput]:
    """Load and resolve a challenge's manifest into a GenerationInput, or None if no manifest."""
    mpath = challenge_path / "manifest.json"
    if not mpath.is_file():
        return None
    try:
        manifest = json.loads(mpath.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Bad manifest at {}: {}", mpath, e)
        return None

    gi = manifest.get("generation_input", {})
    student_prompt = _resolve_student_prompt(challenge_path, manifest)

    student_files: list[StudentFile] = []
    for rel in gi.get("student_files", []):
        p = challenge_path / rel
        if not p.is_file():
            continue  # archive members listed but missing, etc.
        if _is_text_file(p):
            student_files.append(
                StudentFile(rel, "text", _read_text(p, _MAX_STUDENT_FILE_CHARS))
            )
        else:
            meta = _binary_metadata_snapshot(p)
            student_files.append(StudentFile(rel, "binary", meta))

    author_writeup = _resolve_author_writeup(
        challenge_path, manifest, gi.get("author_writeup")
    )
    solver_rel = gi.get("author_solver")
    author_solver = (
        _read_text(challenge_path / solver_rel, _MAX_AUTHOR_CHARS)
        if solver_rel
        else None
    )

    return GenerationInput(
        challenge_id=manifest.get("challenge_id", challenge_path.name),
        tier=manifest.get("tier", "unknown"),
        eligible=bool(manifest.get("eligible", False)),
        student_prompt=student_prompt,
        student_files=student_files,
        author_writeup=author_writeup,
        author_solver=author_solver,
        flag_status=gi.get("flag_status", manifest.get("flag", "unknown")),
        notes=list(gi.get("notes", [])),
    )
