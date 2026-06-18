"""Tests for the manifest -> generation-input loader (G1).

The loader resolves a challenge's manifest.json into the clean contract the generators
consume: course-gen reads student_prompt + student_files ONLY; the author writeup/solver
are resolved separately for solver-gen. Dual-role descriptions are split so the student
prompt never contains the '## Solution' portion. Binary student files are flagged, not inlined.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.agents.generation_input import GenerationInput, load_generation_input


def _make_challenge(root: Path, manifest: dict, files: dict[str, bytes]) -> Path:
    ch = root / "ch"
    ch.mkdir(parents=True)
    for rel, data in files.items():
        p = ch / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    (ch / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return ch


def test_no_manifest_returns_none(tmp_path):
    (tmp_path / "ch").mkdir()
    assert load_generation_input(tmp_path / "ch") is None


def test_loads_well_formed(tmp_path):
    manifest = {
        "challenge_id": "pwn/x",
        "tier": "well_formed",
        "eligible": True,
        "description": "cyberedu/write-up/description.md",
        "description_split": None,
        "flag": "present",
        "generation_input": {
            "student_prompt": "...path...",
            "student_files": ["public/chall.py"],
            "author_writeup": "cyberedu/write-up/writeup.md",
            "author_solver": "cyberedu/write-up/solve.py",
            "flag_status": "present",
            "notes": [],
        },
    }
    ch = _make_challenge(
        tmp_path,
        manifest,
        {
            "cyberedu/write-up/description.md": b"# Buffy\nSmash the stack to win.",
            "public/chall.py": b"import os\nprint('vuln')\n",
            "cyberedu/write-up/writeup.md": b"Author writeup: overflow at offset 64.",
            "cyberedu/write-up/solve.py": b"from pwn import *\n# exploit\n",
        },
    )
    gi = load_generation_input(ch)
    assert isinstance(gi, GenerationInput)
    assert gi.tier == "well_formed" and gi.eligible is True
    assert "Smash the stack" in gi.student_prompt
    assert len(gi.student_files) == 1
    sf = gi.student_files[0]
    assert sf.path == "public/chall.py" and sf.kind == "text" and "vuln" in sf.content
    # author refs resolved (for solver-gen ONLY)
    assert "offset 64" in gi.author_writeup
    assert "pwn" in gi.author_solver


def test_dual_role_description_split(tmp_path):
    desc = "# Presigned\nThe cloud is all you need.\n\n## Solution\nLeak the AWS creds, sign the URL."
    boundary = desc.index("## Solution")
    manifest = {
        "challenge_id": "web/presigned",
        "tier": "writeup_only",
        "eligible": True,
        "description": "description.md",
        "description_split": {
            "file": "description.md",
            "solution_starts_at_char": boundary,
        },
        "flag": "present",
        "generation_input": {
            "student_prompt": "...",
            "student_files": [],
            "author_writeup": "description.md#solution",
            "author_solver": None,
            "flag_status": "present",
            "notes": [],
        },
    }
    ch = _make_challenge(tmp_path, manifest, {"description.md": desc.encode()})
    gi = load_generation_input(ch)
    # student sees ONLY the prompt portion — never the solution
    assert "cloud is all you need" in gi.student_prompt
    assert "## Solution" not in gi.student_prompt
    assert "AWS creds" not in gi.student_prompt
    # the author_writeup resolves to the solution portion (solver-gen only)
    assert "AWS creds" in gi.author_writeup


def test_binary_student_file_not_inlined(tmp_path):
    manifest = {
        "challenge_id": "rev/x",
        "tier": "well_formed",
        "eligible": True,
        "description": "desc.md",
        "description_split": None,
        "flag": "present",
        "generation_input": {
            "student_prompt": "...",
            "student_files": ["public/binary"],
            "author_writeup": None,
            "author_solver": None,
            "flag_status": "present",
            "notes": [],
        },
    }
    ch = _make_challenge(
        tmp_path,
        manifest,
        {
            "desc.md": b"Reverse the binary.",
            "public/binary": b"\x7fELF\x00\x01\x02\x03\x00\x00binarydata",
        },
    )
    gi = load_generation_input(ch)
    assert len(gi.student_files) == 1
    sf = gi.student_files[0]
    assert sf.kind == "binary"
    assert sf.content is not None, "binary metadata snapshot expected"
    assert (
        "type:" in sf.content
    ), f"expected file type in metadata, got: {sf.content[:100]}"
    assert "size:" in sf.content, f"expected size in metadata, got: {sf.content[:100]}"


def test_missing_student_file_skipped(tmp_path):
    manifest = {
        "challenge_id": "x/y",
        "tier": "well_formed",
        "eligible": True,
        "description": "desc.md",
        "description_split": None,
        "flag": "absent",
        "generation_input": {
            "student_prompt": "...",
            "student_files": ["public/gone.py"],
            "author_writeup": None,
            "author_solver": None,
            "flag_status": "absent",
            "notes": [],
        },
    }
    ch = _make_challenge(tmp_path, manifest, {"desc.md": b"desc"})
    gi = load_generation_input(ch)
    assert gi.student_files == []
