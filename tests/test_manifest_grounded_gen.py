"""G2 — manifest-grounded course generation: the course prompt must be grounded ONLY in
student materials. When generation_input is supplied, the author writeup/solver must NOT
appear in the course prompt (the context-leak fix), and the frame-leak "student does NOT
have ..." line must be gone. Legacy mode (generation_input=None) keeps the old behaviour.
"""

from __future__ import annotations

import src.agents.content_generation_agent as cga
from src.agents.generation_input import GenerationInput, StudentFile

_AW = "AUTHOR_WRITEUP_INTERNAL_DATA_BIN_0x5A"
_AS = "AUTHOR_SOLVER_SECRET_CODE"


def _run(monkeypatch, generation_input):
    monkeypatch.setattr(cga.app_settings, "RAG_ENABLED", False)
    captured = {}

    def fake_llm(system, user, **kw):
        captured["user"] = user
        return "# course\n## 1. Title and Context\nbody"

    monkeypatch.setattr(cga, "generate_response_with_system", fake_llm)
    cga._generate_writeup_for_challenge(
        challenge_id="pwn/x",
        category="pwn",
        challenge_name="x",
        description="Smash the stack to win the flag.",
        author_writeup=_AW,
        author_solver=_AS,
        solver_for_section_9="V4_SOLVER_ANCHOR_CODE",
        generation_input=generation_input,
    )
    return captured["user"]


def test_manifest_mode_excludes_author_includes_student(monkeypatch):
    gi = GenerationInput(
        challenge_id="pwn/x",
        tier="well_formed",
        eligible=True,
        student_prompt="Smash the stack to win the flag.",
        student_files=[StudentFile("public/chall.py", "text", "STUDENT_SOURCE_CODE")],
        author_writeup=_AW,
        author_solver=_AS,
        flag_status="present",
    )
    u = _run(monkeypatch, gi)
    # student materials are present
    assert "Smash the stack to win the flag." in u
    assert "STUDENT_SOURCE_CODE" in u
    # THE LEAK FIX: author writeup + solver are NOT in the course prompt
    assert _AW not in u
    assert _AS not in u
    # frame-leak line gone
    assert "does NOT have source code, author writeup, or the flag" not in u
    assert "Reference only (do NOT cite" not in u
    # v4 solver anchor (what the student receives in Section 9) is still present — correctness
    assert "V4_SOLVER_ANCHOR_CODE" in u


def test_binary_student_file_referenced_not_inlined(monkeypatch):
    gi = GenerationInput(
        challenge_id="rev/x",
        tier="well_formed",
        eligible=True,
        student_prompt="Reverse it.",
        student_files=[StudentFile("public/chall", "binary", None)],
        author_writeup=None,
        author_solver=None,
        flag_status="present",
    )
    u = _run(monkeypatch, gi)
    assert "public/chall" in u and "binary file" in u


def test_legacy_mode_unchanged(monkeypatch):
    # generation_input=None → author writeup/solver injected, lacks line present (reproducible)
    u = _run(monkeypatch, None)
    assert _AW in u
    assert _AS in u
    assert "does NOT have source code, author writeup, or the flag" in u
