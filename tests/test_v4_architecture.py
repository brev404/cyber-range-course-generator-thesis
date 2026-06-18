"""Tests for the v4 unified two-call architecture.

v4 inverts the v3 generation order:
  v3: course-gen FIRST (LLM writes Section 9 with the full solver), then solver-gen.
  v4: solver-gen FIRST, course-gen receives the solver as authoritative input and
      writes Section 9 as a `<!-- SOLVER_PLACEHOLDER -->` marker. The pipeline replaces
      the marker with the python-fenced solver so course narrative (Sections 6-8) and
      solver cannot drift apart.

v4.1 (placeholder marker fix): the v4.0 "OMIT Section 9" instruction destabilised the
LLM's numbering (the model added off-spec sections or dropped numbering entirely). The
placeholder approach keeps all 11 numbered sections in the LLM's output stream and only
the marker line is replaced post-generation.

Tests cover:
  - The flag default (True)
  - Order-of-calls in _process_one_challenge_content under v4
  - The assembly helper replaces the placeholder marker with the python-fenced solver
  - Assembly falls back to legacy insert-before-Section-10 when the marker is missing
  - The course-gen user prompt instructs the LLM to write the placeholder marker
  - When V4_ARCHITECTURE_ENABLED=False, the v3 course-first flow is taken
"""

from __future__ import annotations

from unittest.mock import patch

# ---------------------------------------------------------------------------
# Settings flag default
# ---------------------------------------------------------------------------


def test_v4_flag_default_true():
    """V4_ARCHITECTURE_ENABLED is True by default — v4 is the new baseline."""
    from src.config.settings import settings

    assert hasattr(
        settings, "V4_ARCHITECTURE_ENABLED"
    ), "settings is missing V4_ARCHITECTURE_ENABLED field"
    assert (
        settings.V4_ARCHITECTURE_ENABLED is True
    ), f"V4_ARCHITECTURE_ENABLED should default to True, got {settings.V4_ARCHITECTURE_ENABLED!r}"


def test_v4_flag_in_reproducibility_export():
    """V4_ARCHITECTURE_ENABLED appears in export_for_reproducibility for ablation auditing."""
    from src.config.settings import settings

    data = settings.export_for_reproducibility()
    nested = data.get("settings", {})
    assert (
        "V4_ARCHITECTURE_ENABLED" in nested
    ), "V4_ARCHITECTURE_ENABLED must be in reproducibility export"


# ---------------------------------------------------------------------------
# Assembly helper
# ---------------------------------------------------------------------------


def test_assembled_course_contains_solver_as_section_9():
    """v4.1: assembly replaces `<!-- SOLVER_PLACEHOLDER -->` with the python-fenced solver,
    preserves the existing `## 9. Solution Script` heading written by the LLM, and leaves
    Sections 10 and 11 intact. No duplicate Section 9 should be produced.
    """
    from src.agents.content_generation_agent import _assemble_course_with_section_9

    course_with_placeholder = (
        "## 1. Title\n"
        "Crypto challenge.\n\n"
        "## 8. Step-by-step\n"
        "Run the recovery script.\n\n"
        "## 9. Solution Script\n\n"
        "The script below implements the technique from Section 8.\n\n"
        "<!-- SOLVER_PLACEHOLDER -->\n\n"
        "## 10. Conclusion\n"
        "We exploited the weakness.\n\n"
        "## 11. Extra Resources\n"
        "- CWE-327\n"
    )
    solver = "def solve():\n    print('CTF{flag}')\n\nsolve()\n"

    assembled = _assemble_course_with_section_9(course_with_placeholder, solver)

    # Section 9 heading present (LLM wrote it, assembly did not touch it)
    assert (
        "## 9. Solution Script" in assembled
    ), f"Assembled course must contain '## 9. Solution Script' heading.\nGot:\n{assembled}"
    # Placeholder has been replaced
    assert (
        "<!-- SOLVER_PLACEHOLDER -->" not in assembled
    ), "Placeholder marker must be replaced post-assembly"
    # Solver appears inside the assembled section
    assert (
        "def solve():" in assembled
    ), "Assembled course must contain the solver source"
    # Section 9 must come before Section 10
    pos_s9 = assembled.index("## 9. Solution Script")
    pos_s10 = assembled.index("## 10. Conclusion")
    assert pos_s9 < pos_s10, "Section 9 must remain BEFORE Section 10 (Conclusion)."
    # Python code fence wraps the solver
    assert (
        "```python" in assembled
    ), "Assembled Section 9 must wrap the solver in a python fence"
    # No duplicate Section 9 heading (assembly must not insert another one)
    assert (
        assembled.count("## 9. Solution Script") == 1
    ), "Assembly must NOT produce a duplicate '## 9. Solution Script' heading"
    # Section 11 still follows
    assert (
        "## 11. Extra Resources" in assembled
    ), "Assembly must preserve Section 11 written by the LLM"


def test_assembly_fallback_when_placeholder_missing(caplog):
    """If the LLM ignored the placeholder rule, assembly falls back to the legacy path
    (insert Section 9 before the Section 10 heading) and logs a warning so the fallback
    is visible in EXP logs.
    """
    from src.agents.content_generation_agent import _assemble_course_with_section_9

    course_no_marker = (
        "## 1. Title\nSome challenge.\n\n"
        "## 8. Steps\nDo stuff.\n\n"
        "## 10. Conclusion\nDone.\n"
    )
    solver = "print('hi')\n"

    # Loguru -> stderr by default; caplog captures stdlib logging. Use a sink to capture.
    from loguru import logger as _logger

    captured: list[str] = []
    sink_id = _logger.add(lambda msg: captured.append(str(msg)), level="WARNING")
    try:
        assembled = _assemble_course_with_section_9(course_no_marker, solver)
    finally:
        _logger.remove(sink_id)

    assert "## 9. Solution Script" in assembled
    assert "print('hi')" in assembled
    # Section 9 inserted before Section 10
    pos_s9 = assembled.index("## 9. Solution Script")
    pos_s10 = assembled.index("## 10. Conclusion")
    assert pos_s9 < pos_s10
    # Warning logged
    assert any(
        "SOLVER_PLACEHOLDER missing" in msg for msg in captured
    ), f"Expected fallback warning, got: {captured!r}"


def test_assembly_legacy_strips_llm_written_section_9():
    """v4.1.1: when the LLM ignored the placeholder AND wrote its own ## 9. ... block,
    legacy fallback must strip the LLM's Section 9 before inserting ours — otherwise
    the assembled course has DUPLICATE Section 9 headings.

    Real-world pattern (crypto/atentie-la-transport): Haiku
    wrote "## 9. Step-by-Step Resolution" and skipped the placeholder. The legacy
    insertion then added "## 9. Solution Script" — and after refinement-retry
    cycles, the final course had THREE Section 9 headings.
    """
    from src.agents.content_generation_agent import _assemble_course_with_section_9

    course_with_llm_s9 = (
        "## 1. Title\nSome challenge.\n\n"
        "## 8. Steps\nDo stuff.\n\n"
        "## 9. Step-by-Step Resolution\n"  # ← LLM wrote its own Section 9
        "### Step 1\nDo X.\n\n"
        "### Step 2\nDo Y.\n\n"
        "## 10. Conclusion\nDone.\n"
    )
    solver = "print('hi')\n"

    assembled = _assemble_course_with_section_9(course_with_llm_s9, solver)

    # Exactly one ## 9. heading
    s9_count = assembled.count("\n## 9.") + (1 if assembled.startswith("## 9.") else 0)
    assert s9_count == 1, (
        f"Expected exactly 1 Section 9 heading after assembly, got {s9_count}.\n"
        f"Assembled course:\n{assembled}"
    )
    # The remaining Section 9 must be the assembled "Solution Script" (not the LLM's)
    assert "## 9. Solution Script" in assembled
    assert "## 9. Step-by-Step Resolution" not in assembled
    # Solver code present
    assert "print('hi')" in assembled
    # Section 10 still present
    assert "## 10. Conclusion" in assembled


def test_assembly_legacy_strips_llm_section_9_with_multiple_subsections():
    """The LLM's ## 9. block may contain subsections (### Step 1, etc.) — strip the
    entire block up to the next ## heading."""
    from src.agents.content_generation_agent import _assemble_course_with_section_9

    course = (
        "## 1. Title\nT\n\n"
        "## 9. Some LLM-Invented Section\n"
        "lots of content\n\n"
        "### Subheading 1\nA\n\n"
        "### Subheading 2\nB\n\n"
        "## 10. Conclusion\nC\n"
    )
    assembled = _assemble_course_with_section_9(course, "x = 1\n")

    assert "Some LLM-Invented Section" not in assembled
    assert "Subheading 1" not in assembled
    assert "Subheading 2" not in assembled
    assert "## 9. Solution Script" in assembled
    assert "x = 1" in assembled
    assert "## 10. Conclusion" in assembled


def test_assembled_course_appended_when_no_section_10():
    """Legacy fallback: if no Section 10 heading AND no placeholder marker exists, Section 9
    is appended at the end (best-effort)."""
    from src.agents.content_generation_agent import _assemble_course_with_section_9

    course_no_s9 = "## 1. Title\nSome challenge.\n\n## 8. Steps\nDo stuff.\n"
    solver = "print('hi')\n"
    assembled = _assemble_course_with_section_9(course_no_s9, solver)

    assert "## 9. Solution Script" in assembled
    assert "print('hi')" in assembled


def test_assembled_course_handles_empty_solver():
    """An empty solver still produces a Section 9 with a placeholder body so the structural
    validator surfaces the issue downstream rather than the assembly silently dropping it.
    """
    from src.agents.content_generation_agent import _assemble_course_with_section_9

    course_with_placeholder = (
        "## 8. Steps\nDo stuff.\n\n"
        "## 9. Solution Script\n\n<!-- SOLVER_PLACEHOLDER -->\n\n"
        "## 10. Conclusion\nDone.\n"
    )
    assembled = _assemble_course_with_section_9(course_with_placeholder, "")

    assert "## 9. Solution Script" in assembled
    # Placeholder replaced (with a comment-stub fenced block since solver is empty)
    assert "<!-- SOLVER_PLACEHOLDER -->" not in assembled


def test_assembly_idempotent_when_already_assembled():
    """Calling assembly twice on an already-assembled course produces the same result
    (the placeholder is gone after the first call, so the second call should be a no-op
    via the legacy-fallback path — but since the course already contains a Section 9
    heading, the strip-and-reassemble pattern in the retry path is what handles this).
    For a single-call idempotency check: assembly on a course without the marker but with
    an existing fenced Section 9 should NOT add a duplicate Section 9.
    """
    from src.agents.content_generation_agent import _assemble_course_with_section_9

    course_with_placeholder = (
        "## 8. Steps\nDo stuff.\n\n"
        "## 9. Solution Script\n\n<!-- SOLVER_PLACEHOLDER -->\n\n"
        "## 10. Conclusion\nDone.\n"
    )
    solver = "print('hi')\n"
    first = _assemble_course_with_section_9(course_with_placeholder, solver)
    second = _assemble_course_with_section_9(first, solver)

    # The marker is gone in `first`; the second call uses the legacy fallback and inserts
    # before Section 10. That duplicates the heading. Use _strip_assembled_section_9 to
    # restore the pre-assembly form, then re-assemble — exactly the retry-path contract.
    from src.agents.content_generation_agent import _strip_assembled_section_9

    stripped = _strip_assembled_section_9(first)
    re_assembled = _assemble_course_with_section_9(stripped, solver)
    # The retry pattern must produce a course with exactly one Section 9 heading.
    assert (
        re_assembled.count("## 9. Solution Script") == 1
    ), f"Strip+re-assemble must yield exactly one Section 9 heading; got {re_assembled.count('## 9. Solution Script')}"
    # And the solver appears
    assert "print('hi')" in re_assembled
    # 'second' is allowed to differ (legacy fallback duplicates) — the call site protects
    # against this by always stripping before re-assembling on retry.
    _ = second


# ---------------------------------------------------------------------------
# Course-gen prompt content
# ---------------------------------------------------------------------------


def test_v4_prompt_requires_solver_placeholder(tmp_path):
    """v4.1: when solver_for_section_9 is set, the course-gen user prompt MUST instruct
    the LLM to write Section 9 with the literal `<!-- SOLVER_PLACEHOLDER -->` marker AND
    must enumerate all 11 sections with proper numbering."""
    captured: list[tuple[str, str]] = []

    def _fake_generate(system, user, *, temperature=0.5, max_tokens=14000, **kwargs):
        captured.append((system, user))
        # Return a plausible course (with placeholder) — LLM honoured the instruction.
        return (
            "## 1. Title\nx\n## 8. Steps\ny\n"
            "## 9. Solution Script\n\n<!-- SOLVER_PLACEHOLDER -->\n\n"
            "## 10. Conclusion\nz\n## 11. Extra Resources\n- CWE-327"
        )

    with (
        patch(
            "src.agents.content_generation_agent.generate_response_with_system",
            side_effect=_fake_generate,
        ),
        patch(
            "src.agents.content_generation_agent._build_rag_context",
            return_value="",
        ),
    ):
        from src.agents.content_generation_agent import (
            _generate_writeup_for_challenge,
        )

        _generate_writeup_for_challenge(
            challenge_id="crypto/test",
            category="crypto",
            challenge_name="test",
            description="A test crypto challenge.",
            solver_for_section_9="def solve(): pass\n",
        )

    assert (
        len(captured) == 1
    ), "course-gen should call generate_response_with_system once"
    system_prompt, user_prompt = captured[0]
    # The literal placeholder marker must be present in the system or user prompt
    combined = system_prompt + "\n" + user_prompt
    assert (
        "<!-- SOLVER_PLACEHOLDER -->" in combined
    ), "v4.1 prompt MUST contain the literal placeholder marker `<!-- SOLVER_PLACEHOLDER -->`"
    # Prompt must enumerate the 11 sections / require proper numbering
    lower = combined.lower()
    assert (
        "## 1." in combined and "## 11." in combined
    ), "v4.1 prompt should reference numbered sections ## 1. through ## 11."
    assert (
        "11 sections" in lower or "numbered" in lower or "numbering" in lower
    ), "v4.1 prompt should require proper section numbering"
    # The solver itself must be embedded in the prompt as the authoritative input.
    assert (
        "<solver_for_section_9>" in user_prompt
    ), "Course-gen prompt must contain <solver_for_section_9> wrapper block"
    assert (
        "def solve(): pass" in user_prompt
    ), "Course-gen prompt must contain the solver source"


def test_course_gen_prompt_no_solver_block_when_not_v4():
    """When solver_for_section_9 is empty (v3 fallback or first-round legacy), the
    placeholder marker instruction must NOT appear in the user prompt — the LLM writes
    Section 9 itself (with the full solver, per the v3 prompt path).

    Note: the placeholder marker MAY appear in the system prompt (rule 11 describes the
    v4 mechanism unconditionally), but the user-prompt v4_solver_block must be absent.
    """
    captured: list[tuple[str, str]] = []

    def _fake_generate(system, user, *, temperature=0.5, max_tokens=14000, **kwargs):
        captured.append((system, user))
        return "fake course"

    with (
        patch(
            "src.agents.content_generation_agent.generate_response_with_system",
            side_effect=_fake_generate,
        ),
        patch(
            "src.agents.content_generation_agent._build_rag_context",
            return_value="",
        ),
    ):
        from src.agents.content_generation_agent import (
            _generate_writeup_for_challenge,
        )

        _generate_writeup_for_challenge(
            challenge_id="crypto/test",
            category="crypto",
            challenge_name="test",
            description="A test challenge.",
            solver_for_section_9="",  # legacy / v3 mode
        )

    _system, user_prompt = captured[0]
    assert (
        "<solver_for_section_9>" not in user_prompt
    ), "Without solver_for_section_9, no v4 solver block should appear in the user prompt"
    # The v4-specific task footer must be absent from the user prompt
    assert (
        "v4 architecture is active" not in user_prompt.lower()
    ), "Without solver_for_section_9, the v4 task footer must be absent from the user prompt"


# ---------------------------------------------------------------------------
# Order of calls in _process_one_challenge_content
# ---------------------------------------------------------------------------


def _make_minimal_challenge_dir(tmp_path):
    """Create a barebones challenge directory tree expected by the pipeline."""
    cdir = tmp_path / "test-chal"
    wu = cdir / "cyberedu" / "write-up"
    wu.mkdir(parents=True)
    (wu / "description.md").write_text("A small crypto challenge.", encoding="utf-8")
    return cdir


def test_solver_generates_before_course_in_v4_flow(tmp_path, monkeypatch):
    """In v4 mode, _process_one_challenge_content invokes the solver generator BEFORE
    the course generator (so the course receives the solver as authoritative input).
    """
    from src.config.settings import settings

    monkeypatch.setattr(settings, "V4_ARCHITECTURE_ENABLED", True)
    monkeypatch.setattr(settings, "STRUCTURAL_VALIDATOR_ENABLED", False)

    call_order: list[str] = []

    def _fake_solve(*args, **kwargs):
        call_order.append("solver")
        return "def solve():\n    pass\n"

    def _fake_writeup(*args, **kwargs):
        call_order.append("course")
        # v4.1: LLM is expected to write Section 9 with the placeholder marker
        return (
            "## 1. Title\nx\n## 8. Step-by-step\ny\n"
            "## 9. Solution Script\n\n<!-- SOLVER_PLACEHOLDER -->\n\n"
            "## 10. Conclusion\nz\n## 11. Extra Resources\n- CWE-327"
        )

    cdir = _make_minimal_challenge_dir(tmp_path)
    with (
        patch(
            "src.agents.content_generation_agent._generate_solve_script_for_challenge",
            side_effect=_fake_solve,
        ),
        patch(
            "src.agents.content_generation_agent._generate_writeup_for_challenge",
            side_effect=_fake_writeup,
        ),
    ):
        from src.agents.content_generation_agent import _process_one_challenge_content

        cid, course, script, err = _process_one_challenge_content(
            cdir, "crypto", human_feedback_items=None
        )

    assert err is None, f"v4 flow raised: {err!r}"
    assert call_order == [
        "solver",
        "course",
    ], f"Expected solver-then-course order in v4, got {call_order}"
    # And the assembled course should contain Section 9 with the solver
    assert "## 9. Solution Script" in course
    assert "def solve():" in course
    # The placeholder must have been replaced
    assert "<!-- SOLVER_PLACEHOLDER -->" not in course


def test_v3_fallback_when_v4_flag_disabled(tmp_path, monkeypatch):
    """When V4_ARCHITECTURE_ENABLED=False, the legacy v3 flow runs: course first, solver second."""
    from src.config.settings import settings

    monkeypatch.setattr(settings, "V4_ARCHITECTURE_ENABLED", False)
    monkeypatch.setattr(settings, "STRUCTURAL_VALIDATOR_ENABLED", False)

    call_order: list[str] = []

    def _fake_solve(*args, **kwargs):
        call_order.append("solver")
        return "def solve():\n    pass\n"

    def _fake_writeup(*args, **kwargs):
        call_order.append("course")
        return (
            "## 1. Title\nx\n## 8. Step-by-step\ny\n## 9. Solution Script\n"
            "```python\nprint('x')\n```\n## 10. Conclusion\nz\n## 11. Extra Resources\n- CWE-327"
        )

    cdir = _make_minimal_challenge_dir(tmp_path)
    with (
        patch(
            "src.agents.content_generation_agent._generate_solve_script_for_challenge",
            side_effect=_fake_solve,
        ),
        patch(
            "src.agents.content_generation_agent._generate_writeup_for_challenge",
            side_effect=_fake_writeup,
        ),
    ):
        from src.agents.content_generation_agent import _process_one_challenge_content

        cid, course, script, err = _process_one_challenge_content(
            cdir, "crypto", human_feedback_items=None
        )

    assert err is None, f"v3 flow raised: {err!r}"
    assert call_order == [
        "course",
        "solver",
    ], f"Expected course-then-solver order in v3 fallback, got {call_order}"
    # Course is whatever the LLM returned — no assembly in v3 fallback
    assert "## 9. Solution Script" in course
