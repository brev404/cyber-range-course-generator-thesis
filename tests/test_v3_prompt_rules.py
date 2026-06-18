"""Test F1: _WRITEUP_SYSTEM contains all required v3 anti-pattern MUST rules."""


def test_no_truncation_rule():
    """_WRITEUP_SYSTEM must include a 'No truncation' rule."""
    from src.agents.content_generation_agent import _WRITEUP_SYSTEM

    assert (
        "No truncation" in _WRITEUP_SYSTEM or "no truncation" in _WRITEUP_SYSTEM.lower()
    ), "_WRITEUP_SYSTEM must contain 'No truncation' rule (F1)"


def test_attack_reference_rule():
    """_WRITEUP_SYSTEM must reference ATT&CK technique IDs."""
    from src.agents.content_generation_agent import _WRITEUP_SYSTEM

    assert (
        "MITRE ATT&CK" in _WRITEUP_SYSTEM or "ATT&CK" in _WRITEUP_SYSTEM
    ), "_WRITEUP_SYSTEM must contain 'ATT&CK' reference (F1)"


def test_conclusion_section_rule():
    """_WRITEUP_SYSTEM must require a Conclusion section."""
    from src.agents.content_generation_agent import _WRITEUP_SYSTEM

    assert "Conclusion section" in _WRITEUP_SYSTEM or (
        "Conclusion" in _WRITEUP_SYSTEM and "section" in _WRITEUP_SYSTEM
    ), "_WRITEUP_SYSTEM must mention 'Conclusion section' (F1)"


def test_expected_output_rule():
    """_WRITEUP_SYSTEM must require expected output in every step."""
    from src.agents.content_generation_agent import _WRITEUP_SYSTEM

    assert (
        "expected output" in _WRITEUP_SYSTEM.lower()
    ), "_WRITEUP_SYSTEM must contain 'expected output' rule (F1)"


def test_full_solver_rule():
    """v4 update: _WRITEUP_SYSTEM no longer asks the LLM to write Section 9 — Section 9
    is auto-assembled. Rule 11 must mark the v4 architecture explicitly so we cannot
    silently revert to the v3 'embed FULL solver' phrasing.
    """
    from src.agents.content_generation_agent import _WRITEUP_SYSTEM

    lower = _WRITEUP_SYSTEM.lower()
    has_v4_marker = (
        "auto-assembled" in lower
        or "do not write section 9" in lower
        or "omit section 9" in lower
    )
    assert has_v4_marker, (
        "_WRITEUP_SYSTEM must mark v4 architecture explicitly: rule 11 should contain "
        "'auto-assembled' / 'DO NOT write Section 9' / 'OMIT Section 9' so the LLM does "
        "not write Section 9 itself (Section 9 is assembled from the solver by the pipeline)."
    )


def test_narrative_agreement_rule():
    """_WRITEUP_SYSTEM must require solver and narrative to agree."""
    from src.agents.content_generation_agent import _WRITEUP_SYSTEM

    has_agree = (
        "narrative must agree" in _WRITEUP_SYSTEM
        or "Solver and narrative" in _WRITEUP_SYSTEM
    )
    assert (
        has_agree
    ), "_WRITEUP_SYSTEM must contain 'narrative must agree' or 'Solver and narrative' rule (F1)"


def test_no_placeholder_arrays_rule():
    """_WRITEUP_SYSTEM must forbid placeholder arrays / unknown constants in the solver (C6)."""
    from src.agents.content_generation_agent import _WRITEUP_SYSTEM

    has_placeholder_rule = (
        "placeholder array" in _WRITEUP_SYSTEM.lower()
        or "empty-stub" in _WRITEUP_SYSTEM.lower()
        or ("KEY = []" in _WRITEUP_SYSTEM and "ENCODED = []" in _WRITEUP_SYSTEM)
    )
    assert (
        has_placeholder_rule
    ), "_WRITEUP_SYSTEM must contain a 'no placeholder arrays' rule (F1, C6)"
