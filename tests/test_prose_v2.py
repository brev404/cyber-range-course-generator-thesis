"""G4 — prose overhaul (PROSE_V2). The flag post-processes the course system prompt to fix
the audit's prose defects: Section 6 rename (D4), frame-leak line removed, and an override
block banning the formulaic per-step template (D3), em-dashes/AI-tells (D2), and flag-in-prose
(D5). Default off => baseline prose unchanged (reproducible)."""

from __future__ import annotations

import src.agents.content_generation_agent as cga


def test_apply_prose_v2_transforms():
    out = cga._apply_prose_v2(cga._WRITEUP_SYSTEM)
    # D4: Section 6 renamed; old name gone
    assert "**Setup**" in out
    assert "Reproducibility (Step 0)" not in out
    # frame-leak instruction removed
    assert "Do not assume they have source, writeup, or the flag" not in out
    # override block present (D2/D3/D5)
    assert "## 6. Setup" in out
    assert "Never repeat the" in out and "Expected: ... This shows that" in out
    assert "avoid em-dashes" in out
    assert "Never print a flag value" in out


def test_baseline_unchanged_when_flag_off(monkeypatch):
    monkeypatch.setattr(cga.app_settings, "PROSE_V2", False)
    monkeypatch.setattr(cga.app_settings, "PROMPT_VARIANT", "baseline")
    s = cga._resolve_writeup_system()
    assert s == cga._WRITEUP_SYSTEM
    assert "Reproducibility (Step 0)" in s  # baseline prose intact


def test_resolve_applies_prose_v2_when_on(monkeypatch):
    monkeypatch.setattr(cga.app_settings, "PROSE_V2", True)
    monkeypatch.setattr(cga.app_settings, "PROMPT_VARIANT", "baseline")
    s = cga._resolve_writeup_system()
    assert "**Setup**" in s
    assert "Reproducibility (Step 0)" not in s
    assert "Do not assume they have source, writeup, or the flag" not in s
    assert "avoid em-dashes" in s


# --- deterministic post-processor (D2 em-dash + residual frame-leak) ---
def test_postprocess_em_dash_outside_code_only():
    txt = "Apply the key — recover plaintext.\n\n```python\nx = 1  # a — b in code stays\n```\n"
    out = cga._postprocess_prose_v2(txt)
    assert "—" not in out.split("```")[0]  # prose em-dash gone
    assert "a — b in code stays" in out  # code block untouched
    assert "Apply the key, recover plaintext." in out


def test_postprocess_strips_kittyos_frame_leak():
    # the exact residual from the KittyOS G4 run
    txt = "You have the binary; the flag, source code, and any author writeup are not provided.\n"
    out = cga._postprocess_prose_v2(txt)
    assert "author writeup" not in out
    assert "not provided" not in out
    assert "You have the binary." in out  # positive content preserved


def test_postprocess_noop_when_clean():
    txt = (
        "Run `file KittyOS` to confirm the format. The XOR key recovers the strings.\n"
    )
    assert cga._postprocess_prose_v2(txt) == txt
