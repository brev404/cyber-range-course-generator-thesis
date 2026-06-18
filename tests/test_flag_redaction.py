"""Tests for _redact_flags: strips CTF flag-format tokens from author reference
context before it is injected into the generation prompt (fair-generator mode).

Motivated by an observed failure mode: the generator hardcoded the real flag it was shown in
the author writeup/solver context. Redaction removes the spoiler so any generator
must implement the technique to produce the flag.
"""

from src.agents.content_generation_agent import _redact_flags


def test_redacts_braced_flag_tokens():
    text = "The flag is CTF{4CA9A1_NOS7026} found via EXIF."
    out = _redact_flags(text)
    assert "CTF{4CA9A1_NOS7026}" not in out
    assert "[REDACTED_FLAG]" in out


def test_redacts_multiple_and_varied_prefixes():
    text = "CTF{abc} then flag{def_ghi} and FLAG{ZZZ-123}."
    out = _redact_flags(text)
    for tok in ("CTF{abc}", "flag{def_ghi}", "FLAG{ZZZ-123}"):
        assert tok not in out
    assert out.count("[REDACTED_FLAG]") == 3


def test_preserves_non_flag_braces():
    # python set/dict/f-string-ish braces (no word immediately before '{') must survive
    text = "scores = {1, 2, 3}; d = {'k': 'v'}; print(f'{x}')"
    out = _redact_flags(text)
    assert out == text


def test_empty_and_none_safe():
    assert _redact_flags("") == ""
    assert _redact_flags("no flags here") == "no flags here"
