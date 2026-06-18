"""Tests for _strip_code_blocks: the pedagogical judge evaluates teaching quality
(structure, language, progression) and does not need the solver code. Stripping
fenced code blocks shrinks the judge input (faster haiku calls) without changing
the technical judge (J3) — stripping code improves token efficiency."""

from src.agents.ranking_agent import _strip_code_blocks


def test_strips_fenced_python_block():
    course = "## 8. Steps\nDo X.\n\n```python\nimport os\nFLAG='x'\nprint(FLAG)\n```\n\n## 10. Conclusion\nDone."
    out = _strip_code_blocks(course)
    assert "import os" not in out
    assert "print(FLAG)" not in out
    assert "## 8. Steps" in out and "## 10. Conclusion" in out  # headings/prose kept
    assert "[code omitted" in out


def test_strips_multiple_blocks_keeps_prose():
    course = "Intro\n```bash\nls -la\n```\nmiddle\n```python\nx=1\n```\nend"
    out = _strip_code_blocks(course)
    assert "ls -la" not in out and "x=1" not in out
    assert "Intro" in out and "middle" in out and "end" in out


def test_preserves_inline_code():
    # single-backtick inline code is prose-level context, not a big block — keep it
    course = "Run `nmap -sV` to scan; expected `open` ports."
    out = _strip_code_blocks(course)
    assert out == course


def test_empty_safe():
    assert _strip_code_blocks("") == ""
    assert _strip_code_blocks("no code here") == "no code here"
