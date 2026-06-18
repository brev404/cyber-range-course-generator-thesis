"""Test _strip_markdown_fence handles all common LLM fence patterns.

Bug discovered in v4 smoke trial: Haiku wraps solver output in ```python ... ```
fences despite the prompt explicitly forbidding them. Original _strip_markdown_fence
only handled ```markdown / ```md / ``` — leaving ```python through to ast.parse,
which fails on line 1 ("invalid syntax").
"""

from src.agents.content_generation_agent import _strip_markdown_fence


def test_strip_python_fence():
    text = "```python\ndef solve():\n    pass\n```"
    assert _strip_markdown_fence(text) == "def solve():\n    pass"


def test_strip_py_fence():
    text = "```py\nx = 1\n```"
    assert _strip_markdown_fence(text) == "x = 1"


def test_strip_markdown_fence_still_works():
    text = "```markdown\n# Title\n```"
    assert _strip_markdown_fence(text) == "# Title"


def test_strip_bare_fence():
    text = "```\nhello\n```"
    assert _strip_markdown_fence(text) == "hello"


def test_strip_truncated_open_fence_no_close():
    """When solver output is truncated, opener may be present but closer missing.
    We must still strip the opener so ast.parse doesn't choke on line 1.
    """
    text = "```python\ndef solve():\n    return 42\n"
    assert _strip_markdown_fence(text) == "def solve():\n    return 42"


def test_no_fence_returns_unchanged():
    text = "def solve():\n    pass\n"
    assert _strip_markdown_fence(text) == "def solve():\n    pass"


def test_strip_with_trailing_whitespace():
    text = "```python\nx = 1\n```   \n\n"
    assert _strip_markdown_fence(text) == "x = 1"


def test_strip_language_with_hyphen():
    text = "```c++\nint main() {}\n```"
    assert _strip_markdown_fence(text) == "int main() {}"


def test_stripped_solver_parses_with_ast():
    """End-to-end: the stripped solver must parse with ast.parse (F8 validator path)."""
    import ast

    text = (
        "```python\nimport sys\n\ndef solve():\n    return sys.argv[1]\n\nsolve()\n```"
    )
    stripped = _strip_markdown_fence(text)
    ast.parse(stripped)  # raises SyntaxError if fence still present
