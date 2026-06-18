"""CourseViewerScreen — renders course.md files for a given experiment.

Keys:
  [  — previous challenge
  ]  — next challenge
  Escape — back
"""

from __future__ import annotations

from pathlib import Path

from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import Screen
from textual.widgets import Footer, Header, Label

from src.tui.widgets.markdown_view import MarkdownView


def _experiments_base() -> Path:
    return Path(__file__).resolve().parents[3] / "output" / "experiments"


def _load_courses(exp_id: str) -> list[tuple[str, str]]:
    """Return list of (label, markdown_content) for all course.md files in the experiment.

    Sorted by relative path (category/name). Returns empty list if none found.
    """
    courses_dir = _experiments_base() / exp_id / "courses"
    if not courses_dir.exists():
        return []
    results: list[tuple[str, str]] = []
    for md_path in sorted(courses_dir.rglob("course.md")):
        try:
            rel = md_path.relative_to(courses_dir)
            # label: category/name  (drop the trailing course.md segment)
            label = str(rel.parent)
            content = md_path.read_text(encoding="utf-8")
            results.append((label, content))
        except Exception:
            pass
    return results


class CourseViewerScreen(Screen):
    BINDINGS = [
        Binding("[", "prev_challenge", "Prev"),
        Binding("]", "next_challenge", "Next"),
        Binding("escape", "app.pop_screen", "Back"),
    ]

    def __init__(self, exp_id: str) -> None:
        super().__init__()
        self._exp_id = exp_id
        self._courses: list[tuple[str, str]] = []
        self._index: int = 0

    def compose(self) -> ComposeResult:
        yield Header()
        yield Label("", id="course-header")
        yield MarkdownView(id="course-content")
        yield Footer()

    def on_mount(self) -> None:
        self._courses = _load_courses(self._exp_id)
        self._render()

    def _render(self) -> None:
        header: Label = self.query_one("#course-header", Label)
        mv: MarkdownView = self.query_one("#course-content", MarkdownView)

        if not self._courses:
            header.update(f"[{self._exp_id}] No courses found.")
            mv.update("*(no course.md files found for this experiment)*")
            return

        label, content = self._courses[self._index]
        total = len(self._courses)
        header.update(f"[{self._exp_id}] Challenge {self._index + 1}/{total}: {label}")
        mv.update(content)

    def action_prev_challenge(self) -> None:
        if not self._courses:
            return
        self._index = (self._index - 1) % len(self._courses)
        self._render()

    def action_next_challenge(self) -> None:
        if not self._courses:
            return
        self._index = (self._index + 1) % len(self._courses)
        self._render()
