"""NodeProgress — 8-node pipeline progress bar widget."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Static

NODES = [
    "coordinator",
    "validator",
    "content_generation",
    "course_terminology_checker",
    "mapping",
    "ranking",
    "refinement_step",
    "hitl",
]

# Short labels so the bar fits on one line at any reasonable terminal width.
_SHORT: dict[str, str] = {
    "coordinator": "coord",
    "validator": "valid",
    "content_generation": "gen",
    "course_terminology_checker": "term",
    "mapping": "map",
    "ranking": "rank",
    "refinement_step": "refine",
    "hitl": "hitl",
}

_SYMBOLS: dict[str, str] = {
    "pending": "□",
    "active": "▶",
    "done": "■",
    "failed": "✗",
}


class NodeProgress(Widget):
    """Visual pipeline bar showing pending/active/done/failed state for each node."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._states: dict[str, str] = {n: "pending" for n in NODES}

    def compose(self) -> ComposeResult:
        yield Static(self._render_bar(), id="node-bar")
        yield Static("", id="run-status-label")

    def _render_bar(self) -> str:
        parts = []
        for n in NODES:
            sym = _SYMBOLS[self._states[n]]
            parts.append(f"{sym}{_SHORT[n]}")
        return "  ".join(parts)

    def set_node_active(self, node: str) -> None:
        if node in self._states:
            self._states[node] = "active"
            self.query_one("#node-bar", Static).update(self._render_bar())

    def set_node_done(self, node: str, error: str | None = None) -> None:
        if node in self._states:
            self._states[node] = "failed" if error else "done"
            self.query_one("#node-bar", Static).update(self._render_bar())

    def mark_run_end(self, success: bool) -> None:
        """Finalize node states when a run ends.

        If success: all active/pending nodes become done.
        If failure: active nodes become failed; pending stays pending.
        """
        for n in NODES:
            if success:
                if self._states[n] in ("active", "pending"):
                    self._states[n] = "done"
            else:
                if self._states[n] == "active":
                    self._states[n] = "failed"
        self.query_one("#node-bar", Static).update(self._render_bar())
        label = "RUN COMPLETE" if success else "RUN FAILED"
        self.query_one("#run-status-label", Static).update(label)

    def reset(self) -> None:
        self._states = {n: "pending" for n in NODES}
        self.query_one("#node-bar", Static).update(self._render_bar())
        self.query_one("#run-status-label", Static).update("")
