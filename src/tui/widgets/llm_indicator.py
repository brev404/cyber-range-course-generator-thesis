"""LLMIndicator — shows the currently active LLM node and cumulative token count."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Static


class LLMIndicator(Widget):
    """Displays the active LLM node name and running token count."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._node = ""
        self._tokens = 0

    def compose(self) -> ComposeResult:
        yield Static("", id="llm-indicator-text")

    def set_active(self, node: str) -> None:
        """Set the currently active node and reset token counter."""
        self._node = node
        self._tokens = 0
        self._update()

    def add_token(self) -> None:
        """Increment the token counter by one."""
        self._tokens += 1
        self._update()

    def clear(self) -> None:
        """Clear the indicator (run finished or idle)."""
        self._node = ""
        self._tokens = 0
        self._update()

    def _update(self) -> None:
        text = f"LLM: {self._node}  {self._tokens} tokens" if self._node else ""
        self.query_one("#llm-indicator-text", Static).update(text)
