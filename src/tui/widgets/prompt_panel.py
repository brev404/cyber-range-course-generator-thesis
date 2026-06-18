"""PromptPanel — collapsible panel showing last LLM prompt and live token stream."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widget import Widget
from textual.widgets import Static


class PromptPanel(Widget):
    """Collapsible panel (P key toggle) displaying the last captured prompt and streaming tokens."""

    def compose(self) -> ComposeResult:
        yield Static("Prompt Panel (P to toggle)", id="prompt-header")
        with VerticalScroll():
            yield Static("", id="prompt-text")
            yield Static("", id="token-text")

    def set_prompt(self, prompt: str) -> None:
        """Display a new prompt, truncating to 2000 chars."""
        display = prompt[:2000] + "..." if len(prompt) > 2000 else prompt
        self.query_one("#prompt-text", Static).update(display)
        self.query_one("#token-text", Static).update("")

    def append_token(self, token: str) -> None:
        """Append a streaming token to the live token display."""
        current = self.query_one("#token-text", Static).renderable
        self.query_one("#token-text", Static).update(str(current) + token)

    def clear(self) -> None:
        """Clear both prompt and token displays."""
        self.query_one("#prompt-text", Static).update("")
        self.query_one("#token-text", Static).update("")
