"""MarkdownView — scrollable Markdown widget wrapper."""

from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Markdown


class MarkdownView(Widget):
    """A scrollable wrapper around Textual's Markdown widget."""

    def __init__(self, content: str = "", **kwargs) -> None:
        super().__init__(**kwargs)
        self._content = content

    def compose(self) -> ComposeResult:
        yield Markdown(self._content)

    def update(self, content: str) -> None:
        """Replace displayed markdown content."""
        self._content = content
        self.query_one(Markdown).update(content)
