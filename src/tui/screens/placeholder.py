from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Footer, Static


class PlaceholderScreen(Screen):
    """Generic placeholder used for screens not yet implemented."""

    def __init__(self, label: str = "Coming soon") -> None:
        super().__init__()
        self._label = label

    def compose(self) -> ComposeResult:
        yield Static(self._label, id="title")
        yield Static("This screen is not yet implemented.", id="nav-hint")
        yield Footer()
