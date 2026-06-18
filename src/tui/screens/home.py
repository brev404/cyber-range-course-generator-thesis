from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import Screen
from textual.widgets import Footer, Static


class HomeScreen(Screen):
    BINDINGS = [
        Binding("h", "app.show_home", "Home"),
        Binding("r", "app.show_run_config", "Run Config"),
    ]

    def compose(self) -> ComposeResult:
        yield Static("Context Creator TUI", id="title")
        yield Static("[H] Home  [R] Run Config  [Q] Quit", id="nav-hint")
        yield Footer()
