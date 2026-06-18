from textual.app import App
from textual.binding import Binding

from src.tui.events import RunFinished
from src.tui.screens.home import HomeScreen
from src.tui.screens.run_config import RunConfigScreen


class ContextCreatorApp(App):
    CSS_PATH = "tui.tcss"

    # q is the only global binding that must always work.
    # H / R are handled by action methods below so we can use
    # switch_screen (replace) instead of push_screen (stack).
    # RunScreen is always push_screen'd by RunConfigScreen so Esc works.
    BINDINGS = [
        Binding("q", "quit", "Quit"),
    ]

    SCREENS = {
        "home": HomeScreen,
        "run_config": RunConfigScreen,
    }

    def on_mount(self) -> None:
        self.push_screen("home")

    def action_show_home(self) -> None:
        self.switch_screen("home")

    def action_show_run_config(self) -> None:
        self.switch_screen("run_config")

    def on_run_finished(self, event: RunFinished) -> None:
        pass
