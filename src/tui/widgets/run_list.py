"""RunList — ListView of active and recent runs with status badges."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Label, ListItem, ListView


class RunList(Widget):
    """Displays a list of pipeline runs with their current status."""

    def compose(self) -> ComposeResult:
        yield ListView(id="run-list-view")

    def add_run(self, run_id: str, status: str = "running") -> None:
        """Add a new run entry to the list."""
        lv = self.query_one(ListView)
        lv.append(ListItem(Label(f"▶ {run_id}  {status}"), id=f"run-{run_id}"))

    def update_run(self, run_id: str, status: str) -> None:
        """Update an existing run entry's status label."""
        try:
            item = self.query_one(f"#run-{run_id}", ListItem)
            symbol = "✓" if status == "complete" else "✗"
            item.query_one(Label).update(f"{symbol} {run_id}  {status}")
        except Exception:
            pass
