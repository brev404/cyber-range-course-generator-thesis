"""RankingViewerScreen — DataTable of ranking scores for a given experiment.

Loads output/experiments/{exp_id}/ranking_reports.json.
Handles missing file gracefully.
"""

from __future__ import annotations

import json
from pathlib import Path

from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import Screen
from textual.widgets import Footer, Header, Label

from src.tui.widgets.ranking_table import RankingTable


def _experiments_base() -> Path:
    return Path(__file__).resolve().parents[3] / "output" / "experiments"


def _load_ranking_reports(exp_id: str) -> list[dict]:
    """Load ranking_reports.json for the given experiment.

    Returns empty list if missing or malformed.
    """
    path = _experiments_base() / exp_id / "ranking_reports.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            # Some formats wrap the list in a key
            for v in data.values():
                if isinstance(v, list):
                    return v
        return []
    except Exception:
        return []


class RankingViewerScreen(Screen):
    BINDINGS = [
        Binding("escape", "app.pop_screen", "Back"),
    ]

    def __init__(self, exp_id: str) -> None:
        super().__init__()
        self._exp_id = exp_id

    def compose(self) -> ComposeResult:
        yield Header()
        yield Label("", id="ranking-header")
        yield RankingTable(id="ranking-table")
        yield Footer()

    def on_mount(self) -> None:
        reports = _load_ranking_reports(self._exp_id)
        header: Label = self.query_one("#ranking-header", Label)
        table: RankingTable = self.query_one("#ranking-table", RankingTable)

        if not reports:
            header.update(
                f"[{self._exp_id}] No ranking reports found (ranking_reports.json missing or empty)."
            )
            return

        header.update(f"[{self._exp_id}] Ranking — {len(reports)} challenges")
        table.load_reports(reports)
