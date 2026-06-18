"""RankingTable — DataTable widget for displaying challenge ranking scores."""

from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import DataTable


class RankingTable(Widget):
    """DataTable widget displaying ranking scores per challenge."""

    def compose(self) -> ComposeResult:
        table = DataTable()
        table.add_columns("Challenge", "Overall", "Technical", "Pedagogical")
        yield table

    def load_reports(self, reports: list[dict]) -> None:
        """Populate the table from a list of ranking report dicts."""
        table = self.query_one(DataTable)
        table.clear()
        for r in reports:
            table.add_row(
                r.get("challenge_id", ""),
                str(r.get("overall_score", r.get("overall", "—"))),
                str(r.get("technical_score", r.get("technical", "—"))),
                str(r.get("pedagogical_score", r.get("pedagogical", "—"))),
            )
