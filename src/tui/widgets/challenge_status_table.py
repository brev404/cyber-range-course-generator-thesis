"""ChallengeStatusTable — DataTable showing challenge_id, status, and score."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import DataTable


class ChallengeStatusTable(Widget):
    """Displays a table of challenges with their pipeline status and ranking score."""

    def compose(self) -> ComposeResult:
        table = DataTable()
        # Explicit string keys so update_cell("rsb", "status", ...) works reliably.
        table.add_column("Challenge", key="challenge")
        table.add_column("Status", key="status", width=20)
        table.add_column("Score", key="score", width=6)
        yield table

    def add_challenge(self, challenge_id: str) -> None:
        """Add a new challenge row in pending state."""
        self.query_one(DataTable).add_row(
            challenge_id, "pending", "—", key=challenge_id
        )

    def update_score(self, challenge_id: str, score: float) -> None:
        """Update a challenge row with its overall ranking score."""
        try:
            table = self.query_one(DataTable)
            table.update_cell(challenge_id, "score", f"{score:.1f}")
            table.update_cell(challenge_id, "status", "done")
        except Exception:
            pass

    def update_status(self, challenge_id: str, status: str) -> None:
        """Update only the status column for a challenge."""
        try:
            self.query_one(DataTable).update_cell(challenge_id, "status", status)
        except Exception:
            pass
