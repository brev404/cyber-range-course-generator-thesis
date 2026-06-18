"""PromptViewerScreen — llm_calls.jsonl reader for a given experiment.

Loads output/experiments/{exp_id}/llm_calls.jsonl.
Shows each LLM call with node name, prompt (truncated in list), token count.
Filter cycling through node names via F key.
Full prompt displayed in a scrollable static area.
Handles missing file gracefully.
"""

from __future__ import annotations

import json
from pathlib import Path

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Footer, Header, Label, ListItem, ListView, Static


def _experiments_base() -> Path:
    return Path(__file__).resolve().parents[3] / "output" / "experiments"


def _load_llm_calls(exp_id: str) -> list[dict]:
    """Load llm_calls.jsonl for the given experiment.

    Returns empty list if missing or malformed.
    """
    path = _experiments_base() / exp_id / "llm_calls.jsonl"
    if not path.exists():
        return []
    records: list[dict] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    except Exception:
        pass
    return records


class PromptViewerScreen(Screen):
    BINDINGS = [
        Binding("f", "cycle_filter", "Filter node"),
        Binding("escape", "app.pop_screen", "Back"),
    ]

    def __init__(self, exp_id: str) -> None:
        super().__init__()
        self._exp_id = exp_id
        self._all_calls: list[dict] = []
        self._visible_calls: list[dict] = []
        self._filter_nodes: list[str] = []
        self._node_filter_idx: int = 0  # 0 = "all"
        self._selected_idx: int = 0

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            with Vertical(id="left-panel"):
                yield Label("", id="calls-header")
                yield ListView(id="calls-list")
            with Vertical(id="right-panel"):
                yield Label("Full prompt", id="detail-title")
                yield Static("Select a call from the list.", id="prompt-detail")
        yield Footer()

    def on_mount(self) -> None:
        self._all_calls = _load_llm_calls(self._exp_id)
        # Build node list for filter cycling
        seen: list[str] = []
        for c in self._all_calls:
            n = c.get("node", "")
            if n and n not in seen:
                seen.append(n)
        self._filter_nodes = seen
        self._apply_filter()

    def _apply_filter(self) -> None:
        if self._node_filter_idx == 0 or not self._filter_nodes:
            self._visible_calls = list(self._all_calls)
            current_filter = "all"
        else:
            node_name = self._filter_nodes[self._node_filter_idx - 1]
            self._visible_calls = [
                c for c in self._all_calls if c.get("node") == node_name
            ]
            current_filter = node_name

        header: Label = self.query_one("#calls-header", Label)
        exp_label = f"[{self._exp_id}]"
        if not self._all_calls:
            header.update(f"{exp_label} No llm_calls.jsonl found.")
        else:
            header.update(
                f"{exp_label} LLM calls — filter: {current_filter} "
                f"({len(self._visible_calls)}/{len(self._all_calls)})"
            )

        lv: ListView = self.query_one("#calls-list", ListView)
        lv.clear()
        if not self._visible_calls:
            lv.append(ListItem(Label("(none)")))
            return

        for i, call in enumerate(self._visible_calls):
            node = call.get("node", "?")
            prompt = call.get("prompt", "")
            prompt_preview = (prompt[:60] + "...") if len(prompt) > 60 else prompt
            # Replace newlines for preview
            prompt_preview = prompt_preview.replace("\n", " ")
            tokens = call.get("response_tokens", [])
            token_count = len(tokens)
            duration = call.get("duration_s", 0)
            lv.append(
                ListItem(
                    Label(
                        f"[{i}] {node}  tok={token_count}  {duration:.1f}s  {prompt_preview}"
                    )
                )
            )

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        self._selected_idx = event.list_view.index or 0
        self._refresh_detail()

    def _refresh_detail(self) -> None:
        detail: Static = self.query_one("#prompt-detail", Static)
        if not self._visible_calls or self._selected_idx >= len(self._visible_calls):
            detail.update("(no call selected)")
            return
        call = self._visible_calls[self._selected_idx]
        node = call.get("node", "?")
        prompt = call.get("prompt", "(empty)")
        tokens = call.get("response_tokens", [])
        duration = call.get("duration_s", 0)
        token_count = len(tokens)
        # Show first 2000 chars of prompt to avoid overwhelming the widget
        prompt_display = prompt[:2000] + ("..." if len(prompt) > 2000 else "")
        detail.update(
            f"Node: {node}\nDuration: {duration:.2f}s\nResponse tokens: {token_count}\n\n"
            f"--- Prompt ---\n{prompt_display}"
        )

    def action_cycle_filter(self) -> None:
        """Cycle through node filter: all -> node1 -> node2 -> ... -> all."""
        if not self._filter_nodes:
            return
        # +1 for "all" option at index 0
        self._node_filter_idx = (self._node_filter_idx + 1) % (
            len(self._filter_nodes) + 1
        )
        self._apply_filter()
        # Reset selection
        self._selected_idx = 0
        self._refresh_detail()
