"""RunConfigScreen — form that builds a RunConfig dataclass. No pipeline execution here."""

from __future__ import annotations

import re
from pathlib import Path

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.screen import Screen
from textual.widgets import (
    Button,
    Checkbox,
    Footer,
    Input,
    Label,
    RadioButton,
    RadioSet,
    Static,
)

from src.tui.challenge_loader import get_available_categories
from src.tui.run_config import RunConfig

# (display_label, provider, model_id)
# "claude-code" entries use the local `claude --print` CLI (Pro/Max subscription).
# The model_id is passed as `--model <id>` so generation AND ranking use the same model.
_MODEL_CHOICES: list[tuple[str, str, str]] = [
    # ── Local CLI (Claude Code) ──────────────────────────────────────────
    ("claude-code · sonnet-4-6", "claude-code", "claude-sonnet-4-6"),
    ("claude-code · haiku-4-5", "claude-code", "claude-haiku-4-5-20251001"),
    ("claude-code · opus-4-7", "claude-code", "claude-opus-4-7"),
    # ── Anthropic API ───────────────────────────────────────────────────
    ("anthropic · sonnet-4-6", "anthropic", "claude-sonnet-4-6"),
    ("anthropic · haiku-4-5", "anthropic", "claude-haiku-4-5-20251001"),
    ("anthropic · opus-4-7", "anthropic", "claude-opus-4-7"),
    # ── Google ──────────────────────────────────────────────────────────
    ("gemini-2.5-flash", "google", "gemini-2.5-flash"),
    # ── OpenRouter (free tier) ──────────────────────────────────────────
    ("gemma-4-27b (openrouter)", "openrouter", "google/gemma-4-27b-it:free"),
    # ── OpenAI ──────────────────────────────────────────────────────────
    ("gpt-4o", "openai", "gpt-4o"),
    ("gpt-4o-mini", "openai", "gpt-4o-mini"),
]


def _model_label_to_provider_model(label: str) -> tuple[str, str]:
    for lbl, provider, model_id in _MODEL_CHOICES:
        if lbl == label:
            return provider, model_id
    return "openai", label  # fallback for custom


def _derive_provider_from_override(override: str) -> str:
    """Derive provider string from a free-text model override.

    Priority: claude → anthropic, gemma → openrouter, gemini → google, else openai.
    """
    if "claude" in override:
        return "anthropic"
    if "gemma" in override:
        return "openrouter"
    if "gemini" in override:
        return "google"
    return "openai"


# Path to output/experiments/ — two levels up from src/tui/
_EXPERIMENTS_DIR = Path(__file__).parent.parent.parent.parent / "output" / "experiments"


def _next_exp_id() -> str:
    """Scan output/experiments/ and return the next sequential run-NNN id."""
    if not _EXPERIMENTS_DIR.exists():
        return "run-001"

    pattern = re.compile(r"^run-(\d+)$")
    highest = 0
    for entry in _EXPERIMENTS_DIR.iterdir():
        if entry.is_dir():
            m = pattern.match(entry.name)
            if m:
                n = int(m.group(1))
                if n > highest:
                    highest = n
    return f"run-{highest + 1:03d}"


class RunConfigScreen(Screen):
    """Form screen for configuring a pipeline run.

    On submit, builds a RunConfig and notifies the user (wired to push RunScreen
    on the next iteration).
    """

    BINDINGS = [
        Binding("h", "app.show_home", "Home"),
        Binding("e", "app.show_experiments", "Experiments"),
        Binding("escape", "app.pop_screen", "Back"),
    ]

    def compose(self) -> ComposeResult:
        yield Static("New Run Configuration", id="run-config-title")

        with VerticalScroll(id="form-scroll"):
            # --- Experiment ID --------------------------------------------------
            yield Label("Experiment ID")
            yield Input(
                placeholder="EXP-NNN",
                id="exp-id",
            )

            # --- Model (provider is derived from selection) ---------------------
            yield Label("Model")
            yield RadioSet(
                *[
                    RadioButton(lbl, value=(i == 0))
                    for i, (lbl, _, _) in enumerate(_MODEL_CHOICES)
                ],
                id="model-radio",
            )
            yield Label("Custom model override (leave blank to use selection above)")
            yield Input(
                placeholder="e.g. claude-sonnet-4-6 / gpt-4o", id="model-override"
            )

            # --- Numeric parameters --------------------------------------------
            yield Label("Temperature")
            yield Input(value="0.0", id="temperature-input")

            yield Label("Ranking threshold")
            yield Input(value="9.0", id="threshold-input")

            yield Label("Max refinements")
            yield Input(value="5", id="max-refinements-input")

            # --- Challenge source -----------------------------------------------
            yield Label("Challenge source")
            yield RadioSet(
                RadioButton("local", value=True),
                RadioButton("processed"),
                id="source-radio",
            )

            # --- Category checkboxes -------------------------------------------
            yield Label("Categories (empty = all)")
            yield Static(id="category-container")

            # --- Challenge IDs override ----------------------------------------
            yield Label("Challenge IDs override (comma-separated, optional)")
            yield Input(
                placeholder="e.g. rsb, careflow",
                id="challenge-ids-input",
            )

            # --- Skip ranking --------------------------------------------------
            yield Checkbox("Skip ranking", id="skip-ranking")

        # Buttons outside the scroll so they are always visible.
        with Horizontal(id="button-row"):
            yield Button("Launch Run", id="submit", variant="primary")
            yield Button("Cancel", id="cancel", variant="default")

        yield Footer()

    def on_mount(self) -> None:
        # Pre-fill next sequential experiment ID
        self.query_one("#exp-id", Input).value = _next_exp_id()
        # Populate categories for the default source
        self._populate_categories("local")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _populate_categories(self, source: str) -> None:
        """Clear and repopulate the category-container with Checkboxes."""
        container = self.query_one("#category-container", Static)
        # Remove existing checkboxes
        for child in list(container.children):
            child.remove()
        # Add fresh ones
        cats = get_available_categories(source)
        for cat in cats:
            container.mount(Checkbox(cat, id=f"cat-{cat}"))

    def _selected_categories(self) -> list[str]:
        """Return category names whose Checkbox is checked."""
        selected: list[str] = []
        for checkbox in self.query(Checkbox):
            # skip the skip-ranking checkbox
            if checkbox.id == "skip-ranking":
                continue
            if checkbox.value:
                # strip the "cat-" prefix that was added as the widget id
                cat_name = (
                    checkbox.label.plain
                    if hasattr(checkbox.label, "plain")
                    else str(checkbox.label)
                )
                selected.append(cat_name)
        return selected

    def _selected_model_and_provider(self) -> tuple[str, str]:
        """Return (model_id, provider) from the current form state."""
        override = self.query_one("#model-override", Input).value.strip()
        if override:
            # derive provider from override text if possible, else default anthropic
            for lbl, provider, model_id in _MODEL_CHOICES:
                if override == model_id or override == lbl:
                    return model_id, provider
            return override, _derive_provider_from_override(override)

        radio_set = self.query_one("#model-radio", RadioSet)
        btn = radio_set.pressed_button
        if btn is None:
            return _MODEL_CHOICES[0][2], _MODEL_CHOICES[0][1]
        lbl = btn.label.plain if hasattr(btn.label, "plain") else str(btn.label)
        provider, model_id = _model_label_to_provider_model(lbl)
        return model_id, provider

    def _source_label(self) -> str:
        radio_set = self.query_one("#source-radio", RadioSet)
        btn = radio_set.pressed_button
        if btn is None:
            return "local"
        label = btn.label
        return label.plain if hasattr(label, "plain") else str(label)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        """React to source changes (model radio needs no secondary action)."""
        if event.radio_set.id == "source-radio":
            source = (
                event.pressed.label.plain
                if hasattr(event.pressed.label, "plain")
                else str(event.pressed.label)
            )
            self._populate_categories(source)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel":
            self.app.pop_screen()
            return

        if event.button.id != "submit":
            return

        # --- Validate --------------------------------------------------
        exp_id = self.query_one("#exp-id", Input).value.strip()
        if not exp_id:
            self.notify("Experiment ID cannot be empty.", severity="error")
            return

        model, provider = self._selected_model_and_provider()
        if not model:
            self.notify("Model cannot be empty.", severity="error")
            return

        try:
            temperature = float(
                self.query_one("#temperature-input", Input).value.strip()
            )
        except ValueError:
            self.notify(
                "Temperature must be a valid float (e.g. 0.0).", severity="error"
            )
            return

        try:
            threshold = float(self.query_one("#threshold-input", Input).value.strip())
        except ValueError:
            self.notify("Threshold must be a valid float (e.g. 9.0).", severity="error")
            return

        try:
            max_refinements = int(
                self.query_one("#max-refinements-input", Input).value.strip()
            )
        except ValueError:
            self.notify("Max refinements must be an integer.", severity="error")
            return

        source = self._source_label()
        categories = self._selected_categories()

        ids_raw = self.query_one("#challenge-ids-input", Input).value.strip()
        challenge_ids: list[str] = (
            [cid.strip() for cid in ids_raw.split(",") if cid.strip()]
            if ids_raw
            else []
        )

        skip_ranking = self.query_one("#skip-ranking", Checkbox).value

        # --- Build RunConfig -------------------------------------------
        cfg = RunConfig(
            exp_id=exp_id,
            provider=provider,
            model=model,
            temperature=temperature,
            threshold=threshold,
            challenge_ids=challenge_ids,
            categories=categories,
            source=source,  # type: ignore[arg-type]
            max_refinements=max_refinements,
            skip_ranking=skip_ranking,
        )

        from src.tui.screens.run import RunScreen

        self.app.push_screen(RunScreen(cfg))
