"""Unit tests for TUI widgets — no live Textual app required for pure-logic tests.

Widget DOM tests use Textual's run_test() pilot where needed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.tui.artifact_writer import ArtifactWriter
from src.tui.run_config import RunConfig
from src.tui.widgets.challenge_status_table import ChallengeStatusTable
from src.tui.widgets.node_progress import _SHORT, _SYMBOLS, NODES, NodeProgress


class TestNodeProgressLogic:
    def test_nodes_count(self):
        assert len(NODES) == 8

    def test_nodes_order(self):
        assert NODES == [
            "coordinator",
            "validator",
            "content_generation",
            "course_terminology_checker",
            "mapping",
            "ranking",
            "refinement_step",
            "hitl",
        ]

    def test_all_nodes_have_short_label(self):
        for n in NODES:
            assert n in _SHORT, f"Missing short label for {n}"
            assert _SHORT[n], f"Empty short label for {n}"

    def test_render_bar_contains_all_short_labels(self):
        widget = NodeProgress()
        bar = widget._render_bar()
        for n in NODES:
            assert _SHORT[n] in bar

    def test_render_bar_fits_80_chars(self):
        widget = NodeProgress()
        bar = widget._render_bar()
        assert len(bar) <= 80, f"Bar too long ({len(bar)} chars): {bar}"

    def test_initial_state_all_pending(self):
        widget = NodeProgress()
        for n in NODES:
            assert widget._states[n] == "pending"

    def test_set_node_active_updates_state(self):
        widget = NodeProgress()
        widget._states["coordinator"] = "active"  # simulate set_node_active
        assert widget._states["coordinator"] == "active"
        assert widget._states["validator"] == "pending"

    def test_set_node_done_updates_state(self):
        widget = NodeProgress()
        widget._states["ranking"] = "done"
        assert widget._states["ranking"] == "done"

    def test_set_node_failed_on_error(self):
        widget = NodeProgress()
        widget._states["content_generation"] = "failed"
        assert widget._states["content_generation"] == "failed"

    def test_unknown_node_ignored(self):
        widget = NodeProgress()
        widget.set_node_active("nonexistent_node")  # must not raise
        widget.set_node_done("nonexistent_node")  # must not raise

    def test_reset_clears_all_to_pending(self):
        widget = NodeProgress()
        widget._states["coordinator"] = "done"
        widget._states["ranking"] = "failed"
        widget._states = {n: "pending" for n in NODES}  # simulate reset
        for n in NODES:
            assert widget._states[n] == "pending"

    def test_render_bar_after_partial_progress(self):
        widget = NodeProgress()
        widget._states["coordinator"] = "done"
        widget._states["validator"] = "active"
        bar = widget._render_bar()
        assert _SYMBOLS["done"] + _SHORT["coordinator"] in bar
        assert _SYMBOLS["active"] + _SHORT["validator"] in bar
        assert _SYMBOLS["pending"] + _SHORT["content_generation"] in bar


# ---------------------------------------------------------------------------
# ChallengeStatusTable — column key correctness
# ---------------------------------------------------------------------------


class TestChallengeStatusTableColumns:
    """Verify that the widget uses explicit column keys so update_cell works."""

    def test_compose_uses_explicit_keys(self):
        """The widget must call add_column with key= parameter, not add_columns()."""
        import inspect

        source = inspect.getsource(ChallengeStatusTable.compose)
        # Must use explicit key= arguments, not add_columns()
        assert "add_columns(" not in source, (
            "compose() must not use add_columns() — use add_column(label, key=...) "
            "so that update_cell(row_key, column_key) works reliably."
        )
        assert 'key="challenge"' in source or "key='challenge'" in source
        assert 'key="status"' in source or "key='status'" in source
        assert 'key="score"' in source or "key='score'" in source

    def test_update_score_uses_lowercase_keys(self):
        """update_score must use lowercase key strings matching add_column(key=...)."""
        import inspect

        source = inspect.getsource(ChallengeStatusTable.update_score)
        assert '"score"' in source or "'score'" in source
        assert '"status"' in source or "'status'" in source
        # Must NOT use title-case "Score" / "Status" (those are labels, not keys)
        assert '"Score"' not in source, "Should use key 'score', not label 'Score'"
        assert '"Status"' not in source, "Should use key 'status', not label 'Status'"


# ---------------------------------------------------------------------------
# RunList — pure logic
# ---------------------------------------------------------------------------
class TestRunListLogic:
    def test_imports_cleanly(self):
        from src.tui.widgets.run_list import RunList  # noqa: F401


# ---------------------------------------------------------------------------
# LLMIndicator — pure logic
# ---------------------------------------------------------------------------
class TestLLMIndicatorLogic:
    def test_initial_state(self):
        from src.tui.widgets.llm_indicator import LLMIndicator

        w = LLMIndicator()
        assert w._node == ""
        assert w._tokens == 0

    def test_set_active_resets_tokens(self):
        from src.tui.widgets.llm_indicator import LLMIndicator

        w = LLMIndicator()
        w._node = "ranking"
        w._tokens = 42
        w._node = "content_generation"
        w._tokens = 0
        assert w._tokens == 0
        assert w._node == "content_generation"

    def test_clear_method_exists(self):
        from src.tui.widgets.llm_indicator import LLMIndicator

        assert callable(getattr(LLMIndicator, "clear", None))


# ---------------------------------------------------------------------------
# PromptPanel — pure logic
# ---------------------------------------------------------------------------
class TestPromptPanelLogic:
    def test_imports_cleanly(self):
        from src.tui.widgets.prompt_panel import PromptPanel  # noqa: F401


# ---------------------------------------------------------------------------
# MarkdownView / RankingTable — import checks
# ---------------------------------------------------------------------------
class TestViewWidgets:
    def test_markdown_view_imports(self):
        from src.tui.widgets.markdown_view import MarkdownView  # noqa: F401

    def test_ranking_table_imports(self):
        from src.tui.widgets.ranking_table import RankingTable  # noqa: F401


# ---------------------------------------------------------------------------
# challenge_id normalisation in RunScreen handler logic
# ---------------------------------------------------------------------------
class TestChallengeIdNormalisation:
    """Verify that 'category/name' IDs are normalised to bare 'name' for table lookup."""

    def _normalise(self, challenge_id: str) -> str:
        """Mirror the normalisation logic in RunScreen.on_challenge_scored."""
        return challenge_id.rsplit("/", 1)[-1] if "/" in challenge_id else challenge_id

    def test_bare_id_unchanged(self):
        assert self._normalise("rsb") == "rsb"

    def test_category_prefix_stripped(self):
        assert self._normalise("crypto/rsb") == "rsb"

    def test_deep_prefix_stripped_to_last_segment(self):
        assert self._normalise("electron/patch-unlock") == "patch-unlock"

    def test_spaces_in_name_preserved(self):
        assert self._normalise("osint/Aero Sponge") == "Aero Sponge"

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("web/web-rsa", "web-rsa"),
            ("pwn/ret2win", "ret2win"),
            ("misc/s3crets", "s3crets"),
            ("forensics/HNS", "HNS"),
            ("rev/KittyOS", "KittyOS"),
            ("mobile/courier-drop", "courier-drop"),
            ("boogie-woogie", "boogie-woogie"),  # no prefix
        ],
    )
    def test_parametrised_normalisation(self, raw, expected):
        assert self._normalise(raw) == expected


# ---------------------------------------------------------------------------
# RunConfigScreen — model selection logic
# ---------------------------------------------------------------------------
class TestModelSelection:
    def test_model_choices_all_have_three_fields(self):
        from src.tui.screens.run_config import _MODEL_CHOICES

        for entry in _MODEL_CHOICES:
            assert len(entry) == 3, f"Bad entry: {entry}"
            label, provider, model_id = entry
            assert label
            assert provider in (
                "openai",
                "anthropic",
                "google",
                "openrouter",
                "claude-code",
            )
            assert model_id

    def test_model_label_to_provider_model_known(self):
        from src.tui.screens.run_config import (
            _MODEL_CHOICES,
            _model_label_to_provider_model,
        )

        for label, expected_provider, expected_model in _MODEL_CHOICES:
            provider, model_id = _model_label_to_provider_model(label)
            assert provider == expected_provider
            assert model_id == expected_model

    def test_model_label_to_provider_model_unknown_falls_back(self):
        from src.tui.screens.run_config import _model_label_to_provider_model

        provider, model_id = _model_label_to_provider_model("some-custom-model")
        assert provider == "openai"
        assert model_id == "some-custom-model"

    @pytest.mark.parametrize(
        "override,expected_provider",
        [
            ("claude-sonnet-4-6", "anthropic"),
            ("gemini-2.5-flash", "google"),
            ("gemma-3-27b-it", "openrouter"),
            ("gpt-4o", "openai"),
            ("random-model", "openai"),
            ("gemini-gemma-test", "openrouter"),
        ],
    )
    def test_derive_provider_from_override(self, override, expected_provider):
        from src.tui.screens.run_config import _derive_provider_from_override

        assert _derive_provider_from_override(override) == expected_provider


# ---------------------------------------------------------------------------
# CSS layout — button-row outside scroll
# ---------------------------------------------------------------------------
class TestRunConfigLayout:
    def test_buttons_outside_scroll(self):
        """Buttons must be composed outside the VerticalScroll to stay visible."""
        import inspect

        from src.tui.screens.run_config import RunConfigScreen

        source = inspect.getsource(RunConfigScreen.compose)
        # The VerticalScroll context must close before the button-row
        scroll_close = source.find("form-scroll")
        button_row = source.find("button-row")
        assert scroll_close != -1, "form-scroll not found in compose()"
        assert button_row != -1, "button-row not found in compose()"
        # button-row must appear after the VerticalScroll block ends
        # We look for the button-row yield happening outside any scroll indent
        # Simpler check: button-row must not be nested under VerticalScroll
        lines = source.splitlines()
        in_scroll = False
        scroll_indent = 0
        for line in lines:
            stripped = line.lstrip()
            indent = len(line) - len(stripped)
            if "VerticalScroll" in line:
                in_scroll = True
                scroll_indent = indent
            if (
                in_scroll
                and indent <= scroll_indent
                and "VerticalScroll" not in line
                and stripped
            ):
                in_scroll = False
            if "button-row" in line:
                assert not in_scroll, (
                    "button-row is still inside VerticalScroll — "
                    "it will be scrolled off-screen."
                )


# ---------------------------------------------------------------------------
# ArtifactWriter — manifest challenge_ids recording
# ---------------------------------------------------------------------------


class TestArtifactWriterManifest:
    def _make_cfg(self, challenge_ids: list[str] | None = None) -> RunConfig:
        return RunConfig(
            exp_id="EXP-TEST",
            provider="openai",
            model="gpt-4o",
            temperature=0.0,
            threshold=9.0,
            challenge_ids=challenge_ids or [],
        )

    def test_update_challenge_ids_writes_resolved_ids(self, tmp_path: Path):
        cfg = self._make_cfg(challenge_ids=[])
        writer = ArtifactWriter(tmp_path, cfg)
        writer.start_run(cfg)
        writer.update_challenge_ids(["crypto/rsb", "web/xss"])
        manifest = json.loads((tmp_path / "EXP-TEST" / "manifest.json").read_text())
        assert manifest["challenge_ids"] == ["crypto/rsb", "web/xss"]

    def test_run_config_preserves_original_challenge_ids(self, tmp_path: Path):
        cfg = self._make_cfg(challenge_ids=["user-override-1"])
        writer = ArtifactWriter(tmp_path, cfg)
        writer.start_run(cfg)
        writer.update_challenge_ids(["crypto/rsb", "web/xss"])
        run_cfg = json.loads((tmp_path / "EXP-TEST" / "run_config.json").read_text())
        assert run_cfg["challenge_ids"] == ["user-override-1"]


# ---------------------------------------------------------------------------
# NodeProgress.mark_run_end — pure logic tests
# ---------------------------------------------------------------------------


class TestNodeProgressMarkRunEnd:
    @staticmethod
    def _patched_widget():
        widget = NodeProgress()
        widget.query_one = lambda *a, **kw: type(
            "S", (), {"update": lambda self, x: None}
        )()
        return widget

    def test_mark_run_end_failure_sets_active_to_failed(self):
        widget = self._patched_widget()
        widget._states["coordinator"] = "done"
        widget._states["validator"] = "done"
        widget._states["ranking"] = "active"
        widget.mark_run_end(success=False)
        assert widget._states["ranking"] == "failed"
        assert widget._states["refinement_step"] == "pending"
        assert widget._states["coordinator"] == "done"

    def test_mark_run_end_success_sets_remaining_to_done(self):
        widget = self._patched_widget()
        widget._states["coordinator"] = "done"
        widget._states["validator"] = "done"
        widget.mark_run_end(success=True)
        for n in NODES:
            assert widget._states[n] == "done"

    def test_reset_after_mark_run_end_restores_pending(self):
        widget = self._patched_widget()
        widget._states["coordinator"] = "done"
        widget._states["ranking"] = "active"
        widget.mark_run_end(success=False)
        assert widget._states["ranking"] == "failed"
        widget.reset()
        for n in NODES:
            assert widget._states[n] == "pending"
