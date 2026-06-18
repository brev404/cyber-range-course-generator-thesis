"""RunScreen — live pipeline execution with astream and all live-update widgets."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Footer

from src.services.gen_progress import register as _gp_register
from src.services.gen_progress import unregister as _gp_unregister
from src.tui.artifact_writer import ArtifactWriter
from src.tui.callback import PromptCaptureHandler
from src.tui.events import (
    ChallengeScored,
    HITLPaused,
    LLMToken,
    NodeFinished,
    NodeStarted,
    PromptCaptured,
    RunFinished,
    RunStarted,
)
from src.tui.run_config import RunConfig
from src.tui.widgets.challenge_status_table import ChallengeStatusTable
from src.tui.widgets.llm_indicator import LLMIndicator
from src.tui.widgets.node_progress import NodeProgress
from src.tui.widgets.prompt_panel import PromptPanel
from src.tui.widgets.run_list import RunList


class RunScreen(Screen):
    """Screen that streams a live LangGraph pipeline run.

    Launched by RunConfigScreen after a RunConfig is built.  One worker
    per run; multiple concurrent runs are supported (exclusive=False).
    """

    BINDINGS = [
        Binding("p", "toggle_prompt", "Toggle Prompt"),
        Binding("ctrl+c", "cancel_run", "Cancel"),
        Binding("h", "app.show_home", "Home"),
        Binding("e", "app.show_experiments", "Experiments"),
        Binding("escape", "app.pop_screen", "Back"),
    ]

    def __init__(self, cfg: RunConfig) -> None:
        super().__init__()
        self._cfg = cfg
        self._prompt_visible = False
        self._writer: ArtifactWriter | None = None
        self._token_buffer: list[str] = []
        self._current_node: str = ""
        self._node_timings: dict[str, float] = {}
        self._node_start_times: dict[str, float] = {}
        self._last_prompt: str = ""
        self._challenge_ids: list[str] = []
        self._done_challenges: set[str] = set()
        self._seen_courses: set[str] = set()
        self._courses_dir: Path | None = None

    def compose(self) -> ComposeResult:
        with Horizontal(id="run-horizontal"):
            yield RunList(id="run-list")
            with Vertical(id="main-panel"):
                yield NodeProgress(id="node-progress")
                yield ChallengeStatusTable(id="challenge-table")
                yield LLMIndicator(id="llm-indicator")
        yield PromptPanel(id="prompt-panel", classes="hidden")
        yield Footer()

    def on_mount(self) -> None:
        base_dir = Path("output/experiments")
        self._writer = ArtifactWriter(base_dir, self._cfg)
        self._writer.start_run(self._cfg)
        self._courses_dir = base_dir / self._cfg.exp_id / "courses"
        self.set_interval(2.0, self._poll_new_courses)
        _gp_register(self._on_gen_progress)
        self.run_worker(
            self._stream(self._cfg),
            name=self._cfg.exp_id,
            exclusive=False,
        )

    def on_unmount(self) -> None:
        _gp_unregister(self._on_gen_progress)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def action_toggle_prompt(self) -> None:
        panel = self.query_one("#prompt-panel", PromptPanel)
        self._prompt_visible = not self._prompt_visible
        if self._prompt_visible:
            panel.remove_class("hidden")
        else:
            panel.add_class("hidden")

    def action_cancel_run(self) -> None:
        """Cancel all workers on this screen."""
        for worker in self.workers:
            worker.cancel()
        self.notify("Run cancelled.")

    # ------------------------------------------------------------------
    # Per-challenge generation progress callback (called from agent thread)
    # ------------------------------------------------------------------

    def _on_gen_progress(self, challenge_id: str, status: str, error: str) -> None:
        bare = challenge_id.rsplit("/", 1)[-1] if "/" in challenge_id else challenge_id
        if bare in self._done_challenges:
            return
        if status == "start":
            label = f"▶ {bare[:10]}"
        elif status == "failed":
            short_err = error[:18] if error else "?"
            # strip common verbose prefix from llm errors
            for prefix in ("LLM call failed after", "Command '", "claude --print"):
                if short_err.startswith(prefix):
                    short_err = "timeout/err"
                    break
            label = f"failed: {short_err}"
        else:
            return  # "done" is handled by filesystem poll
        self.call_from_thread(self._safe_update_status, bare, label)

    def _safe_update_status(self, bare_id: str, label: str) -> None:
        try:
            self.query_one("#challenge-table", ChallengeStatusTable).update_status(
                bare_id, label
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Course filesystem polling (fires every 2 s)
    # ------------------------------------------------------------------

    def _poll_new_courses(self) -> None:
        """Detect newly written course.md files and mark challenges generated ✓."""
        if self._courses_dir is None or not self._courses_dir.exists():
            return
        tbl = self.query_one("#challenge-table", ChallengeStatusTable)
        for course_file in self._courses_dir.rglob("course.md"):
            # Path: courses/{category}/{name}/course.md → parent.name == bare challenge id
            cid = course_file.parent.name
            if cid not in self._seen_courses:
                self._seen_courses.add(cid)
                if cid not in self._done_challenges:
                    tbl.update_status(cid, "generated ✓")

    # ------------------------------------------------------------------
    # Stream worker
    # ------------------------------------------------------------------

    async def _stream(self, cfg: RunConfig) -> None:
        """Run the LangGraph pipeline asynchronously and dispatch events."""

        from langchain_core.runnables import RunnableConfig as LGConfig

        from src.core.graph import app as graph
        from src.core.state import AgentState
        from src.tui.challenge_loader import load_challenges

        # Discover challenges
        challenges = load_challenges(
            cfg.source,
            cfg.categories if cfg.categories else None,
        )

        # If specific challenge IDs were requested, filter to those
        if cfg.challenge_ids:
            override_ids = {c.strip() for c in cfg.challenge_ids}
            challenges = [c for c in challenges if c.challenge_id in override_ids]

        # Build parallel lists for AgentState
        organized_paths: list[Path] = [c.path for c in challenges]
        challenge_ids: list[str] = [c.challenge_id for c in challenges]

        # If challenge_ids override was given but no challenges found via loader,
        # still record them for UI display purposes (coordinator will handle loading)
        if not challenge_ids and cfg.challenge_ids:
            challenge_ids = [c.strip() for c in cfg.challenge_ids]

        # Register run in the UI
        run_list = self.query_one("#run-list", RunList)
        run_list.add_run(cfg.exp_id)
        self.post_message(
            RunStarted(run_id=cfg.exp_id, challenge_count=len(challenge_ids))
        )

        # Populate challenge table
        table = self.query_one("#challenge-table", ChallengeStatusTable)
        for cid in challenge_ids:
            table.add_challenge(cid)
        self._challenge_ids = list(challenge_ids)
        if self._writer:
            self._writer.update_challenge_ids(self._challenge_ids)

        # Build prompt capture queue and LangGraph config
        prompt_q: asyncio.Queue = asyncio.Queue()
        callback = PromptCaptureHandler(prompt_q)
        lg_config = LGConfig(
            callbacks=[callback],
            configurable={"thread_id": cfg.exp_id},
        )

        # Build initial AgentState.
        # The coordinator agent will skip its own scanning when organized_challenges
        # and challenge_ids are already populated.
        state = AgentState(
            organized_challenges=organized_paths,
            challenge_ids=challenge_ids,
            skip_ranking=cfg.skip_ranking,
            max_hitl_iterations=3,
        )

        # Set per-task provider/model overrides so every LLM call inside this
        # asyncio task (including nested graph nodes) uses the selected model.
        from src.services.gen_progress import set_exp_dir as _gp_set_exp_dir
        from src.services.llm_service import _run_model, _run_provider

        _gp_set_exp_dir(Path("output/experiments") / cfg.exp_id)
        tok_p = _run_provider.set(cfg.provider)
        tok_m = _run_model.set(cfg.model)

        start = time.time()
        try:
            async for part in graph.astream(
                state,
                config=lg_config,
                stream_mode=["updates", "tasks", "messages"],
                version="v2",
            ):
                # Drain prompt queue opportunistically
                while not prompt_q.empty():
                    item = prompt_q.get_nowait()
                    self.post_message(
                        PromptCaptured(
                            run_id=cfg.exp_id,
                            prompt=item.get("prompt", ""),
                        )
                    )

                # Dispatch stream event
                part_type = part.get("type") if isinstance(part, dict) else None
                part_data = part.get("data") if isinstance(part, dict) else part

                if part_type == "tasks":
                    task = part_data if isinstance(part_data, dict) else {}
                    node_name = task.get("name", "")
                    if "result" in task:
                        # TaskResultPayload — node finished
                        self.post_message(
                            NodeFinished(
                                run_id=cfg.exp_id,
                                node_name=node_name,
                                duration_s=0.0,
                                error=task.get("error"),
                            )
                        )
                    else:
                        # TaskPayload — node started
                        self.post_message(
                            NodeStarted(run_id=cfg.exp_id, node_name=node_name)
                        )

                elif part_type == "updates":
                    data = part_data if isinstance(part_data, dict) else {}
                    for node_name, delta in data.items():
                        if node_name == "ranking" and isinstance(delta, dict):
                            for report in delta.get("ranking_reports") or []:
                                if isinstance(report, dict):
                                    self.post_message(
                                        ChallengeScored(
                                            run_id=cfg.exp_id,
                                            challenge_id=report.get("challenge_id", ""),
                                            overall=float(
                                                report.get(
                                                    "overall_score",
                                                    report.get("overall", 0),
                                                )
                                            ),
                                            technical=float(
                                                report.get(
                                                    "technical_score",
                                                    report.get("technical", 0),
                                                )
                                            ),
                                            pedagogical=float(
                                                report.get(
                                                    "pedagogical_score",
                                                    report.get("pedagogical", 0),
                                                )
                                            ),
                                        )
                                    )
                            if self._writer:
                                reports = delta.get("ranking_reports") or []
                                if reports:
                                    self._writer.write_ranking(reports)
                        if isinstance(delta, dict):
                            # Write courses from content_generation output
                            # State field: generated_courses (Dict[str, str])
                            course_data = delta.get("generated_courses")
                            if course_data and isinstance(course_data, dict):
                                tbl = self.query_one(
                                    "#challenge-table", ChallengeStatusTable
                                )
                                for challenge_id, content in course_data.items():
                                    if self._writer:
                                        self._writer.write_course(
                                            challenge_id, str(content)
                                        )
                                    cid = (
                                        challenge_id.rsplit("/", 1)[-1]
                                        if "/" in challenge_id
                                        else challenge_id
                                    )
                                    if cid not in self._done_challenges:
                                        tbl.update_status(cid, "generated ✓")

                elif part_type == "messages":
                    if isinstance(part_data, (list, tuple)) and len(part_data) == 2:
                        msg_chunk, metadata = part_data
                        node_name = (
                            metadata.get("langgraph_node", "")
                            if isinstance(metadata, dict)
                            else ""
                        )
                        token = getattr(msg_chunk, "content", "") or ""
                        self.post_message(
                            LLMToken(
                                run_id=cfg.exp_id,
                                node_name=node_name,
                                token=token,
                            )
                        )

            elapsed = time.time() - start
            self.post_message(
                RunFinished(run_id=cfg.exp_id, success=True, elapsed_s=elapsed)
            )

        except Exception as exc:
            # Check for GraphInterrupt (HITL pause) without hard-importing
            try:
                from langgraph.errors import GraphInterrupt

                if isinstance(exc, GraphInterrupt):
                    self.post_message(HITLPaused(run_id=cfg.exp_id))
                    return
            except ImportError:
                pass

            elapsed = time.time() - start
            self.post_message(
                RunFinished(
                    run_id=cfg.exp_id,
                    success=False,
                    elapsed_s=elapsed,
                    error=str(exc),
                )
            )
        finally:
            _run_provider.reset(tok_p)
            _run_model.reset(tok_m)

    # ------------------------------------------------------------------
    # Message handlers
    # ------------------------------------------------------------------

    _NODE_STATUS_LABELS: dict[str, str] = {
        "coordinator": "loading…",
        "validator": "validating…",
        "content_generation": "generating…",
        "course_terminology_checker": "checking…",
        "mapping": "mapping…",
        "ranking": "scoring…",
        "refinement_step": "refining…",
        "hitl": "awaiting review",
    }

    def on_node_started(self, event: NodeStarted) -> None:
        self.query_one("#node-progress", NodeProgress).set_node_active(event.node_name)
        self.query_one("#llm-indicator", LLMIndicator).set_active(event.node_name)
        self._node_start_times[event.node_name] = time.time()
        self._current_node = event.node_name
        self._token_buffer = []
        label = self._NODE_STATUS_LABELS.get(event.node_name)
        if label:
            tbl = self.query_one("#challenge-table", ChallengeStatusTable)
            for cid in self._challenge_ids:
                if cid not in self._done_challenges:
                    tbl.update_status(cid, label)

    def on_node_finished(self, event: NodeFinished) -> None:
        self.query_one("#node-progress", NodeProgress).set_node_done(
            event.node_name, event.error
        )
        # Compute duration and flush token buffer to artifact writer
        start = self._node_start_times.get(event.node_name)
        duration = (time.time() - start) if start is not None else 0.0
        self._node_timings[event.node_name] = duration
        if self._writer and self._token_buffer:
            self._writer.append_llm_call(
                event.node_name,
                self._last_prompt,
                list(self._token_buffer),
                duration,
            )
        self._token_buffer = []

    def on_llm_token(self, event: LLMToken) -> None:
        self.query_one("#llm-indicator", LLMIndicator).add_token()
        if self._prompt_visible:
            self.query_one("#prompt-panel", PromptPanel).append_token(event.token)
        self._token_buffer.append(event.token)

    def on_prompt_captured(self, event: PromptCaptured) -> None:
        self.query_one("#prompt-panel", PromptPanel).set_prompt(event.prompt)
        self._last_prompt = event.prompt

    def on_challenge_scored(self, event: ChallengeScored) -> None:
        # Ranking reports use "category/name"; table rows use bare "name".
        cid = (
            event.challenge_id.rsplit("/", 1)[-1]
            if "/" in event.challenge_id
            else event.challenge_id
        )
        self._done_challenges.add(cid)
        self.query_one("#challenge-table", ChallengeStatusTable).update_score(
            cid, event.overall
        )

    def on_run_finished(self, event: RunFinished) -> None:
        status = "complete" if event.success else "failed"
        self.query_one("#run-list", RunList).update_run(event.run_id, status)
        self.query_one("#llm-indicator", LLMIndicator).clear()
        self.query_one("#node-progress", NodeProgress).mark_run_end(event.success)
        if not event.success:
            tbl = self.query_one("#challenge-table", ChallengeStatusTable)
            for cid in self._challenge_ids:
                if cid not in self._done_challenges:
                    tbl.update_status(cid, "run failed")
        if self._writer:
            self._writer.finish_run(event.success, self._node_timings)
        elapsed = f"{event.elapsed_s:.1f}s"
        msg = f"Run {event.run_id} {status} in {elapsed}"
        if not event.success and event.error:
            msg += f": {event.error[:80]}"
        self.notify(msg)
