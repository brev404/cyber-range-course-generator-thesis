"""Textual Message subclasses — one per pipeline event.

These are pure data containers. No logic, no I/O.

Textual is an optional dependency. A minimal fallback Message
is provided so the contract tests run without the textual package installed.
"""

from __future__ import annotations

try:
    from textual.message import Message
except ImportError:  # pragma: no cover — textual is an optional dependency

    class Message:  # type: ignore[no-redef]
        """Minimal stub so events.py imports without textual installed."""

        def __init__(self, **kwargs: object) -> None:
            pass


class NodeStarted(Message):
    """Fired when the graph enters a new node."""

    def __init__(self, run_id: str, node_name: str) -> None:
        super().__init__()
        self.run_id = run_id
        self.node_name = node_name


class NodeFinished(Message):
    """Fired when a graph node completes (or fails)."""

    def __init__(
        self,
        run_id: str,
        node_name: str,
        duration_s: float,
        error: str | None = None,
    ) -> None:
        super().__init__()
        self.run_id = run_id
        self.node_name = node_name
        self.duration_s = duration_s
        self.error = error


class LLMToken(Message):
    """One streamed token from an LLM call."""

    def __init__(self, run_id: str, node_name: str, token: str) -> None:
        super().__init__()
        self.run_id = run_id
        self.node_name = node_name
        self.token = token


class PromptCaptured(Message):
    """Full input prompt captured via PromptCaptureHandler callback."""

    def __init__(self, run_id: str, prompt: str) -> None:
        super().__init__()
        self.run_id = run_id
        self.prompt = prompt


class ChallengeScored(Message):
    """Ranking scores for a single challenge, emitted after the ranking node."""

    def __init__(
        self,
        run_id: str,
        challenge_id: str,
        overall: float,
        technical: float,
        pedagogical: float,
    ) -> None:
        super().__init__()
        self.run_id = run_id
        self.challenge_id = challenge_id
        self.overall = overall
        self.technical = technical
        self.pedagogical = pedagogical


class RunStarted(Message):
    """Fired once at the start of a pipeline run."""

    def __init__(self, run_id: str, challenge_count: int) -> None:
        super().__init__()
        self.run_id = run_id
        self.challenge_count = challenge_count


class RunFinished(Message):
    """Fired once when the pipeline run ends (success or failure)."""

    def __init__(
        self,
        run_id: str,
        success: bool,
        elapsed_s: float,
        error: str | None = None,
    ) -> None:
        super().__init__()
        self.run_id = run_id
        self.success = success
        self.elapsed_s = elapsed_s
        self.error = error


class HITLPaused(Message):
    """Fired when the pipeline hits a GraphInterrupt and awaits human input."""

    def __init__(self, run_id: str) -> None:
        super().__init__()
        self.run_id = run_id
