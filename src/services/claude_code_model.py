"""LangChain-compatible chat model backed by the `claude -p` CLI subprocess.

Delegates LLM calls to the local `claude` binary (Claude Code) so the
pipeline uses your Claude Pro/Max subscription instead of the API.

Streaming is not supported (the CLI exits after printing the full response),
so the TUI's token counter will stay at 0, but ranking and artifact writes
work normally.

G — Per-call telemetry: each _generate call appends one JSON line to
    <telemetry_dir>/llm_calls.jsonl when a telemetry_dir is configured.
    Set via set_telemetry_dir(path) and reset with set_telemetry_dir(None).
    Use set_llm_call_role("judge-tech") / set_telemetry_challenge_id("crypto/x")
    as ContextVars (thread-safe for concurrent workers).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from contextvars import ContextVar
from pathlib import Path
from threading import Lock
from typing import Any, Iterator, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from loguru import logger

# ---------------------------------------------------------------------------
# Telemetry ContextVars — thread-safe; each worker context carries its own values
# ---------------------------------------------------------------------------

#: Role of the current LLM call: gen-course / gen-solver / judge-tech / judge-ped / other
_llm_call_role: ContextVar[str] = ContextVar("_llm_call_role", default="other")
#: Challenge ID being processed in the current thread/task context
_llm_call_challenge_id: ContextVar[str] = ContextVar(
    "_llm_call_challenge_id", default=""
)

# Process-level telemetry directory (not a ContextVar — shared across all threads)
_telemetry_dir: Optional[Path] = None
_telemetry_dir_lock = Lock()
_telemetry_file_lock = Lock()


def set_telemetry_dir(path: Optional[Path | str]) -> None:
    """Set the directory where llm_calls.jsonl will be written.

    Args:
        path: Directory path (must exist or will be created). Pass None to disable.
    """
    global _telemetry_dir
    with _telemetry_dir_lock:
        if path is None:
            _telemetry_dir = None
        else:
            _telemetry_dir = Path(path)


def set_llm_call_role(role: str) -> None:
    """Set the role label for subsequent LLM calls in the current thread/task context.

    Valid roles: gen-course, gen-solver, judge-tech, judge-ped, other.
    """
    _llm_call_role.set(role)


def set_telemetry_challenge_id(challenge_id: str) -> None:
    """Set the challenge_id label for subsequent LLM calls in the current thread/task."""
    _llm_call_challenge_id.set(challenge_id)


def _write_telemetry_entry(entry: dict) -> None:
    """Append one JSON line to llm_calls.jsonl. No-op if telemetry_dir is None."""
    with _telemetry_dir_lock:
        tdir = _telemetry_dir

    if tdir is None:
        return

    try:
        tdir.mkdir(parents=True, exist_ok=True)
        jsonl_path = tdir / "llm_calls.jsonl"
        line = json.dumps(entry, ensure_ascii=False)
        with _telemetry_file_lock:
            with open(jsonl_path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
    except Exception as exc:  # never let telemetry crash the pipeline
        logger.debug("Telemetry write failed: {}", exc)


def _is_available() -> bool:
    return shutil.which("claude") is not None


def _format_messages(messages: List[BaseMessage]) -> str:
    """Flatten LangChain messages into a single prompt string for stdin."""
    parts: list[str] = []
    for msg in messages:
        role = (
            type(msg).__name__.replace("Message", "").lower()
        )  # "system", "human", "ai"
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        if role == "system":
            parts.append(f"<system>\n{content}\n</system>")
        elif role == "human":
            parts.append(content)
        else:
            parts.append(f"[{role}]\n{content}")
    return "\n\n".join(parts)


class ClaudeCodeModel(BaseChatModel):
    """Calls `claude --print` via subprocess — no API key required.

    Uses the `claude` CLI that comes with Claude Code.  The full prompt is
    piped via stdin to avoid shell argument-length limits on large prompts.
    """

    model_name: str = "claude-code"
    timeout: int = 300

    @property
    def _llm_type(self) -> str:
        return "claude-code"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model_name": self.model_name}

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        if not _is_available():
            raise ValueError(
                "claude CLI not found in PATH. "
                "Install Claude Code: https://claude.ai/code"
            )

        prompt = _format_messages(messages)
        prompt_chars = len(prompt)
        logger.debug("ClaudeCodeModel: sending {} chars via stdin", prompt_chars)

        cmd = ["claude", "--print"]
        # Pass explicit model when specified (e.g. "claude-sonnet-4-6").
        # Fall back to whatever Claude Code's default is when model_name=="claude-code".
        if self.model_name and self.model_name != "claude-code":
            cmd.extend(["--model", self.model_name])

        t0 = time.monotonic()
        result = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=self.timeout,
        )
        duration_ms = int((time.monotonic() - t0) * 1000)

        if result.returncode != 0:
            # claude --print writes its error text to STDOUT (stderr is usually
            # empty), so surface stdout when stderr is absent. Only a genuinely
            # silent failure (no stderr AND no stdout) keeps the "(no stderr)"
            # marker that the quota detector treats as a real usage-limit exit.
            err = (result.stderr or "").strip()
            out = (result.stdout or "").strip()
            detail = err[:500] or out[:500] or "(no stderr)"
            raise ValueError(
                f"claude --print exited with code {result.returncode}: {detail}"
            )

        content = result.stdout.strip()
        if not content:
            raise ValueError("claude --print returned empty response")

        response_chars = len(content)
        logger.debug("ClaudeCodeModel: received {} chars", response_chars)

        # G — write telemetry entry.  Persisting the full response text enables
        # downstream re-parsing (e.g. scripts/recover_dim_scores.py) when judge
        # output is later found to need recovery (truncated, quota-failed, etc.).
        _write_telemetry_entry(
            {
                "ts": time.time(),
                "challenge_id": _llm_call_challenge_id.get(""),
                "role": _llm_call_role.get("other"),
                "prompt_chars": prompt_chars,
                "response_chars": response_chars,
                "duration_ms": duration_ms,
                "model": self.model_name,
                "response": content,
            }
        )

        if run_manager:
            run_manager.on_llm_new_token(content)
        message = AIMessage(content=content)
        return ChatResult(generations=[ChatGeneration(message=message)])

    # LangGraph's astream uses stream() when available; fall back to _generate.
    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        result = self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        content = result.generations[0].message.content
        if run_manager:
            run_manager.on_llm_new_token(content)
        yield ChatGenerationChunk(message=AIMessageChunk(content=content))
