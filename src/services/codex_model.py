"""LangChain-compatible chat model backed by the `codex exec` CLI subprocess.

Runs GPT-5.5 (or another model) via the local `codex` binary in single-shot,
read-only-sandbox mode so the pipeline can use a Codex generator baseline
without an API key. The final answer is read from codex's
``--output-last-message`` file rather than parsed from the transcript on stdout.

Telemetry (llm_calls.jsonl), prompt formatting, and the per-call ContextVars are
REUSED from claude_code_model so codex calls share the same telemetry stream.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Iterator, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from loguru import logger

from src.services.claude_code_model import (
    _format_messages,
    _llm_call_challenge_id,
    _llm_call_role,
    _write_telemetry_entry,
)


def _is_available() -> bool:
    """True if the codex CLI is on PATH."""
    return shutil.which("codex") is not None


class CodexModel(BaseChatModel):
    """Calls `codex exec` via subprocess — no API key required.

    Single-shot, read-only sandbox. The prompt is piped via stdin (the `-` arg);
    the final message is read from a per-call --output-last-message temp file.
    """

    model_name: str = "gpt-5.5"
    timeout: int = 300

    @property
    def _llm_type(self) -> str:
        return "codex"

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
                "codex CLI not found in PATH. Install the Codex CLI to use the "
                "'codex' provider."
            )

        prompt = _format_messages(messages)
        prompt_chars = len(prompt)
        logger.debug("CodexModel: sending {} chars via stdin", prompt_chars)

        fd, out_name = tempfile.mkstemp(suffix=".txt")
        os.close(fd)
        out_path = Path(out_name)
        try:
            cmd = [
                "codex",
                "exec",
                "--sandbox",
                "read-only",
                "--skip-git-repo-check",
                "-m",
                self.model_name,
                "-o",
                str(out_path),
                "-",
            ]
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
                err = (result.stderr or "").strip()
                out = (result.stdout or "").strip()
                detail = err[:500] or out[:500] or "(no stderr)"
                raise ValueError(
                    f"codex exec exited with code {result.returncode}: {detail}"
                )

            content = (
                out_path.read_text(encoding="utf-8").strip()
                if out_path.exists()
                else ""
            )
            if not content:
                raise ValueError("codex exec returned empty response")
        finally:
            out_path.unlink(missing_ok=True)

        response_chars = len(content)
        logger.debug("CodexModel: received {} chars", response_chars)

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
