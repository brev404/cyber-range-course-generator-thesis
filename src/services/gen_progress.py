"""Per-challenge progress bus for content_generation node.

The agent calls report() after each challenge starts or finishes.
The TUI registers a callback and updates the UI from those calls.
Thread-safe: callbacks fire from agent threads.

Also provides _exp_output_dir ContextVar so the agent can write courses
to disk immediately (enabling the TUI filesystem poll to show progress
per-challenge rather than waiting for the whole node to finish).
asyncio.run_in_executor copies the context snapshot, so the ContextVar
set in the async _stream worker is visible in the thread pool.
"""

from __future__ import annotations

import threading
from contextvars import ContextVar
from pathlib import Path
from typing import Callable

_lock = threading.Lock()
_callbacks: list[Callable[[str, str, str], None]] = []

# Set by the TUI before graph.astream(); read by the agent to write
# courses to disk as they finish (not waiting for the whole node).
_exp_output_dir: ContextVar[Path | None] = ContextVar("_exp_output_dir", default=None)


def set_exp_dir(path: Path) -> None:
    _exp_output_dir.set(path)


def get_exp_dir() -> Path | None:
    return _exp_output_dir.get()


def register(cb: Callable[[str, str, str], None]) -> None:
    with _lock:
        _callbacks.append(cb)


def unregister(cb: Callable[[str, str, str], None]) -> None:
    with _lock:
        try:
            _callbacks.remove(cb)
        except ValueError:
            pass


def report(challenge_id: str, status: str, error: str = "") -> None:
    """Called by the agent. status: 'start' | 'done' | 'failed'."""
    with _lock:
        cbs = list(_callbacks)
    for cb in cbs:
        try:
            cb(challenge_id, status, error)
        except Exception:
            pass
