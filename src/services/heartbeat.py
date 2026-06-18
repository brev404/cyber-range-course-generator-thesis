"""Heartbeat status file for external monitors.

Module-level registry for the active HeartbeatState so agent code can update
phase/completed_challenges without needing a dependency injection chain.
Call `set_active_state()` after creating a HeartbeatState to register it, and
`get_active_state()` from agent code.  When no run is active the getter
returns None; agents must guard against that case.

Spawns a daemon thread that writes output/jobs/_run_<run_id>.heartbeat.json
every INTERVAL seconds so watchdogs and monitoring scripts can check liveness
without grepping logs.

Schema:
    {
        "exp_id": "EXP-EXAMPLE",
        "pid": 12345,
        "started_at": "2026-05-26T14:30:00+03:00",
        "last_update": "2026-05-26T14:38:00+03:00",
        "last_llm_call_at": "2026-05-26T14:37:42+03:00",
        "completed_challenges": 12,
        "total_challenges": 29,
        "current_phase": "content_generation"
    }

Writes are atomic (write .tmp + rename) to avoid corrupt reads.

Usage:
    from src.services.heartbeat import HeartbeatThread, HeartbeatState

    state = HeartbeatState(exp_id="EXP-EXAMPLE", total_challenges=29)
    thread = HeartbeatThread(jobs_dir=Path("output/jobs"), state=state)
    thread.start()

    # Update from agents/pipeline:
    state.current_phase = "ranking"
    state.completed_challenges = 12

    # On clean exit:
    thread.stop()
    thread.join()
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Module-level active-state registry
# ---------------------------------------------------------------------------
# Agents import get_active_state() to update phase/completed_challenges without
# needing the HeartbeatState passed through every call chain.
# Call set_active_state() once after creating the state to register it.
# Thread safety: the reference assignment is atomic in CPython (GIL).

_active_state: Optional["HeartbeatState"] = None
_registry_lock = threading.Lock()


def set_active_state(state: Optional["HeartbeatState"]) -> None:
    """Register the active HeartbeatState for the current pipeline run.

    Call once after creating HeartbeatState to make it available to agents.
    Pass None to deregister on shutdown.
    """
    global _active_state
    with _registry_lock:
        _active_state = state


def get_active_state() -> Optional["HeartbeatState"]:
    """Return the currently active HeartbeatState, or None if no run is active."""
    return _active_state


class HeartbeatState:
    """Mutable state dict shared between the pipeline and the heartbeat thread.

    The pipeline updates fields directly; the thread reads them on each tick.
    Thread safety: individual field writes are atomic in CPython (GIL); we do
    not need a lock for the simple integer/string fields used here.
    """

    def __init__(
        self,
        exp_id: str,
        total_challenges: int = 0,
    ) -> None:
        self.exp_id = exp_id
        self.total_challenges = total_challenges
        self.completed_challenges: int = 0
        self.current_phase: str = "init"
        self.last_llm_call_at: Optional[str] = None
        self.started_at: str = datetime.now(timezone.utc).isoformat()


class HeartbeatThread(threading.Thread):
    """Daemon thread that writes heartbeat JSON every *interval_seconds* seconds.

    Lifecycle:
      1. Spawned as daemon=True after the pipeline state is created.
      2. Writes heartbeat every interval_seconds until stop() is called.
      3. On stop(): writes final heartbeat with phase="complete", then exits.
    """

    def __init__(
        self,
        jobs_dir: Path,
        state: HeartbeatState,
        interval_seconds: float = 30.0,
    ) -> None:
        super().__init__(daemon=True, name=f"heartbeat-{state.exp_id}")
        self._jobs_dir = jobs_dir
        self._state = state
        self._interval = interval_seconds
        self._stop_event = threading.Event()
        self._pid = os.getpid()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def stop(self) -> None:
        """Signal the thread to stop after the next write."""
        self._stop_event.set()

    def run(self) -> None:
        """Write heartbeat repeatedly until stop() is called."""
        self._jobs_dir.mkdir(parents=True, exist_ok=True)

        # Write immediately on first tick
        self._write()

        while not self._stop_event.wait(timeout=self._interval):
            self._write()

        # Final write with phase="complete"
        self._state.current_phase = "complete"
        self._write()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _write(self) -> None:
        """Write the heartbeat file atomically."""
        state = self._state
        data = {
            "exp_id": state.exp_id,
            "pid": self._pid,
            "started_at": state.started_at,
            "last_update": datetime.now(timezone.utc).isoformat(),
            "last_llm_call_at": state.last_llm_call_at,
            "completed_challenges": state.completed_challenges,
            "total_challenges": state.total_challenges,
            "current_phase": state.current_phase,
        }
        filename = f"_run_{state.exp_id}.heartbeat.json"
        final_path = self._jobs_dir / filename
        tmp_path = self._jobs_dir / (filename + ".tmp")

        payload = json.dumps(data, indent=2).encode("utf-8")
        try:
            fd = os.open(str(tmp_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
            try:
                os.write(fd, payload)
                os.fsync(fd)
            finally:
                os.close(fd)
            os.rename(str(tmp_path), str(final_path))
        except OSError:
            # Non-fatal: watchdog will notice stale timestamp
            pass
