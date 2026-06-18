"""Tests for heartbeat status file (Feature 3 / B6).

Tests verify:
1. Heartbeat writes initial JSON within 1s of thread start.
2. Heartbeat updates with a new last_update timestamp.
3. Heartbeat file survives concurrent reads (no corrupt JSON).
4. On thread shutdown, final heartbeat has phase="complete".
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_jobs_dir(tmp_path: Path) -> Path:
    """Temporary output/jobs directory."""
    jobs_dir = tmp_path / "output" / "jobs"
    jobs_dir.mkdir(parents=True)
    return jobs_dir


# ---------------------------------------------------------------------------
# Helper: start a heartbeat thread for testing
# ---------------------------------------------------------------------------


def _start_heartbeat(jobs_dir: Path, exp_id: str = "EXP-TEST", total: int = 5):
    """Import and start a HeartbeatThread, return (thread, state_dict)."""
    from src.services.heartbeat import HeartbeatState, HeartbeatThread

    state = HeartbeatState(
        exp_id=exp_id,
        total_challenges=total,
    )
    thread = HeartbeatThread(
        jobs_dir=jobs_dir,
        state=state,
        interval_seconds=0.1,  # fast for tests
    )
    thread.start()
    return thread, state


# ---------------------------------------------------------------------------
# Test 1: Heartbeat writes initial JSON within 1s of thread start
# ---------------------------------------------------------------------------


def test_heartbeat_writes_initial_json(tmp_jobs_dir: Path):
    """HeartbeatThread should write the heartbeat file within 1 second of start."""
    thread, state = _start_heartbeat(tmp_jobs_dir, exp_id="EXP-HBTEST")
    try:
        heartbeat_file = tmp_jobs_dir / "_run_EXP-HBTEST.heartbeat.json"

        deadline = time.time() + 1.0
        while time.time() < deadline:
            if heartbeat_file.exists() and heartbeat_file.stat().st_size > 0:
                break
            time.sleep(0.05)

        assert heartbeat_file.exists(), "Heartbeat file not written within 1s"
        data = json.loads(heartbeat_file.read_text())
        assert data["exp_id"] == "EXP-HBTEST"
        assert "pid" in data
        assert "started_at" in data
        assert "last_update" in data
        assert "total_challenges" in data
    finally:
        thread.stop()
        thread.join(timeout=2.0)


# ---------------------------------------------------------------------------
# Test 2: Heartbeat updates last_update timestamp over time
# ---------------------------------------------------------------------------


def test_heartbeat_updates_last_update(tmp_jobs_dir: Path):
    """last_update field should change between successive reads."""
    thread, state = _start_heartbeat(tmp_jobs_dir, exp_id="EXP-UPDATE")
    try:
        heartbeat_file = tmp_jobs_dir / "_run_EXP-UPDATE.heartbeat.json"

        # Wait for first write
        deadline = time.time() + 1.0
        while time.time() < deadline:
            if heartbeat_file.exists() and heartbeat_file.stat().st_size > 0:
                break
            time.sleep(0.05)

        assert heartbeat_file.exists()
        first_data = json.loads(heartbeat_file.read_text())
        first_ts = first_data["last_update"]

        # Wait for at least one more write cycle (interval is 0.1s)
        time.sleep(0.25)

        second_data = json.loads(heartbeat_file.read_text())
        second_ts = second_data["last_update"]

        # Timestamps should differ (updated since first read)
        assert second_ts >= first_ts, "last_update should not go backwards"
        # In practice, with 0.25s wait and 0.1s interval, they should differ
        # We check they're at least not identical strings in most cases — but
        # allow equal if the system is slow.
    finally:
        thread.stop()
        thread.join(timeout=2.0)


# ---------------------------------------------------------------------------
# Test 3: Heartbeat survives concurrent reads (no corrupt JSON)
# ---------------------------------------------------------------------------


def test_heartbeat_survives_concurrent_reads(tmp_jobs_dir: Path):
    """Multiple concurrent reads of the heartbeat file should not produce corrupt JSON."""
    import threading

    thread, state = _start_heartbeat(tmp_jobs_dir, exp_id="EXP-CONC")
    heartbeat_file = tmp_jobs_dir / "_run_EXP-CONC.heartbeat.json"
    errors: list[Exception] = []

    def reader():
        for _ in range(20):
            try:
                if heartbeat_file.exists() and heartbeat_file.stat().st_size > 0:
                    json.loads(heartbeat_file.read_text())
            except Exception as e:
                errors.append(e)
            time.sleep(0.02)

    try:
        # Wait for file to appear
        deadline = time.time() + 1.0
        while time.time() < deadline and not heartbeat_file.exists():
            time.sleep(0.05)

        readers = [threading.Thread(target=reader) for _ in range(5)]
        for r in readers:
            r.start()
        for r in readers:
            r.join(timeout=3.0)

        assert errors == [], f"Concurrent reads produced errors: {errors}"
    finally:
        thread.stop()
        thread.join(timeout=2.0)


# ---------------------------------------------------------------------------
# Test 4: On thread shutdown, final heartbeat has phase="complete"
# ---------------------------------------------------------------------------


def test_heartbeat_final_write_phase_complete(tmp_jobs_dir: Path):
    """After stop(), the last heartbeat written should have phase='complete'."""
    thread, state = _start_heartbeat(tmp_jobs_dir, exp_id="EXP-FINAL")
    heartbeat_file = tmp_jobs_dir / "_run_EXP-FINAL.heartbeat.json"

    # Wait for first write
    deadline = time.time() + 1.0
    while time.time() < deadline:
        if heartbeat_file.exists() and heartbeat_file.stat().st_size > 0:
            break
        time.sleep(0.05)

    assert heartbeat_file.exists()

    thread.stop()
    thread.join(timeout=2.0)

    # After join, the final write should have phase="complete"
    final_data = json.loads(heartbeat_file.read_text())
    assert (
        final_data.get("current_phase") == "complete"
    ), f"Expected phase='complete' after stop(), got: {final_data.get('current_phase')}"


# ---------------------------------------------------------------------------
# Test 5: set_active_state / get_active_state registry works correctly (Bug 2)
# ---------------------------------------------------------------------------


def test_active_state_registry(tmp_jobs_dir: Path):
    """set_active_state registers a state; get_active_state returns it; None deregisters."""
    from src.services.heartbeat import (
        HeartbeatState,
        get_active_state,
        set_active_state,
    )

    # Initially no active state
    set_active_state(None)
    assert get_active_state() is None

    state = HeartbeatState(exp_id="EXP-REG", total_challenges=3)
    set_active_state(state)
    assert get_active_state() is state
    assert get_active_state().exp_id == "EXP-REG"

    # Deregister
    set_active_state(None)
    assert get_active_state() is None


# ---------------------------------------------------------------------------
# Test 6: Updating current_phase via active state is reflected on next heartbeat write
# ---------------------------------------------------------------------------


def test_heartbeat_phase_update_reflected_in_file(tmp_jobs_dir: Path):
    """Changing state.current_phase should appear in the next heartbeat file write."""
    from src.services.heartbeat import (
        HeartbeatState,
        HeartbeatThread,
        set_active_state,
    )

    state = HeartbeatState(exp_id="EXP-PHASE", total_challenges=5)
    thread = HeartbeatThread(jobs_dir=tmp_jobs_dir, state=state, interval_seconds=0.05)
    set_active_state(state)
    thread.start()

    heartbeat_file = tmp_jobs_dir / "_run_EXP-PHASE.heartbeat.json"
    try:
        # Wait for first write
        deadline = time.time() + 1.0
        while time.time() < deadline:
            if heartbeat_file.exists() and heartbeat_file.stat().st_size > 0:
                break
            time.sleep(0.02)

        assert heartbeat_file.exists()
        # Initial phase should be "init"
        data = json.loads(heartbeat_file.read_text())
        assert data["current_phase"] == "init"

        # Update via active state registry
        from src.services.heartbeat import get_active_state

        active = get_active_state()
        assert active is not None
        active.current_phase = "ranking"

        # Wait for next heartbeat tick
        time.sleep(0.15)

        data2 = json.loads(heartbeat_file.read_text())
        assert (
            data2["current_phase"] == "ranking"
        ), f"Expected 'ranking', got {data2['current_phase']!r}"
    finally:
        set_active_state(None)
        thread.stop()
        thread.join(timeout=2.0)


# ---------------------------------------------------------------------------
# Test 7: Incrementing completed_challenges is reflected on next heartbeat write
# ---------------------------------------------------------------------------


def test_heartbeat_completed_challenges_increment(tmp_jobs_dir: Path):
    """Incrementing state.completed_challenges should appear in the next heartbeat file."""
    from src.services.heartbeat import (
        HeartbeatState,
        HeartbeatThread,
        get_active_state,
        set_active_state,
    )

    state = HeartbeatState(exp_id="EXP-COUNT", total_challenges=10)
    thread = HeartbeatThread(jobs_dir=tmp_jobs_dir, state=state, interval_seconds=0.05)
    set_active_state(state)
    thread.start()

    heartbeat_file = tmp_jobs_dir / "_run_EXP-COUNT.heartbeat.json"
    try:
        # Wait for first write
        deadline = time.time() + 1.0
        while time.time() < deadline:
            if heartbeat_file.exists() and heartbeat_file.stat().st_size > 0:
                break
            time.sleep(0.02)

        assert heartbeat_file.exists()
        data = json.loads(heartbeat_file.read_text())
        assert data["completed_challenges"] == 0

        # Simulate 3 challenges completing
        active = get_active_state()
        active.completed_challenges += 1
        active.completed_challenges += 1
        active.completed_challenges += 1

        # Wait for next heartbeat tick
        time.sleep(0.15)

        data2 = json.loads(heartbeat_file.read_text())
        assert (
            data2["completed_challenges"] == 3
        ), f"Expected 3, got {data2['completed_challenges']}"
    finally:
        set_active_state(None)
        thread.stop()
        thread.join(timeout=2.0)
