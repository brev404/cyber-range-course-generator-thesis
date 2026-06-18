"""Tests for quota sleep-until-reset (C10) feature.

Tests verify:
1. seconds_until_next_reset math at various times.
2. Quota signal with reset within QUOTA_SLEEP_MAX_MINUTES: sleeps, then retries;
   if second attempt succeeds, returns the result.
3. Quota signal with reset within max: if second attempt also hits quota, raises
   QuotaExhaustedError.
4. Quota signal with reset > QUOTA_SLEEP_MAX_MINUTES away: no sleep, raises immediately.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helper: patch datetime.now to a fixed wall-clock time
# ---------------------------------------------------------------------------


def _make_now_patch(hour: int, minute: int, second: int = 0):
    """Return a datetime object at today's date with the given time (local naive)."""
    # We use a naive datetime so it matches the server-local behaviour
    # documented in the implementation (no DST conversion).
    return datetime(2026, 5, 25, hour, minute, second)


# ---------------------------------------------------------------------------
# Test 1: seconds_until_next_reset math
# ---------------------------------------------------------------------------


class TestSecondsUntilNextReset:
    """Unit-tests for the quota_helper.seconds_until_next_reset() function."""

    def _call(self, now: datetime, anchor: str = "00:30", cycle_hours: int = 5) -> int:
        from src.services.quota_helper import seconds_until_next_reset

        return seconds_until_next_reset(now=now, anchor=anchor, cycle_hours=cycle_hours)

    def test_at_23h00_anchor_00h30_returns_about_9000(self):
        """At 23:00 with anchor 00:30, last reset was at 20:30, next at 01:30 = 9000s."""
        now = _make_now_patch(23, 0, 0)
        result = self._call(now)
        # 23:00 → last reset 20:30, next reset 01:30 next day = 2.5h = 9000s
        assert abs(result - 9000) <= 5, f"Expected ~9000 but got {result}"

    def test_at_00h00_anchor_00h30_returns_about_1800(self):
        """At 00:00 with anchor 00:30, reset is in 30 min = 1800s."""
        now = _make_now_patch(0, 0, 0)
        result = self._call(now)
        assert abs(result - 1800) <= 5, f"Expected ~1800 but got {result}"

    def test_at_00h35_anchor_00h30_returns_about_18000(self):
        """At 00:35 with anchor 00:30, we just passed the last reset;
        next reset is ~5h away = 18000s."""
        now = _make_now_patch(0, 35, 0)
        result = self._call(now)
        # 00:35 - 00:30 = 5min into window; next reset is 5h - 5min = 4h55m = 17700s
        assert abs(result - 17700) <= 10, f"Expected ~17700 but got {result}"

    def test_result_always_positive(self):
        """Return value must always be positive (> 0)."""
        for h in range(0, 24):
            now = _make_now_patch(h, 0, 0)
            result = self._call(now)
            assert result > 0, f"Got non-positive result {result} at hour {h}"

    def test_result_never_exceeds_cycle(self):
        """Return value must never exceed cycle_hours * 3600."""
        for h in range(0, 24):
            now = _make_now_patch(h, 0, 0)
            result = self._call(now)
            assert result <= 5 * 3600 + 1, f"Result {result} exceeds one cycle"


# ---------------------------------------------------------------------------
# Test 2: Quota within max → sleep then retry succeeds
# ---------------------------------------------------------------------------

_QUOTA_SIGNAL_MSG = "claude --print exited with code 1: (no stderr)"


@patch("src.services.llm_service.get_chat_model")
@patch("src.services.llm_service.time.sleep")
@patch("src.services.quota_helper.seconds_until_next_reset")
def test_quota_within_max_sleep_then_retry_succeeds(
    mock_secs, mock_sleep, mock_get_chat_model
):
    """Quota exhausted with reset 30 min away: sleeps then retries;
    if second attempt succeeds, returns the result."""
    from src.services.llm_service import generate_response

    # seconds_until_next_reset returns 1800s (30 min) — within 90-min max
    mock_secs.return_value = 1800

    quota_err = ValueError(_QUOTA_SIGNAL_MSG)
    success_response = MagicMock()
    success_response.content = "LLM response after sleep"

    chat = MagicMock()
    # First call: quota error; second call: success
    chat.invoke.side_effect = [quota_err, success_response]
    mock_get_chat_model.return_value = chat

    result = generate_response("prompt", max_retries=2)

    assert result == "LLM response after sleep"
    # Sleep should be called once with ~1860s (1800 + 60 buffer)
    assert mock_sleep.call_count >= 1
    sleep_args = [c.args[0] for c in mock_sleep.call_args_list]
    # At least one sleep call should be > 60s (the quota sleep)
    assert any(s >= 60 for s in sleep_args), f"Expected quota sleep, got: {sleep_args}"
    # invoke() called twice (first = quota error, second = success)
    assert chat.invoke.call_count == 2


# ---------------------------------------------------------------------------
# Test 3: Quota within max → sleep then retry also hits quota → raises
# ---------------------------------------------------------------------------


@patch("src.services.llm_service.get_chat_model")
@patch("src.services.llm_service.time.sleep")
@patch("src.services.quota_helper.seconds_until_next_reset")
def test_quota_within_max_retry_also_hits_quota_raises(
    mock_secs, mock_sleep, mock_get_chat_model
):
    """Quota exhausted within max, second attempt also hits quota → raises QuotaExhaustedError."""
    from src.services.llm_service import QuotaExhaustedError, generate_response

    mock_secs.return_value = 1800  # 30 min, within max

    quota_err = ValueError(_QUOTA_SIGNAL_MSG)
    chat = MagicMock()
    # Both attempts hit quota
    chat.invoke.side_effect = [quota_err, quota_err]
    mock_get_chat_model.return_value = chat

    with pytest.raises(QuotaExhaustedError):
        generate_response("prompt", max_retries=2)

    # Sleep must have been called (quota sleep before retry)
    assert mock_sleep.call_count >= 1


# ---------------------------------------------------------------------------
# Test 4: Quota > max → no sleep, raise immediately
# ---------------------------------------------------------------------------


@patch("src.services.llm_service.get_chat_model")
@patch("src.services.llm_service.time.sleep")
@patch("src.services.quota_helper.seconds_until_next_reset")
def test_quota_exceeds_max_no_sleep_raises_immediately(
    mock_secs, mock_sleep, mock_get_chat_model
):
    """Quota reset > QUOTA_SLEEP_MAX_MINUTES away: no quota sleep, raise immediately."""
    from src.services.llm_service import QuotaExhaustedError, generate_response

    # Must exceed QUOTA_SLEEP_MAX_MINUTES (360 since d4c1655) → 21600s max.
    # 7 hours = 25200s safely above.
    mock_secs.return_value = 25200

    quota_err = ValueError(_QUOTA_SIGNAL_MSG)
    chat = MagicMock()
    chat.invoke.side_effect = [quota_err]
    mock_get_chat_model.return_value = chat

    with pytest.raises(QuotaExhaustedError):
        generate_response("prompt", max_retries=2)

    # The only sleep that may have been called is the transient backoff before the
    # first attempt. Since this is the very first attempt (attempt=0) and there's
    # no transient backoff before attempt 0, no sleep should be called.
    # But more importantly: no LONG quota sleep.
    quota_sleep_calls = [
        c.args[0] for c in mock_sleep.call_args_list if c.args[0] > 100
    ]
    assert (
        quota_sleep_calls == []
    ), f"Should not sleep for quota reset; got long sleeps: {quota_sleep_calls}"
    # invoke() called only once
    assert chat.invoke.call_count == 1
