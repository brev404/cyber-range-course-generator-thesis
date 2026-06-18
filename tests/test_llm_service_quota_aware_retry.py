"""Tests for quota-aware retry logic in src/services/llm_service.py.

Tests verify (C10 contract: sleep-then-retry on quota):
1. Exponential backoff between retries on transient errors.
2. Quota-exhausted signal triggers _handle_quota_signal, which sleeps until the
   next subscription reset window (when within QUOTA_SLEEP_MAX_MINUTES) and then
   makes ONE post-sleep attempt. A second consecutive quota signal raises
   QuotaExhaustedError; a successful post-sleep response is returned normally.
3. Quota signal raises a distinguishable QuotaExhaustedError.
4. Transient errors still retry the full 3 attempts (max_retries=2 → 3 total).
5. Happy path: first call succeeds, no retry.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.services.llm_service import LLMServiceError, generate_response

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# The exact ValueError message raised by claude_code_model._generate when
# claude --print exits with returncode=1 and no stderr.
_QUOTA_SIGNAL_MSG = "claude --print exited with code 1: (no stderr)"


def _make_chat_model(side_effects):
    """Return a mock chat model whose .invoke() raises side_effects in order."""
    chat = MagicMock()
    chat.invoke.side_effect = side_effects
    return chat


# ---------------------------------------------------------------------------
# Import QuotaExhaustedError (must exist after implementation)
# ---------------------------------------------------------------------------


def test_quota_exhausted_error_importable():
    """QuotaExhaustedError must be importable from llm_service."""
    from src.services.llm_service import QuotaExhaustedError  # noqa: F401


def test_quota_exhausted_error_is_subclass_of_llm_service_error():
    """QuotaExhaustedError must be a subclass of LLMServiceError."""
    from src.services.llm_service import QuotaExhaustedError

    assert issubclass(QuotaExhaustedError, LLMServiceError)


# ---------------------------------------------------------------------------
# Test 1: Exponential backoff applied on transient errors
# ---------------------------------------------------------------------------


@patch("src.services.llm_service.get_chat_model")
@patch("src.services.llm_service.time.sleep")
def test_backoff_applied_on_transient_errors(mock_sleep, mock_get_chat_model):
    """Between retries for transient errors, time.sleep is called with exponential delays."""
    transient = RuntimeError("network blip")
    chat = _make_chat_model([transient, transient, transient])
    mock_get_chat_model.return_value = chat

    with pytest.raises(LLMServiceError):
        generate_response("prompt", max_retries=2)

    # With max_retries=2 there are 3 attempts (0, 1, 2).
    # Sleep should be called TWICE (after attempt 0 and after attempt 1).
    # Delays: 2**0=1s, 2**1=2s
    assert mock_sleep.call_count == 2
    mock_sleep.assert_any_call(1)
    mock_sleep.assert_any_call(2)


# ---------------------------------------------------------------------------
# Test 2: Quota signal then success — sleep once, return post-sleep response
# ---------------------------------------------------------------------------


@patch("src.services.llm_service.seconds_until_next_reset", create=True)
@patch("src.services.llm_service.get_chat_model")
@patch("src.services.llm_service.time.sleep")
def test_quota_signal_sleeps_then_retries_on_success(
    mock_sleep, mock_get_chat_model, _mock_secs_unused
):
    """A quota signal sleeps until reset (+60s) then makes ONE post-sleep attempt.

    If the post-sleep attempt succeeds, the response is returned and no exception
    is raised. The transient backoff loop is NOT engaged — sleep happens exactly
    once (the quota-reset sleep).
    """
    # Force the quota reset window to be small (well under QUOTA_SLEEP_MAX_MINUTES)
    # via the local import inside _handle_quota_signal.
    quota_err = ValueError(_QUOTA_SIGNAL_MSG)
    response = MagicMock()
    response.content = "Recovered after quota sleep"
    chat = _make_chat_model([quota_err, response])
    mock_get_chat_model.return_value = chat

    with patch("src.services.quota_helper.seconds_until_next_reset", return_value=10):
        result = generate_response("prompt", max_retries=2)

    assert result == "Recovered after quota sleep"
    # invoke() called twice: initial (quota) + one post-sleep attempt (success)
    assert chat.invoke.call_count == 2
    # time.sleep called exactly once with reset_secs + 60s buffer
    assert mock_sleep.call_count == 1
    mock_sleep.assert_called_once_with(70)


# ---------------------------------------------------------------------------
# Test 3: Quota signal raises a distinguishable QuotaExhaustedError
# ---------------------------------------------------------------------------


@patch("src.services.llm_service.get_chat_model")
@patch("src.services.llm_service.time.sleep")
def test_quota_signal_raises_quota_exhausted_error(mock_sleep, mock_get_chat_model):
    """Two consecutive quota signals → QuotaExhaustedError (subclass of LLMServiceError).

    First quota signal triggers _handle_quota_signal → sleep → one post-sleep
    attempt. If that attempt also returns a quota signal, QuotaExhaustedError is
    raised with a clear "subscription usage window exhausted" message.
    """
    from src.services.llm_service import QuotaExhaustedError

    quota_err_1 = ValueError(_QUOTA_SIGNAL_MSG)
    quota_err_2 = ValueError(_QUOTA_SIGNAL_MSG)
    chat = _make_chat_model([quota_err_1, quota_err_2])
    mock_get_chat_model.return_value = chat

    with patch("src.services.quota_helper.seconds_until_next_reset", return_value=10):
        with pytest.raises(QuotaExhaustedError) as exc_info:
            generate_response("prompt", max_retries=2)

    # Must be catchable as LLMServiceError (subclass contract)
    assert isinstance(exc_info.value, LLMServiceError)
    # Message should contain a meaningful hint
    assert (
        "quota" in str(exc_info.value).lower()
        or "exhausted" in str(exc_info.value).lower()
    )
    # invoke() called twice (initial + post-sleep), sleep called once
    assert chat.invoke.call_count == 2
    assert mock_sleep.call_count == 1


# ---------------------------------------------------------------------------
# Test 4: Transient errors still retry the full 3 attempts
# ---------------------------------------------------------------------------


@patch("src.services.llm_service.get_chat_model")
@patch("src.services.llm_service.time.sleep")
def test_transient_errors_retry_full_attempts(mock_sleep, mock_get_chat_model):
    """Generic transient errors should retry all 3 attempts (max_retries=2)."""
    transient = OSError("temporary network failure")
    chat = _make_chat_model([transient, transient, transient])
    mock_get_chat_model.return_value = chat

    with pytest.raises(LLMServiceError):
        generate_response("prompt", max_retries=2)

    assert chat.invoke.call_count == 3  # 3 total attempts: 0, 1, 2


# ---------------------------------------------------------------------------
# Test 5: Happy path — first call succeeds, no retry or sleep
# ---------------------------------------------------------------------------


@patch("src.services.llm_service.get_chat_model")
@patch("src.services.llm_service.time.sleep")
def test_happy_path_no_retry(mock_sleep, mock_get_chat_model):
    """When the first call succeeds, invoke() is called once and sleep is never called."""
    response = MagicMock()
    response.content = "All good!"
    chat = _make_chat_model([response])
    mock_get_chat_model.return_value = chat

    result = generate_response("prompt", max_retries=2)

    assert result == "All good!"
    assert chat.invoke.call_count == 1
    mock_sleep.assert_not_called()


# ---------------------------------------------------------------------------
# Test 6: Transient succeeds on second attempt — sleep called once
# ---------------------------------------------------------------------------


@patch("src.services.llm_service.get_chat_model")
@patch("src.services.llm_service.time.sleep")
def test_transient_succeeds_on_second_attempt(mock_sleep, mock_get_chat_model):
    """If first call fails transiently but second succeeds, sleep is called once."""
    transient = RuntimeError("first try blip")
    response = MagicMock()
    response.content = "Recovered!"
    chat = _make_chat_model([transient, response])
    mock_get_chat_model.return_value = chat

    result = generate_response("prompt", max_retries=2)

    assert result == "Recovered!"
    assert chat.invoke.call_count == 2
    assert mock_sleep.call_count == 1
    mock_sleep.assert_called_once_with(1)  # 2**0 = 1s after first failure


# ---------------------------------------------------------------------------
# Test 7: generate_response_with_system honours sleep-then-retry on quota
# ---------------------------------------------------------------------------


@patch("src.services.llm_service.get_chat_model")
@patch("src.services.llm_service.time.sleep")
def test_generate_response_with_system_quota_sleep_then_retry(
    mock_sleep, mock_get_chat_model
):
    """generate_response_with_system must apply the C10 sleep-then-retry contract.

    Two consecutive quota signals on the Anthropic cached-content path raise
    QuotaExhaustedError after exactly one sleep + one post-sleep attempt.
    """
    from src.services.llm_service import (
        QuotaExhaustedError,
        generate_response_with_system,
    )

    quota_err_1 = ValueError(_QUOTA_SIGNAL_MSG)
    quota_err_2 = ValueError(_QUOTA_SIGNAL_MSG)
    chat = _make_chat_model([quota_err_1, quota_err_2])
    mock_get_chat_model.return_value = chat

    # Force anthropic path so the inner retry loop is exercised
    with patch("src.services.llm_service._run_provider") as mock_cv:
        mock_cv.get.return_value = "anthropic"
        with patch(
            "src.services.quota_helper.seconds_until_next_reset", return_value=10
        ):
            with pytest.raises(QuotaExhaustedError):
                generate_response_with_system(
                    "system prompt",
                    "user prompt",
                    provider="anthropic",
                    max_retries=2,
                )

    # invoke() called twice (initial + post-sleep), sleep called once
    assert chat.invoke.call_count == 2
    assert mock_sleep.call_count == 1


# ---------------------------------------------------------------------------
# Test 8: Auth errors (401/token_revoked) propagate WITHOUT sleeping
# (observed: a codex 401/token_revoked was misclassified as quota → false sleep)
# ---------------------------------------------------------------------------

# Codex CLI surfaces a revoked/expired OAuth token as a non-zero exit whose
# detail text carries an auth phrasing (NOT a quota/429 phrasing).
_AUTH_SIGNAL_MSG = "codex exec exited with code 1: 401 Unauthorized: token_revoked"


def test_auth_error_importable_and_subclass():
    from src.services.llm_service import LLMAuthError

    assert issubclass(LLMAuthError, LLMServiceError)


def test_is_auth_error_distinct_from_quota():
    from src.services.llm_service import _is_auth_error, _is_quota_signal

    auth = ValueError(_AUTH_SIGNAL_MSG)
    quota = ValueError("Error: 429 too many requests (usage limit)")
    assert _is_auth_error(auth) is True
    assert _is_quota_signal(auth) is False  # auth text carries no quota vocabulary
    assert _is_auth_error(quota) is False  # quota text carries no auth vocabulary
    assert _is_quota_signal(quota) is True


@patch("src.services.llm_service.get_chat_model")
@patch("src.services.llm_service.time.sleep")
def test_auth_error_propagates_without_sleeping(mock_sleep, mock_get_chat_model):
    """A 401/token_revoked must raise LLMAuthError immediately — never sleep."""
    from src.services.llm_service import LLMAuthError

    auth_err = ValueError(_AUTH_SIGNAL_MSG)
    chat = _make_chat_model([auth_err, auth_err, auth_err])
    mock_get_chat_model.return_value = chat

    with pytest.raises(LLMAuthError):
        generate_response("prompt", max_retries=2)

    # No reset-window sleep, and no post-error retry of the failing call.
    mock_sleep.assert_not_called()
    assert chat.invoke.call_count == 1


@patch("src.services.llm_service.get_chat_model")
@patch("src.services.llm_service.time.sleep")
def test_auth_error_propagates_in_with_system_path(mock_sleep, mock_get_chat_model):
    """generate_response_with_system must also propagate auth errors without sleeping."""
    from src.services.llm_service import LLMAuthError, generate_response_with_system

    auth_err = ValueError(_AUTH_SIGNAL_MSG)
    chat = _make_chat_model([auth_err])
    mock_get_chat_model.return_value = chat

    with patch("src.services.llm_service._run_provider") as mock_cv:
        mock_cv.get.return_value = "anthropic"
        with pytest.raises(LLMAuthError):
            generate_response_with_system(
                "system prompt", "user prompt", provider="anthropic", max_retries=2
            )

    mock_sleep.assert_not_called()
    assert chat.invoke.call_count == 1
