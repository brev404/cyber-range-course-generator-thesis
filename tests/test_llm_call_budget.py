"""Tests for the per-challenge LLM call budget (Fix 1).

Covers:
1. Budget allows up to MAX_LLM_CALLS_PER_CHALLENGE then raises LLMCallBudgetExceeded.
2. Counter resets between challenges via reset_challenge_llm_budget().
3. LLMCallBudgetExceeded is a subclass of LLMServiceError.
4. Warning logged at 50% threshold.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from src.services.llm_service import (
    LLMCallBudgetExceeded,
    LLMServiceError,
    _challenge_llm_call_count,
    _increment_and_check_budget,
    reset_challenge_llm_budget,
)


class TestLLMCallBudgetHierarchy(unittest.TestCase):
    """Test 3: LLMCallBudgetExceeded is a subclass of LLMServiceError."""

    def test_is_subclass_of_llm_service_error(self) -> None:
        assert issubclass(LLMCallBudgetExceeded, LLMServiceError), (
            "LLMCallBudgetExceeded must subclass LLMServiceError so existing "
            "except LLMServiceError catches still work."
        )

    def test_instance_caught_by_llm_service_error(self) -> None:
        exc = LLMCallBudgetExceeded("test")
        assert isinstance(exc, LLMServiceError)


class TestLLMCallBudgetCap(unittest.TestCase):
    """Test 1: Budget allows up to cap, then raises LLMCallBudgetExceeded."""

    def setUp(self) -> None:
        reset_challenge_llm_budget("test-challenge")

    def test_raises_after_cap(self) -> None:
        cap = 5
        with patch("src.services.llm_service.settings") as mock_settings:
            mock_settings.MAX_LLM_CALLS_PER_CHALLENGE = cap
            # Calls 1..cap should NOT raise.
            for i in range(cap):
                _increment_and_check_budget()
            # Call cap+1 SHOULD raise.
            with self.assertRaises(LLMCallBudgetExceeded):
                _increment_and_check_budget()

    def test_does_not_raise_before_cap(self) -> None:
        cap = 5
        with patch("src.services.llm_service.settings") as mock_settings:
            mock_settings.MAX_LLM_CALLS_PER_CHALLENGE = cap
            reset_challenge_llm_budget("c1")
            for _ in range(cap):
                _increment_and_check_budget()  # Must not raise
            # Verify counter is exactly at cap
            assert _challenge_llm_call_count.get(0) == cap

    def test_cap_zero_means_unlimited(self) -> None:
        """cap=0 disables the budget check — no exception even after many calls."""
        with patch("src.services.llm_service.settings") as mock_settings:
            mock_settings.MAX_LLM_CALLS_PER_CHALLENGE = 0
            reset_challenge_llm_budget("unlimited")
            for _ in range(100):
                _increment_and_check_budget()  # Must not raise

    def test_exception_message_contains_cap(self) -> None:
        cap = 3
        with patch("src.services.llm_service.settings") as mock_settings:
            mock_settings.MAX_LLM_CALLS_PER_CHALLENGE = cap
            reset_challenge_llm_budget("msg-test")
            for _ in range(cap):
                _increment_and_check_budget()
            try:
                _increment_and_check_budget()
                self.fail("Expected LLMCallBudgetExceeded")
            except LLMCallBudgetExceeded as e:
                assert str(cap) in str(
                    e
                ), f"Cap value {cap} should appear in exception: {e}"


class TestLLMCallBudgetReset(unittest.TestCase):
    """Test 2: Counter resets between challenges."""

    def test_reset_clears_counter(self) -> None:
        cap = 4
        with patch("src.services.llm_service.settings") as mock_settings:
            mock_settings.MAX_LLM_CALLS_PER_CHALLENGE = cap
            reset_challenge_llm_budget("challenge-a")
            for _ in range(cap):
                _increment_and_check_budget()
            # Would raise on next call — but after reset it should not
            reset_challenge_llm_budget("challenge-b")
            for _ in range(cap):
                _increment_and_check_budget()  # Must not raise

    def test_reset_sets_counter_to_zero(self) -> None:
        reset_challenge_llm_budget("any-challenge")
        assert _challenge_llm_call_count.get(0) == 0

    def test_different_challenge_ids_reset_independently(self) -> None:
        """Resetting with a different challenge_id clears the counter."""
        cap = 3
        with patch("src.services.llm_service.settings") as mock_settings:
            mock_settings.MAX_LLM_CALLS_PER_CHALLENGE = cap
            reset_challenge_llm_budget("challenge-x")
            for _ in range(cap):
                _increment_and_check_budget()
            # Reset for next challenge — counter back to 0
            reset_challenge_llm_budget("challenge-y")
            assert _challenge_llm_call_count.get(0) == 0


class TestLLMCallBudgetWarning(unittest.TestCase):
    """Test 4: Warning logged at 50% threshold."""

    def test_warning_at_50_percent(self) -> None:
        cap = 10
        halfway = cap // 2  # 5
        with patch("src.services.llm_service.settings") as mock_settings:
            mock_settings.MAX_LLM_CALLS_PER_CHALLENGE = cap
            reset_challenge_llm_budget("warn-test")
            with patch("src.services.llm_service.logger") as mock_logger:
                # Calls 1..4: no warning
                for _ in range(halfway - 1):
                    _increment_and_check_budget()
                assert (
                    mock_logger.warning.call_count == 0
                ), "Should not warn before hitting halfway mark"
                # Call 5 (= halfway): warning
                _increment_and_check_budget()
                assert (
                    mock_logger.warning.call_count == 1
                ), f"Expected 1 warning at {halfway}/{cap} calls"
                # Calls 6..9: no additional warnings
                for _ in range(halfway - 1):
                    _increment_and_check_budget()
                assert (
                    mock_logger.warning.call_count == 1
                ), "Should not warn again after halfway mark"

    def test_no_warning_when_cap_disabled(self) -> None:
        with patch("src.services.llm_service.settings") as mock_settings:
            mock_settings.MAX_LLM_CALLS_PER_CHALLENGE = 0
            reset_challenge_llm_budget("no-warn")
            with patch("src.services.llm_service.logger") as mock_logger:
                for _ in range(20):
                    _increment_and_check_budget()
                assert mock_logger.warning.call_count == 0


class TestLLMCallBudgetCrossPhaseReset(unittest.TestCase):
    """Test that resetting between content_generation and ranking phases prevents spurious cap errors.

    Bug 3 (B5): A challenge with 12 gen calls + 10 ranking calls = 22 total could exceed
    cap=20 if the counter was NOT reset between phases.  After the fix, ranking_agent calls
    reset_challenge_llm_budget() at the start of each challenge, giving each phase its own
    fresh budget of cap calls.
    """

    def test_21_calls_across_phases_no_exception_with_reset(self) -> None:
        """21 calls split across gen (12) + ranking (9) should NOT raise with inter-phase reset."""
        cap = 20
        with patch("src.services.llm_service.settings") as mock_settings:
            mock_settings.MAX_LLM_CALLS_PER_CHALLENGE = cap

            # content_generation — 12 calls
            reset_challenge_llm_budget("challenge-x")
            for _ in range(12):
                _increment_and_check_budget()

            # Inter-stage reset (ranking_agent does this)
            reset_challenge_llm_budget("challenge-x")

            # ranking — 9 calls
            for _ in range(9):
                _increment_and_check_budget()  # Must NOT raise

    def test_21_calls_across_phases_raises_without_reset(self) -> None:
        """Without an inter-phase reset, 21 calls DO exceed cap=20 and raise."""
        cap = 20
        with patch("src.services.llm_service.settings") as mock_settings:
            mock_settings.MAX_LLM_CALLS_PER_CHALLENGE = cap

            # content_generation — 12 calls (no reset between stages)
            reset_challenge_llm_budget("challenge-y")
            for _ in range(cap):
                _increment_and_check_budget()

            # ranking call #21 — should raise because no reset
            with self.assertRaises(LLMCallBudgetExceeded):
                _increment_and_check_budget()

    def test_each_phase_gets_full_cap_after_reset(self) -> None:
        """Each phase can use the full cap independently after a reset."""
        cap = 10
        with patch("src.services.llm_service.settings") as mock_settings:
            mock_settings.MAX_LLM_CALLS_PER_CHALLENGE = cap

            # Gen phase: use full cap
            reset_challenge_llm_budget("chal-z")
            for _ in range(cap):
                _increment_and_check_budget()

            # After reset, ranking phase can also use full cap
            reset_challenge_llm_budget("chal-z")
            for _ in range(cap):
                _increment_and_check_budget()

            # Counter should be exactly cap after second phase
            assert _challenge_llm_call_count.get(0) == cap


if __name__ == "__main__":
    unittest.main()
