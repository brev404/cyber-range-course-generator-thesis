"""Pytest configuration and shared fixtures for E2E tests.

Fixtures live under tests/fixtures/challenges/ (raw) and tests/fixtures/processed/
(processed layout). Used to verify env and mapping without external machine-specific paths.
"""

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _reset_llm_call_budget():
    """Reset the per-challenge LLM call counter before every test.

    The counter lives in a module-level ContextVar.  Without this fixture,
    tests that call generate_response* (including mocked ones) accumulate a
    shared counter across the test session and can trigger LLMCallBudgetExceeded
    mid-session.  Resetting before each test keeps tests independent.
    """
    from src.services.llm_service import reset_challenge_llm_budget

    reset_challenge_llm_budget("")
    yield
    # No teardown needed — next test resets via setUp.


# Base paths for fixtures (pathlib throughout per CONVENTIONS.md)
TESTS_DIR = Path(__file__).resolve().parent
FIXTURES_DIR = TESTS_DIR / "fixtures"
RAW_CHALLENGES_FIXTURE = FIXTURES_DIR / "challenges"
PROCESSED_FIXTURE = FIXTURES_DIR / "processed"
PROCESSED_ROOT = PROCESSED_FIXTURE / "raw_challenges"


@pytest.fixture
def fixtures_dir() -> Path:
    """Root of tests/fixtures."""
    return FIXTURES_DIR


@pytest.fixture
def raw_challenges_fixture() -> Path:
    """Path to raw challenge fixtures (category/challenge_name/cyberedu/...)."""
    return RAW_CHALLENGES_FIXTURE


@pytest.fixture
def processed_fixture() -> Path:
    """Path to processed fixtures root (contains raw_challenges/)."""
    return PROCESSED_FIXTURE


@pytest.fixture
def processed_root() -> Path:
    """Path to processed challenges: fixtures/processed/raw_challenges/."""
    return PROCESSED_ROOT
