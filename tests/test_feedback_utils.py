"""Tests for feedback loop utilities (reward signal, run history, prompt version).

Covers:
- compute_reward returns reward=True only when judge matches FEEDBACK_JUDGE_MODEL and threshold met
- reward=False when judge does not match (self-judge guard)
- reward=False when score below threshold
- append_run_history creates file on first call, appends on subsequent calls
- Prompt version hash is stable (same prompt = same hash)
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.models.report_models import RankingReport, RankingScore
from src.utils.feedback_utils import RewardRecord, append_run_history, compute_reward

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_score(persona: str, score: int) -> RankingScore:
    return RankingScore(
        score=score,
        persona=persona,
        justification="Test score",
        improvements=[],
        dimension_scores=None,
    )


def _make_report(challenge_id: str, tech: int, ped: int) -> RankingReport:
    t = _make_score("Technical", tech)
    p = _make_score("Pedagogical", ped)
    overall = round((tech + ped) / 2.0, 1)
    return RankingReport(
        challenge_id=challenge_id,
        overall_score=float(overall),
        technical_review=t,
        pedagogical_review=p,
        technical_rank="Intermediate",
        dimension_scores=None,
    )


# ---------------------------------------------------------------------------
# compute_reward: judge guard
# ---------------------------------------------------------------------------


def test_compute_reward_pass_when_judge_matches_and_threshold_met():
    """reward=True when judge matches FEEDBACK_JUDGE_MODEL and both means >= threshold."""
    reports = [_make_report("c1", 8, 8), _make_report("c2", 9, 9)]
    from src.config.settings import settings

    record = compute_reward(reports, judge_model=settings.FEEDBACK_JUDGE_MODEL)

    assert record.reward is True
    assert record.judge_model == settings.FEEDBACK_JUDGE_MODEL
    assert record.mean_tech >= settings.FEEDBACK_REWARD_THRESHOLD
    assert record.mean_ped >= settings.FEEDBACK_REWARD_THRESHOLD


def test_compute_reward_fail_when_judge_does_not_match():
    """reward=False when judge_model differs from FEEDBACK_JUDGE_MODEL (self-judge guard)."""
    reports = [_make_report("c1", 9, 9)]
    record = compute_reward(reports, judge_model="gpt-4o")

    assert record.reward is False
    assert record.judge_model == "gpt-4o"


def test_compute_reward_fail_when_score_below_threshold():
    """reward=False when mean score < FEEDBACK_REWARD_THRESHOLD even with correct judge."""
    from src.config.settings import settings

    # Both scores below threshold (7.0 default)
    reports = [_make_report("c1", 5, 5)]
    record = compute_reward(reports, judge_model=settings.FEEDBACK_JUDGE_MODEL)

    assert record.reward is False
    assert record.mean_tech < settings.FEEDBACK_REWARD_THRESHOLD


def test_compute_reward_fail_when_only_tech_below_threshold():
    """reward=False when tech is below threshold even if ped is above."""
    from src.config.settings import settings

    reports = [_make_report("c1", 4, 9)]
    record = compute_reward(reports, judge_model=settings.FEEDBACK_JUDGE_MODEL)

    assert record.reward is False


def test_compute_reward_fail_when_only_ped_below_threshold():
    """reward=False when ped is below threshold even if tech is above."""
    from src.config.settings import settings

    reports = [_make_report("c1", 9, 4)]
    record = compute_reward(reports, judge_model=settings.FEEDBACK_JUDGE_MODEL)

    assert record.reward is False


# ---------------------------------------------------------------------------
# compute_reward: record fields
# ---------------------------------------------------------------------------


def test_compute_reward_fields_populated():
    """RewardRecord has all required fields correctly populated."""
    from src.config.settings import settings

    reports = [_make_report("ch1", 8, 7), _make_report("ch2", 9, 8)]
    record = compute_reward(reports, judge_model=settings.FEEDBACK_JUDGE_MODEL)

    assert isinstance(record, RewardRecord)
    assert record.run_id  # non-empty UUID
    assert record.timestamp  # non-empty ISO string
    assert "ch1" in record.per_challenge_scores
    assert "ch2" in record.per_challenge_scores
    assert record.mean_tech == pytest.approx((8 + 9) / 2.0, abs=0.01)
    assert record.mean_ped == pytest.approx((7 + 8) / 2.0, abs=0.01)
    assert 0.0 <= record.pass_rate <= 1.0
    assert len(record.prompt_version) == 8  # first 8 chars of sha256


def test_compute_reward_empty_reports():
    """compute_reward with empty list returns reward=False and zero means."""
    from src.config.settings import settings

    record = compute_reward([], judge_model=settings.FEEDBACK_JUDGE_MODEL)

    assert record.reward is False
    assert record.mean_tech == 0.0
    assert record.mean_ped == 0.0
    assert record.pass_rate == 0.0


# ---------------------------------------------------------------------------
# append_run_history
# ---------------------------------------------------------------------------


def _make_record() -> RewardRecord:
    from src.config.settings import settings

    reports = [_make_report("c1", 8, 8)]
    return compute_reward(reports, judge_model=settings.FEEDBACK_JUDGE_MODEL)


def test_append_run_history_creates_file_on_first_call(tmp_path: Path):
    """append_run_history creates the directory and file on first call."""
    history_path = tmp_path / "feedback" / "run_history.jsonl"
    assert not history_path.exists()

    record = _make_record()

    with patch("src.utils.feedback_utils.settings") as mock_settings:
        mock_settings.FEEDBACK_ENABLED = True
        mock_settings.DATA_DIR = tmp_path

        append_run_history(record)

    assert history_path.exists()
    lines = history_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["run_id"] == record.run_id


def test_append_run_history_appends_on_subsequent_calls(tmp_path: Path):
    """append_run_history appends records; file grows by one line per call."""
    record1 = _make_record()
    record2 = _make_record()

    with patch("src.utils.feedback_utils.settings") as mock_settings:
        mock_settings.FEEDBACK_ENABLED = True
        mock_settings.DATA_DIR = tmp_path

        append_run_history(record1)
        append_run_history(record2)

    history_path = tmp_path / "feedback" / "run_history.jsonl"
    lines = history_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2

    ids = [json.loads(line)["run_id"] for line in lines]
    assert ids[0] == record1.run_id
    assert ids[1] == record2.run_id


def test_append_run_history_noop_when_disabled(tmp_path: Path):
    """append_run_history does nothing when FEEDBACK_ENABLED=False."""
    record = _make_record()

    with patch("src.utils.feedback_utils.settings") as mock_settings:
        mock_settings.FEEDBACK_ENABLED = False
        mock_settings.DATA_DIR = tmp_path

        append_run_history(record)

    history_path = tmp_path / "feedback" / "run_history.jsonl"
    assert not history_path.exists()


# ---------------------------------------------------------------------------
# Prompt version stability
# ---------------------------------------------------------------------------


def test_prompt_version_is_stable():
    """Same content-generation system prompt always produces the same hash."""
    from src.config.settings import settings

    reports = [_make_report("c1", 8, 8)]
    r1 = compute_reward(reports, judge_model=settings.FEEDBACK_JUDGE_MODEL)
    r2 = compute_reward(reports, judge_model=settings.FEEDBACK_JUDGE_MODEL)

    assert r1.prompt_version == r2.prompt_version


def test_prompt_version_is_8_chars():
    """Prompt version is exactly 8 hex characters."""
    from src.config.settings import settings

    record = compute_reward(
        [_make_report("c1", 8, 8)], judge_model=settings.FEEDBACK_JUDGE_MODEL
    )
    assert len(record.prompt_version) == 8
    int(record.prompt_version, 16)  # raises ValueError if not valid hex


def test_prompt_version_changes_with_different_prompt():
    """Different prompt string produces different hash."""

    from src.utils.feedback_utils import _get_prompt_version

    original_version = _get_prompt_version()

    with patch(
        "src.agents.content_generation_agent._WRITEUP_SYSTEM",
        "A completely different system prompt.",
    ):
        modified_version = _get_prompt_version()

    assert original_version != modified_version
