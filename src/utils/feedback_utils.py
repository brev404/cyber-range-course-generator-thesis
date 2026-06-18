"""Feedback loop utilities: reward signal computation and run history tracking.

Reward validity is judge-model-gated: a RewardRecord is only reward=True when the
ranking judge matches settings.FEEDBACK_JUDGE_MODEL. This prevents self-judge noise
(e.g. the same model that generated the content also scoring it) from polluting the
feedback store and invalidating A/B comparisons across prompt versions.

Run history is written to data/feedback/run_history.jsonl (JSONL, one record per line).
The directory is created on first write. History is only appended when
settings.FEEDBACK_ENABLED is True — the default is False (opt-in for experiments).
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Dict, List

from pydantic import BaseModel

from src.config.settings import settings

if TYPE_CHECKING:
    from src.models.report_models import RankingReport


class RewardRecord(BaseModel):
    """One reward signal observation produced by compute_reward."""

    run_id: str
    timestamp: str
    judge_model: str
    per_challenge_scores: Dict[str, float]
    mean_tech: float
    mean_ped: float
    pass_rate: float
    reward: bool
    prompt_version: str
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0


def _get_prompt_version() -> str:
    """Return first-8-chars sha256 of the content-generation system prompt."""
    from src.agents.content_generation_agent import _WRITEUP_SYSTEM

    return hashlib.sha256(_WRITEUP_SYSTEM.encode("utf-8")).hexdigest()[:8]


def compute_reward(
    ranking_reports: List["RankingReport"],
    judge_model: str,
    cache_read_tokens: int = 0,
    cache_creation_tokens: int = 0,
) -> RewardRecord:
    """Build a RewardRecord from a list of RankingReports.

    Reward is True only when:
    - judge_model matches settings.FEEDBACK_JUDGE_MODEL (self-judge guard)
    - mean technical score >= FEEDBACK_REWARD_THRESHOLD
    - mean pedagogical score >= FEEDBACK_REWARD_THRESHOLD

    Args:
        ranking_reports: RankingReport list produced by the ranking agent.
        judge_model: Model identifier that produced the scores (e.g. RANKING_MODEL).

    Returns:
        RewardRecord with reward=True only when the guard conditions above hold.
    """
    per_challenge_scores: Dict[str, float] = {}
    tech_scores: List[float] = []
    ped_scores: List[float] = []
    passes = 0

    for report in ranking_reports:
        per_challenge_scores[report.challenge_id] = float(report.overall_score)
        t_score = report.technical_review.score if report.technical_review else 0.0
        p_score = report.pedagogical_review.score if report.pedagogical_review else 0.0
        tech_scores.append(float(t_score))
        ped_scores.append(float(p_score))
        if (
            t_score >= settings.FEEDBACK_REWARD_THRESHOLD
            and p_score >= settings.FEEDBACK_REWARD_THRESHOLD
        ):
            passes += 1

    mean_tech = round(sum(tech_scores) / len(tech_scores), 3) if tech_scores else 0.0
    mean_ped = round(sum(ped_scores) / len(ped_scores), 3) if ped_scores else 0.0
    pass_rate = round(passes / len(ranking_reports), 3) if ranking_reports else 0.0

    reward = (
        judge_model == settings.FEEDBACK_JUDGE_MODEL
        and mean_tech >= settings.FEEDBACK_REWARD_THRESHOLD
        and mean_ped >= settings.FEEDBACK_REWARD_THRESHOLD
    )

    return RewardRecord(
        run_id=str(uuid.uuid4()),
        timestamp=datetime.now(timezone.utc).isoformat(),
        judge_model=judge_model,
        per_challenge_scores=per_challenge_scores,
        mean_tech=mean_tech,
        mean_ped=mean_ped,
        pass_rate=pass_rate,
        reward=reward,
        prompt_version=_get_prompt_version(),
        cache_read_tokens=cache_read_tokens,
        cache_creation_tokens=cache_creation_tokens,
    )


def append_run_history(record: RewardRecord) -> None:
    """Append a RewardRecord to data/feedback/run_history.jsonl.

    Creates the directory and file on first call; appends on subsequent calls.
    Does nothing when settings.FEEDBACK_ENABLED is False.
    """
    if not settings.FEEDBACK_ENABLED:
        return

    feedback_dir = settings.DATA_DIR / "feedback"
    feedback_dir.mkdir(parents=True, exist_ok=True)
    history_path = feedback_dir / "run_history.jsonl"

    with history_path.open("a", encoding="utf-8") as f:
        f.write(record.model_dump_json() + "\n")
