"""LangSmith integration for agent workflow monitoring and debugging.

LangSmith is a platform for observing, debugging, and monitoring LLM applications.
This module integrates LangSmith tracing with LANGCHAIN_TRACING_V2: when enabled,
LangChain/LangGraph automatically send traces to LangSmith for:
- LLM calls (prompts, responses, tokens, latency)
- Agent execution flow and state
- Tool calls and errors

Usage:
    from src.services.langsmith_service import enable_tracing, disable_tracing, is_tracing_available

    # At application startup (e.g. in main.py), enable if configured
    if is_tracing_available():
        enable_tracing()
    # All subsequent LangChain/LangGraph calls are traced automatically

    # Or disable tracing
    disable_tracing()
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional

from loguru import logger

from src.config.settings import settings

if TYPE_CHECKING:
    from src.utils.feedback_utils import RewardRecord

# LangChain reads these env vars at runtime for tracing
_TRACING_V2_KEY = "LANGCHAIN_TRACING_V2"
_ENDPOINT_KEY = "LANGCHAIN_ENDPOINT"
_API_KEY_KEY = "LANGCHAIN_API_KEY"
_PROJECT_KEY = "LANGCHAIN_PROJECT"


def is_tracing_available() -> bool:
    """Return True if LangSmith tracing can be enabled (API key is set)."""
    return bool(settings.LANGCHAIN_API_KEY)


def enable_tracing(
    project_name: Optional[str] = None,
    api_key: Optional[str] = None,
    endpoint: Optional[str] = None,
) -> bool:
    """Turn on LangSmith tracing by setting LangChain env vars.

    Call this at application startup so that all subsequent LangChain/LangGraph
    calls (LLM, agents, chains) are traced to LangSmith. Uses settings values
    unless overrides are passed.

    Args:
        project_name: LangSmith project name. Default: settings.LANGCHAIN_PROJECT.
        api_key: LangSmith API key. Default: settings.LANGCHAIN_API_KEY.
        endpoint: LangSmith endpoint. Default: settings.LANGCHAIN_ENDPOINT.

    Returns:
        True if tracing was enabled, False if no API key (tracing not set).
    """
    key = api_key or settings.LANGCHAIN_API_KEY
    if not key:
        logger.debug("LangSmith tracing not enabled: LANGCHAIN_API_KEY not set")
        return False

    os.environ[_TRACING_V2_KEY] = "true"
    os.environ[_API_KEY_KEY] = key
    os.environ[_ENDPOINT_KEY] = endpoint or settings.LANGCHAIN_ENDPOINT
    os.environ[_PROJECT_KEY] = project_name or settings.LANGCHAIN_PROJECT
    logger.info(
        "LangSmith tracing enabled (project={})",
        project_name or settings.LANGCHAIN_PROJECT,
    )
    return True


def disable_tracing() -> None:
    """Turn off LangSmith tracing by unsetting LANGCHAIN_TRACING_V2."""
    os.environ.pop(_TRACING_V2_KEY, None)
    logger.debug("LangSmith tracing disabled")


def is_tracing_enabled() -> bool:
    """Return True if tracing is currently on (env LANGCHAIN_TRACING_V2=true)."""
    return os.environ.get(_TRACING_V2_KEY, "").lower() in ("true", "1", "yes")


def apply_tracing_from_settings() -> bool:
    """Enable or disable tracing based on settings (e.g. at startup).

    If settings.LANGCHAIN_TRACING_V2 is True and LANGCHAIN_API_KEY is set,
    enables tracing. Otherwise disables it.

    Returns:
        True if tracing is now enabled, False otherwise.
    """
    if settings.LANGCHAIN_TRACING_V2 and is_tracing_available():
        return enable_tracing()
    disable_tracing()
    return False


def post_ranking_feedback_to_langsmith(reward_record: "RewardRecord") -> None:
    """Post reward signal metadata to LangSmith and (optionally) append run history.

    Tags the current LangSmith run with:
      - "reward:pass" or "reward:fail" depending on reward_record.reward
      - "judge:<model_slug>" using the first segment of the judge model name

    When settings.FEEDBACK_ENABLED is True, also appends the record to
    data/feedback/run_history.jsonl via append_run_history().

    Args:
        reward_record: RewardRecord produced by compute_reward().
    """
    reward_tag = "reward:pass" if reward_record.reward else "reward:fail"
    model_slug = reward_record.judge_model.split("/")[-1]
    judge_tag = f"judge:{model_slug}"

    if is_tracing_available() and is_tracing_enabled():
        try:
            from langsmith import Client as LangSmithClient

            client = LangSmithClient(
                api_key=settings.LANGCHAIN_API_KEY,
                api_url=settings.LANGCHAIN_ENDPOINT,
            )
            runs = list(
                client.list_runs(
                    project_name=settings.LANGCHAIN_PROJECT,
                    limit=1,
                )
            )
            if runs:
                client.update_run(
                    runs[0].id,
                    tags=[reward_tag, judge_tag],
                    extra={"feedback": reward_record.model_dump()},
                )
                logger.info(
                    "LangSmith run {} tagged: {}, {}",
                    runs[0].id,
                    reward_tag,
                    judge_tag,
                )
        except Exception as e:
            logger.debug("LangSmith tag update skipped ({}): {}", type(e).__name__, e)
    else:
        logger.info(
            "Feedback signal: {} {} mean_tech={} mean_ped={} pass_rate={}",
            reward_tag,
            judge_tag,
            reward_record.mean_tech,
            reward_record.mean_ped,
            reward_record.pass_rate,
        )

    if settings.FEEDBACK_ENABLED:
        from src.utils.feedback_utils import append_run_history

        append_run_history(reward_record)
        logger.debug(
            "Run history appended (prompt_version={}, reward={})",
            reward_record.prompt_version,
            reward_record.reward,
        )
