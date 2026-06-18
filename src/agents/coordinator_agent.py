"""Coordinator agent: graph entry point and high-level execution.

The coordinator runs as the first node in the LangGraph pipeline. It prepares
initial state (e.g. organized_challenges, challenge_ids from PROCESSED_DIR) so
validation, content generation, mapping, and ranking agents have inputs. It does
not run the full pipeline itself; the graph invokes the other agents in sequence.

Responsibilities:
    - Prepare state at pipeline start (scan PROCESSED_DIR for challenge paths).
    - Log pipeline start and challenge count.
    - Support end-of-run summary when invoked (caller can use final_state from
      app.invoke() to log errors, ranking_reports, etc.).

Workflow:
    START → coordinator (this) → validator → content_generation → mapping → ranking → (hitl?) → END

Usage:
    from src.core.graph import app
    from src.core.state import AgentState

    state = AgentState()
    final_state = app.invoke(state)
    # End-of-run: inspect final_state.errors, final_state.ranking_reports, etc.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import List

from loguru import logger

from src.config.settings import settings
from src.core.state import AgentState


def _prepare_organized_challenges() -> tuple[List[Path], List[str]]:
    """Scan PROCESSED_DIR for raw_challenges category/challenge dirs.

    Returns:
        (paths, challenge_ids): List of challenge paths and list of "category/name" IDs.
    """
    paths: List[Path] = []
    ids: List[str] = []
    processed = getattr(settings, "PROCESSED_DIR", None)
    if not processed or not processed.is_dir():
        return paths, ids
    root = processed / settings.RAW_CHALLENGES_SOURCE.name
    if not root.is_dir():
        logger.debug(
            "Coordinator: no {} under PROCESSED_DIR",
            settings.RAW_CHALLENGES_SOURCE.name,
        )
        return paths, ids
    for category_dir in sorted(root.iterdir()):
        if not category_dir.is_dir():
            continue
        for challenge_dir in sorted(category_dir.iterdir()):
            if challenge_dir.is_dir():
                paths.append(challenge_dir)
                ids.append(f"{category_dir.name}/{challenge_dir.name}")
    return paths, ids


def run_coordinator_agent(state: AgentState) -> AgentState:
    """Prepare initial state for the pipeline (entry node).

    If organized_challenges is empty, scans PROCESSED_DIR/{RAW_CHALLENGES_SOURCE.name} and
    sets organized_challenges and challenge_ids. Otherwise leaves state as-is.
    Sets current_agent to "coordinator" and logs the challenge count.

    Args:
        state: Current pipeline state (may be minimal at start).

    Returns:
        Updated state with organized_challenges and challenge_ids set when empty.
    """
    if state.organized_challenges and state.challenge_ids:
        logger.info(
            "Coordinator: state already has {} challenges; proceeding",
            len(state.challenge_ids),
        )
        return replace(state, current_agent="coordinator")

    paths, ids = _prepare_organized_challenges()
    if not paths:
        logger.warning(
            "Coordinator: no challenge paths found under PROCESSED_DIR/{RAW_CHALLENGES_SOURCE.name}"
        )
        return replace(
            state,
            organized_challenges=paths,
            challenge_ids=ids,
            current_agent="coordinator",
        )

    # When content_generation_subset_ids is set, restrict to that subset so validator
    # and other nodes only process those challenges (e.g. --content-subset pwn/p-xml,crypto/msghash)
    subset = state.content_generation_subset_ids
    if subset and len(subset) > 0:
        subset_set = set(subset)
        paths = [p for p, i in zip(paths, ids) if i in subset_set]
        ids = [i for i in ids if i in subset_set]
        if not paths:
            logger.warning(
                "Coordinator: no challenges match content_subset %s; check IDs exist",
                subset,
            )
        else:
            logger.info(
                "Coordinator: restricted to %d challenges from content_subset: %s",
                len(ids),
                ids,
            )

    logger.info(
        "Coordinator: prepared state with {} challenges for validate → content_generation → mapping → ranking",
        len(ids),
    )
    return replace(
        state,
        organized_challenges=paths,
        challenge_ids=ids,
        current_agent="coordinator",
    )


async def coordinator_agent(state: AgentState) -> AgentState:
    """Entry-point node for the LangGraph pipeline.

    Prepares initial state (organized_challenges, challenge_ids) so downstream
    agents (validation, content generation, mapping, ranking) can run. The graph
    main entry goes through this coordinator.

    Args:
        state: Current pipeline state (typically minimal at start).

    Returns:
        AgentState: Updated state with organized_challenges and challenge_ids set.
    """
    return run_coordinator_agent(state)
