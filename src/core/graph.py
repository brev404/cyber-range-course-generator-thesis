"""LangGraph agent workflow orchestration.

This module defines the StateGraph that orchestrates the multi-agent content
creation pipeline. LangGraph manages agent sequencing, state transitions,
routing decisions, and error handling.

Architecture Overview:
    The content creation pipeline is implemented as a directed acyclic graph
    where each agent is a node and transitions are edges. The graph:
    1. Routes challenges through agents in sequence
    2. Manages state updates at each step
    3. Handles errors and logs transitions
    4. Supports conditional routing (e.g., HITL feedback)

Agent Network:

    START
      ↓
    ┌─────────────────────────────────────┐
    │  Coordinator Agent (entry)          │
    │  Prepares state (organized paths)   │
    │  Output: organized_challenges, ids  │
    └─────────────────┬───────────────────┘
                      ↓
    ┌─────────────────────────────────────┐
    │  Validator Agent (Python-only, no LLM) │
    │  Checks structural requirements     │
    │  Uses: validate_challenge + rule-based Step 0 │
    │  Input: organized_challenges        │
    │  Output: validation_reports         │
    └─────────────────┬───────────────────┘
                      ↓
    ┌─────────────────────────────────────┐
    │  Content Generation Agent (LLM)     │
    │  Creates generated courses & solve scripts │
    │  Input: organized_challenges        │
    │  Output: generated_courses, code     │
    └─────────────────┬───────────────────┘
                      ↓
    ┌─────────────────────────────────────┐
    │  Course Terminology Checker (Python-only)             │
    │  Validates ATT&CK/CWE/OWASP IDs in generated courses │
    │  Input: generated_courses           │
    │  Output: course_terminology_issues   │
    │  Mode: off/annotate/warn/block → mapping or refinement_step │
    └─────────────────┬───────────────────┘
                      ↓
    ┌─────────────────────────────────────┐
    │  Mapping Agent (Python-only, no LLM) │
    │  Tags generated courses: ATT&CK, CWE, WSTG │
    │  Rule-based ID load, parse, validate │
    │  Input: generated_courses           │
    │  Output: writeup_mappings           │
    └─────────────────┬───────────────────┘
                      ↓
    ┌─────────────────────────────────────┐
    │  Ranking Agent (LLM)                │
    │  Evaluates quality & difficulty     │
    │  Input: generated_courses, etc.     │
    │  Output: ranking_reports            │
    └──────────┬──────────────────┬───────┘
               │                  │
         (all scores ≥              (any score <
      RANKING_PASS_THRESHOLD)   RANKING_PASS_THRESHOLD)
               │                  │
               ↓                  ↓ (up to MAX_REFINEMENT_ROUNDS)
             END         ┌──────────────────────┐
                         │  refinement_step     │
                         │  increment count;    │
                         │  set subset IDs      │
                         └──────────┬───────────┘
                                    │
                                    ↓
                          content_generation → mapping → ranking
                          (if max rounds reached → HITL instead)
                                    │
                        ┌──────────────────────┐
                        │  HITL Agent          │
                        │  interrupt/resume    │
                        └──────────┬───────────┘
                                   │
                    approved /     │    not approved &
                    max iter       │    under max iter
                         ↓         ↓
                       END    content_generation

State Flow:
    State flows left-to-right through the graph. Each agent:
    1. Receives AgentState with input data
    2. Processes based on its role
    3. Updates relevant fields in state
    4. Returns state to graph for routing

    Example state updates:
    - Coordinator: (empty state) → organized_challenges, challenge_ids
    - Validator: organized_challenges → validation_reports
    - Content Gen: organized_challenges → generated_courses, generated_solve_scripts
    - Mapping: generated_courses → writeup_mappings
    - Ranking: generated_courses, writeup_mappings → ranking_reports
    - HITL: ranking_reports → human_feedback + iteration_count

Routing Decisions:
    The graph uses conditional routing at the Ranking Agent and after HITL:
    - Ranking: checks per-persona scores (technical_review.score, pedagogical_review.score)
      against RANKING_PASS_THRESHOLD (default 9.0, set in .env).
      All scores ≥ threshold → END.
      Any score < threshold AND refinement_count < MAX_REFINEMENT_ROUNDS →
        refinement_step → content_generation → mapping → ranking (loop).
      Any score < threshold AND max rounds reached → HITL.
    - HITL: If human approved or iteration_count >= max_hitl_iterations → END;
      else → content_generation (refinement), then mapping → ranking again
    - HITL node uses interrupt() when available; resume with Command(resume=...)
      and same config (thread_id) so feedback is merged and loop continues

Error Handling:
    - Each agent catches exceptions and adds to state.errors
    - Graph logs errors and continues processing other challenges
    - Error states don't cause pipeline termination
    - Errors surfaced in final report

Functions:
    create_graph(): Factory function building and compiling the StateGraph

Classes:
    StateGraph: LangGraph orchestrator (imported from langgraph)

Usage:
    from src.core.graph import app

    # Run a batch of challenges through the pipeline
    input_state = AgentState(
        challenges=[challenge1, challenge2],
        challenge_ids=["crypto_001", "crypto_002"]
    )

    # Execute the graph (invoker: synchronous, stream: streaming output)
    final_state = app.invoke(input_state)

    # Access results
    for report in final_state.ranking_reports:
        logger.info("%s: %s/10", report.challenge_id, report.overall_score)

    # Check for errors
    if final_state.errors:
        for error in final_state.errors:
            logger.warning(f"Error in {error['agent']}: {error['message']}")

Configuration:
    LangSmith Tracing (optional debugging):
        Set LANGCHAIN_TRACING_V2=true in .env to trace all agent calls
        Traces visible at https://smith.langchain.com under "cyber-range-validator"

    Logger Configuration:
        Each agent should log progress:
        logger.info(f"Processing {state.challenge_ids}")
        logger.debug(f"Found {len(state.challenges)} challenges")
        logger.error(f"Failed on {challenge_id}", exc_info=True)

Extensibility:
    To add a new agent to the graph:
    1. Define new agent function in src/agents/
    2. Register in StateGraph with graph.add_node("agent_name", agent_fn)
    3. Add edge from previous agent: graph.add_edge("previous", "agent_name")
    4. Update routing if needed: graph.add_conditional_edges(...)

Notes:
    - LLM vs Python-only: Validator and Mapping are Python-only (no LLM).
      Content Generation and Ranking use the LLM. See docs/reference/FLOW_AND_LLM.md.
    - Graph is compiled once at import time for performance
    - State is immutable: agents create new state objects
    - Parallel processing not implemented (sequential for now)
    - Can be extended to parallel execution using send_keys parameter
"""

import time
from dataclasses import replace
from typing import Callable

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.agents.content_generation_agent import run_content_generation_agent
from src.agents.coordinator_agent import run_coordinator_agent
from src.agents.course_terminology_agent import run_course_terminology_checker
from src.agents.hitl_agent import run_hitl_agent
from src.agents.mapping_agent import run_mapping_agent
from src.agents.ranking_agent import run_ranking_agent
from src.agents.validation_agent import run_validation_agent
from src.config.settings import settings as app_settings
from src.core.state import AgentState
from src.models.report_models import IssueSeverity
from src.pipeline.repair_challenge_structure import (
    has_repairable_issues,
    repair_challenge,
)
from src.utils.logging_config import setup_logging

# Configure logging for graph orchestration
logger = setup_logging("graph")

# Node wrappers log "Graph: entering node 'X'" and "Graph: finished node 'X' (Ns)".
# If the pipeline gets stuck, the last log line will be "entering node 'X'" without a "finished" line.
# Run with: venv/bin/python -c "from src.core.graph import app; app.invoke(src.core.state.AgentState())"


def _wrap_sync_node(
    node_name: str, fn: Callable[[AgentState], AgentState]
) -> Callable[[AgentState], AgentState]:
    """Wrap a sync node to log enter/finish and duration; helps pinpoint where the pipeline gets stuck.
    GraphInterrupt (HITL pause) is re-raised without logging as ERROR."""
    try:
        from langgraph.errors import GraphInterrupt
    except ImportError:
        GraphInterrupt = type(
            "_NeverRaised", (BaseException,), {}
        )  # sentinel: never raised

    def wrapper(state: AgentState) -> AgentState:
        logger.info("Graph: entering node '%s'", node_name)
        t0 = time.perf_counter()
        try:
            result = fn(state)
            elapsed = time.perf_counter() - t0
            logger.info("Graph: finished node '%s' (%.2fs)", node_name, elapsed)
            return result
        except GraphInterrupt:
            elapsed = time.perf_counter() - t0
            logger.info(
                "Graph: node '%s' paused for human input after %.2fs (resume with Command(resume=...))",
                node_name,
                elapsed,
            )
            raise
        except Exception as e:
            elapsed = time.perf_counter() - t0
            logger.error(
                "Graph: node '%s' failed after %.2fs: %s", node_name, elapsed, e
            )
            raise

    return wrapper


def _route_ranking_decision(state: AgentState) -> str:
    """Route to END, refinement_step (retry content gen + only failing persona(s)), or HITL.

    Uses per-persona scores: refine if technical_review.score or pedagogical_review.score < threshold
    (not just overall). Refinement step sets ranking_retest_technical_ids / ranking_retest_pedagogical_ids
    so ranking re-runs only the persona(s) that were below threshold.

    When MAX_REFINEMENT_ROUNDS is reached, branches on MAX_REFINEMENT_STRATEGY:
    - "hitl": route to HITL (default)
    - "accept_best": accept current scores and route to END
    - "soft_accept": accept if overall_score >= SOFT_ACCEPT_THRESHOLD, else HITL

    R2 (Wave 2): Early-exit per challenge — if score gain < 0.5 since last round,
    that challenge is considered stuck and excluded from further refinement. When ALL
    still-failing challenges are stuck (early-exit), route to END instead of refinement.
    """
    if not state.ranking_reports:
        return END
    threshold = getattr(app_settings, "RANKING_PASS_THRESHOLD", 9.0)
    tech_threshold = getattr(app_settings, "RANKING_TECHNICAL_THRESHOLD", threshold)
    ped_threshold = getattr(app_settings, "RANKING_PEDAGOGICAL_THRESHOLD", threshold)
    max_rounds = getattr(app_settings, "MAX_REFINEMENT_ROUNDS", 5)
    strategy = getattr(app_settings, "MAX_REFINEMENT_STRATEGY", "hitl")
    soft_threshold = getattr(app_settings, "SOFT_ACCEPT_THRESHOLD", 7.0)
    score_history: dict = dict(
        getattr(state, "score_history_per_challenge", None) or {}
    )

    # Track whether any challenge still needs refinement (after early-exit check)
    needs_refinement = False

    for report in state.ranking_reports:
        # K2: per-dimension thresholds (defaults equal RANKING_PASS_THRESHOLD → backward compat)
        tech_ok = (
            report.technical_review.score >= threshold
            and report.technical_review.score >= tech_threshold
        )
        ped_ok = (
            report.pedagogical_review.score >= threshold
            and report.pedagogical_review.score >= ped_threshold
        )
        if tech_ok and ped_ok:
            continue

        # Max rounds — apply configured strategy
        if state.refinement_count >= max_rounds:
            best_score = report.overall_score
            if strategy == "accept_best":
                logger.info(
                    "Ranking: %s tech=%s ped=%s; max refinement rounds (%s) reached; strategy=accept_best; accepting score=%.1f → END",
                    report.challenge_id,
                    report.technical_review.score,
                    report.pedagogical_review.score,
                    max_rounds,
                    best_score,
                )
                return END
            elif strategy == "soft_accept":
                if best_score >= soft_threshold:
                    logger.info(
                        "Ranking: %s overall=%.1f >= soft_accept_threshold=%.1f; max rounds (%s) reached; accepting → END",
                        report.challenge_id,
                        best_score,
                        soft_threshold,
                        max_rounds,
                    )
                    continue
                else:
                    logger.info(
                        "Ranking: %s overall=%.1f < soft_accept_threshold=%.1f; max rounds (%s) reached; routing to HITL",
                        report.challenge_id,
                        best_score,
                        soft_threshold,
                        max_rounds,
                    )
                    return "hitl"
            else:
                logger.info(
                    "Ranking: %s tech=%s ped=%s; max refinement rounds (%s) reached; routing to HITL",
                    report.challenge_id,
                    report.technical_review.score,
                    report.pedagogical_review.score,
                    max_rounds,
                )
                return "hitl"

        # R2: early-exit if score hasn't improved enough since last round
        cid = report.challenge_id
        history = score_history.get(cid, [])
        if len(history) >= 2:
            gain = history[-1] - history[-2]
            if gain < 0.5:
                logger.info(
                    "Ranking R2: %s score_history=%s gain=%.2f < 0.5 → early-exit (accepting best score)",
                    cid,
                    history,
                    gain,
                )
                continue  # skip this challenge from refinement, don't set needs_refinement

        # This challenge still needs refinement
        logger.info(
            "Ranking: %s tech=%s ped=%s (threshold %.1f); auto-refine (round %s/%s) → content_generation",
            report.challenge_id,
            report.technical_review.score,
            report.pedagogical_review.score,
            threshold,
            state.refinement_count + 1,
            max_rounds,
        )
        needs_refinement = True

    if needs_refinement:
        return "refinement_step"
    return END


def _refinement_step_node_fn(state: AgentState) -> AgentState:
    """Increment refinement count; collect judge feedback; update score history.

    Sets content_generation_subset_ids from terminology issues or
    per-persona scores (ranking). If any report has truncation in justification,
    bumps content_max_tokens_override for that challenge.

    R1 (Wave 2): collects improvements lists from failing persona(s) and stores
    them in state.prior_improvements_per_challenge so content_generation can
    inject "MUST address" constraints into the gen prompt.

    R2 (Wave 2): appends current overall_score to state.score_history_per_challenge
    for early-exit detection in _route_ranking_decision.
    """
    # Terminology-driven refinement: when routed from course_terminology_checker (block mode)
    terminology_bad_ids = (
        list(state.course_terminology_issues.keys())
        if state.course_terminology_issues
        else []
    )
    # Score-driven refinement: when routed from ranking
    threshold = getattr(app_settings, "RANKING_PASS_THRESHOLD", 9.0)
    tech_threshold = getattr(app_settings, "RANKING_TECHNICAL_THRESHOLD", threshold)
    ped_threshold = getattr(app_settings, "RANKING_PEDAGOGICAL_THRESHOLD", threshold)
    reports = state.ranking_reports or []
    low_tech_ids = [
        r.challenge_id
        for r in reports
        if r.technical_review.score < max(threshold, tech_threshold)
    ]
    low_ped_ids = [
        r.challenge_id
        for r in reports
        if r.pedagogical_review.score < max(threshold, ped_threshold)
    ]
    low_ids = list(dict.fromkeys(low_tech_ids + low_ped_ids))  # union, preserve order
    content_subset = (
        terminology_bad_ids
        if terminology_bad_ids
        else (low_ids if low_ids else state.content_generation_subset_ids)
    )

    content_max_override = dict(state.content_max_tokens_override or {})
    bump = getattr(app_settings, "CONTENT_GENERATION_MAX_TOKENS", 12000) + 2000
    for r in reports:
        just = (getattr(r.technical_review, "justification", "") or "") + (
            getattr(r.pedagogical_review, "justification", "") or ""
        )
        if "truncat" in just.lower() or "partial json" in just.lower():
            content_max_override[r.challenge_id] = max(
                content_max_override.get(r.challenge_id, 0), bump
            )

    # R1: collect judge improvements from failing persona(s) per challenge
    prior_improvements: dict = dict(
        getattr(state, "prior_improvements_per_challenge", None) or {}
    )
    # D2: collect per-dim scores for EVERY challenge with a report (regardless of
    # pass/fail) so the next gen round can surface the actual scoreline alongside
    # the improvement list. This is parallel to prior_improvements_per_challenge
    # but kept as a separate map (mirrors state-field naming).
    prior_dim_scores: dict = dict(
        getattr(state, "prior_dim_scores_per_challenge", None) or {}
    )
    for r in reports:
        cid = r.challenge_id
        combined: list = []
        if r.technical_review.score < threshold:
            combined.extend(r.technical_review.improvements or [])
        if r.pedagogical_review.score < threshold:
            combined.extend(r.pedagogical_review.improvements or [])
        if combined:
            prior_improvements[cid] = combined
            logger.debug(
                "Refinement R1: collected %d improvements for %s", len(combined), cid
            )
        # D2: record both scores so content_generation sees "tech=8.5, ped=6.5"
        prior_dim_scores[cid] = {
            "technical": float(r.technical_review.score),
            "pedagogical": float(r.pedagogical_review.score),
        }

    # R2: append current overall scores to score_history_per_challenge
    score_history: dict = dict(
        getattr(state, "score_history_per_challenge", None) or {}
    )
    for r in reports:
        cid = r.challenge_id
        existing_hist = list(score_history.get(cid, []))
        existing_hist.append(r.overall_score)
        score_history[cid] = existing_hist
        logger.debug("Refinement R2: %s score_history=%s", cid, existing_hist)

    return replace(
        state,
        refinement_count=state.refinement_count + 1,
        current_agent="refinement_step",
        content_generation_subset_ids=content_subset,
        ranking_retest_technical_ids=low_tech_ids if low_tech_ids else None,
        ranking_retest_pedagogical_ids=low_ped_ids if low_ped_ids else None,
        content_max_tokens_override=(
            content_max_override
            if content_max_override
            else state.content_max_tokens_override
        ),
        prior_improvements_per_challenge=prior_improvements,
        prior_dim_scores_per_challenge=prior_dim_scores,
        score_history_per_challenge=score_history,
    )


def create_graph() -> StateGraph:
    """Create and compile the agent workflow StateGraph.

    Builds the directed graph of agents and their connections, then compiles
    it for execution. The graph routes data through agents in sequence and
    handles state updates at each step.

    The resulting graph:
    - Takes AgentState as input
    - Routes through agents: Analyzer → Organizer → Validator → Content Gen → Ranking → (HITL)
    - Returns final AgentState with all results

    Returns:
        StateGraph: Compiled and ready-to-run agent workflow
        The graph supports:
        - invoke(state): Synchronous execution returning final state
        - stream(state): Generator yielding state updates as pipeline runs
        - batch(states): Process multiple batches concurrently

    Raises:
        ImportError: If agent modules not properly installed
        ValueError: If agent functions have invalid signatures

    Example:
        >>> from src.core.graph import app  # Pre-compiled at module level
        >>> input_state = AgentState(challenges=[challenge1, challenge2])
        >>> final_state = app.invoke(input_state)
        >>> print(f"Processed {len(final_state.ranking_reports)} challenges")

    Notes:
        - Graph is compiled once at module import time
        - Compilation includes type checking and signature validation
        - Compiled graph is 10-100x faster than uncompiled
        - Most users should use the pre-compiled 'app' rather than calling this
    """
    # Create graph with AgentState as state schema
    graph = StateGraph(AgentState)

    # Add agent nodes with enter/finish logging so we can see where the pipeline gets stuck
    graph.add_node("coordinator", _wrap_sync_node("coordinator", run_coordinator_agent))
    graph.add_node("validator", _wrap_sync_node("validator", run_validation_agent))

    def _content_generation_node(state: AgentState) -> AgentState:
        return run_content_generation_agent(
            state, write_to_disk=app_settings.WRITE_COURSES_TO_DISK
        )

    graph.add_node(
        "content_generation",
        _wrap_sync_node("content_generation", _content_generation_node),
    )
    graph.add_node(
        "course_terminology_checker",
        _wrap_sync_node("course_terminology_checker", run_course_terminology_checker),
    )
    graph.add_node("mapping", _wrap_sync_node("mapping", run_mapping_agent))
    graph.add_node("ranking", _wrap_sync_node("ranking", run_ranking_agent))
    graph.add_node("hitl", _wrap_sync_node("hitl", run_hitl_agent))

    graph.add_node(
        "refinement_step", _wrap_sync_node("refinement_step", _refinement_step_node_fn)
    )

    def _repair_node(state: AgentState) -> AgentState:
        """Run structural repair on all organized challenges, then re-validate.

        Calls repair_challenge per challenge path, then re-runs the validation
        agent inline so updated validation_reports reflect the repaired filesystem.
        Sets repair_attempted=True to prevent infinite repair loops.
        """
        from pathlib import Path

        from src.agents.validation_agent import run_validation_agent

        for challenge_path in state.organized_challenges:
            path = Path(challenge_path)
            if path.is_dir():
                repair_challenge(path)

        updated_state = replace(state, repair_attempted=True, current_agent="repair")
        return run_validation_agent(updated_state)

    graph.add_node("repair", _wrap_sync_node("repair", _repair_node))

    # Edges: START → coordinator → validator → (conditional) → content_generation → mapping → ranking → (conditional)
    graph.add_edge(START, "coordinator")
    graph.add_edge("coordinator", "validator")

    # Route after validation — repair, early exit, or content_generation
    def route_after_validation(state: AgentState) -> str:
        """Route to repair, content_generation, or END after validation.

        Priority:
        1. If attempt_repair is True and repair not yet attempted and
           there are repairable issues → repair node (which re-validates).
        2. If stop_on_validation_fail and any HIGH/CRITICAL issue → END.
        3. Default: content_generation.
        """
        if state.attempt_repair and not state.repair_attempted:
            if has_repairable_issues(state.validation_reports):
                logger.info("Repair: repairable issues found; routing to repair node")
                return "repair"

        if not state.stop_on_validation_fail:
            return "content_generation"
        for report in state.validation_reports:
            for issue in report.issues:
                if issue.severity in (IssueSeverity.HIGH, IssueSeverity.CRITICAL):
                    logger.info(
                        "Validation early exit: critical issue in %s (%s); routing to END",
                        report.challenge_id,
                        issue.severity.value,
                    )
                    return END
        return "content_generation"

    graph.add_conditional_edges("validator", route_after_validation)
    # After repair, re-enter routing (repair node re-ran validation internally)
    graph.add_conditional_edges("repair", route_after_validation)
    graph.add_edge("content_generation", "course_terminology_checker")

    def route_after_course_terminology(state: AgentState) -> str:
        """Route to mapping or refinement_step based on TERMINOLOGY_CHECK_MODE and issues.
        When mode=block and any challenge has terminology issues, route to refinement_step.
        When max refinement rounds reached, proceed to mapping to avoid infinite loop.
        """
        mode = getattr(app_settings, "TERMINOLOGY_CHECK_MODE", "warn")
        if mode != "block":
            return "mapping"
        issues = state.course_terminology_issues or {}
        bad_ids = [cid for cid, iss in issues.items() if iss]
        if not bad_ids:
            return "mapping"
        max_rounds = getattr(app_settings, "MAX_REFINEMENT_ROUNDS", 5)
        if state.refinement_count >= max_rounds:
            logger.info(
                "Course terminology: %s challenge(s) with issues but max refinement rounds (%s) reached; proceeding to mapping",
                len(bad_ids),
                max_rounds,
            )
            return "mapping"
        logger.info(
            "Course terminology (block): routing %s challenge(s) to refinement_step",
            len(bad_ids),
        )
        return "refinement_step"

    graph.add_conditional_edges(
        "course_terminology_checker", route_after_course_terminology
    )
    graph.add_edge("mapping", "ranking")
    graph.add_edge("refinement_step", "content_generation")

    graph.add_conditional_edges("ranking", _route_ranking_decision)

    # After HITL, either END (approved or max iterations) or refine (content_generation)
    def route_after_hitl(state: AgentState) -> str:
        """Route to content_generation for refinement or END if approved / max iterations."""
        if state.hitl_approved:
            logger.info("HITL: human approved; routing to END")
            return END
        if state.iteration_count >= state.max_hitl_iterations:
            logger.info(
                "HITL: max iterations (%s) reached; routing to END",
                state.max_hitl_iterations,
            )
            return END
        logger.info("HITL: routing to content_generation for refinement")
        return "content_generation"

    graph.add_conditional_edges("hitl", route_after_hitl)

    # Compile graph with checkpointer so HITL interrupt can pause and resume
    checkpointer = MemorySaver()
    compiled_graph = graph.compile(checkpointer=checkpointer)

    logger.info("Agent workflow graph created and compiled successfully")
    logger.debug(f"Graph has {len(graph.nodes)} nodes and {len(graph.edges)} edges")

    return compiled_graph


# Pre-compile graph at module import time for performance
# Most code should use this 'app' rather than calling create_graph()
app = create_graph()
"""Pre-compiled StateGraph for the content creation pipeline.

Usage:
    from src.core.graph import app

    # Execute pipeline
    input_state = AgentState(challenges=[...])
    final_state = app.invoke(input_state)

    # Or stream results as they process
    for step in app.stream(input_state):
        logger.info("Agent stepped: %s", step)
"""
