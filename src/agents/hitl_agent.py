"""Human-in-the-Loop Agent for iterative content improvement.

This agent enables human feedback integration into the automated pipeline.
When AI-generated content scores below quality thresholds, this agent collects
human feedback and coordinates refinements through the Content Generation Agent.

The graph can interrupt at the HITL node for human feedback (LangGraph interrupt).
When compiled with a checkpointer, execution pauses and the caller receives
the review payload in result["__interrupt__"]. The caller then resumes with
Command(resume=human_response) so the resume value is merged into state and
routing continues (refine → content_generation or approve → END).

Feedback Collection:
    Presented to Human (JSON-serializable payload for interrupt):
    - Challenge IDs and scores needing review (Rich table or plain text)
    - Routing cause: quality failure vs max refinements reached
    - Ranking summary and writeup excerpts

    Human Provides (resume payload) — new action format:
    - {"action": "approve_all"}              — accept all, route to END
    - {"action": "approve_ids", "ids": [...]} — accept subset; rest re-refine
    - {"action": "edit_retry", "hint": "..."}— inject operator hint, re-run
    - {"action": "abort"}                    — clean stop (SystemExit 0)

    Legacy format still accepted:
    - {"approved": True}                     — same as approve_all
    - {"approved": False, "human_feedback": {...}} — same as edit_retry

Interrupt/Checkpoint:
    - interrupt() is called inside the HITL node so the graph pauses for human input.
    - Requires: graph compiled with a checkpointer (e.g. MemorySaver) and
      config={"configurable": {"thread_id": "..."}} so resume uses the same thread.
    - Resume: app.invoke(Command(resume={"action": "approve_all"}), config=config)
    - If LangGraph does not provide interrupt/Command (older versions), the agent
      falls back to updating state only (no pause); document use of checkpointer for HITL.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from loguru import logger

from src.config.settings import settings as app_settings
from src.core.state import AgentState

# Optional: LangGraph interrupt for human-in-the-loop pause/resume
try:
    from langgraph.types import interrupt

    _HAS_INTERRUPT = True
except ImportError:
    _HAS_INTERRUPT = False
    interrupt = None  # type: ignore[misc, assignment]


def _build_review_payload(state: AgentState) -> dict[str, Any]:
    """Build a JSON-serializable payload for human review (idempotent; no side effects).

    Includes routing_cause, per-challenge scores, dimension scores, and writeup excerpts.
    """
    max_rounds = getattr(app_settings, "MAX_REFINEMENT_ROUNDS", 5)
    threshold = getattr(app_settings, "RANKING_PASS_THRESHOLD", 9.0)
    at_max_rounds = state.refinement_count >= max_rounds

    if at_max_rounds:
        routing_cause = f"max refinements reached (round {state.refinement_count})"
        best_score: float = max(
            (r.overall_score for r in state.ranking_reports), default=0.0
        )
    else:
        routing_cause = "quality failure"
        best_score = 0.0

    payload: dict[str, Any] = {
        "instruction": (
            "Resume with action: approve_all / approve <id1,id2> / "
            "edit_retry / abort"
        ),
        "iteration": state.iteration_count + 1,
        "max_iterations": state.max_hitl_iterations,
        "routing_cause": routing_cause,
        "ranking_pass_threshold": threshold,
        "challenge_dimension_scores": state.challenge_dimension_scores,
        "challenges": [],
    }
    if at_max_rounds:
        payload["max_rounds_note"] = (
            f"Max refinement rounds reached (round {state.refinement_count}). "
            f"Best score: {best_score:.1f}."
        )

    writeup_excerpt_len = 400
    for report in state.ranking_reports:
        if report.overall_score > 7:
            continue
        challenge_id = report.challenge_id
        writeup = (state.generated_courses.get(challenge_id) or "")[
            :writeup_excerpt_len
        ]
        if (
            state.generated_courses.get(challenge_id)
            and len(state.generated_courses[challenge_id]) > writeup_excerpt_len
        ):
            writeup += "..."
        payload["challenges"].append(
            {
                "challenge_id": challenge_id,
                "overall_score": report.overall_score,
                "pedagogical_score": getattr(report.pedagogical_review, "score", None),
                "technical_score": getattr(report.technical_review, "score", None),
                "pedagogical_justification": (
                    getattr(report.pedagogical_review, "justification", "")[:300]
                    if report.pedagogical_review
                    else ""
                ),
                "technical_justification": (
                    getattr(report.technical_review, "justification", "")[:300]
                    if report.technical_review
                    else ""
                ),
                "writeup_excerpt": writeup,
            }
        )

    if not payload["challenges"]:
        payload["challenges"] = [
            {
                "challenge_id": r.challenge_id,
                "overall_score": r.overall_score,
                "pedagogical_score": getattr(r.pedagogical_review, "score", None),
                "technical_score": getattr(r.technical_review, "score", None),
                "pedagogical_justification": getattr(
                    r.pedagogical_review, "justification", ""
                )[:200],
                "technical_justification": getattr(
                    r.technical_review, "justification", ""
                )[:200],
                "writeup_excerpt": (state.generated_courses.get(r.challenge_id) or "")[
                    :300
                ],
            }
            for r in state.ranking_reports
        ]

    return payload


def print_hitl_summary(payload: dict) -> None:
    """Print HITL review summary to stdout.

    Shows routing cause header, iteration counter, and a per-challenge table
    with technical score, pedagogical score, overall score, and failed dimensions.
    Safe to call from CLI after catching GraphInterrupt.
    """
    routing_cause = payload.get("routing_cause", "quality failure")
    iteration = payload.get("iteration", "?")
    max_iterations = payload.get("max_iterations", "?")
    challenges = payload.get("challenges", [])
    threshold = float(payload.get("ranking_pass_threshold", 9.0))
    dim_scores: dict = payload.get("challenge_dimension_scores") or {}

    print(f"\nRouting cause: {routing_cause}")
    print(f"HITL iteration {iteration} / {max_iterations}\n")

    if not challenges:
        return

    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        has_dims = bool(dim_scores)

        table = Table(
            title="Challenge Review Summary", show_header=True, header_style="bold"
        )
        table.add_column("challenge_id", style="cyan", no_wrap=True)
        table.add_column("technical", justify="right")
        table.add_column("pedagogical", justify="right")
        table.add_column("overall", justify="right")
        if has_dims:
            table.add_column("failed_dimensions")

        for c in challenges:
            cid = str(c.get("challenge_id", ""))
            tech = str(c.get("technical_score", "-"))
            ped = str(c.get("pedagogical_score", "-"))
            overall = str(c.get("overall_score", "-"))
            if has_dims:
                dims = dim_scores.get(cid, {})
                failed = sorted(
                    k
                    for k, v in dims.items()
                    if isinstance(v, (int, float)) and v < threshold
                )
                failed_str = ", ".join(failed) if failed else "-"
                table.add_row(cid, tech, ped, overall, failed_str)
            else:
                table.add_row(cid, tech, ped, overall)

        console.print(table)

    except ImportError:
        header = f"{'challenge_id':<40}  {'technical':>10}  {'pedagogical':>12}  {'overall':>8}"
        print(header)
        print("-" * len(header))
        for c in challenges:
            cid = str(c.get("challenge_id", ""))[:40]
            tech = str(c.get("technical_score", "-"))
            ped = str(c.get("pedagogical_score", "-"))
            overall = str(c.get("overall_score", "-"))
            failed_str = ""
            if dim_scores:
                dims = dim_scores.get(cid, {})
                failed = sorted(
                    k
                    for k, v in dims.items()
                    if isinstance(v, (int, float)) and v < threshold
                )
                if failed:
                    failed_str = f"  failed: {', '.join(failed)}"
            print(f"{cid:<40}  {tech:>10}  {ped:>12}  {overall:>8}{failed_str}")


def _parse_resume_value_extended(value: Any) -> dict:
    """Parse resume payload supporting legacy and new action formats.

    Returns a dict with keys:
        action:         "approve_all" | "approve_ids" | "edit_retry" | "abort"
                        | "legacy_approve" | "legacy_refine"
        ids:            list[str]              (approve_ids)
        hint:           str                    (edit_retry)
        human_feedback: dict[str, list[str]]   (legacy_refine)
    """
    if value is None:
        return {"action": "legacy_refine", "human_feedback": {}}
    if isinstance(value, bool):
        return {
            "action": "legacy_approve" if value else "legacy_refine",
            "human_feedback": {},
        }
    if not isinstance(value, dict):
        return {"action": "legacy_refine", "human_feedback": {}}

    action = value.get("action")

    if action == "approve_all":
        return {"action": "approve_all"}

    if action == "approve_ids":
        ids = value.get("ids", [])
        if isinstance(ids, str):
            ids = [s.strip() for s in ids.split(",") if s.strip()]
        return {"action": "approve_ids", "ids": list(ids)}

    if action == "edit_retry":
        return {"action": "edit_retry", "hint": str(value.get("hint", ""))}

    if action == "abort":
        return {"action": "abort"}

    # Legacy format: {"approved": bool, "human_feedback": {...}}
    approved = bool(value.get("approved", False))
    feedback = value.get("human_feedback") or {}
    if not isinstance(feedback, dict):
        feedback = {}
    else:
        feedback = {
            k: v if isinstance(v, list) else [str(v)] for k, v in feedback.items()
        }
    return {
        "action": "legacy_approve" if approved else "legacy_refine",
        "human_feedback": feedback,
    }


def run_hitl_agent(state: AgentState) -> AgentState:
    """Sync entry point for HITL node: interrupt for human feedback, then merge resume value.

    When LangGraph provides interrupt():
    1. Build review payload (idempotent).
    2. Call interrupt(payload); execution pauses and state is checkpointed.
    3. Caller resumes with Command(resume=human_response); interrupt() returns that value.
    4. Parse resume value (new action format or legacy), update state accordingly.

    Actions supported in resume value:
    - approve_all:              set hitl_approved=True → route to END
    - approve_ids (subset):     set content/ranking subset to non-approved IDs → refine rest
    - edit_retry:               inject operator hint into human_feedback → re-refine
    - abort:                    raise SystemExit(0)
    - legacy {"approved": bool}: backward-compatible

    Args:
        state: Pipeline state with ranking_reports and generated content for review.

    Returns:
        AgentState: Updated state with human_feedback, hitl_approved, iteration_count.
    """
    payload = _build_review_payload(state)

    if _HAS_INTERRUPT and interrupt is not None:
        try:
            human_response = interrupt(payload)
        except RuntimeError as e:
            if "runnable context" in str(e).lower() or "get_config" in str(e).lower():
                logger.debug(
                    "HITL: interrupt called outside graph context ({}); using state only",
                    e,
                )
                return replace(
                    state,
                    current_agent="hitl",
                    iteration_count=state.iteration_count + 1,
                )
            raise

        parsed = _parse_resume_value_extended(human_response)
        action = parsed["action"]

        if action == "abort":
            logger.info("HITL: operator aborted run cleanly")
            raise SystemExit(0)

        threshold = getattr(app_settings, "RANKING_PASS_THRESHOLD", 9.0)
        human_feedback = dict(state.human_feedback)
        hitl_approved = False
        content_subset = state.content_generation_subset_ids
        ranking_subset = state.ranking_subset_ids

        if action == "approve_all":
            hitl_approved = True

        elif action == "approve_ids":
            approved_ids = set(parsed.get("ids", []))
            all_ids = [r.challenge_id for r in state.ranking_reports]
            remaining = [cid for cid in all_ids if cid not in approved_ids]
            content_subset = remaining if remaining else None
            ranking_subset = remaining if remaining else None

        elif action == "edit_retry":
            hint = parsed.get("hint", "")
            if hint:
                for report in state.ranking_reports:
                    if report.overall_score < threshold:
                        cid = report.challenge_id
                        human_feedback[cid] = human_feedback.get(cid, []) + [
                            f"Operator hint: {hint}"
                        ]

        else:  # legacy_approve or legacy_refine
            if action == "legacy_approve":
                hitl_approved = True
            for cid, items in parsed.get("human_feedback", {}).items():
                human_feedback[cid] = human_feedback.get(cid, []) + list(items)

        return replace(
            state,
            current_agent="hitl",
            iteration_count=state.iteration_count + 1,
            human_feedback=human_feedback,
            hitl_approved=hitl_approved,
            content_generation_subset_ids=content_subset,
            ranking_subset_ids=ranking_subset,
        )

    # Fallback: no interrupt API (e.g. older LangGraph); merge any existing state.human_feedback
    logger.info(
        "HITL: interrupt not available (langgraph.types.interrupt); using state only. "
        "Compile graph with checkpointer and use interrupt for real HITL pause/resume."
    )
    return replace(
        state,
        current_agent="hitl",
        iteration_count=state.iteration_count + 1,
    )


async def hitl_agent(state: AgentState) -> AgentState:
    """Async entry point for HITL node; delegates to run_hitl_agent."""
    return run_hitl_agent(state)
