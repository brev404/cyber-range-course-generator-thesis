"""LangGraph state definition for the agent workflow.

This module defines the state object that flows through the LangGraph agent
network. The state represents the shared data structure passed between agents
in the content creation pipeline.

Information flow between agents (validate → content generation → mapping → ranking):
    - Coordinator: Prepares state (e.g. organized_challenges, challenge_ids).
    - Validation Agent: Reads organized_challenges; writes validation_reports.
    - Content Generation Agent: Reads organized_challenges / validation_reports;
      writes generated_courses, generated_solve_scripts.
    - Mapping Agent: Reads generated_courses; writes writeup_mappings.
    - Ranking Agent: Reads generated_courses, generated_solve_scripts, writeup_mappings;
      writes ranking_reports.
    - HITL Agent: Reads ranking_reports; writes human_feedback, iteration_count.

All agent outputs are part of state and typed consistently with src/models/report_models.

State Management:
    - State is immutable: agents return updated state via replace(), not in-place.
    - Each agent reads relevant fields and returns state with its outputs set.

Usage:
    State is managed internally by LangGraph. Each agent:
    1. Receives current state
    2. Performs computation
    3. Updates state with results
    4. Returns to graph for routing decision

    from src.core.graph import app  # Use compiled StateGraph

    # State is passed internally between agents
    # State updates accumulate as pipeline progresses

Classes:
    StateGraph: Main orchestrator connecting all agents (defined in graph.py)
    State: Data class representing pipeline state (to be defined here)

Notes:
    - State is immutable: agents return new state, don't modify in place
    - All state changes are logged for debugging and reproducibility
    - State snapshots can be saved for resuming interrupted pipelines
    - Vector embeddings stored separately in vector database
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.models.challenge_models import Challenge
from src.models.report_models import (
    EvaluationReport,
    RankingReport,
    ValidationIssue,
    ValidationReport,
    WriteupMapping,
)


@dataclass
class AgentState:
    """Shared state flowing through the agent workflow in LangGraph.

    This data class represents all information shared between agents in the
    content creation pipeline. Each agent reads relevant fields and updates
    the state with its results.

    Attributes:
        # Input Data
        challenges (List[Challenge]): Challenges being processed through pipeline
            Populated by: Analyzer Agent
            Consumed by: Organizer, Validator, Content Generation, Ranking agents

        challenge_ids (List[str]): IDs of challenges in current batch
            Used for tracking and logging

        # Analyzer Results
        analyzed_challenges (List[Dict[str, Any]]): File structure analysis results
            Contains: file paths, types, sizes, structure details
            Produced by: Analyzer Agent

        # Organizer Results
        organized_challenges (List[Path]): Paths to reorganized challenges
            Standard directory structure: cyberedu/{src,write-up,public,deploy}
            Produced by: Organizer Agent

        # Validator Results
        validation_reports (List[ValidationReport]): Completeness checks
            Contains: ChallengeChecklist, issues, structure_score
            Produced by: Validator Agent

        # Content Generation Results (course = project-generated output; see TERMINOLOGY.md)
        generated_courses (Dict[str, str]): Challenge ID → generated course text
            Contains: Step-by-step solution explanations
            Produced by: Content Generation Agent

        generated_solve_scripts (Dict[str, str]): Challenge ID → solver code mapping
            Contains: Working exploit code or solution scripts
            Produced by: Content Generation Agent

        # Course terminology: challenge_id → list of ValidationIssue from check_terminology
        course_terminology_issues (Dict[str, List[ValidationIssue]]): Terminology issues per challenge.
            terminology_ok[challenge_id] = len(issues) == 0 (derived).

        # Writeup Mapping
        writeup_mappings (Dict[str, WriteupMapping]): Challenge ID → taxonomy tags
            Contains: ATT&CK technique IDs, CWE IDs, OWASP WSTG scenario IDs
            Produced by: Mapping Agent

        # Ranking Results
        ranking_reports (List[RankingReport]): Quality evaluations
            Contains: overall_score, pedagogical_review, technical_review
            Produced by: Ranking Agent

        challenge_dimension_scores (Optional[Dict[str, Dict[str, float]]]): Per-challenge
            aggregated dimension scores from ranking agent. Maps challenge_id to
            a flat dict of dimension_name → score (float). Available for refinement and HITL.

        # Evaluation: optional automated evaluation
        evaluation_reports (List[EvaluationReport]): Rubric + optional Ragas/LLM
            Produced by: evaluation_service.evaluate_writeup(); can feed reporting
            or Ranking Agent.

        # Human-in-the-Loop (HITL)
        human_feedback (Dict[str, List[str]]): Challenge ID → feedback items
            User-provided suggestions for improvement
            Updated by: Human-in-the-Loop Agent (or injected on resume)
        hitl_approved (bool): When True, human approved content; route to END.
            Set by HITL agent from resume payload.
        max_hitl_iterations (int): Max refinement cycles before forcing END.
            Default 3; stopping condition with iteration_count.

        iteration_count (int): Number of improvement iterations performed
            Tracks how many refinement cycles have occurred
        refinement_count (int): Number of auto-refinement loops (ranking → content_generation)
            Used to cap retries before HITL when score < threshold.
        content_max_tokens_override (Optional[Dict[str, int]]): Challenge ID → max tokens for course output
            If set, overrides CONTENT_GENERATION_MAX_TOKENS for those challenges only.

        # Flow optimization: reduce token usage
        skip_ranking (bool): If True, ranking node skips LLM calls and leaves
            ranking_reports unchanged. Default False.
        ranking_subset_ids (Optional[List[str]]): If set, ranking runs only for
            these challenge IDs; others are skipped. Reduces tokens when set.
        content_generation_subset_ids (Optional[List[str]]): If set, content
            generation runs LLM only for these challenge IDs; others are skipped.
        stop_on_validation_fail (bool): If True, graph routes to END after
            validation when any report has a critical (e.g. HIGH) issue.

        attempt_repair (bool): If True, run the repair node after validation
            when repairable issues exist. Default False.
        repair_attempted (bool): Set to True by repair node after one repair
            pass; prevents infinite repair loops. Default False.

        # Status and Logging
        current_agent (str): Name of agent currently processing
            Examples: "analyzer", "validator", "content_generation"

        errors (List[Dict[str, Any]]): Errors encountered during processing
            Contains: agent name, error message, challenge ID, timestamp

        is_complete (bool): Whether pipeline execution is finished
            Used by StateGraph for termination condition

    Example:
        >>> state = AgentState(
        ...     challenges=[challenge1, challenge2],
        ...     challenge_ids=["crypto_001", "crypto_002"],
        ...     current_agent="analyzer"
        ... )
        >>>
        >>> # Agent processes and updates state
        >>> state.analyzed_challenges = [...]
        >>> state.current_agent = "organizer"
        >>>
        >>> # Next agent receives updated state
    """

    # Input challenges
    challenges: List[Challenge] = field(default_factory=list)
    challenge_ids: List[str] = field(default_factory=list)

    # Pipeline results
    analyzed_challenges: List[Dict[str, Any]] = field(default_factory=list)
    organized_challenges: List[Path] = field(default_factory=list)
    validation_reports: List[ValidationReport] = field(default_factory=list)
    generated_courses: Dict[str, str] = field(default_factory=dict)
    generated_solve_scripts: Dict[str, str] = field(default_factory=dict)
    course_terminology_issues: Dict[str, List[ValidationIssue]] = field(
        default_factory=dict
    )
    writeup_mappings: Dict[str, WriteupMapping] = field(default_factory=dict)
    ranking_reports: List[RankingReport] = field(default_factory=list)
    # Granular grading: challenge_id → aggregated dimension_name → score
    challenge_dimension_scores: Optional[Dict[str, Dict[str, float]]] = None
    evaluation_reports: List[EvaluationReport] = field(default_factory=list)

    # Human feedback and iteration (HITL & refinement)
    human_feedback: Dict[str, List[str]] = field(default_factory=dict)
    hitl_approved: bool = False
    max_hitl_iterations: int = 3
    iteration_count: int = 0
    refinement_count: int = 0
    content_max_tokens_override: Optional[Dict[str, int]] = None

    # Flow optimization: skip or subset ranking to reduce tokens
    skip_ranking: bool = False
    ranking_subset_ids: Optional[List[str]] = None
    # Token-saving: content generation subset and validation early exit
    content_generation_subset_ids: Optional[List[str]] = None
    stop_on_validation_fail: bool = False
    # Refinement: only re-run the persona(s) that scored below threshold (set by refinement_step)
    ranking_retest_technical_ids: Optional[List[str]] = None
    ranking_retest_pedagogical_ids: Optional[List[str]] = None

    # Wave 2 (R1): judge improvements collected after each ranking round, keyed by challenge_id.
    # content_generation uses these to inject "MUST address" constraints into the gen prompt.
    prior_improvements_per_challenge: Dict[str, List[str]] = field(default_factory=dict)

    # D2: per-dimension scores from the prior ranking round, keyed by challenge_id.
    # Inner dict shape: {"technical": <float>, "pedagogical": <float>}.
    # Populated by ranking_agent after a refinement-triggering round; consumed by
    # content_generation_agent so the gen prompt knows which dimension failed
    # alongside the improvement list. Empty on first round.
    prior_dim_scores_per_challenge: Dict[str, Dict[str, float]] = field(
        default_factory=dict
    )

    # Wave 2 (R2): per-challenge score history (overall_score per round) for early-exit detection.
    # Appended by _refinement_step after each ranking pass; keyed by challenge_id.
    score_history_per_challenge: Dict[str, List[float]] = field(default_factory=dict)

    # Repair: attempt structural repair after validation before proceeding
    attempt_repair: bool = False
    repair_attempted: bool = False

    # Language configuration
    output_language: str = "en"

    # Metadata KB: contest context for ranking prompt injection
    contest_metadata: Dict[str, Any] = field(default_factory=dict)

    # Execution tracking
    current_agent: str = "start"
    errors: List[Dict[str, Any]] = field(default_factory=list)
    is_complete: bool = False

    def add_error(self, agent_name: str, challenge_id: str, error_message: str) -> None:
        """Record an error that occurred during processing.

        Args:
            agent_name (str): Name of agent where error occurred
            challenge_id (str): ID of challenge being processed
            error_message (str): Description of the error

        Example:
            >>> state.add_error("validator", "crypto_001", "Missing solver script")
        """
        self.errors.append(
            {
                "agent": agent_name,
                "challenge_id": challenge_id,
                "message": error_message,
            }
        )

    def is_valid_state(self) -> bool:
        """Check if state has the minimum required data to proceed.

        Returns:
            bool: True if state is initialized with challenges
        """
        return len(self.challenges) > 0 and len(self.challenge_ids) > 0
