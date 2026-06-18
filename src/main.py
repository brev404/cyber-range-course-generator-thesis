"""CyberEdu Challenge Processing Pipeline - Main Entry Point.

This module is the single main entry for the project CLI. It orchestrates
(1) the processing pipeline (organize → analyze → map-docs → validate) and
(2) the LangGraph agent pipeline (validation, content generation, ranking).

Primary modes:
    --analyse-contest: Run all 4 contest steps (organize, analyze, map-docs, validate).
    --generate-courses: Run the LangGraph pipeline (validation, content generation, mapping, ranking).

Pipeline (individual steps):
    --organize, --analyze, --map-docs, --validate run specific steps.
    --map-docs-jobs N: parallel workers for PDF extraction (e.g. --map-docs --map-docs-jobs 8).
    --all is equivalent to --analyse-contest.

Graph (LangGraph app):
    --run-graph is equivalent to --generate-courses. Optional flags:
    --skip-ranking, --ranking-subset id1,id2, --content-subset id1,id2,
    --stop-on-validation-fail. See docs/reference/FLOW_AND_LLM.md for options.

Alternative entry for graph only:
    python -m src.run_graph [same optional flags]


Usage:
    # Contest analysis (all 4 steps)
    python src/main.py --analyse-contest

    # Generate courses (LangGraph)
    python src/main.py --generate-courses
    python src/main.py --generate-courses --skip-ranking --content-subset web/foo,crypto/bar

    # Individual pipeline steps
    python src/main.py --organize --validate
    python src/main.py --map-docs --map-docs-jobs 8

    # Legacy: --all same as --analyse-contest, --run-graph same as --generate-courses

    # Graph (alternative entry)
    python -m src.run_graph
    python -m src.run_graph --skip-ranking

    # Help
    python src/main.py --help

Command-Line Arguments:
    Primary: --analyse-contest (all 4 steps), --generate-courses (LangGraph).
    Pipeline: --organize, --analyze, --map-docs, --validate; --map-docs-jobs N (default 4).
    Graph options: --run-graph (same as --generate-courses), --all (same as --analyse-contest).
    For --generate-courses: --skip-ranking, --ranking-subset id1,id2, --content-subset id1,id2,
    --stop-on-validation-fail. See docs/reference/FLOW_AND_LLM.md.

Pipeline Execution Flow (with --all):

    [START]
        ↓
    [1/4] Challenge Organization
        Reorganizes raw files into: cyberedu/src, cyberedu/write-up, cyberedu/public
        Input: RAW_CHALLENGES_SOURCE
        Output: PROCESSED_DIR
        ↓
    [2/4] Challenge Analysis
        Analyzes structure, file types, dependencies
        Input: PROCESSED_DIR
        Output: Logs + statistics
        ↓
    [3/4] Official Documentation Mapping
        Links official PDFs to challenges via text matching
        Input: OFFICIAL_DOCS_SOURCE + challenges
        Output: Challenge → Documentation mapping
        ↓
    [4/4] Challenge Structure Validation
        Validates structure and completeness
        Input: Processed challenges
        Output: Validation reports
        ↓
    [SUMMARY]
        Report successes and failures

Configuration:
    Uses settings from src/config/settings.py:
    - DATA_DIR, INPUT_DIR, PROCESSED_DIR, OUTPUT_DIR
    - RAW_CHALLENGES_SOURCE (challenge files)
    - OFFICIAL_DOCS_SOURCE (PDF documentation)
    - LOG_LEVEL (logging verbosity)

Logging:
    - Configured via src/utils/logging_config.py
    - Creates logs/main_runner.log; rotates previous logs (keeps last 5 runs: .log.1–.log.5)
    - Stdout and stderr are teed to the same log file so all main output is visible in the log
    - Console output shows INFO and above; file output shows DEBUG and above plus any stdout/stderr

    Log example:
    2024-01-15 14:23:45 - main_runner - INFO - CyberEDU Challenge Processing Started
    2024-01-15 14:23:45 - main_runner - INFO - [1/4] STARTING: Challenge Organization
    2024-01-15 14:23:47 - main_runner - INFO -       ✓ Challenge Organization completed

Error Handling:
    - Failed scripts logged with error message
    - Pipeline continues even if script fails
    - Summary at end shows completed vs failed
    - Logs can be examined for root cause analysis

    Example failed output:
    2024-01-15 14:23:50 - main_runner - ERROR -       ✗ Challenge Analysis failed: Permission denied

    At end:
    ✓ Completed Scripts: 3
      • Challenge Organization
      • Challenge Validation
      • Documentation Mapping
    ✗ Failed Scripts: 1
      • Challenge Analysis: Permission denied

Dependencies:
    - src/config/settings.py: Configuration management
    - src/utils/logging_config.py: Logging setup
    - src/pipeline/*.py: Individual processing scripts

    Each script independently imports:
    - loguru for logging
    - pathlib for file paths
    - Models and validators as needed

Exit Behavior:
    - Exit code 0 if all scripts succeed
    - Exit code 0 even if some scripts fail (still completes summary)
    - Exception in main() will exit with code 1

    Check logs for error details if individual scripts fail

Notes:
    - Designed for stand-alone execution or cron jobs
    - Idempotent (safe to run multiple times)
    - Previous outputs preserved (not overwritten)
    - Can run subset of pipeline via command-line flags
    - Logging enables debugging without code changes
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

# Add project root to Python path for imports
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.config.settings import settings  # noqa: E402
from src.services.langsmith_service import apply_tracing_from_settings  # noqa: E402
from src.utils.logging_config import get_log_file_path, setup_logging  # noqa: E402

# Logger will be initialized in main() after setup_logging is called
logger = None


class _TeeOutput:
    """Writes to both the original stream and the main runner log file (keeps output visible and logged)."""

    def __init__(self, stream: TextIO, log_path: Path) -> None:
        self._stream = stream
        self._log_path = log_path
        self._file: TextIO | None = None

    def _ensure_file(self) -> None:
        if self._file is None and self._log_path.exists():
            self._file = open(self._log_path, "a", encoding="utf-8")

    def write(self, data: str) -> int:
        n = self._stream.write(data)
        self._ensure_file()
        if self._file and data:
            try:
                self._file.write(data)
                self._file.flush()
            except OSError:
                pass
        return n

    def flush(self) -> None:
        self._stream.flush()
        if self._file:
            try:
                self._file.flush()
            except OSError:
                pass

    def close(self) -> None:
        if self._file:
            try:
                self._file.close()
            except OSError:
                pass
            self._file = None


# Import the main functions from each pipeline step with error handling
try:
    from src.pipeline.analyze_challenges import run_analysis
except ImportError as e:
    logger.warning(f"Could not import analyze_challenges from src.pipeline: {e}")
    run_analysis = None

try:
    from src.pipeline.organize_challenges import run_organizer
except ImportError as e:
    logger.warning(f"Could not import organize_challenges from src.pipeline: {e}")
    run_organizer = None

try:
    from src.pipeline.map_official_docs import run_mapping
except ImportError as e:
    logger.warning(f"Could not import map_official_docs from src.pipeline: {e}")
    run_mapping = None

try:
    from src.pipeline.validate_challenge_structure import run_validator
except ImportError as e:
    logger.warning(
        f"Could not import validate_challenge_structure from src.pipeline: {e}"
    )
    run_validator = None


def _run_graph(
    skip_ranking: bool = False,
    ranking_subset: str | None = None,
    content_subset: str | None = None,
    stop_on_validation_fail: bool = False,
    one_challenge_at_a_time: bool = False,
    attempt_repair: bool = False,
) -> None:
    """Build AgentState from processed challenges and invoke the LangGraph pipeline.

    State is prepared by the coordinator node (organized_challenges from PROCESSED_DIR).
    Token-saving options are applied to state before invocation.

    Args:
        skip_ranking: Set state.skip_ranking = True.
        ranking_subset: Comma-separated challenge IDs; set state.ranking_subset_ids.
        content_subset: Comma-separated challenge IDs; set state.content_generation_subset_ids.
        stop_on_validation_fail: Set state.stop_on_validation_fail = True.
        one_challenge_at_a_time: If True, run the pipeline once per challenge and merge reports.
        attempt_repair: Set state.attempt_repair = True; runs repair node after validation.
    """
    from src.agents.coordinator_agent import run_coordinator_agent
    from src.core.graph import app
    from src.core.state import AgentState

    def _ranking_table(rankings: list) -> str:
        """Format ranking reports (or interrupt payload challenges) as a text table.

        Shows overall scores and optionally dimension breakdowns if dimension_scores are available.
        """
        rows = []
        has_dimensions = False
        for r in rankings:
            if hasattr(r, "challenge_id"):
                cid = r.challenge_id
                tech = getattr(r.technical_review, "score", "")
                ped = getattr(r.pedagogical_review, "score", "")
                overall = getattr(r, "overall_score", "")
                # Check if dimension_scores are available
                tech_dims = getattr(r.technical_review, "dimension_scores", None)
                ped_dims = getattr(r.pedagogical_review, "dimension_scores", None)
                if tech_dims or ped_dims:
                    has_dimensions = True
            elif isinstance(r, dict):
                cid = r.get("challenge_id", "")
                tech = r.get("technical_score", "")
                ped = r.get("pedagogical_score", "")
                overall = r.get("overall_score", "")
                tech_dims = r.get("technical_dimension_scores")
                ped_dims = r.get("pedagogical_dimension_scores")
                if tech_dims or ped_dims:
                    has_dimensions = True
            else:
                continue
            rows.append((str(cid), str(tech), str(ped), str(overall)))
        if not rows:
            return ""
        widths = [max(len(r[i]) for r in rows) for i in range(4)]
        widths = [
            max(w, len(h))
            for w, h in zip(
                widths, ("challenge_id", "technical", "pedagogical", "overall")
            )
        ]
        header = "  ".join(
            h.ljust(widths[i])
            for i, h in enumerate(
                ("challenge_id", "technical", "pedagogical", "overall")
            )
        )
        sep = "-" * len(header)
        lines = [header, sep] + [
            "  ".join(rows[i][j].ljust(widths[j]) for j in range(4))
            for i in range(len(rows))
        ]

        # Add dimension breakdowns if available (optional, shown below main table)
        if has_dimensions:
            lines.append("")
            lines.append("Dimension breakdowns (where available):")
            for r in rankings:
                if hasattr(r, "challenge_id"):
                    cid = r.challenge_id
                    tech_dims = getattr(r.technical_review, "dimension_scores", None)
                    ped_dims = getattr(r.pedagogical_review, "dimension_scores", None)
                    if tech_dims:
                        dim_str = ", ".join(
                            f"{k}: {v}" for k, v in sorted(tech_dims.items())
                        )
                        lines.append(f"  {cid} [Technical]: {dim_str}")
                    if ped_dims:
                        dim_str = ", ".join(
                            f"{k}: {v}" for k, v in sorted(ped_dims.items())
                        )
                        lines.append(f"  {cid} [Pedagogical]: {dim_str}")
                elif isinstance(r, dict):
                    cid = r.get("challenge_id", "")
                    tech_dims = r.get("technical_dimension_scores")
                    ped_dims = r.get("pedagogical_dimension_scores")
                    if tech_dims:
                        dim_str = ", ".join(
                            f"{k}: {v}" for k, v in sorted(tech_dims.items())
                        )
                        lines.append(f"  {cid} [Technical]: {dim_str}")
                    if ped_dims:
                        dim_str = ", ".join(
                            f"{k}: {v}" for k, v in sorted(ped_dims.items())
                        )
                        lines.append(f"  {cid} [Pedagogical]: {dim_str}")

        return "\n".join(lines)

    # Ingest knowledge base so RAG has context for content generation (avoids "No RAG context retrieved")
    try:
        from src.services.vector_db_service import VectorDBService

        vb = VectorDBService()
        n = vb.ingest_knowledge_base()
        if n > 0:
            logger.info("RAG: ingested %s chunks from knowledge base", n)
        else:
            logger.warning(
                "RAG: no chunks ingested (check data/knowledge_base has .md/.txt files). "
                "Content generation will use fallback prompt without KB context."
            )
    except Exception as e:
        logger.warning("RAG ingest failed: %s; continuing without KB context.", e)

    try:
        from langgraph.errors import GraphInterrupt
        from langgraph.types import Command
    except ImportError:
        GraphInterrupt = type("_Never", (BaseException,), {})
        Command = None
    config = {"configurable": {"thread_id": "run_graph_1"}}

    def _invoke_graph_once(initial_state: AgentState):
        """Invoke the graph once (with HITL loop) and return final state dict or object."""
        resume_value = None
        raw_result = None
        while True:
            try:
                if resume_value is None:
                    raw_result = app.invoke(initial_state, config=config)
                else:
                    if Command is None:
                        logger.warning(
                            "HITL: langgraph.types.Command not available; cannot resume. Exiting."
                        )
                        raise SystemExit(1)
                    raw_result = app.invoke(Command(resume=resume_value), config=config)
                return raw_result
            except Exception as e:
                if not isinstance(e, GraphInterrupt) or not e.args:
                    raise
                # Extract payload robustly from LangGraph's Interrupt wrapper
                raw = e.args[0] if e.args else {}
                if isinstance(raw, dict):
                    payload = raw
                elif hasattr(raw, "value"):
                    payload = raw.value if isinstance(raw.value, dict) else {}
                elif isinstance(raw, (list, tuple)) and raw:
                    first = raw[0]
                    payload = (
                        first.value
                        if hasattr(first, "value")
                        else (first if isinstance(first, dict) else {})
                    )
                else:
                    payload = {}

                # Display Rich summary table + routing cause
                from src.agents.hitl_agent import print_hitl_summary as _print_hitl

                _print_hitl(payload)
                logger.info(
                    "HITL paused (iteration %s/%s; cause: %s)",
                    payload.get("iteration", "?"),
                    payload.get("max_iterations", "?"),
                    payload.get("routing_cause", "unknown"),
                )
                # Also log tabular data for the log file
                challenges = payload.get("challenges", [])
                if challenges:
                    table = _ranking_table(challenges)
                    if table:
                        logger.info("Ranking grades (HITL paused):\n%s", table)

                # New action menu (CONVENTIONS: print acceptable for interactive CLI)
                challenge_ids = [
                    c.get("challenge_id", "")
                    for c in challenges
                    if c.get("challenge_id")
                ]
                print("\nAvailable actions:")
                print("  approve_all            — accept all challenges, route to END")
                if challenge_ids:
                    print(
                        "  approve <id1,id2,...>  — accept subset; remaining re-enter refinement"
                    )
                print(
                    "  edit_retry             — add operator hint and re-run refinement"
                )
                print("  abort                  — stop the run cleanly")
                print()
                try:
                    action_input = input("Enter action [approve_all]: ").strip()
                except EOFError:
                    action_input = ""
                    print("(no input, defaulting to approve_all)")

                action_lower = action_input.lower()
                if not action_lower or action_lower in ("y", "yes", "approve_all"):
                    resume_value = {"action": "approve_all"}
                elif action_lower == "abort":
                    logger.info("HITL: operator aborted run")
                    raise SystemExit(0)
                elif action_lower.startswith("approve "):
                    ids_str = action_input[len("approve ") :].strip()
                    ids = [s.strip() for s in ids_str.split(",") if s.strip()]
                    if ids:
                        resume_value = {"action": "approve_ids", "ids": ids}
                    else:
                        logger.warning(
                            "HITL: 'approve' with no IDs; defaulting to approve_all"
                        )
                        resume_value = {"action": "approve_all"}
                elif action_lower in ("n", "no", "edit_retry", "refine"):
                    try:
                        hint = input("Operator hint (Enter to skip): ").strip()
                    except EOFError:
                        hint = ""
                    resume_value = {"action": "edit_retry", "hint": hint}
                else:
                    logger.warning(
                        "HITL: unknown action '%s'; defaulting to approve_all",
                        action_input,
                    )
                    resume_value = {"action": "approve_all"}

                logger.info("Resuming graph with action=%s", resume_value.get("action"))

    if one_challenge_at_a_time:
        coord_state = run_coordinator_agent(AgentState())
        paths = list(coord_state.organized_challenges or [])
        ids = list(coord_state.challenge_ids or [])
        if content_subset:
            subset_set = {s.strip() for s in content_subset.split(",") if s.strip()}
            paths = [p for p, i in zip(paths, ids) if i in subset_set]
            ids = [i for i in ids if i in subset_set]
        if not paths or not ids:
            logger.warning(
                "One-challenge-at-a-time: no challenges to run (check PROCESSED_DIR or --content-subset)."
            )
            return
        logger.info(
            "One-challenge-at-a-time: running pipeline for %d challenges sequentially",
            len(ids),
        )
        all_rankings = []
        for idx, (path, cid) in enumerate(zip(paths, ids)):
            logger.info("One-challenge-at-a-time: [%d/%d] %s", idx + 1, len(ids), cid)
            state = AgentState(
                organized_challenges=[path],
                challenge_ids=[cid],
                skip_ranking=skip_ranking,
                stop_on_validation_fail=stop_on_validation_fail,
                attempt_repair=attempt_repair,
                output_language=settings.OUTPUT_LANGUAGE,
            )
            if ranking_subset:
                state.ranking_subset_ids = [
                    s.strip() for s in ranking_subset.split(",") if s.strip()
                ]
            raw_result = _invoke_graph_once(state)
            if raw_result:
                reports = (
                    raw_result.get("ranking_reports", [])
                    if isinstance(raw_result, dict)
                    else getattr(raw_result, "ranking_reports", [])
                )
                if reports:
                    all_rankings.extend(reports)
        final_reports = []
        final_courses = {}
        final_rankings = all_rankings
        final_errors = []
    else:
        state = AgentState()
        state.output_language = settings.OUTPUT_LANGUAGE
        state.skip_ranking = skip_ranking
        state.stop_on_validation_fail = stop_on_validation_fail
        state.attempt_repair = attempt_repair
        if ranking_subset:
            state.ranking_subset_ids = [
                s.strip() for s in ranking_subset.split(",") if s.strip()
            ]
        if content_subset:
            state.content_generation_subset_ids = [
                s.strip() for s in content_subset.split(",") if s.strip()
            ]
        logger.info(
            "Invoking LangGraph pipeline (coordinator will load challenges from PROCESSED_DIR)"
        )
        raw_result = _invoke_graph_once(state)
        if isinstance(raw_result, dict):
            final_reports = raw_result.get("validation_reports", [])
            final_courses = raw_result.get("generated_courses", {})
            final_rankings = raw_result.get("ranking_reports", [])
            final_errors = raw_result.get("errors", [])
        else:
            final_reports = raw_result.validation_reports
            final_courses = raw_result.generated_courses
            final_rankings = raw_result.ranking_reports
            final_errors = getattr(raw_result, "errors", []) or []
    if final_rankings:
        table = _ranking_table(final_rankings)
        if table:
            logger.info("Ranking grades:\n%s", table)
            # CLI output (user-facing)
            print("\n--- Ranking grades ---\n" + table + "\n")
        # Export ranking report for LangSmith feedback script (see docs/reference/LANGSMITH_EVALUATORS.md)
        try:
            from src.config.settings import settings as _s

            out_dir = getattr(_s, "OUTPUT_DIR", None)
            if out_dir:
                out_dir = Path(out_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                scores_path = out_dir / "ranking_reports_latest.json"
                rows = []
                for r in final_rankings:
                    if hasattr(r, "challenge_id"):
                        tech_review = getattr(r, "technical_review", None)
                        ped_review = getattr(r, "pedagogical_review", None)
                        row = {
                            "challenge_id": r.challenge_id,
                            "overall_score": float(getattr(r, "overall_score", 0)),
                            "technical_score": int(getattr(tech_review, "score", 0)),
                            "pedagogical_score": int(getattr(ped_review, "score", 0)),
                            "technical_rank": getattr(r, "technical_rank", ""),
                        }
                        # Include dimension_scores if available (rubric anchoring)
                        tech_dims = getattr(tech_review, "dimension_scores", None)
                        ped_dims = getattr(ped_review, "dimension_scores", None)
                        if tech_dims:
                            row["technical_dimension_scores"] = tech_dims
                        if ped_dims:
                            row["pedagogical_dimension_scores"] = ped_dims
                        rows.append(row)
                if rows:
                    import json

                    scores_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
                    logger.info(
                        "Wrote ranking report to %s (for post_ranking_feedback_to_langsmith)",
                        scores_path,
                    )
        except Exception as ex:
            logger.debug("Could not write ranking report JSON: %s", ex)
    logger.info(
        "Pipeline finished: %d validation reports, %d generated courses, %d ranking reports",
        len(final_reports),
        len(final_courses),
        len(final_rankings),
    )
    if final_errors:
        for err in final_errors:
            logger.warning(
                "Pipeline error [%s]: %s", err.get("agent"), err.get("message")
            )


def main():
    """Main entry point for running challenge processing pipeline.

    Provides command-line interface to run individual processing scripts
    or all scripts in sequence. Handles argument parsing, logging setup,
    error management, and final reporting.

    Available Scripts:
        - Challenge Organization: Reorganize files into standard structure
        - Challenge Analysis: Analyze structure and identify patterns
        - Official Documentation Mapping: Link official PDFs to challenges
        - Challenge Validation: Check structure against requirements

    Exit Behavior:
        - Reports completion status and any errors
        - Always exits normally (check logs for error details)
        - Exit code 0 regardless of script failures

    Example Usage:
        >>> # Run all steps
        >>> python src/main.py --all

        >>> # Run specific steps
        >>> python src/main.py --organize --validate

        >>> # Run analysis only
        >>> python src/main.py --analyze

    Logging:
        - Configured via setup_logging("main_runner")
        - Logs written to logs/main_runner.log
        - Previous runs archived as .log.1, .log.2, etc.
        - Console shows progress, file shows details

    Error Handling:
        - Catches exceptions from individual scripts
        - Logs errors with full context
        - Continues pipeline even if script fails
        - Reports summary of successes and failures
    """
    parser = argparse.ArgumentParser(
        description="Main entry: --analyse-contest (all 4 steps) or --generate-courses (LangGraph); "
        "or individual steps (--organize, --analyze, --map-docs, --validate) and --run-graph with options."
    )
    # Primary modes
    parser.add_argument(
        "--analyse-contest",
        action="store_true",
        help="Run all 4 contest analysis steps: organize, analyze, map-docs, validate.",
    )
    parser.add_argument(
        "--generate-courses",
        action="store_true",
        help="Run the LangGraph pipeline to generate courses (validation, content generation, mapping, ranking).",
    )
    # Individual pipeline steps
    parser.add_argument(
        "--analyze", action="store_true", help="Run challenge folder analysis."
    )
    parser.add_argument(
        "--organize", action="store_true", help="Run challenge organization."
    )
    parser.add_argument(
        "--map-docs", action="store_true", help="Run official documentation mapping."
    )
    parser.add_argument(
        "--map-docs-jobs",
        type=int,
        default=4,
        metavar="N",
        help="Parallel workers for PDF text extraction in map-docs (default: 4).",
    )
    parser.add_argument(
        "--validate", action="store_true", help="Run challenge structure validation."
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all 4 processing steps in order (same as --analyse-contest).",
    )
    # LangGraph pipeline
    parser.add_argument(
        "--run-graph",
        action="store_true",
        help="Run the LangGraph agent pipeline (same as --generate-courses).",
    )
    parser.add_argument(
        "--skip-ranking",
        action="store_true",
        help="Set state.skip_ranking=True; ranking node skips LLM calls.",
    )
    parser.add_argument(
        "--ranking-subset",
        type=str,
        metavar="id1,id2",
        help="Run ranking only for these challenge IDs (e.g. web/foo,crypto/bar).",
    )
    parser.add_argument(
        "--content-subset",
        type=str,
        metavar="id1,id2",
        help="Run content generation LLM only for these challenge IDs.",
    )
    parser.add_argument(
        "--stop-on-validation-fail",
        action="store_true",
        help="Route to END after validation if any report has a critical (HIGH or CRITICAL) issue.",
    )
    parser.add_argument(
        "--one-challenge-at-a-time",
        action="store_true",
        help="Run the pipeline once per challenge (validate → content → mapping → ranking per challenge). Reduces token pressure and can avoid truncation.",
    )
    parser.add_argument(
        "--attempt-repair",
        action="store_true",
        help="After validation, attempt structural repair (rename/move files, create dirs) then re-validate. Default False.",
    )
    parser.add_argument(
        "--ranking-model",
        type=str,
        default=None,
        metavar="MODEL_ID",
        help="Override RANKING_MODEL setting at runtime for the LangGraph pipeline judge (e.g. claude-haiku-4-5-20251001).",
    )
    parser.add_argument(
        "--ranking-provider",
        type=str,
        default=None,
        metavar="PROVIDER",
        help="Override RANKING_PROVIDER setting at runtime for the LangGraph pipeline judge (openai, anthropic, google).",
    )

    args = parser.parse_args()

    # Configure logging for the main runner and get the logger (keeps 5 previous runs: .log.1–.log.5)
    # Apply runtime overrides for ranking judge before any LLM code is imported
    if args.ranking_model:
        settings.RANKING_MODEL = args.ranking_model
    if args.ranking_provider:
        settings.RANKING_PROVIDER = args.ranking_provider

    global logger
    logger = setup_logging("main_runner")
    log_path = get_log_file_path("main_runner")
    # Tee stdout/stderr to the same log file so all main output is visible in the log
    _stdout_tee = _TeeOutput(sys.stdout, log_path)
    _stderr_tee = _TeeOutput(sys.stderr, log_path)
    sys.stdout = _stdout_tee
    sys.stderr = _stderr_tee
    settings.create_directories()  # Ensure all necessary directories are created
    apply_tracing_from_settings()  # Enable LangSmith tracing if LANGCHAIN_TRACING_V2=true and key set

    run_started_at = datetime.now(timezone.utc).isoformat()
    cli_overrides = {
        "analyse_contest": args.analyse_contest or args.all,
        "generate_courses": args.run_graph or args.generate_courses,
        "skip_ranking": getattr(args, "skip_ranking", False),
        "content_subset": getattr(args, "content_subset", None),
        "ranking_subset": getattr(args, "ranking_subset", None),
        "stop_on_validation_fail": getattr(args, "stop_on_validation_fail", False),
        "one_challenge_at_a_time": getattr(args, "one_challenge_at_a_time", False),
        "attempt_repair": getattr(args, "attempt_repair", False),
        "map_docs_jobs": getattr(args, "map_docs_jobs", None),
        "ranking_model": getattr(args, "ranking_model", None),
        "ranking_provider": getattr(args, "ranking_provider", None),
    }
    cli_filtered = {
        k: v for k, v in cli_overrides.items() if v is not None and v is not False
    }
    config_data = settings.export_for_reproducibility(
        run_started_at=run_started_at,
        cli_overrides=cli_filtered or None,
    )
    logger.info("Run config (reproducibility): %s", json.dumps(config_data, indent=2))
    saved_path = settings.save_run_config(
        run_started_at=run_started_at, cli_overrides=cli_filtered or None
    )
    if saved_path:
        logger.info("Saved run config to %s", saved_path)

    logger.info("=" * 70)
    logger.info("CyberEDU Challenge Processing Pipeline Started")
    logger.info("=" * 70)

    # Generate-courses / run-graph: invoke LangGraph and exit
    if args.run_graph or args.generate_courses:
        _run_graph(
            skip_ranking=args.skip_ranking,
            ranking_subset=args.ranking_subset,
            content_subset=args.content_subset,
            stop_on_validation_fail=args.stop_on_validation_fail,
            one_challenge_at_a_time=getattr(args, "one_challenge_at_a_time", False),
            attempt_repair=getattr(args, "attempt_repair", False),
        )
        return

    # Analyse-contest / all: run all 4 steps; or individual steps
    run_all_four = args.all or args.analyse_contest
    # If only --map-docs-jobs was given, run map-docs with that job count
    map_docs_jobs_requested = any("map-docs-jobs" in a for a in sys.argv)
    if map_docs_jobs_requested and not (
        args.analyze or args.organize or args.map_docs or args.validate or run_all_four
    ):
        args.map_docs = True
    steps_selected = (
        args.analyze or args.organize or args.map_docs or args.validate or run_all_four
    )
    if not steps_selected:
        logger.warning(
            "No script selected. Examples: --analyse-contest (all 4 steps), --generate-courses (LangGraph), "
            "--map-docs (or --map-docs --map-docs-jobs 8 for parallel PDF mapping). Use --help for more."
        )
        return

    completed_scripts = []
    failed_scripts = []

    if run_all_four:
        logger.info("Running all 4 contest analysis steps in sequence...")
        logger.info("-" * 70)

        # Step 1: Organization
        if run_organizer:
            try:
                logger.info("[1/4] STARTING: Challenge Organization")
                logger.info(
                    "      Purpose: Organizing raw challenge files into standardized structure"
                )
                run_organizer()
                logger.info("      ✓ Challenge Organization completed successfully")
                completed_scripts.append("Challenge Organization")
            except Exception as e:
                logger.error(f"      ✗ Challenge Organization failed: {e}")
                failed_scripts.append(("Challenge Organization", str(e)))
        logger.info("-" * 70)

        # Step 2: Analysis
        if run_analysis:
            try:
                logger.info("[2/4] STARTING: Challenge Folder Analysis")
                logger.info(
                    "      Purpose: Analyzing structure and consistency of challenge folders"
                )
                run_analysis()
                logger.info("      ✓ Challenge Analysis completed successfully")
                completed_scripts.append("Challenge Analysis")
            except Exception as e:
                logger.error(f"      ✗ Challenge Analysis failed: {e}")
                failed_scripts.append(("Challenge Analysis", str(e)))
        logger.info("-" * 70)

        # Step 3: Documentation Mapping
        if run_mapping:
            try:
                logger.info("[3/4] STARTING: Official Documentation Mapping")
                logger.info(
                    "      Purpose: Mapping challenges to official documentation"
                )
                run_mapping(max_workers=getattr(args, "map_docs_jobs", 4))
                logger.info("      ✓ Documentation Mapping completed successfully")
                completed_scripts.append("Documentation Mapping")
            except Exception as e:
                logger.error(f"      ✗ Documentation Mapping failed: {e}")
                failed_scripts.append(("Documentation Mapping", str(e)))
        logger.info("-" * 70)

        # Step 4: Validation
        if run_validator:
            try:
                logger.info("[4/4] STARTING: Challenge Structure Validation")
                logger.info(
                    "      Purpose: Validating challenge structure and completeness"
                )
                run_validator()
                logger.info("      ✓ Challenge Validation completed successfully")
                completed_scripts.append("Challenge Validation")
            except Exception as e:
                logger.error(f"      ✗ Challenge Validation failed: {e}")
                failed_scripts.append(("Challenge Validation", str(e)))
        logger.info("-" * 70)

    else:
        step_count = sum([args.organize, args.analyze, args.map_docs, args.validate])
        current_step = 0

        if args.organize and run_organizer:
            current_step += 1
            try:
                logger.info(
                    f"[{current_step}/{step_count}] STARTING: Challenge Organization"
                )
                logger.info(
                    "      Purpose: Organizing raw challenge files into standardized structure"
                )
                run_organizer()
                logger.info("      ✓ Challenge Organization completed successfully")
                completed_scripts.append("Challenge Organization")
            except Exception as e:
                logger.error(f"      ✗ Challenge Organization failed: {e}")
                failed_scripts.append(("Challenge Organization", str(e)))
            logger.info("-" * 70)

        if args.analyze and run_analysis:
            current_step += 1
            try:
                logger.info(
                    f"[{current_step}/{step_count}] STARTING: Challenge Folder Analysis"
                )
                logger.info(
                    "      Purpose: Analyzing structure and consistency of challenge folders"
                )
                run_analysis()
                logger.info("      ✓ Challenge Analysis completed successfully")
                completed_scripts.append("Challenge Analysis")
            except Exception as e:
                logger.error(f"      ✗ Challenge Analysis failed: {e}")
                failed_scripts.append(("Challenge Analysis", str(e)))
            logger.info("-" * 70)

        if args.map_docs and run_mapping:
            current_step += 1
            try:
                logger.info(
                    f"[{current_step}/{step_count}] STARTING: Official Documentation Mapping"
                )
                logger.info(
                    "      Purpose: Mapping challenges to official documentation"
                )
                run_mapping(max_workers=getattr(args, "map_docs_jobs", 4))
                logger.info("      ✓ Documentation Mapping completed successfully")
                completed_scripts.append("Documentation Mapping")
            except Exception as e:
                logger.error(f"      ✗ Documentation Mapping failed: {e}")
                failed_scripts.append(("Documentation Mapping", str(e)))
            logger.info("-" * 70)

        if args.validate and run_validator:
            current_step += 1
            try:
                logger.info(
                    f"[{current_step}/{step_count}] STARTING: Challenge Structure Validation"
                )
                logger.info(
                    "      Purpose: Validating challenge structure and completeness"
                )
                run_validator()
                logger.info("      ✓ Challenge Validation completed successfully")
                completed_scripts.append("Challenge Validation")
            except Exception as e:
                logger.error(f"      ✗ Challenge Validation failed: {e}")
                failed_scripts.append(("Challenge Validation", str(e)))
            logger.info("-" * 70)

    # Final Summary
    logger.info("=" * 70)
    logger.info("PIPELINE EXECUTION SUMMARY")
    logger.info("=" * 70)
    logger.info(f"✓ Completed Scripts: {len(completed_scripts)}")
    for script in completed_scripts:
        logger.info(f"  • {script}")

    if failed_scripts:
        logger.info(f"✗ Failed Scripts: {len(failed_scripts)}")
        for script, error in failed_scripts:
            logger.info(f"  • {script}: {error}")

    logger.info("=" * 70)

    if failed_scripts:
        logger.error(
            f"Pipeline completed with {len(failed_scripts)} error(s). Check logs for details."
        )
    else:
        logger.info("Pipeline completed successfully! All scripts ran without errors.")

    logger.info("=" * 70)


if __name__ == "__main__":
    main()
