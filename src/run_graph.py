"""Entry point to run the LangGraph agent pipeline.

Use: python -m src.run_graph [--skip-ranking] [--ranking-subset id1,id2]
     [--content-subset id1,id2] [--stop-on-validation-fail]

State is built with token-saving options and coordinator loads challenges
from PROCESSED_DIR. See docs/reference/FLOW_AND_LLM.md for options.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.config.settings import settings  # noqa: E402
from src.core.graph import app  # noqa: E402
from src.core.state import AgentState  # noqa: E402
from src.services.langsmith_service import apply_tracing_from_settings  # noqa: E402
from src.utils.consistency_checker import (  # noqa: E402
    check_consistency,
    write_consistency_report,
)
from src.utils.logging_config import setup_logging  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the LangGraph agent pipeline (coordinator loads from PROCESSED_DIR)."
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
        "--exp-id",
        type=str,
        default="",
        metavar="EXPERIMENT_ID",
        help="Experiment output isolation: courses written to OUTPUT_DIR/EXPERIMENT_ID/<challenge_id>/course.md.",
    )
    args = parser.parse_args()

    if args.exp_id:
        settings.EXPERIMENT_ID = args.exp_id

    logger = setup_logging("run_graph")
    settings.create_directories()
    apply_tracing_from_settings()

    state = AgentState()
    state.output_language = settings.OUTPUT_LANGUAGE
    state.skip_ranking = args.skip_ranking
    state.stop_on_validation_fail = args.stop_on_validation_fail
    if args.ranking_subset:
        state.ranking_subset_ids = [
            s.strip() for s in args.ranking_subset.split(",") if s.strip()
        ]
    if args.content_subset:
        state.content_generation_subset_ids = [
            s.strip() for s in args.content_subset.split(",") if s.strip()
        ]

    # For HITL interrupt/resume, pass config with thread_id and handle __interrupt__ / Command(resume=...)
    config = {"configurable": {"thread_id": "run_graph_1"}}
    logger.info(
        "Invoking LangGraph pipeline (coordinator will load challenges from PROCESSED_DIR)"
    )
    final_state = app.invoke(state, config=config)
    # LangGraph invoke() returns a plain dict; use .get() throughout.
    fs = final_state if isinstance(final_state, dict) else vars(final_state)
    logger.info(
        "Pipeline finished: %d validation reports, %d generated courses, %d ranking reports",
        len(fs.get("validation_reports", [])),
        len(fs.get("generated_courses", {})),
        len(fs.get("ranking_reports", [])),
    )
    for err in fs.get("errors", []):
        logger.warning("Pipeline error [%s]: %s", err.get("agent"), err.get("message"))

    # Consistency pass — run after all courses are generated
    if fs.get("generated_courses"):
        run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        report = check_consistency(fs["generated_courses"], run_id)
        report_path = write_consistency_report(report, settings.OUTPUT_DIR)
        logger.info("Consistency report: %s", report_path)
        if report.deviations:
            logger.warning(
                "Consistency: %d deviation(s) detected across %d course(s)",
                len(report.deviations),
                report.challenge_count,
            )
    else:
        logger.info("Consistency check skipped: no generated courses in final state")


if __name__ == "__main__":
    main()
