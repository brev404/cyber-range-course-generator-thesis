"""Post ranking report scores to LangSmith as feedback on a run.

Use this after running the graph (--generate-courses) so that technical_score,
pedagogical_score, and overall_score appear as feedback on the trace in LangSmith.

Prerequisites:
  - LANGCHAIN_API_KEY (or LANGSMITH_API_KEY) set in .env
  - langsmith installed: pip install langsmith

Usage:
  # Post to the most recent run in your project (recommended after a graph run)
  ./venv/bin/python -m src.scripts.post_ranking_feedback_to_langsmith --latest

  # Post to a specific run (copy run_id from LangSmith trace URL)
  ./venv/bin/python -m src.scripts.post_ranking_feedback_to_langsmith --run-id <uuid> --scores data/outputs/ranking_reports_latest.json

  # Use custom project or scores file
  ./venv/bin/python -m src.scripts.post_ranking_feedback_to_langsmith --latest --project my-project --scores path/to/scores.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Project root for imports
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


def _load_scores(path: Path) -> list[dict]:
    """Load ranking scores from JSON file. Expected: list of {challenge_id, technical_score, pedagogical_score, overall_score}."""
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8")
    data = json.loads(text)
    if not isinstance(data, list):
        return []
    return [x for x in data if isinstance(x, dict) and "challenge_id" in x]


def _averages(rows: list[dict]) -> tuple[float, float, float]:
    tech = [
        r.get("technical_score") for r in rows if r.get("technical_score") is not None
    ]
    ped = [
        r.get("pedagogical_score")
        for r in rows
        if r.get("pedagogical_score") is not None
    ]
    ov = [r.get("overall_score") for r in rows if r.get("overall_score") is not None]
    n = len(rows) or 1
    return (
        sum(tech) / n if tech else 0.0,
        sum(ped) / n if ped else 0.0,
        sum(ov) / n if ov else 0.0,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Post ranking report scores to LangSmith as feedback on a run.",
        epilog="See docs/reference/LANGSMITH_EVALUATORS.md for full setup.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="LangSmith run UUID to attach feedback to (from trace URL).",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Use the most recent root run in the project instead of --run-id.",
    )
    parser.add_argument(
        "--scores",
        type=Path,
        default=None,
        help="Path to ranking_reports_latest.json (default: data/outputs/ranking_reports_latest.json).",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="LangSmith project name (default: from LANGCHAIN_PROJECT or settings).",
    )
    args = parser.parse_args()

    if not args.run_id and not args.latest:
        print("Error: provide either --run-id <uuid> or --latest.", file=sys.stderr)
        return 1

    try:
        from langsmith import Client
    except ImportError:
        print(
            "Error: langsmith not installed. Run: pip install langsmith",
            file=sys.stderr,
        )
        return 1

    from src.config.settings import settings

    api_key = getattr(settings, "LANGCHAIN_API_KEY", None) or getattr(
        settings, "LANGSMITH_API_KEY", None
    )
    if not api_key:
        print(
            "Error: LANGCHAIN_API_KEY or LANGSMITH_API_KEY not set (e.g. in .env).",
            file=sys.stderr,
        )
        return 1

    client = Client(api_key=api_key)

    scores_path = (
        args.scores or Path(settings.OUTPUT_DIR) / "ranking_reports_latest.json"
    )
    rows = _load_scores(scores_path)
    if not rows:
        print(
            f"Warning: no scores in {scores_path}. Run the graph first (--generate-courses) to generate ranking_reports_latest.json.",
            file=sys.stderr,
        )
        return 1

    tech_avg, ped_avg, overall_avg = _averages(rows)
    project_name = args.project or getattr(
        settings, "LANGCHAIN_PROJECT", "cyber-range-validator"
    )

    run_id = args.run_id
    if args.latest:
        runs = list(client.list_runs(project_name=project_name, is_root=True, limit=1))
        if not runs:
            print(
                f"Error: no runs found in project '{project_name}'. Run the graph with tracing on first.",
                file=sys.stderr,
            )
            return 1
        run_id = str(runs[0].id)
        print(f"Using latest run: {run_id}")

    try:
        client.create_feedback(
            run_id,
            key="technical_score_avg",
            score=tech_avg,
            comment=f"From {len(rows)} challenges",
        )
        client.create_feedback(
            run_id,
            key="pedagogical_score_avg",
            score=ped_avg,
            comment=f"From {len(rows)} challenges",
        )
        client.create_feedback(
            run_id,
            key="overall_score_avg",
            score=overall_avg,
            comment=f"From {len(rows)} challenges",
        )
        print(
            f"Posted feedback to run {run_id}: technical_avg={tech_avg:.1f}, pedagogical_avg={ped_avg:.1f}, overall_avg={overall_avg:.1f}"
        )
    except Exception as e:
        print(f"Error posting feedback: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
