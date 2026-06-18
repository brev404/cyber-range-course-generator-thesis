"""CLI for generating REVIEW.md for an experiment run.

Usage:
    uv run python -m src.scripts.generate_review <exp-id>
    uv run python -m src.scripts.generate_review output/experiments/run-0001
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate REVIEW.md for an experiment run."
    )
    parser.add_argument(
        "exp_id",
        help="Experiment ID (e.g. run-0001) or path to experiment directory.",
    )
    args = parser.parse_args()

    exp_input = args.exp_id
    exp_dir = Path(exp_input)
    if not exp_dir.is_dir():
        exp_dir = Path("output/experiments") / exp_input
    if not exp_dir.is_dir():
        logger.error(f"Experiment directory not found: {exp_dir}")
        return 1

    try:
        from src.services.review_generator import generate_review
        from src.services.review_queue_updater import append_to_review_queue

        review_path = generate_review(exp_dir)
        logger.info(f"REVIEW.md generated at {review_path}")

        append_to_review_queue(
            exp_id=exp_dir.name,
            review_path=review_path,
            summary="CLI-generated review",
        )
        return 0
    except Exception as exc:
        logger.error(f"Failed to generate review: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
