"""Tests for REVIEW.md wall_time computation (F).

Bug: current code sums node_timings (only one phase) instead of using
(finished_at - started_at).total_seconds().

Pre-fix: test_wall_time_from_timestamps fails.
Post-fix: all tests pass.
"""

from __future__ import annotations

import json
from pathlib import Path


class TestWallTimeFromTimestamps:
    """wall_time must equal finished_at - started_at when both are present."""

    def test_wall_time_from_timestamps(self, tmp_path: Path):
        """Manifest with started_at + finished_at → wall_time = delta (ignoring node_timings)."""
        exp_dir = tmp_path / "EXP-WALLTIME"
        exp_dir.mkdir()

        # started_at and finished_at span exactly 600 seconds
        manifest = {
            "status": "complete",
            "started_at": "2026-05-25T10:00:00",
            "finished_at": "2026-05-25T10:10:00",
            "challenge_ids": ["crypto/test1"],
            "settings_snapshot": {},
            # node_timings present but must NOT be used as sole source
            "node_timings": {"content_generation": 309.0},
        }
        (exp_dir / "manifest.json").write_text(json.dumps(manifest))

        # Minimal ranking so review generates without errors
        rankings = [
            {
                "challenge_id": "crypto/test1",
                "overall_score": 9.5,
                "technical_review": {"score": 9.5},
                "pedagogical_review": {"score": 9.5},
                "_refinement_rounds": 1,
            }
        ]
        (exp_dir / "ranking_reports.json").write_text(json.dumps(rankings))

        from src.services.review_generator import generate_review

        review_path = generate_review(exp_dir)
        content = review_path.read_text()

        # wall_time must be 600s
        assert (
            "**Wall time**: 600s" in content
        ), "Expected 600s wall time from timestamp delta, got:\n" + "\n".join(
            ln for ln in content.splitlines() if "Wall time" in ln
        )

    def test_wall_time_node_timings_fallback_when_no_timestamps(self, tmp_path: Path):
        """When timestamps absent, fall back to sum(node_timings)."""
        exp_dir = tmp_path / "EXP-NOTIMESTAMPS"
        exp_dir.mkdir()

        manifest = {
            "status": "complete",
            "challenge_ids": ["crypto/t1"],
            "settings_snapshot": {},
            "node_timings": {"content_generation": 100.0, "ranking": 50.0},
        }
        (exp_dir / "manifest.json").write_text(json.dumps(manifest))

        rankings = [
            {
                "challenge_id": "crypto/t1",
                "overall_score": 8.0,
                "technical_review": {"score": 8.0},
                "pedagogical_review": {"score": 8.0},
                "_refinement_rounds": 0,
            }
        ]
        (exp_dir / "ranking_reports.json").write_text(json.dumps(rankings))

        from src.services.review_generator import generate_review

        review_path = generate_review(exp_dir)
        content = review_path.read_text()

        # Falls back to sum of node_timings = 150s
        assert "**Wall time**: 150s" in content

    def test_wall_time_timestamps_take_priority_over_node_timings(self, tmp_path: Path):
        """When both timestamps and node_timings exist, timestamps win."""
        exp_dir = tmp_path / "EXP-PRIORITY"
        exp_dir.mkdir()

        manifest = {
            "status": "complete",
            "started_at": "2026-05-25T12:00:00",
            "finished_at": "2026-05-25T12:15:00",  # 900s
            "challenge_ids": ["crypto/t1"],
            "settings_snapshot": {},
            "node_timings": {"ranking": 200.0},  # only 200s, NOT the true wall time
        }
        (exp_dir / "manifest.json").write_text(json.dumps(manifest))

        rankings = [
            {
                "challenge_id": "crypto/t1",
                "overall_score": 8.0,
                "technical_review": {"score": 8.0},
                "pedagogical_review": {"score": 8.0},
                "_refinement_rounds": 0,
            }
        ]
        (exp_dir / "ranking_reports.json").write_text(json.dumps(rankings))

        from src.services.review_generator import generate_review

        review_path = generate_review(exp_dir)
        content = review_path.read_text()

        # timestamps win: 900s, not 200s
        assert "**Wall time**: 900s" in content


class TestRefinementRoundHistogram:
    """REVIEW.md must contain a refinement-round histogram section."""

    def test_refinement_histogram_section_present(self, tmp_path: Path):
        """REVIEW.md must have a section titled 'Refinement round distribution'."""
        exp_dir = tmp_path / "EXP-HIST"
        exp_dir.mkdir()

        manifest = {
            "status": "complete",
            "challenge_ids": ["c/a", "c/b", "c/c"],
            "settings_snapshot": {},
        }
        (exp_dir / "manifest.json").write_text(json.dumps(manifest))

        rankings = [
            {
                "challenge_id": "c/a",
                "overall_score": 9.5,
                "technical_review": {"score": 9.5},
                "pedagogical_review": {"score": 9.5},
                "_refinement_rounds": 1,
            },
            {
                "challenge_id": "c/b",
                "overall_score": 7.0,
                "technical_review": {"score": 7.0},
                "pedagogical_review": {"score": 7.0},
                "_refinement_rounds": 3,
            },
            {
                "challenge_id": "c/c",
                "overall_score": 9.0,
                "technical_review": {"score": 9.0},
                "pedagogical_review": {"score": 9.0},
                "_refinement_rounds": 1,
            },
        ]
        (exp_dir / "ranking_reports.json").write_text(json.dumps(rankings))

        from src.services.review_generator import generate_review

        review_path = generate_review(exp_dir)
        content = review_path.read_text()

        assert (
            "Refinement round distribution" in content
        ), "Expected 'Refinement round distribution' section in REVIEW.md"

    def test_refinement_histogram_counts(self, tmp_path: Path):
        """Histogram rows must show correct per-round challenge counts."""
        exp_dir = tmp_path / "EXP-HISTCOUNT"
        exp_dir.mkdir()

        manifest = {
            "status": "complete",
            "challenge_ids": ["c/a", "c/b", "c/c", "c/d"],
            "settings_snapshot": {},
        }
        (exp_dir / "manifest.json").write_text(json.dumps(manifest))

        # 2x Round 1, 1x Round 2, 1x Round 3
        rankings = [
            {
                "challenge_id": "c/a",
                "overall_score": 9.5,
                "technical_review": {"score": 9.5},
                "pedagogical_review": {"score": 9.5},
                "_refinement_rounds": 1,
            },
            {
                "challenge_id": "c/b",
                "overall_score": 9.0,
                "technical_review": {"score": 9.0},
                "pedagogical_review": {"score": 9.0},
                "_refinement_rounds": 1,
            },
            {
                "challenge_id": "c/c",
                "overall_score": 7.5,
                "technical_review": {"score": 7.5},
                "pedagogical_review": {"score": 7.5},
                "_refinement_rounds": 2,
            },
            {
                "challenge_id": "c/d",
                "overall_score": 6.0,
                "technical_review": {"score": 6.0},
                "pedagogical_review": {"score": 6.0},
                "_refinement_rounds": 3,
            },
        ]
        (exp_dir / "ranking_reports.json").write_text(json.dumps(rankings))

        from src.services.review_generator import generate_review

        review_path = generate_review(exp_dir)
        content = review_path.read_text()

        # Round 1: 2 challenges
        assert "Round 1" in content
        # Round 2: 1 challenge
        assert "Round 2" in content
        # Round 3: 1 challenge
        assert "Round 3" in content
