"""Tests for review_generator and review_queue_updater."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures" / "exp_reviews"


class TestGenerateReviewModernComplete:
    """Test with synthetic run-complete using modern flat-list ranking format."""

    def test_generate_review_modern_complete(self, tmp_path: Path):
        exp_dir = tmp_path / "run-complete"
        shutil.copytree(FIXTURES / "review-complete", exp_dir)
        (exp_dir / "courses" / "crypto" / "aes").mkdir(parents=True)
        (exp_dir / "courses" / "crypto" / "aes" / "course.md").write_text("# AES")
        (exp_dir / "courses" / "web" / "xss").mkdir(parents=True)
        (exp_dir / "courses" / "web" / "xss" / "course.md").write_text("# XSS")
        (exp_dir / "courses" / "misc" / "trivia").mkdir(parents=True)
        (exp_dir / "courses" / "misc" / "trivia" / "course.md").write_text("# Trivia")

        from src.services.review_generator import generate_review

        review_path = generate_review(exp_dir)

        assert review_path.exists()
        content = review_path.read_text()
        assert "run-complete" in content
        assert "complete" in content
        assert "Pass rate" in content
        assert "Score distribution" in content

        review_json = exp_dir / "REVIEW.json"
        assert review_json.exists()
        data = json.loads(review_json.read_text())
        assert data["metrics"]["challenges_total"] == 3
        assert data["metrics"]["challenges_passed"] == 1
        assert data["distribution"]["below_5"] == 1
        assert data["distribution"]["range_7_9"] == 1
        assert data["distribution"]["above_9"] == 1


class TestGenerateReviewFailedEmpty:
    """Test with run-024 failed run (empty rankings)."""

    def test_generate_review_failed_empty(self, tmp_path: Path):
        exp_dir = tmp_path / "run-024"
        shutil.copytree(FIXTURES / "review-failed", exp_dir)

        from src.services.review_generator import generate_review

        review_path = generate_review(exp_dir)

        assert review_path.exists()
        content = review_path.read_text()
        assert "failed" in content
        assert "0.0%" in content

        data = json.loads((exp_dir / "REVIEW.json").read_text())
        assert data["metrics"]["pass_rate"] == 0.0
        assert data["metrics"]["challenges_total"] == 0
        assert data["anomalies"]["empty_challenge_ids"] is True
        assert data["anomalies"]["ranking_missing"] is False
        assert data["anomalies"]["mid_graph_halt"] is True


class TestGenerateReviewLegacyNoManifest:
    """Test with legacy format (no manifest.json, per_challenge rankings)."""

    def test_generate_review_legacy_no_manifest(self, tmp_path: Path):
        exp_dir = tmp_path / "run-019"
        shutil.copytree(FIXTURES / "review-legacy", exp_dir)

        from src.services.review_generator import generate_review

        review_path = generate_review(exp_dir)

        assert review_path.exists()
        data = json.loads((exp_dir / "REVIEW.json").read_text())
        assert data["metrics"]["challenges_total"] == 2
        assert data["metrics"]["mean_overall"] == pytest.approx(8.75, abs=0.01)
        assert data["status"] == "complete (inferred)"


class TestReviewQueueAppendAtomic:
    """Test atomic append to REVIEW_QUEUE.md."""

    def test_review_queue_append_atomic(self, tmp_path: Path):
        queue = tmp_path / "REVIEW_QUEUE.md"
        queue.write_text(
            "## \U0001f4ca Experiment runs (output/experiments/EXP-NNN/REVIEW.md)\n\n_None._\n\n---\n"
        )

        from src.services.review_queue_updater import append_to_review_queue

        append_to_review_queue(
            exp_id="run-test",
            review_path=tmp_path / "REVIEW.md",
            summary="test summary",
            queue_path=queue,
        )

        content = queue.read_text()
        assert "run-test" in content
        assert "_None._" not in content
        assert "test summary" in content


class TestReviewQueueIdempotentDedup:
    """Test idempotent dedup in REVIEW_QUEUE.md."""

    def test_review_queue_idempotent_dedup(self, tmp_path: Path):
        queue = tmp_path / "REVIEW_QUEUE.md"
        queue.write_text(
            "## \U0001f4ca Experiment runs (output/experiments/EXP-NNN/REVIEW.md)\n\n_None._\n\n---\n"
        )

        from src.services.review_queue_updater import append_to_review_queue

        append_to_review_queue(
            exp_id="run-dup",
            review_path=tmp_path / "REVIEW.md",
            summary="first",
            queue_path=queue,
        )
        append_to_review_queue(
            exp_id="run-dup",
            review_path=tmp_path / "REVIEW.md",
            summary="second",
            queue_path=queue,
        )

        content = queue.read_text()
        assert content.count("run-dup") == 1
        assert "second" in content
        assert "first" not in content


class TestJudgeModelField:
    """D13 regression: REVIEW.md Judge field must read from RANKING_MODEL, not generator model."""

    def test_judge_shows_ranking_model_not_generator(self, tmp_path: Path):
        """When reproducibility.json has RANKING_MODEL=haiku, Judge: must show haiku not sonnet."""
        exp_dir = tmp_path / "run-judgetest"
        shutil.copytree(FIXTURES / "review-judgetest", exp_dir)

        from src.services.review_generator import generate_review

        review_path = generate_review(exp_dir)
        content = review_path.read_text()

        # Generator model must appear in Model field
        assert "claude-sonnet-4-6" in content
        # Judge must show haiku (RANKING_MODEL), not sonnet
        assert "**Judge**: claude-haiku-4-5" in content
        assert "**Judge**: claude-sonnet-4-6" not in content

    def test_judge_fallback_cli_overrides(self, tmp_path: Path):
        """Fallback: when RANKING_MODEL absent, use reproducibility.cli_overrides.judge_model."""
        exp_dir = tmp_path / "run-judgetest-fallback"
        shutil.copytree(FIXTURES / "review-judgetest", exp_dir)
        # Remove RANKING_MODEL from settings, keep cli_overrides.judge_model
        repro_path = exp_dir / "reproducibility.json"
        repro = json.loads(repro_path.read_text())
        del repro["settings"]["RANKING_MODEL"]
        repro_path.write_text(json.dumps(repro, indent=2))

        from src.services.review_generator import generate_review

        review_path = generate_review(exp_dir)
        content = review_path.read_text()
        assert "**Judge**: claude-haiku-4-5" in content

    def test_judge_fallback_run_config(self, tmp_path: Path):
        """Fallback: when reproducibility.json absent, use run_config.json:ranking_model."""
        exp_dir = tmp_path / "run-judgetest-rc"
        shutil.copytree(FIXTURES / "review-judgetest", exp_dir)
        # Remove reproducibility.json, add ranking_model to run_config.json
        (exp_dir / "reproducibility.json").unlink()
        run_config = {
            "exp_id": "run-judgetest-rc",
            "provider": "anthropic",
            "model": "claude-sonnet-4-6",
            "ranking_model": "claude-haiku-4-5-rc-fallback",
        }
        (exp_dir / "run_config.json").write_text(json.dumps(run_config))

        from src.services.review_generator import generate_review

        review_path = generate_review(exp_dir)
        content = review_path.read_text()
        assert "**Judge**: claude-haiku-4-5-rc-fallback" in content

    def test_judge_unknown_when_no_sources(self, tmp_path: Path):
        """When no reproducibility.json and no run_config ranking_model, Judge: is 'unknown'."""
        exp_dir = tmp_path / "run-judgetest-nojudge"
        shutil.copytree(FIXTURES / "review-judgetest", exp_dir)
        (exp_dir / "reproducibility.json").unlink()
        # run_config without ranking_model
        run_config = {"exp_id": "run-judgetest-nojudge", "model": "claude-sonnet-4-6"}
        (exp_dir / "run_config.json").write_text(json.dumps(run_config))

        from src.services.review_generator import generate_review

        generate_review(exp_dir)
        data = json.loads((exp_dir / "REVIEW.json").read_text())
        assert data["judge_model"] == "unknown"


class TestHistogramAsciiWidths:
    """Test histogram rendering."""

    def test_histogram_ascii_widths(self):
        from src.services.review_generator import ScoreDistribution, _histogram_ascii

        dist = ScoreDistribution(below_5=0, range_5_7=2, range_7_9=5, above_9=10)
        result = _histogram_ascii(dist)
        lines = result.strip().split("\n")
        assert len(lines) == 4
        for line in lines:
            bar_section = line.split("|")[1]
            assert len(bar_section) == 30

    def test_histogram_empty_buckets(self):
        from src.services.review_generator import ScoreDistribution, _histogram_ascii

        dist = ScoreDistribution(below_5=0, range_5_7=0, range_7_9=0, above_9=0)
        result = _histogram_ascii(dist)
        assert "0" in result
