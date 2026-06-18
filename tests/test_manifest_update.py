"""Tests for Bug 1 fix: manifest.pass_count and mean_overall_score update from ranking_reports.

Covers:
1. After finish_run, pass_count and mean_overall_score reflect ranking_reports.json.
2. Empty ranking_reports.json → both fields stay at 0.
3. Partial reports (some with overall_score=None) → only valid entries counted.
4. No ranking_reports.json file → fields stay at 0 (skip_ranking scenario).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.tui.artifact_writer import ArtifactWriter
from src.tui.run_config import RunConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_exp_dir(tmp_path: Path) -> Path:
    """Temporary output/experiments base directory."""
    return tmp_path / "output" / "experiments"


def _make_cfg(exp_id: str = "EXP-TEST") -> RunConfig:
    return RunConfig(
        exp_id=exp_id,
        provider="anthropic",
        model="claude-haiku",
        temperature=0.0,
        threshold=9.0,
        challenge_ids=["cat/ch1", "cat/ch2"],
        categories=["cat"],
        source="cyberedu",
        max_refinements=5,
        skip_ranking=False,
    )


def _make_ranking_reports(scores: list) -> list:
    """Build minimal ranking_reports list from a list of (challenge_id, overall_score) tuples."""
    reports = []
    for cid, score in scores:
        entry = {
            "challenge_id": cid,
            "overall_score": score,
            "pedagogical_review": {
                "score": score,
                "persona": "Pedagogical",
                "justification": "ok",
                "improvements": [],
            },
            "technical_review": {
                "score": score,
                "persona": "Technical",
                "justification": "ok",
                "improvements": [],
            },
            "technical_rank": "Intermediate",
        }
        reports.append(entry)
    return reports


# ---------------------------------------------------------------------------
# Test 1: Normal case — pass_count and mean_overall_score are set correctly
# ---------------------------------------------------------------------------


def test_manifest_pass_count_and_mean_from_ranking_reports(tmp_exp_dir: Path):
    """After finish_run, manifest reflects ranking_reports scores."""
    cfg = _make_cfg("EXP-M1")
    writer = ArtifactWriter(tmp_exp_dir, cfg)
    writer.start_run(cfg)

    # Simulate ranking: 3 challenges, scores 10.0, 9.0, 5.0
    # threshold=9.0 → 2 pass (10.0 and 9.0)
    reports = _make_ranking_reports(
        [
            ("cat/ch1", 10.0),
            ("cat/ch2", 9.0),
            ("cat/ch3", 5.0),
        ]
    )
    writer.write_ranking(reports)

    with patch("src.config.settings.settings") as mock_settings:
        mock_settings.RANKING_PASS_THRESHOLD = 9.0
        writer.finish_run(success=True, node_timings={})

    manifest_path = tmp_exp_dir / "EXP-M1" / "manifest.json"
    assert manifest_path.exists()
    data = json.loads(manifest_path.read_text())

    assert data["pass_count"] == 2, f"Expected 2 passes, got {data['pass_count']}"
    assert (
        abs(data["mean_overall_score"] - 8.0) < 0.01
    ), f"Expected mean=8.0, got {data['mean_overall_score']}"
    assert data["status"] == "complete"


# ---------------------------------------------------------------------------
# Test 2: Empty ranking_reports → fields stay at 0
# ---------------------------------------------------------------------------


def test_manifest_empty_ranking_reports_keeps_zeros(tmp_exp_dir: Path):
    """Empty ranking_reports.json → pass_count=0 and mean_overall_score=0.0."""
    cfg = _make_cfg("EXP-M2")
    writer = ArtifactWriter(tmp_exp_dir, cfg)
    writer.start_run(cfg)
    writer.write_ranking([])  # empty list

    with patch("src.config.settings.settings") as mock_settings:
        mock_settings.RANKING_PASS_THRESHOLD = 9.0
        writer.finish_run(success=True, node_timings={})

    data = json.loads((tmp_exp_dir / "EXP-M2" / "manifest.json").read_text())
    assert data["pass_count"] == 0
    assert data["mean_overall_score"] == 0.0


# ---------------------------------------------------------------------------
# Test 3: Partial reports (some with overall_score=None) → only valid counted
# ---------------------------------------------------------------------------


def test_manifest_partial_reports_skips_none_scores(tmp_exp_dir: Path):
    """Entries with overall_score=None are excluded from pass_count and mean."""
    cfg = _make_cfg("EXP-M3")
    writer = ArtifactWriter(tmp_exp_dir, cfg)
    writer.start_run(cfg)

    # 2 valid, 1 with None overall_score
    reports = _make_ranking_reports([("cat/ch1", 10.0), ("cat/ch2", 8.0)])
    reports.append({"challenge_id": "cat/ch3", "overall_score": None})
    writer.write_ranking(reports)

    with patch("src.config.settings.settings") as mock_settings:
        mock_settings.RANKING_PASS_THRESHOLD = 9.0
        writer.finish_run(success=True, node_timings={})

    data = json.loads((tmp_exp_dir / "EXP-M3" / "manifest.json").read_text())
    # Only ch1 (10.0) passes; ch2 (8.0) does not
    assert data["pass_count"] == 1
    assert (
        abs(data["mean_overall_score"] - 9.0) < 0.01
    ), f"Expected mean=9.0 (from 10.0+8.0 / 2), got {data['mean_overall_score']}"


# ---------------------------------------------------------------------------
# Test 4: No ranking_reports.json (skip_ranking scenario)
# ---------------------------------------------------------------------------


def test_manifest_no_ranking_reports_file_keeps_zeros(tmp_exp_dir: Path):
    """When ranking_reports.json doesn't exist (skip_ranking), fields stay at 0."""
    cfg = _make_cfg("EXP-M4")
    writer = ArtifactWriter(tmp_exp_dir, cfg)
    writer.start_run(cfg)
    # Do NOT call write_ranking — simulates skip_ranking=True

    with patch("src.config.settings.settings") as mock_settings:
        mock_settings.RANKING_PASS_THRESHOLD = 9.0
        writer.finish_run(success=True, node_timings={})

    data = json.loads((tmp_exp_dir / "EXP-M4" / "manifest.json").read_text())
    assert data["pass_count"] == 0
    assert data["mean_overall_score"] == 0.0


# ---------------------------------------------------------------------------
# Test 5: All entries pass threshold
# ---------------------------------------------------------------------------


def test_manifest_all_pass(tmp_exp_dir: Path):
    """When all challenges pass, pass_count equals total challenges."""
    cfg = _make_cfg("EXP-M5")
    writer = ArtifactWriter(tmp_exp_dir, cfg)
    writer.start_run(cfg)

    reports = _make_ranking_reports(
        [
            ("cat/ch1", 9.5),
            ("cat/ch2", 10.0),
            ("cat/ch3", 9.0),
        ]
    )
    writer.write_ranking(reports)

    with patch("src.config.settings.settings") as mock_settings:
        mock_settings.RANKING_PASS_THRESHOLD = 9.0
        writer.finish_run(success=True, node_timings={})

    data = json.loads((tmp_exp_dir / "EXP-M5" / "manifest.json").read_text())
    assert data["pass_count"] == 3
    expected_mean = round((9.5 + 10.0 + 9.0) / 3, 4)
    assert abs(data["mean_overall_score"] - expected_mean) < 0.01


# ---------------------------------------------------------------------------
# Test 6: failed run still updates manifest fields
# ---------------------------------------------------------------------------


def test_manifest_failed_run_still_computes_scores(tmp_exp_dir: Path):
    """Even on failed runs, if ranking_reports exists, scores are computed."""
    cfg = _make_cfg("EXP-M6")
    writer = ArtifactWriter(tmp_exp_dir, cfg)
    writer.start_run(cfg)

    reports = _make_ranking_reports([("cat/ch1", 7.0)])
    writer.write_ranking(reports)

    with patch("src.config.settings.settings") as mock_settings:
        mock_settings.RANKING_PASS_THRESHOLD = 9.0
        writer.finish_run(success=False, node_timings={})

    data = json.loads((tmp_exp_dir / "EXP-M6" / "manifest.json").read_text())
    assert data["status"] == "failed"
    assert data["pass_count"] == 0  # 7.0 < 9.0
    assert abs(data["mean_overall_score"] - 7.0) < 0.01
