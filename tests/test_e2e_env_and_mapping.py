"""E2E tests: env configuration, pipeline (organize + validate), mapping, and graph.

Fixtures live under tests/fixtures/challenges/ and tests/fixtures/processed/.
These tests verify env and mapping without requiring external real paths.
Run with: ./venv/bin/python -m pytest tests/ -v
"""

from pathlib import Path

import pytest

from src.config import settings as settings_module
from src.models.report_models import ValidationReport


def test_settings_defined() -> None:
    """Settings load from src/config/settings; key paths are defined."""
    s = settings_module.settings
    assert s.RAW_CHALLENGES_SOURCE is not None
    assert s.PROCESSED_DIR is not None
    assert s.OFFICIAL_DOCS_SOURCE is not None
    assert isinstance(s.RAW_CHALLENGES_SOURCE, Path)
    assert isinstance(s.PROCESSED_DIR, Path)
    assert isinstance(s.OFFICIAL_DOCS_SOURCE, Path)


def test_export_for_reproducibility() -> None:
    """export_for_reproducibility returns dict with settings and redacted secrets."""
    s = settings_module.settings
    data = s.export_for_reproducibility(cli_overrides={"generate_courses": True})
    assert "run_started_at" in data
    assert "settings" in data
    assert "secrets_redacted" in data
    assert data["settings"]["LLM_TEMPERATURE"] == s.LLM_TEMPERATURE
    assert data["settings"]["RAG_CHUNK_SIZE"] == s.RAG_CHUNK_SIZE
    assert data["cli_overrides"] == {"generate_courses": True}
    for key in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "LANGCHAIN_API_KEY",
    ):
        assert data["secrets_redacted"][key] in ("set", "unset")


def test_validation_report_shape(raw_challenges_fixture: Path) -> None:
    """Structural validation produces ValidationReport with challenge_id and issues."""
    from src.pipeline.validate_challenge_structure import validate_challenge

    challenge_path = raw_challenges_fixture / "crypto" / "test_crypto_01"
    if not challenge_path.is_dir():
        pytest.skip("Fixture crypto/test_crypto_01 not found")
    report = validate_challenge(challenge_path, "crypto")
    assert isinstance(report, ValidationReport)
    assert report.challenge_id == "crypto/test_crypto_01"
    assert hasattr(report, "issues")
    assert hasattr(report, "structure_score")


def test_pipeline_organize_validate_with_fixtures(
    monkeypatch: pytest.MonkeyPatch,
    raw_challenges_fixture: Path,
    tmp_path: Path,
) -> None:
    """Run organize and validate on fixture paths; assert no config errors and report shape."""
    if not raw_challenges_fixture.is_dir():
        pytest.skip("Raw fixtures not found")
    dest_root = tmp_path / "raw_challenges"
    dest_root.mkdir(parents=True, exist_ok=True)
    out_dir = tmp_path / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        settings_module.settings, "RAW_CHALLENGES_SOURCE", raw_challenges_fixture
    )
    monkeypatch.setattr(settings_module.settings, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(settings_module.settings, "OUTPUT_DIR", out_dir)
    import src.pipeline.organize_challenges as org_mod
    import src.pipeline.validate_challenge_structure as val_mod

    monkeypatch.setattr(org_mod, "SOURCE_ROOT_DIR", raw_challenges_fixture)
    monkeypatch.setattr(org_mod, "DESTINATION_ROOT_DIR", dest_root)
    org_mod.run_organizer()

    monkeypatch.setattr(val_mod, "ORGANIZED_CHALLENGES_ROOT", dest_root)
    val_mod.run_validator()

    out_summary = out_dir / "validation_reports" / "validation_summary.json"
    assert out_summary.exists(), "Validation summary should be written"
    import json

    data = json.loads(out_summary.read_text(encoding="utf-8"))
    assert len(data) >= 1
    for item in data:
        assert "challenge_id" in item
        assert "issues" in item


def test_map_docs_completes_with_fixture_processed(
    monkeypatch: pytest.MonkeyPatch,
    processed_fixture: Path,
) -> None:
    """map-docs step runs without unhandled exception when PROCESSED_DIR points at fixtures."""
    if not (processed_fixture / "raw_challenges").is_dir():
        pytest.skip("Processed fixtures not found")
    monkeypatch.setattr(settings_module.settings, "PROCESSED_DIR", processed_fixture)
    monkeypatch.setattr(
        settings_module.settings, "OFFICIAL_DOCS_SOURCE", processed_fixture
    )
    import src.pipeline.map_official_docs as map_mod

    monkeypatch.setattr(map_mod, "OFFICIAL_DOCS_DIR", processed_fixture)
    monkeypatch.setattr(
        map_mod, "PROCESSED_CHALLENGES_ROOT", processed_fixture / "raw_challenges"
    )
    map_mod.map_official_docs()


def test_graph_e2e_with_fixtures(
    monkeypatch: pytest.MonkeyPatch,
    processed_fixture: Path,
) -> None:
    """Graph runs with fixture PROCESSED_DIR; validation_reports and writeup_mappings shape."""
    if not (processed_fixture / "raw_challenges").is_dir():
        pytest.skip("Processed fixtures not found")
    monkeypatch.setattr(settings_module.settings, "PROCESSED_DIR", processed_fixture)

    from src.core.graph import app
    from src.core.state import AgentState

    state = AgentState()
    state.skip_ranking = True
    state.content_generation_subset_ids = []
    config = {"configurable": {"thread_id": "e2e_test_fixtures"}}
    final = app.invoke(state, config=config)
    # With checkpointer, invoke may return state as dict
    if isinstance(final, dict):
        validation_reports = final.get("validation_reports", [])
        writeup_mappings = final.get("writeup_mappings", {})
    else:
        validation_reports = final.validation_reports
        writeup_mappings = final.writeup_mappings

    assert len(validation_reports) >= 1
    for report in validation_reports:
        assert hasattr(report, "challenge_id")
        assert hasattr(report, "issues")
        assert report.challenge_id
    assert isinstance(writeup_mappings, dict)


def test_full_pipeline_real_paths_optional(monkeypatch: pytest.MonkeyPatch) -> None:
    """Optional: run pipeline against real paths from env vars; skip in CI.

    Set REAL_CHALLENGES_DIR and REAL_DOCS_DIR env vars to point at local challenge
    data to run this test. When unset (e.g. CI), the test is skipped.
    """
    import os

    real_challenges = os.environ.get("REAL_CHALLENGES_DIR")
    real_docs = os.environ.get("REAL_DOCS_DIR")
    if not real_challenges or not Path(real_challenges).is_dir():
        pytest.skip("REAL_CHALLENGES_DIR not set or not found (e.g. CI)")
    if not real_docs or not Path(real_docs).is_dir():
        pytest.skip("REAL_DOCS_DIR not set or not found (e.g. CI)")

    from src.pipeline.map_official_docs import run_mapping
    from src.pipeline.organize_challenges import run_organizer
    from src.pipeline.validate_challenge_structure import run_validator

    run_organizer()
    run_validator()
    run_mapping()
