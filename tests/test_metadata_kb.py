"""Tests for the Metadata Knowledge Base."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fixture_tree(base: Path) -> Path:
    """Create a minimal processed-dir fixture tree and return it."""
    # contest1/scripting/hello-flag/cyberedu/write-up/description.md
    ch1 = base / "contest1" / "scripting" / "hello-flag" / "cyberedu" / "write-up"
    ch1.mkdir(parents=True, exist_ok=True)
    (ch1 / "description.md").write_text("# Hello Flag\n\n**Category:** Scripting\n")

    # contest1/crypto/base64-easy/cyberedu/write-up/description.md (with author frontmatter)
    ch2 = base / "contest1" / "crypto" / "base64-easy" / "cyberedu" / "write-up"
    ch2.mkdir(parents=True, exist_ok=True)
    (ch2 / "description.md").write_text(
        "---\nauthor: alice\ntitle: Base64 Easy\n---\n# Base64 Easy\n"
    )

    # contest2/web/xss-reflect/cyberedu/write-up/description.md
    ch3 = base / "contest2" / "web" / "xss-reflect" / "cyberedu" / "write-up"
    ch3.mkdir(parents=True, exist_ok=True)
    (ch3 / "description.md").write_text("# XSS Reflect\n")

    return base


# ---------------------------------------------------------------------------
# Test 1: build_metadata_kb produces the three JSON files
# ---------------------------------------------------------------------------


def test_build_metadata_kb_produces_files(tmp_path):
    processed = _make_fixture_tree(tmp_path / "processed")
    output = tmp_path / "kb_meta"

    from src.pipeline.build_metadata_kb import build_metadata_kb

    counts = build_metadata_kb(processed_dir=processed, output_dir=output)

    assert (output / "contests.json").is_file()
    assert (output / "categories.json").is_file()
    assert (output / "challenges_index.json").is_file()

    contests = json.loads((output / "contests.json").read_text())
    categories = json.loads((output / "categories.json").read_text())
    challenges = json.loads((output / "challenges_index.json").read_text())

    assert counts["contests"] == 2
    assert counts["categories"] == 3  # scripting, crypto, web
    assert counts["challenges"] == 3

    contest_ids = {c["id"] for c in contests}
    assert contest_ids == {"contest1", "contest2"}

    cat_ids = {c["id"] for c in categories}
    assert cat_ids == {"scripting", "crypto", "web"}

    # author extracted from frontmatter for base64-easy
    base64_entry = next(c for c in challenges if c["challenge_id"] == "base64-easy")
    assert base64_entry["author"] == "alice"
    assert base64_entry["contest_id"] == "contest1"
    assert base64_entry["category"] == "crypto"

    # challenge without author has no "author" key
    hello_entry = next(c for c in challenges if c["challenge_id"] == "hello-flag")
    assert "author" not in hello_entry


# ---------------------------------------------------------------------------
# Test 2: similarity_search passes metadata_filter as 'filter' kwarg; omits when None
# ---------------------------------------------------------------------------


def test_similarity_search_passes_filter_when_provided():
    from src.services.vector_db_service import VectorDBService

    svc = VectorDBService.__new__(VectorDBService)
    mock_store = MagicMock()
    mock_store.similarity_search_with_relevance_scores.return_value = []
    svc._vector_store = mock_store
    svc._embedding = MagicMock()
    svc._persist_dir = Path("/tmp/fake")
    svc._collection_name = "test"

    mf = {"contest_id": {"$eq": "ctf-finals-2025"}}
    svc.similarity_search("xss attack", k=3, metadata_filter=mf)

    mock_store.similarity_search_with_relevance_scores.assert_called_once()
    _, kwargs = mock_store.similarity_search_with_relevance_scores.call_args
    assert kwargs.get("filter") == mf


def test_similarity_search_omits_filter_when_none():
    from src.services.vector_db_service import VectorDBService

    svc = VectorDBService.__new__(VectorDBService)
    mock_store = MagicMock()
    mock_store.similarity_search_with_relevance_scores.return_value = []
    svc._vector_store = mock_store
    svc._embedding = MagicMock()
    svc._persist_dir = Path("/tmp/fake")
    svc._collection_name = "test"

    svc.similarity_search("xss attack", k=3)

    mock_store.similarity_search_with_relevance_scores.assert_called_once()
    _, kwargs = mock_store.similarity_search_with_relevance_scores.call_args
    assert "filter" not in kwargs


# ---------------------------------------------------------------------------
# Test 3: ranking prompt includes contest note when contest_metadata populated
# ---------------------------------------------------------------------------

_DUMMY_TECH_JSON = json.dumps(
    {
        "score": 8,
        "justification": "ok",
        "improvements": [],
        "technical_rank": "Intermediate",
        "dimension_scores": {
            "correctness": 8,
            "completeness": 8,
            "technical_accuracy": 8,
            "code_quality": 8,
            "logical_validity": 8,
        },
    }
)
_DUMMY_PED_JSON = json.dumps(
    {
        "score": 8,
        "justification": "ok",
        "improvements": [],
        "dimension_scores": {
            "sections_structure": 8,
            "cognitive_load": 8,
            "scaffolding_reproducibility": 8,
            "relevance_curriculum": 8,
            "skill_level_awareness": 8,
            "human_language_context": 8,
        },
    }
)


def test_ranking_prompt_includes_contest_note_when_metadata_set():
    from src.agents.ranking_agent import _evaluate_one_challenge

    contest_meta = {
        "id": "unr25",
        "name": "CTF Finals 2025",
        "type": "ctf",
        "date": "2025-05",
    }
    captured_systems: list[str] = []

    def mock_generate(system_prompt, user_prompt, **kwargs):
        # Arg 0 = system_prompt; contest note is prepended there
        captured_systems.append(system_prompt)
        if len(captured_systems) == 1:
            return _DUMMY_TECH_JSON
        return _DUMMY_PED_JSON

    with patch(
        "src.agents.ranking_agent.generate_response_with_system",
        side_effect=mock_generate,
    ):
        _evaluate_one_challenge(
            "chall1",
            "writeup text",
            "solve script",
            contest_metadata=contest_meta,
        )

    assert len(captured_systems) == 2
    for system_prompt in captured_systems:
        assert "Contest: CTF Finals 2025" in system_prompt
        assert "ctf" in system_prompt
        assert "2025-05" in system_prompt


# ---------------------------------------------------------------------------
# Test 4: ranking prompt has NO contest note when contest_metadata is empty
# ---------------------------------------------------------------------------


def test_ranking_prompt_excludes_contest_note_when_metadata_empty():
    from src.agents.ranking_agent import _evaluate_one_challenge

    captured_systems: list[str] = []

    def mock_generate(system_prompt, user_prompt, **kwargs):
        captured_systems.append(system_prompt)
        if len(captured_systems) == 1:
            return _DUMMY_TECH_JSON
        return _DUMMY_PED_JSON

    with patch(
        "src.agents.ranking_agent.generate_response_with_system",
        side_effect=mock_generate,
    ):
        _evaluate_one_challenge("chall1", "writeup text", "solve script")

    assert len(captured_systems) == 2
    for system_prompt in captured_systems:
        assert "Contest:" not in system_prompt


# ---------------------------------------------------------------------------
# Test 5: summarise_kb returns correct counts from fixture JSON
# ---------------------------------------------------------------------------


def test_summarise_kb_correct_counts(tmp_path):
    challenges = [
        {"challenge_id": "ch1", "contest_id": "c1", "category": "crypto"},
        {"challenge_id": "ch2", "contest_id": "c1", "category": "web"},
        {"challenge_id": "ch3", "contest_id": "c2", "category": "crypto"},
        {"challenge_id": "ch4", "contest_id": "c2", "category": "crypto"},
    ]
    (tmp_path / "challenges_index.json").write_text(
        json.dumps(challenges), encoding="utf-8"
    )

    from src.utils.kb_analytics import summarise_kb

    result = summarise_kb(tmp_path)

    assert result["total_challenges"] == 4
    assert result["by_category"] == {"crypto": 3, "web": 1}
    assert result["by_contest"] == {"c1": 2, "c2": 2}


def test_summarise_kb_missing_file(tmp_path):
    from src.utils.kb_analytics import summarise_kb

    result = summarise_kb(tmp_path)
    assert result == {"total_challenges": 0, "by_category": {}, "by_contest": {}}
