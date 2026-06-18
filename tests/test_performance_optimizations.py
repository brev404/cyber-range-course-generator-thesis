"""Tests for performance optimisations (prompt caching, batch API, cache tracking).

Covers:
- generate_response_with_system uses SystemMessage with cache_control for Anthropic
- generate_response_with_system falls back to concatenated call for non-Anthropic
- usage_out dict is populated with cache token counts from response metadata
- submit_ranking_batch constructs the correct batch request shape
- RewardRecord has cache_read_tokens and cache_creation_tokens fields
- compute_reward accepts and forwards cache token counts
- RANKING_USE_BATCH_API setting exists and defaults to False
"""

from unittest.mock import MagicMock, patch

from src.utils.feedback_utils import RewardRecord, compute_reward

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ai_message(content: str, cache_read: int = 0, cache_created: int = 0):
    msg = MagicMock()
    msg.content = content
    msg.response_metadata = {
        "usage": {
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_read_input_tokens": cache_read,
            "cache_creation_input_tokens": cache_created,
        }
    }
    return msg


# ---------------------------------------------------------------------------
# generate_response_with_system — Anthropic path
# ---------------------------------------------------------------------------


def test_anthropic_path_uses_system_message_with_cache_control():
    """When provider resolves to anthropic, a SystemMessage with cache_control is sent."""
    from langchain_core.messages import SystemMessage

    mock_chat = MagicMock()
    mock_chat.invoke.return_value = _make_ai_message("ok")

    with (
        patch("src.services.llm_service.get_chat_model", return_value=mock_chat),
        patch("src.services.llm_service.settings") as mock_settings,
    ):
        mock_settings.LLM_DEFAULT_PROVIDER = "anthropic"

        from src.services.llm_service import generate_response_with_system

        result = generate_response_with_system(
            "system instructions",
            "user message",
            provider="anthropic",
        )

    assert result == "ok"
    mock_chat.invoke.assert_called_once()
    call_args = mock_chat.invoke.call_args[0][0]
    system_msg = call_args[0]
    assert isinstance(system_msg, SystemMessage)
    # Content must be a list with a dict that has cache_control
    assert isinstance(system_msg.content, list)
    block = system_msg.content[0]
    assert block["type"] == "text"
    assert block["text"] == "system instructions"
    assert block["cache_control"] == {"type": "ephemeral"}


def test_anthropic_path_second_message_is_human():
    """The second message in the Anthropic path must be a HumanMessage."""
    from langchain_core.messages import HumanMessage

    mock_chat = MagicMock()
    mock_chat.invoke.return_value = _make_ai_message("ok")

    with patch("src.services.llm_service.get_chat_model", return_value=mock_chat):
        from src.services.llm_service import generate_response_with_system

        generate_response_with_system(
            "sys",
            "user content here",
            provider="anthropic",
        )

    call_args = mock_chat.invoke.call_args[0][0]
    human_msg = call_args[1]
    assert isinstance(human_msg, HumanMessage)
    assert human_msg.content == "user content here"


# ---------------------------------------------------------------------------
# generate_response_with_system — non-Anthropic fallback
# ---------------------------------------------------------------------------


def test_non_anthropic_uses_concatenated_generate_response():
    """Non-Anthropic providers fall back to generate_response with concatenated prompt."""
    with patch(
        "src.services.llm_service.generate_response", return_value="resp"
    ) as mock_gr:
        from src.services.llm_service import generate_response_with_system

        result = generate_response_with_system(
            "system part",
            "user part",
            provider="google",
            temperature=0.3,
            max_tokens=512,
        )

    assert result == "resp"
    mock_gr.assert_called_once()
    # The concatenated prompt must contain both parts
    concat_prompt = mock_gr.call_args[0][0]
    assert "system part" in concat_prompt
    assert "user part" in concat_prompt


def test_openrouter_uses_concatenated_fallback():
    """OpenRouter also falls back to concatenated path (not Anthropic)."""
    with patch(
        "src.services.llm_service.generate_response", return_value="r"
    ) as mock_gr:
        from src.services.llm_service import generate_response_with_system

        generate_response_with_system("sys", "usr", provider="openrouter")

    mock_gr.assert_called_once()


# ---------------------------------------------------------------------------
# usage_out populated from response metadata
# ---------------------------------------------------------------------------


def test_usage_out_populated_with_cache_tokens():
    """usage_out dict is updated with cache token counts from response metadata."""
    mock_chat = MagicMock()
    mock_chat.invoke.return_value = _make_ai_message(
        "answer", cache_read=200, cache_created=1500
    )

    usage_acc: dict = {}
    with patch("src.services.llm_service.get_chat_model", return_value=mock_chat):
        from src.services.llm_service import generate_response_with_system

        generate_response_with_system(
            "sys", "usr", provider="anthropic", usage_out=usage_acc
        )

    assert usage_acc["cache_read_input_tokens"] == 200
    assert usage_acc["cache_creation_input_tokens"] == 1500


def test_usage_out_accumulates_across_calls():
    """Calling with the same usage_out dict accumulates across multiple calls."""
    mock_chat = MagicMock()
    mock_chat.invoke.side_effect = [
        _make_ai_message("first", cache_read=100, cache_created=500),
        _make_ai_message("second", cache_read=100, cache_created=0),
    ]

    usage_acc: dict = {}
    with patch("src.services.llm_service.get_chat_model", return_value=mock_chat):
        from src.services.llm_service import generate_response_with_system

        generate_response_with_system(
            "sys", "usr1", provider="anthropic", usage_out=usage_acc
        )
        generate_response_with_system(
            "sys", "usr2", provider="anthropic", usage_out=usage_acc
        )

    assert usage_acc["cache_read_input_tokens"] == 200
    assert usage_acc["cache_creation_input_tokens"] == 500


# ---------------------------------------------------------------------------
# submit_ranking_batch request shape
# ---------------------------------------------------------------------------


def test_submit_ranking_batch_request_shape():
    """submit_ranking_batch builds the expected Anthropic batch request structure."""
    from src.utils.anthropic_batch import submit_ranking_batch

    requests = [
        {
            "custom_id": "abc123_tech",
            "model": "claude-haiku-4-5-20251001",
            "system": "You are a technical reviewer.",
            "user": "Evaluate this course.",
            "temperature": 0.3,
            "max_tokens": 4096,
        }
    ]

    mock_batch = MagicMock()
    mock_batch.id = "batch_xyz"
    mock_client = MagicMock()
    mock_client.messages.batches.create.return_value = mock_batch

    with (
        patch("src.utils.anthropic_batch._get_client", return_value=mock_client),
        patch("src.config.settings.settings") as mock_settings,
    ):
        mock_settings.ANTHROPIC_API_KEY = "sk-ant-test"
        result = submit_ranking_batch(requests)

    assert result == "batch_xyz"
    create_call = mock_client.messages.batches.create.call_args
    sent_requests = create_call[1].get("requests") or create_call[0][0]
    assert len(sent_requests) == 1
    req = sent_requests[0]
    assert req["custom_id"] == "abc123_tech"
    params = req["params"]
    assert params["model"] == "claude-haiku-4-5-20251001"
    assert params["max_tokens"] == 4096
    assert params["temperature"] == 0.3
    # System must be a list with a block that has cache_control
    system_list = params["system"]
    assert isinstance(system_list, list)
    assert system_list[0]["cache_control"] == {"type": "ephemeral"}
    assert system_list[0]["text"] == "You are a technical reviewer."
    # Messages must be a list with one user turn
    messages = params["messages"]
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "Evaluate this course."


# ---------------------------------------------------------------------------
# RewardRecord and compute_reward cache token fields
# ---------------------------------------------------------------------------


def test_reward_record_has_cache_token_fields():
    """RewardRecord has cache_read_tokens and cache_creation_tokens defaulting to 0."""
    record = RewardRecord(
        run_id="r1",
        timestamp="2026-05-08T00:00:00+00:00",
        judge_model="judge",
        per_challenge_scores={},
        mean_tech=8.0,
        mean_ped=8.0,
        pass_rate=1.0,
        reward=False,
        prompt_version="abcd1234",
    )
    assert record.cache_read_tokens == 0
    assert record.cache_creation_tokens == 0


def test_reward_record_cache_tokens_set():
    """RewardRecord stores non-zero cache token values."""
    record = RewardRecord(
        run_id="r2",
        timestamp="2026-05-08T00:00:00+00:00",
        judge_model="judge",
        per_challenge_scores={},
        mean_tech=8.0,
        mean_ped=8.0,
        pass_rate=1.0,
        reward=False,
        prompt_version="abcd1234",
        cache_read_tokens=1500,
        cache_creation_tokens=300,
    )
    assert record.cache_read_tokens == 1500
    assert record.cache_creation_tokens == 300


def test_compute_reward_forwards_cache_tokens():
    """compute_reward populates cache token fields from its arguments."""
    from src.config.settings import settings
    from src.models.report_models import RankingReport, RankingScore

    def _score(persona, s):
        return RankingScore(
            score=s,
            persona=persona,
            justification="ok",
            improvements=[],
            dimension_scores=None,
        )

    report = RankingReport(
        challenge_id="test/ch1",
        overall_score=8.0,
        pedagogical_review=_score("Pedagogical", 8),
        technical_review=_score("Technical", 8),
        technical_rank="Intermediate",
    )

    record = compute_reward(
        [report],
        judge_model=settings.FEEDBACK_JUDGE_MODEL,
        cache_read_tokens=2000,
        cache_creation_tokens=400,
    )

    assert record.cache_read_tokens == 2000
    assert record.cache_creation_tokens == 400


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


def test_ranking_use_batch_api_defaults_to_false():
    """RANKING_USE_BATCH_API is present in settings and defaults to False."""
    from src.config.settings import settings

    assert hasattr(settings, "RANKING_USE_BATCH_API")
    assert settings.RANKING_USE_BATCH_API is False
