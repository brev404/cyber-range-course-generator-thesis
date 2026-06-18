"""Anthropic Batch API helpers for cost-optimised ranking.

Submitting all ranking calls as a single batch job cuts cost by ~50% vs
synchronous requests. The tradeoff is latency: batches are processed within
~1 hour rather than immediately.

Usage (gated by settings.RANKING_USE_BATCH_API=True):
    batch_id = submit_ranking_batch(requests)  # returns immediately
    results  = poll_batch(batch_id)             # blocks until done

Requires ANTHROPIC_API_KEY and the `anthropic` package to be installed.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

from loguru import logger

from src.config.settings import settings

_MAX_POLLS = 40
_POLL_INTERVAL_S = 30


def _get_client() -> Any:
    """Return an authenticated Anthropic client.

    Raises:
        ValueError: If ANTHROPIC_API_KEY is not configured.
        ImportError: If the `anthropic` package is not installed.
    """
    if not settings.ANTHROPIC_API_KEY:
        raise ValueError(
            "ANTHROPIC_API_KEY is not set; required when RANKING_USE_BATCH_API=True"
        )
    try:
        import anthropic

        return anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
    except ImportError as exc:
        raise ImportError(
            "anthropic package is not installed; run: uv add anthropic"
        ) from exc


def submit_ranking_batch(requests: List[Dict[str, Any]]) -> str:
    """Submit ranking requests as an Anthropic Message Batch.

    Each request dict must have:
        custom_id (str): Unique ID for this request (≤64 chars, alphanumeric + _/-).
        model (str): Model identifier (e.g. 'claude-haiku-4-5-20251001').
        system (str): System prompt text (sent as a cached ephemeral block).
        user (str): User message content.
        temperature (float, optional): Sampling temperature. Default 0.3.
        max_tokens (int, optional): Max output tokens. Default 4096.

    Args:
        requests: List of request dicts as described above.

    Returns:
        Batch ID string (used to poll for results).
    """
    client = _get_client()
    batch_requests = []
    for req in requests:
        batch_requests.append(
            {
                "custom_id": req["custom_id"],
                "params": {
                    "model": req["model"],
                    "max_tokens": req.get("max_tokens", 4096),
                    "temperature": req.get("temperature", 0.3),
                    "system": [
                        {
                            "type": "text",
                            "text": req["system"],
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                    "messages": [{"role": "user", "content": req["user"]}],
                },
            }
        )
    batch = client.messages.batches.create(requests=batch_requests)
    logger.info(
        "Anthropic batch submitted: id={}, {} requests", batch.id, len(batch_requests)
    )
    return batch.id


def poll_batch(batch_id: str) -> List[Dict[str, Any]]:
    """Poll until the batch completes and return per-request results.

    Polls every _POLL_INTERVAL_S seconds up to _MAX_POLLS times
    (default: 30s × 40 = 20 minutes). Raises RuntimeError on timeout.

    Args:
        batch_id: Batch ID returned by submit_ranking_batch.

    Returns:
        List of dicts: {custom_id, content (str), error (str|None)}.
        content is the assistant reply text; error is set on request failure.

    Raises:
        RuntimeError: If the batch does not complete within the timeout.
    """
    client = _get_client()
    for poll_num in range(1, _MAX_POLLS + 1):
        batch = client.messages.batches.retrieve(batch_id)
        status = batch.processing_status
        logger.info(
            "Batch {} poll {}/{}: status={}", batch_id, poll_num, _MAX_POLLS, status
        )
        if status == "ended":
            break
        if poll_num >= _MAX_POLLS:
            raise RuntimeError(
                f"Batch {batch_id} timed out after {_MAX_POLLS * _POLL_INTERVAL_S}s"
            )
        time.sleep(_POLL_INTERVAL_S)

    results: List[Dict[str, Any]] = []
    for result in client.messages.batches.results(batch_id):
        custom_id = result.custom_id
        if result.result.type == "succeeded":
            content_blocks = result.result.message.content or []
            content = content_blocks[0].text if content_blocks else ""
            results.append({"custom_id": custom_id, "content": content, "error": None})
        else:
            error = str(getattr(result.result, "error", "unknown"))
            logger.warning("Batch request {} failed: {}", custom_id, error)
            results.append({"custom_id": custom_id, "content": "", "error": error})

    logger.info(
        "Batch {} complete: {} results ({} errors)",
        batch_id,
        len(results),
        sum(1 for r in results if r["error"]),
    )
    return results
