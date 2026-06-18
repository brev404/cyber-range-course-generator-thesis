"""Stateless service layer for the pipeline.

LLM abstraction (``llm_service``), vector DB + RAG (``vector_db_service``),
artifact writing (``artifact_writer``, ``atomic_write``), LangSmith
observability (``langsmith_service``), heartbeat, structural validation
(``structural_validator``), and quota management. ``__init__`` re-exports
``LLMService``, ``VectorDBService``, ``generate_response`` and quota helpers.

Inputs:  Prompts (strings), challenge paths, app_settings
Outputs: LLM responses (strings), vector search results, written artifacts
"""

from src.services.llm_service import (
    LLMCallBudgetExceeded,
    LLMService,
    LLMServiceError,
    QuotaExhaustedError,
    estimate_tokens,
    generate_response,
    get_available_providers,
    get_chat_model,
    reset_challenge_llm_budget,
)
from src.services.vector_db_service import VectorDBService, VectorDBServiceError

__all__ = [
    "LLMCallBudgetExceeded",
    "LLMService",
    "LLMServiceError",
    "QuotaExhaustedError",
    "estimate_tokens",
    "generate_response",
    "get_available_providers",
    "get_chat_model",
    "reset_challenge_llm_budget",
    "VectorDBService",
    "VectorDBServiceError",
]
