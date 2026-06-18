"""Language Model API integration and orchestration service.

This module provides a unified interface to multiple LLM providers:
- OpenAI (GPT-4, GPT-3.5 Turbo)
- Anthropic (Claude 3 Opus, Sonnet)
- Google (Gemini Pro)

The LLMService handles:
- API authentication and credential management
- Model selection and fallback strategies
- Token counting (heuristic) and optional cost estimation
- Response parsing and formatting
- Error handling and retry logic

Supported Models:
    OpenAI: gpt-4, gpt-4-turbo, gpt-3.5-turbo
    Anthropic: claude-3-opus-20240229, claude-3-sonnet-20240229, claude-3-haiku-20240307
    Google: gemini-2.5-flash (default), gemini-2.5-flash-lite; use LLM_DEFAULT_MODEL if your API has different names.

Usage:
    from src.services.llm_service import LLMService, LLMServiceError

    llm = LLMService()
    response = llm.generate_response("Explain XSS in one paragraph.", temperature=0.7)
"""

from __future__ import annotations

import time
from contextvars import ContextVar
from typing import Any, Optional

from loguru import logger

from src.config.settings import settings

# Per-task (async-safe) provider/model overrides set by the TUI before each run.
# ContextVar is safe for concurrent asyncio tasks — each task sees its own value.
_run_provider: ContextVar[str | None] = ContextVar("_run_provider", default=None)
_run_model: ContextVar[str | None] = ContextVar("_run_model", default=None)


class LLMServiceError(Exception):
    """Raised when LLM service operations fail (API errors, missing config, etc.)."""

    pass


class QuotaExhaustedError(LLMServiceError):
    """Raised when the claude --print subprocess exits with code 1 and no stderr.

    This is the signature of an exhausted Anthropic Pro/Max subscription usage
    window. Retrying immediately is futile — callers should halt and resume after
    the quota resets. Subclasses LLMServiceError so existing ``except LLMServiceError``
    catch blocks continue to work.
    """

    pass


class LLMAuthError(LLMServiceError):
    """Raised when a provider rejects the call for AUTHENTICATION reasons (401/403,
    revoked/expired token, not logged in) — NOT quota.

    Distinct from QuotaExhaustedError: an auth failure must NOT trigger the
    reset-window sleep (a codex 401/token_revoked must not be misclassified as
    quota, which would cause a false sleep against the claude reset window). Auth
    errors propagate immediately so the run halts and the user can re-auth.
    """

    pass


class LLMCallBudgetExceeded(LLMServiceError):
    """Raised when the per-challenge LLM call budget is exhausted.

    Subclasses LLMServiceError so existing ``except LLMServiceError`` catch blocks
    continue to work.  The cap prevents runaway refinement loops from consuming
    hundreds of LLM calls for a single challenge.
    """

    pass


# ---------------------------------------------------------------------------
# Per-challenge LLM-call budget
# ---------------------------------------------------------------------------
# A simple integer counter stored in a ContextVar.  Reset it at the start of
# each challenge via reset_challenge_llm_budget().  Each call to the two
# generate_response* functions increments it; if the count exceeds
# MAX_LLM_CALLS_PER_CHALLENGE a LLMCallBudgetExceeded is raised before the
# actual API call is made.
#
# The ContextVar approach is safe for both sync (thread-per-challenge) and
# async (coroutine-per-challenge) execution models because each context has
# its own copy of the counter.

_challenge_llm_call_count: ContextVar[int] = ContextVar(
    "_challenge_llm_call_count", default=0
)
# Sentinel: the challenge ID whose budget we are currently counting.  Used
# only for log messages; not required for correctness.
_challenge_llm_call_id: ContextVar[str] = ContextVar(
    "_challenge_llm_call_id", default=""
)


def reset_challenge_llm_budget(challenge_id: str = "") -> None:
    """Reset the per-challenge LLM call counter.

    Call this once at the start of processing each challenge so the budget is
    fresh.  Typically called from content_generation_agent before generating
    a course.

    Args:
        challenge_id: Human-readable label used in log messages only.
    """
    _challenge_llm_call_count.set(0)
    _challenge_llm_call_id.set(challenge_id)


def _increment_and_check_budget() -> None:
    """Increment the per-challenge call counter and raise if the cap is exceeded.

    Also emits a WARNING when the counter reaches 50 % of the cap so the
    operator gets an early signal before the budget is fully consumed.

    Raises:
        LLMCallBudgetExceeded: when the call count exceeds MAX_LLM_CALLS_PER_CHALLENGE.
    """
    raw = getattr(settings, "MAX_LLM_CALLS_PER_CHALLENGE", 20)
    # Guard: if settings is a MagicMock (unit tests), skip the cap check entirely
    # to avoid false positives.  A real Settings instance always returns an int.
    if not isinstance(raw, int):
        return
    cap: int = raw
    if cap <= 0:
        # Cap disabled (<=0 means unlimited).
        return
    count = _challenge_llm_call_count.get(0) + 1
    _challenge_llm_call_count.set(count)
    cid = _challenge_llm_call_id.get("")
    label = f" [{cid}]" if cid else ""

    # Early-warning at 50 % of cap
    halfway = cap // 2
    if count == halfway:
        logger.warning(
            "LLM budget{}: {} / {} calls used (50%% of cap={}); check for runaway loop.",
            label,
            count,
            cap,
            cap,
        )

    if count > cap:
        logger.error(
            "LLM budget{}: cap={} exceeded at call {}; raising LLMCallBudgetExceeded.",
            label,
            cap,
            count,
        )
        raise LLMCallBudgetExceeded(
            f"Per-challenge LLM call cap ({cap}) exceeded{label}. "
            "This indicates a runaway refinement loop. "
            "Increase MAX_LLM_CALLS_PER_CHALLENGE in settings or fix the loop."
        )


# Substring present in the ValueError raised by ClaudeCodeModel._generate when
# the claude --print subprocess exits with returncode=1 and no stderr output.
# Used to distinguish quota exhaustion from transient network / timeout errors.
_QUOTA_SIGNAL = "exited with code 1: (no stderr)"

# Genuine usage/rate-limit phrasings claude --print may emit (on stdout) when a
# subscription window is actually exhausted. An "unknown model" or HTTP 400
# invalid_request error contains NONE of these, so it is no longer misclassified
# as quota (which previously caused multi-hour false sleeps).
_QUOTA_KEYWORDS = (
    "rate limit",
    "rate_limit",
    "429",
    "too many requests",
    "usage limit",
    "usage_limit",
    "quota",
)

# Exponential backoff base in seconds (2**attempt → 1s, 2s, 4s).  Capped at 8s.
_BACKOFF_BASE = 2
_BACKOFF_CAP = 8


def _is_quota_signal(exc: BaseException) -> bool:
    """Return True if *exc* looks like a genuine claude --print quota exhaustion.

    Matches either the legacy *silent* exit-1 signature (no stderr AND no stdout)
    or an explicit usage/rate-limit keyword in the error text. A non-quota failure
    that carries an error message on stdout (e.g. unknown model, HTTP 400) is NOT
    treated as quota.
    """
    text = str(exc)
    if _QUOTA_SIGNAL in text:
        return True
    lowered = text.lower()
    return any(kw in lowered for kw in _QUOTA_KEYWORDS)


# Authentication-failure phrasings (401/403, revoked/expired token, not logged in).
# Deliberately disjoint from _QUOTA_KEYWORDS: a real 429/usage-limit error contains
# none of these, and these contain no quota vocabulary, so the two never collide.
# Checked BEFORE the quota signal so an auth failure never enters the reset-sleep path.
_AUTH_KEYWORDS = (
    "401",
    "403",
    "unauthorized",
    "forbidden",
    "token_revoked",
    "token revoked",
    "token has expired",
    "token expired",
    "not authenticated",
    "authentication failed",
    "please log in",
    "please run codex login",
    "invalid api key",
    "invalid_api_key",
)


def _is_auth_error(exc: BaseException) -> bool:
    """True if the error is an authentication failure (must propagate, never sleep)."""
    lowered = str(exc).lower()
    return any(kw in lowered for kw in _AUTH_KEYWORDS)


def _handle_quota_signal(exc: BaseException) -> bool | None:
    """Handle a quota-exhausted signal: sleep if reset is soon, else propagate.

    Called immediately when a quota signal is detected.

    Returns:
        True  if we slept and the caller should make ONE more attempt.
        None  if the reset is too far away and we should propagate QuotaExhaustedError.

    Raises:
        QuotaExhaustedError: always when reset is too far away (return None path
            is provided so the caller can raise the original exception cleanly).
    """
    from src.services.quota_helper import seconds_until_next_reset

    max_sleep_secs = getattr(settings, "QUOTA_SLEEP_MAX_MINUTES", 90) * 60
    secs = seconds_until_next_reset()
    if secs <= max_sleep_secs:
        sleep_duration = secs + 60  # 60s buffer past reset
        logger.warning(
            "Quota exhausted; sleeping {}s until next reset window (+60s buffer)",
            sleep_duration,
        )
        time.sleep(sleep_duration)
        return True
    else:
        logger.warning(
            "Quota exhausted; reset is {}s away (>{} max); propagating immediately.",
            secs,
            max_sleep_secs,
        )
        raise QuotaExhaustedError(
            f"subscription usage window exhausted; reset in {secs}s (>{max_sleep_secs}s max)"
        ) from exc


# Default model names per provider when LLM_DEFAULT_MODEL is not set.
# Google: gemini-2.5-flash (gemini-3-flash not yet available for generateContent in v1beta).
_DEFAULT_MODELS = {
    "openai": "gpt-4",
    "anthropic": "claude-3-sonnet-20240229",
    "google": "gemini-2.5-flash",
    "openrouter": "google/gemma-4-26b-a4b-it:free",
    "deepseek": "deepseek-chat",
}


def get_available_providers() -> list[str]:
    """Return list of provider names that have API keys configured.

    Returns:
        List of provider names: ["claude-code", "openai", "anthropic", "google"] (subset).
    """
    import shutil

    available: list[str] = []
    if shutil.which("claude"):  # Claude Code CLI — no API key needed
        available.append("claude-code")
    if shutil.which("codex"):  # Codex CLI — no API key needed
        available.append("codex")
    if settings.OPENROUTER_API_KEY:
        available.append("openrouter")
    if settings.DEEPSEEK_API_KEY:
        available.append("deepseek")
    if settings.OPENAI_API_KEY:
        available.append("openai")
    if settings.ANTHROPIC_API_KEY:
        available.append("anthropic")
    if settings.GOOGLE_API_KEY:
        available.append("google")
    return available


def get_chat_model(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    timeout: Optional[int] = None,
    **kwargs: Any,
) -> Any:
    """Build and return a LangChain chat model for the given provider.

    Uses the first available provider if provider is None. Prefers the
    requested provider if its API key is set.

    Args:
        provider: One of "openai", "anthropic", "google". If None, uses
            settings.LLM_DEFAULT_PROVIDER or first available.
        model: Model name (e.g. gpt-4, claude-3-sonnet-20240229, gemini-pro).
            If None, uses settings.LLM_DEFAULT_MODEL or provider default.
        temperature: Override default temperature.
        max_tokens: Override default max tokens.
        timeout: Override default timeout in seconds.
        **kwargs: Passed through to the underlying chat model constructor.

    Returns:
        A LangChain BaseChatModel instance (ChatOpenAI, ChatAnthropic, or
        ChatGoogleGenerativeAI).

    Raises:
        LLMServiceError: If no provider is available or provider/model build fails.
    """
    available = get_available_providers()
    if not available:
        raise LLMServiceError(
            "No LLM provider configured. Install Claude Code CLI, or set "
            "OPENAI_API_KEY / ANTHROPIC_API_KEY / GOOGLE_API_KEY in .env"
        )

    # Explicit args win over the per-run ContextVar (set by the TUI for the
    # generator). The ContextVar is an ambient *fallback* — e.g. the ranking
    # judge passes provider=RANKING_PROVIDER and must not be clobbered by the
    # generator's run provider (Bug A: refinement re-ranking judged on the
    # generator model because the ContextVar used to override explicit args).
    provider = (
        provider or _run_provider.get() or settings.LLM_DEFAULT_PROVIDER
    ).lower()
    model = model or _run_model.get() or settings.LLM_DEFAULT_MODEL

    if provider not in available:
        provider = available[0]
        logger.debug(
            "Requested provider '{}' not available, using first available: {}",
            provider,
            provider,
        )

    model_name = model or _DEFAULT_MODELS.get(provider, "")
    # DeepSeek: use the dedicated setting if no model was explicitly provided
    if provider == "deepseek" and model_name == _DEFAULT_MODELS.get("deepseek", ""):
        model_name = settings.DEEPSEEK_DEFAULT_MODEL or model_name
    temp = temperature if temperature is not None else settings.LLM_TEMPERATURE
    max_tok = max_tokens if max_tokens is not None else settings.LLM_MAX_TOKENS
    tout = timeout if timeout is not None else settings.LLM_TIMEOUT

    try:
        if provider == "claude-code":
            from src.services.claude_code_model import ClaudeCodeModel

            # LLM_DEFAULT_MODEL may hold a non-Claude model name (e.g. an
            # OpenRouter/Gemma name) intended for another provider. `claude --print`
            # only accepts Claude models; an incompatible name exits 1 with the error
            # on stdout, which used to be misread as quota exhaustion (hours-long
            # false sleep). Ignore it and fall back to the Claude Code CLI default.
            cc_model = model_name
            if (
                cc_model
                and cc_model != "claude-code"
                and not cc_model.lower().startswith("claude")
            ):
                logger.warning(
                    "claude-code provider: ignoring non-Claude model '{}' "
                    "(from LLM_DEFAULT_MODEL?); using Claude Code CLI default. "
                    "Pass an explicit claude-* model to override.",
                    cc_model,
                )
                cc_model = "claude-code"

            return ClaudeCodeModel(
                model_name=cc_model or "claude-code", timeout=tout or 300
            )

        if provider == "codex":
            from src.services.codex_model import CodexModel

            # LLM_DEFAULT_MODEL may hold a non-Codex model name; codex exec -m
            # only accepts Codex/GPT models. Ignore an incompatible name and fall
            # back to gpt-5.5 (mirrors the claude-code guard above).
            cx_model = model_name
            if cx_model and not (
                cx_model.lower().startswith("gpt") or "codex" in cx_model.lower()
            ):
                logger.warning(
                    "codex provider: ignoring non-Codex model '{}' "
                    "(from LLM_DEFAULT_MODEL?); using gpt-5.5. "
                    "Pass an explicit gpt-* model to override.",
                    cx_model,
                )
                cx_model = "gpt-5.5"
            # Codex exec is slow; use a dedicated (larger) default timeout unless an
            # explicit timeout= was passed.
            cx_timeout = timeout if timeout is not None else settings.CODEX_EXEC_TIMEOUT
            return CodexModel(model_name=cx_model or "gpt-5.5", timeout=cx_timeout)

        if provider == "openai":
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=model_name,
                temperature=temp,
                max_tokens=max_tok,
                request_timeout=tout,
                api_key=settings.OPENAI_API_KEY,
                **kwargs,
            )
        if provider == "anthropic":
            from langchain_anthropic import ChatAnthropic

            return ChatAnthropic(
                model=model_name,
                temperature=temp,
                max_tokens=max_tok,
                timeout=tout,
                api_key=settings.ANTHROPIC_API_KEY,
                **kwargs,
            )
        if provider == "google":
            from langchain_google_genai import ChatGoogleGenerativeAI

            return ChatGoogleGenerativeAI(
                model=model_name,
                temperature=temp,
                max_output_tokens=max_tok,
                api_key=settings.GOOGLE_API_KEY,
                **kwargs,
            )
        if provider == "openrouter":
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=model_name,
                temperature=temp,
                max_tokens=max_tok,
                request_timeout=tout,
                api_key=settings.OPENROUTER_API_KEY,
                base_url="https://openrouter.ai/api/v1",
                **kwargs,
            )
        if provider == "deepseek":
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=model_name,
                temperature=temp,
                max_tokens=max_tok,
                request_timeout=tout,
                api_key=settings.DEEPSEEK_API_KEY,
                base_url="https://api.deepseek.com/v1",
                **kwargs,
            )
    except Exception as e:
        raise LLMServiceError(f"Failed to build {provider} chat model: {e}") from e

    raise LLMServiceError(f"Unknown provider: {provider}")


def generate_response(
    prompt: str,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    timeout: Optional[int] = None,
    max_retries: int = 2,
    **kwargs: Any,
) -> str:
    """Call the LLM with a single prompt and return the response text.

    Args:
        prompt: The user message or system prompt to send.
        provider: Optional provider name. If None, uses default or first available.
        model: Optional model name. If None, uses default for the provider.
        temperature: Optional override for sampling temperature.
        max_tokens: Optional override for max response tokens.
        timeout: Optional override for API timeout in seconds.
        max_retries: Number of retries on transient failures (default 2).
        **kwargs: Passed to get_chat_model() and/or invoke().

    Returns:
        The assistant reply as a string.

    Raises:
        LLMServiceError: If no provider is configured or all retries fail.
        LLMCallBudgetExceeded: If the per-challenge call cap is exceeded.
    """
    _increment_and_check_budget()
    chat = get_chat_model(
        provider=provider,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        **kwargs,
    )

    from langchain_core.messages import HumanMessage

    message = HumanMessage(content=prompt)
    last_error: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        try:
            response = chat.invoke([message])
            if hasattr(response, "content") and response.content:
                return str(response.content).strip()
            raise LLMServiceError("Empty or invalid response from LLM")
        except LLMServiceError:
            raise
        except Exception as e:
            if _is_auth_error(e):
                raise LLMAuthError(
                    f"provider authentication failed (not quota — will not sleep): {e}"
                ) from e
            if _is_quota_signal(e):
                result = _handle_quota_signal(e)
                if result is not None:
                    # Slept and retried — make the single post-sleep attempt
                    try:
                        response = chat.invoke([message])
                        if hasattr(response, "content") and response.content:
                            return str(response.content).strip()
                        raise LLMServiceError("Empty or invalid response from LLM")
                    except LLMServiceError:
                        raise
                    except Exception as e2:
                        if _is_quota_signal(e2):
                            raise QuotaExhaustedError(
                                "subscription usage window exhausted after sleep; halt and resume after reset"
                            ) from e2
                        raise LLMServiceError(
                            f"LLM call failed after quota sleep: {e2}"
                        ) from e2
                raise  # reset too far away — propagate immediately
            last_error = e
            logger.warning("LLM call attempt {} failed: {}", attempt + 1, e)
            if attempt < max_retries:
                delay = min(_BACKOFF_BASE**attempt, _BACKOFF_CAP)
                time.sleep(delay)

    raise LLMServiceError(
        f"LLM call failed after {max_retries + 1} attempts"
    ) from last_error


def generate_response_with_system(
    system_prompt: str,
    user_prompt: str,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    timeout: Optional[int] = None,
    max_retries: int = 2,
    use_cache_control: bool = True,
    usage_out: Optional[dict] = None,
    **kwargs: Any,
) -> str:
    """Call LLM with system and user prompts as separate messages.

    When the resolved provider is Anthropic and use_cache_control=True, the
    system prompt is sent as a cached content block (cache_control: ephemeral).
    Subsequent identical system prompts within the same 5-minute TTL window pay
    ~10% of the input token cost.

    For non-Anthropic providers the system and user prompts are concatenated and
    the call falls through to generate_response() — no behaviour change.

    Args:
        system_prompt: System-level instructions (stable across calls → cached).
        user_prompt: Per-call user message (volatile → not cached).
        use_cache_control: Add cache_control to system block when provider is
            Anthropic. Default True.
        usage_out: Optional dict; populated in-place with
            cache_read_input_tokens and cache_creation_input_tokens from
            response.usage when provider is Anthropic. Pass a shared dict across
            multiple calls to accumulate per-run cache token counts.
        All other args: same as generate_response().

    Returns:
        Assistant reply as a string.

    Raises:
        LLMServiceError: If no provider is configured or all retries fail.
        LLMCallBudgetExceeded: If the per-challenge call cap is exceeded.
    """
    _increment_and_check_budget()
    # Explicit provider wins over the per-run ContextVar (see get_chat_model).
    resolved_provider = (
        provider or _run_provider.get() or settings.LLM_DEFAULT_PROVIDER or ""
    ).lower()

    if resolved_provider == "anthropic" and use_cache_control:
        from langchain_core.messages import HumanMessage, SystemMessage

        chat = get_chat_model(
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            **kwargs,
        )
        messages = [
            SystemMessage(
                content=[
                    {
                        "type": "text",
                        "text": system_prompt,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            ),
            HumanMessage(content=user_prompt),
        ]
        last_error: Optional[Exception] = None
        for attempt in range(max_retries + 1):
            try:
                response = chat.invoke(messages)
                if hasattr(response, "content") and response.content:
                    if usage_out is not None:
                        meta = getattr(response, "response_metadata", {}) or {}
                        usage = meta.get("usage", {}) or {}
                        usage_out["cache_read_input_tokens"] = usage_out.get(
                            "cache_read_input_tokens", 0
                        ) + int(usage.get("cache_read_input_tokens", 0))
                        usage_out["cache_creation_input_tokens"] = usage_out.get(
                            "cache_creation_input_tokens", 0
                        ) + int(usage.get("cache_creation_input_tokens", 0))
                    cache_read = (usage_out or {}).get("cache_read_input_tokens", 0)
                    cache_created = (usage_out or {}).get(
                        "cache_creation_input_tokens", 0
                    )
                    if cache_read or cache_created:
                        logger.debug(
                            "Prompt cache: read={} created={} tokens",
                            cache_read,
                            cache_created,
                        )
                    return str(response.content).strip()
                raise LLMServiceError("Empty or invalid response from LLM")
            except LLMServiceError:
                raise
            except Exception as e:
                if _is_auth_error(e):
                    raise LLMAuthError(
                        f"provider authentication failed (not quota — will not sleep): {e}"
                    ) from e
                if _is_quota_signal(e):
                    result = _handle_quota_signal(e)
                    if result is not None:
                        # Slept — make ONE post-sleep attempt
                        try:
                            response = chat.invoke(messages)
                            if hasattr(response, "content") and response.content:
                                if usage_out is not None:
                                    meta = (
                                        getattr(response, "response_metadata", {}) or {}
                                    )
                                    usage = meta.get("usage", {}) or {}
                                    usage_out["cache_read_input_tokens"] = (
                                        usage_out.get("cache_read_input_tokens", 0)
                                        + int(usage.get("cache_read_input_tokens", 0))
                                    )
                                    usage_out["cache_creation_input_tokens"] = (
                                        usage_out.get("cache_creation_input_tokens", 0)
                                        + int(
                                            usage.get("cache_creation_input_tokens", 0)
                                        )
                                    )
                                return str(response.content).strip()
                            raise LLMServiceError("Empty or invalid response from LLM")
                        except LLMServiceError:
                            raise
                        except Exception as e2:
                            if _is_quota_signal(e2):
                                raise QuotaExhaustedError(
                                    "subscription usage window exhausted after sleep; halt and resume after reset"
                                ) from e2
                            raise LLMServiceError(
                                f"LLM call failed after quota sleep: {e2}"
                            ) from e2
                    raise  # reset too far away — _handle_quota_signal already raised
                last_error = e
                logger.warning("LLM call attempt {} failed: {}", attempt + 1, e)
                if attempt < max_retries:
                    delay = min(_BACKOFF_BASE**attempt, _BACKOFF_CAP)
                    time.sleep(delay)
        raise LLMServiceError(
            f"LLM call failed after {max_retries + 1} attempts"
        ) from last_error

    # Non-Anthropic fallback: concatenate and use existing path.
    return generate_response(
        system_prompt + "\n\n---\n\n" + user_prompt,
        provider=provider,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        max_retries=max_retries,
        **kwargs,
    )


def estimate_tokens(text: str, provider: str = "openai") -> int:
    """Estimate the number of tokens in a text string.

    Uses a simple heuristic (~4 characters per token for English). For
    production token counting with OpenAI, consider using tiktoken.

    Args:
        text: Input text to count.
        provider: Provider name (currently only affects logging; heuristic is generic).

    Returns:
        Estimated token count.
    """
    if not text:
        return 0
    # Conservative heuristic: ~4 chars per token for English/code
    return max(1, len(text) // 4)


class LLMService:
    """Unified LLM service facade for agents and pipeline steps.

    Provides generate_response, get_chat_model, and estimate_tokens with
    optional instance-level default provider/model.
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        """Initialize the service with optional default provider and model.

        Args:
            provider: Default provider (openai, anthropic, google).
            model: Default model name. If None, uses settings or provider default.
        """
        self._provider = provider
        self._model = model

    def get_chat_model(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Return a LangChain chat model. Uses instance defaults if arguments are None."""
        return get_chat_model(
            provider=provider or self._provider,
            model=model or self._model,
            **kwargs,
        )

    def generate_response(
        self,
        prompt: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Generate a single response. Uses instance defaults if provider/model are None."""
        return generate_response(
            prompt,
            provider=provider or self._provider,
            model=model or self._model,
            **kwargs,
        )

    @staticmethod
    def estimate_tokens(text: str, provider: str = "openai") -> int:
        """Estimate token count for the given text."""
        return estimate_tokens(text, provider=provider)

    @staticmethod
    def available_providers() -> list[str]:
        """Return list of configured provider names."""
        return get_available_providers()
