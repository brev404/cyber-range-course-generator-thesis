"""Application configuration and settings management.

This module provides centralized configuration for the content creation pipeline.
Settings are loaded from environment variables and .env files using Pydantic,
ensuring type-safe configuration with automatic validation.

The Settings class manages:
- File paths for data and project directories
- External data source locations (WSL Windows paths)
- LLM API credentials (OpenAI, Anthropic, Google)
- LangSmith tracing configuration for agent monitoring
- Logging configuration

Usage:
    from src.config.settings import settings

    # Access project paths
    input_dir = settings.INPUT_DIR
    output_dir = settings.OUTPUT_DIR

    # Ensure directories exist
    settings.create_directories()

    # Check if LLM API is configured
    if settings.OPENAI_API_KEY:
        # Use OpenAI API
        pass

Environment Variables (.env file):
    BASE_DIR: Project root directory (auto-detected from file location)
    DATA_DIR: Data directory (defaults to {BASE_DIR}/data)
    INPUT_DIR: Input data directory (defaults to {DATA_DIR}/input)
    PROCESSED_DIR: Processed data directory (defaults to {DATA_DIR}/processed)
    KNOWLEDGE_BASE_DIR: Knowledge base files (defaults to {DATA_DIR}/knowledge_base)
    OUTPUT_DIR: Output directory (defaults to {DATA_DIR}/outputs)
    RAW_CHALLENGES_SOURCE: path to challenge files
    OFFICIAL_DOCS_SOURCE: path to official documentation PDFs
    OPENAI_API_KEY: OpenAI API key for GPT models
    ANTHROPIC_API_KEY: Anthropic API key for Claude models
    GOOGLE_API_KEY: Google API key for Gemini models
    LANGCHAIN_TRACING_V2: Enable LangSmith tracing (true/false)
    LANGCHAIN_ENDPOINT: LangSmith API endpoint URL
    LANGCHAIN_API_KEY: LangSmith API authentication key
    LANGCHAIN_PROJECT: Project name for organizing traces
    LOG_LEVEL: Logging verbosity (DEBUG, INFO, WARNING, ERROR, CRITICAL)

Example .env file:
    OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxx
    LOG_LEVEL=DEBUG
    LANGCHAIN_TRACING_V2=true
    LANGCHAIN_API_KEY=ls_xxxxxxxxxxxxx
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Literal, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable and .env file support.

    All paths default to sensible values relative to project structure, but can
    be overridden via environment variables.

    Attributes:
        # Project Directory Structure
        BASE_DIR (Path): Project root directory
            Auto-detected as parent of src/config/
            Example: /home/user/context_creator

        DATA_DIR (Path): Main data directory for all project data
            Default: {BASE_DIR}/data

        INPUT_DIR (Path): Directory for raw input files before processing
            Default: {DATA_DIR}/input
            Used by: analyze_challenges.py as source of challenge archives

        PROCESSED_DIR (Path): Directory for processed/organized challenge files
            Default: {DATA_DIR}/processed
            Used by: organize_challenges.py as output, validate_challenges.py as input

        KNOWLEDGE_BASE_DIR (Path): Directory for stored knowledge base documents
            Default: {DATA_DIR}/knowledge_base
            Used by: Vector DB service for document embeddings

        OUTPUT_DIR (Path): Directory for final pipeline outputs
            Default: {DATA_DIR}/outputs
            Used by: Content Generation Agent, Validation Agent

        # External Data Sources
        RAW_CHALLENGES_SOURCE (Path): Path to original challenge files
            Default: data/raw_challenges
            Contains: Original challenge archives

        OFFICIAL_DOCS_SOURCE (Path): Path to PDF documentation
            Default: data/official_docs
            Contains: Official cybersecurity reference documentation (PDF files)

        # LLM API Authentication
        OPENAI_API_KEY (Optional[str]): API key for OpenAI services
            Format: "sk-proj-xxxxx"
            Used for: GPT-4, GPT-3.5 Turbo language model calls
            Default: None (feature disabled if not provided)

        ANTHROPIC_API_KEY (Optional[str]): API key for Anthropic Claude
            Format: "sk-ant-xxxxx"
            Used for: Claude 3 Opus/Sonnet model calls
            Default: None (feature disabled if not provided)

        GOOGLE_API_KEY (Optional[str]): API key for Google Gemini
            Format: "AIza..."
            Used for: Gemini Pro language model calls
            Default: None (feature disabled if not provided)

        # LangSmith Tracing Configuration (Debug/Observability)
        LANGCHAIN_TRACING_V2 (bool): Enable LangSmith tracing
            Traces all LLM calls and agent workflows for debugging
            Default: False (tracing disabled for performance)

        LANGCHAIN_ENDPOINT (str): LangSmith API endpoint
            Default: "https://api.smith.langchain.com"

        LANGCHAIN_API_KEY (Optional[str]): Authentication for LangSmith
            Obtained from LangSmith dashboard
            Required only if LANGCHAIN_TRACING_V2 is True

        LANGCHAIN_PROJECT (str): Project name in LangSmith dashboard
            Used to organize and group traces
            Default: "cyber-range-validator"

        # Logging Configuration
        LOG_LEVEL (str): Minimum logging verbosity level
            Valid values: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
            Default: "INFO"
            Example values:
                - "DEBUG": Very detailed logs, includes variable states
                - "INFO": Normal operation, key events and milestones
                - "WARNING": Only potential issues and deprecations
                - "ERROR": Only errors that affect functionality

    Example:
        >>> from src.config.settings import settings
        >>>
        >>> # Ensure all data directories exist
        >>> settings.create_directories()
        >>>
        >>> # Use paths in your code
        >>> input_files = list(settings.INPUT_DIR.glob("*.zip"))
        >>> output_file = settings.OUTPUT_DIR / "results.json"
        >>>
        >>> # Check API availability
        >>> if settings.OPENAI_API_KEY:
        ...     print("OpenAI API is configured")
        ... else:
        ...     print("Warning: OpenAI API not configured")
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Project Paths
    BASE_DIR: Path = Path(__file__).parent.parent.parent
    """Project root directory, auto-detected from this file's location."""

    DATA_DIR: Path = BASE_DIR / "data"
    """Root data directory for all project data."""

    INPUT_DIR: Path = DATA_DIR / "input"
    """Raw input files before processing."""

    PROCESSED_DIR: Path = DATA_DIR / "processed"
    """Processed and organized challenge files."""

    KNOWLEDGE_BASE_DIR: Path = DATA_DIR / "knowledge_base"
    """Knowledge base documents and embeddings."""

    KB_METADATA_DIR: Path = DATA_DIR / "kb_metadata"
    """Structured metadata KB: contests.json, categories.json, challenges_index.json."""

    OUTPUT_DIR: Path = DATA_DIR / "outputs"
    """Final pipeline outputs and results."""

    # External Data Sources
    RAW_CHALLENGES_SOURCE: Path = Path("data/raw_challenges")
    """Path to original challenge archives."""

    OFFICIAL_DOCS_SOURCE: Path = Path("data/official_docs")
    """Path to official PDF cybersecurity documentation."""

    # LLM API Keys
    OPENAI_API_KEY: Optional[str] = None
    """API key for OpenAI GPT models."""

    ANTHROPIC_API_KEY: Optional[str] = None
    """API key for Anthropic Claude models."""

    GOOGLE_API_KEY: Optional[str] = None
    """API key for Google Gemini models."""

    OPENROUTER_API_KEY: Optional[str] = None
    """API key for OpenRouter (free + paid open-source models via OpenAI-compatible API)."""

    DEEPSEEK_API_KEY: Optional[str] = None
    """API key for DeepSeek (OpenAI-compatible API)."""

    # LLM Service Defaults (used by src/services/llm_service.py)
    LLM_DEFAULT_PROVIDER: str = "openai"
    """Default provider: openai, anthropic, google, openrouter, or deepseek. First available key is used if this provider is not configured."""

    DEEPSEEK_DEFAULT_MODEL: str = "deepseek-chat"
    """Default model for DeepSeek provider (deepseek-chat or deepseek-reasoner)."""

    LLM_DEFAULT_MODEL: Optional[str] = None
    """Default model name per provider (e.g. gpt-4, claude-3-sonnet-20240229, gemini-pro). None uses provider default."""

    LLM_TEMPERATURE: float = 0.7
    """Response randomness 0.0–2.0. Default 0.7."""

    LLM_MAX_TOKENS: int = 2000
    """Max tokens per response. Default 2000."""

    LLM_TIMEOUT: int = 600
    """API call timeout in seconds. Default 600 (10 min), at parity with CODEX_EXEC_TIMEOUT.
    The claude --print path forwards neither max_tokens nor temperature, so course
    generation output length is uncapped/stochastic and can exceed a 300 s cap on a long
    draw."""

    LLM_MAX_CONCURRENT: int = 1
    """Max concurrent LLM calls for content generation and ranking. 1 = sequential. Set to 4–8 for high-RPM APIs (e.g. Gemini 2.5 Flash Lite 4k RPM)."""

    RANKING_MODEL: Optional[str] = None
    """Override model for the ranking LLM judge. Falls back to LLM_DEFAULT_MODEL when unset. Set via env or --ranking-model CLI flag."""

    RANKING_PROVIDER: Optional[str] = None
    """Override provider for the ranking LLM judge (openai, anthropic, google). Falls back to LLM_DEFAULT_PROVIDER when unset. Set via env or --ranking-provider CLI flag."""

    # Vector DB / RAG (used by src/services/vector_db_service.py)
    CHROMA_PERSIST_DIR: Optional[Path] = None
    """Directory for Chroma persistence. Default: {DATA_DIR}/chroma_db. None = in-memory only."""

    CHROMA_COLLECTION_NAME: str = "knowledge_base"
    """Chroma collection name for RAG documents."""

    EMBEDDING_PROVIDER: Literal["auto", "openai", "local"] = "auto"
    """RAG embeddings: auto (OpenAI if key set, else local), openai, or local (free, no API key)."""

    EMBEDDING_MODEL: str = "text-embedding-3-small"
    """OpenAI embedding model when EMBEDDING_PROVIDER is openai (or auto with key)."""

    LOCAL_EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    """Local embedding model when EMBEDDING_PROVIDER is local (or auto without OpenAI key). Free, runs on CPU."""

    RAG_CHUNK_SIZE: int = 1000
    """Character chunk size when splitting knowledge base documents. Default 1000."""

    RAG_CHUNK_OVERLAP: int = 200
    """Overlap between chunks. Default 200."""

    # Pipeline output
    WRITE_COURSES_TO_DISK: bool = True
    """If True, content generation writes each generated course to cyberedu/write-up/course.md (and solve_generated.py) under PROCESSED_DIR. Author's writeup.md is never overwritten."""

    EXPERIMENT_ID: str = ""
    """When non-empty, courses are written to OUTPUT_DIR/EXPERIMENT_ID/<challenge_id>/course.md instead of the default challenge directory path. Set via --exp-id CLI flag."""

    PROMPT_VARIANT: str = ""
    """When non-empty, selects a named prompt variant from src/prompts/variants/. Empty string = baseline. Set via --prompt-variant CLI flag or PROMPT_VARIANT env var."""

    RANKING_PASS_THRESHOLD: float = 9.0
    """Overall ranking score threshold. If any report's overall_score < this, auto-refine (content gen again) until all >= threshold or MAX_REFINEMENT_ROUNDS reached; then HITL."""

    RANKING_TECHNICAL_THRESHOLD: float = 9.0
    """Per-dimension pass threshold for the technical persona score. Default matches RANKING_PASS_THRESHOLD for backward compatibility. A challenge passes only if technical_review.score >= this AND pedagogical_review.score >= RANKING_PEDAGOGICAL_THRESHOLD."""

    RANKING_PEDAGOGICAL_THRESHOLD: float = 9.0
    """Per-dimension pass threshold for the pedagogical persona score. Default matches RANKING_PASS_THRESHOLD for backward compatibility."""

    SCORING_TECHNICAL_WEIGHT: float = 0.5
    """Weight for technical score when SCORING_POLICY='weighted'. Must sum to 1.0 with SCORING_PEDAGOGICAL_WEIGHT. Default 0.5 = equal weights (same as mean)."""

    SCORING_PEDAGOGICAL_WEIGHT: float = 0.5
    """Weight for pedagogical score when SCORING_POLICY='weighted'. Must sum to 1.0 with SCORING_TECHNICAL_WEIGHT. Default 0.5 = equal weights (same as mean)."""

    SCORING_POLICY: Literal["mean", "min", "weighted"] = "mean"
    """Overall score computation policy: mean (current default), min (bottleneck — either-bad-fails), weighted (uses SCORING_TECHNICAL_WEIGHT + SCORING_PEDAGOGICAL_WEIGHT). Default 'mean' preserves all past-data comparability."""

    MAX_REFINEMENT_ROUNDS: int = 5
    """Max automatic refinement loops (content_generation → mapping → ranking) before routing to HITL when score < RANKING_PASS_THRESHOLD."""

    RANKING_MAX_TOKENS: int = 8192
    """Max output tokens for ranking LLM (technical + pedagogical JSON). Increase if justifications are often truncated."""

    CONTENT_GENERATION_MAX_TOKENS: int = 18000
    """Max output tokens for generated course (course.md).
    Raised in v2 after internal evaluation showed judges reporting truncated mid-function
    and missing Extra Resources/Conclusion sections; largest course.md was ~22kB ≈ 11k tokens
    with the 6k cap clearly acting as a ceiling.
    Raised again in v4.1.2 after internal evaluation showed every judge still citing truncation
    in Step 8 / Section 11; F1+v3+v4 prompt growth (~7k chars of system + ~10k of user prompt
    incl. solver context) leaves less room for the course body itself.
    claude-sonnet-4-6 supports up to 64k output.
    Can be overridden per challenge via state.content_max_tokens_override."""

    CONTENT_GENERATION_SOLVE_MAX_TOKENS: int = 6000
    """Max output tokens for generated solve script (solve_generated.py).
    Raised in v2 after internal evaluation showed typical CTF solver is 100–300 lines = 2–4k tokens;
    the lower cap caused truncated scripts; 6k gives generous headroom for complex multi-step exploits."""

    # LangSmith Configuration
    LANGCHAIN_TRACING_V2: bool = False
    """Enable LangSmith tracing for debugging agent workflows."""

    LANGCHAIN_ENDPOINT: str = "https://api.smith.langchain.com"
    """LangSmith API endpoint URL."""

    LANGCHAIN_API_KEY: Optional[str] = None
    """API key for LangSmith authentication."""

    LANGCHAIN_PROJECT: str = "cyber-range-validator"
    """Project name for organizing LangSmith traces."""

    # Granular grading: optional per-dimension weights for weighted aggregation
    DIMENSION_WEIGHTS: Optional[Dict[str, float]] = None
    """Per-dimension weights for aggregated dimension_scores in RankingReport. Keys are dimension
    names (e.g. correctness, sections_structure); values are positive floats. None = equal weights.
    Example: {"correctness": 2.0, "completeness": 1.5, "sections_structure": 2.0}"""

    # Terminology validation
    TERMINOLOGY_CHECK_MODE: Literal["block", "warn", "annotate", "off"] = "warn"
    """Terminology check behavior: block (fail validation), warn (add issues), annotate (log only), off (skip)."""

    # Consistency pass
    CONSISTENCY_TERM_THRESHOLD: float = 0.5
    """Fraction of courses a term must appear in to be considered 'expected'. Terms present in
    more than this fraction are expected in all courses; absence is flagged as a deviation."""

    # Refinement strategy
    MAX_REFINEMENT_STRATEGY: Literal["hitl", "accept_best", "soft_accept"] = "hitl"
    """Strategy when MAX_REFINEMENT_ROUNDS is reached: hitl (route to HITL), accept_best (accept current scores and route to END), soft_accept (accept if best score >= SOFT_ACCEPT_THRESHOLD, else HITL)."""

    SOFT_ACCEPT_THRESHOLD: float = 7.0
    """Minimum overall_score to auto-accept when MAX_REFINEMENT_STRATEGY is soft_accept."""

    # Language configuration
    OUTPUT_LANGUAGE: Literal["en", "ro"] = "en"
    """Output language for generated course content. 'en' = English, 'ro' = Romanian."""

    # RAG control
    RAG_ENABLED: bool = True
    """When False, skip all RAG retrieval in content_generation_agent (--no-rag ablation flag).
    Default True. Set False via --no-rag CLI flag for ablation experiments."""

    # Heartbeat status file
    HEARTBEAT_ENABLED: bool = True
    """When True, the pipeline spawns a daemon thread that writes
    output/jobs/_run_<run_id>.heartbeat.json every 30s for external monitors.
    Set False to disable (e.g. in unit tests that mock settings)."""

    # Quota sleep-until-reset (C10)
    QUOTA_RESET_ANCHOR: str = "00:30"
    """HH:MM of the daily Anthropic Pro quota reset anchor (local server time).
    Mirrors quota_windows.reset_anchor in prompts/jobs/_shared/experiment_matrix.yaml."""

    QUOTA_CYCLE_HOURS: int = 5
    """Length of each Pro quota cycle in hours. Mirrors quota_windows.cycle_hours."""

    QUOTA_SLEEP_MAX_MINUTES: int = 360
    """Maximum minutes to sleep waiting for quota reset.
    If reset is farther away than this, QuotaExhaustedError is raised immediately.
    Set to 360 to cover the full QUOTA_CYCLE_HOURS so back-to-back chained runs always wait."""

    # Per-challenge LLM call budget (belt-and-suspenders against runaway loops)
    MAX_LLM_CALLS_PER_CHALLENGE: int = 30
    """Hard cap on LLM calls attributed to a single challenge.
    Budget breakdown (v4 architecture):
      - Per round: 1 solver-gen + 1 course-gen + 2 judges (tech + ped) = 4 calls
      - MAX_REFINEMENT_ROUNDS = 5 -> 20 base calls
      - STRUCTURAL_VALIDATOR_MAX_RETRIES = 2: up to 2 extra (solve+gen) per round = <=10 extra
      - Total worst-case: ~30; cap set to 30 to catch genuine runaway loops above this.
    When exceeded, LLMCallBudgetExceeded is raised (subclass of LLMServiceError).
    Set to 0 to disable the cap."""

    # Performance optimisations
    RANKING_USE_BATCH_API: bool = False
    """Submit all ranking LLM calls as a single Anthropic Batch API job (50% cost reduction,
    ~1h latency). Only effective when RANKING_PROVIDER=anthropic and ANTHROPIC_API_KEY is set.
    Default False — opt in per experiment run."""

    # Feedback loop / RL foundation
    FEEDBACK_JUDGE_MODEL: str = "claude-haiku-4-5-20251001"
    """Judge model that produces valid reward signals. Reward is only True when the actual
    ranking judge matches this model — prevents self-judge noise from polluting the store."""

    FEEDBACK_REWARD_THRESHOLD: float = 7.0
    """Mean technical AND pedagogical score must both meet this threshold for reward=True."""

    FEEDBACK_ENABLED: bool = False
    """Enable feedback loop: when True, compute_reward results are appended to run_history.jsonl.
    Default False — opt-in for experiments only."""

    # Logging
    LOG_LEVEL: str = "INFO"
    """Minimum logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)."""

    SAVE_RUN_CONFIG: bool = True
    """If True, export run config to data/outputs/run_config_latest.json at pipeline start (for reproducibility)."""

    METHODOLOGY_VERSION: str = "v4.2"
    """Framework methodology version. v1 = pilot (serial judges, 6k token cap,
    solve_generated.py missing from exp dir). v2 = bumped token limits,
    solve script in exp dir, atomic writes, parallel judges. v3 = anti-pattern MUST rules (F1) +
    solver context 8000 + edit-mode (D1) + pre-ranking structural validator (F8).
    v4 = unified two-call architecture: solver generated FIRST, then course explains the solver,
    Section 9 is auto-assembled (not LLM-written) so course narrative and solver cannot drift apart.
    Targets C1 (solver truncation) and C7 (duplicate solver blocks). Bumped 2026-05-30."""

    V4_ARCHITECTURE_ENABLED: bool = True
    """When True, content generation uses the v4 unified architecture: solver-gen runs first,
    course-gen receives the solver as authoritative input and explains it in Sections 6-8,
    Section 9 is auto-assembled from the solver (not LLM-written). When False, falls back to
    the v3 behaviour (course-first; course writes Section 9 itself). Default True. Toggle to
    False via env or CLI for ablation experiments.
    Does NOT change PROMPT_VARIANT or RANKING_PASS_THRESHOLD."""

    RANKING_PEDAGOGICAL_OMIT_CODE: bool = True
    """J3: when True, the pedagogical judge receives the course
    with fenced code blocks stripped and without the separate solver-script section. The
    pedagogical persona grades structure/language/progression, not code — so this shrinks
    its haiku input (faster calls) without changing the technical judge. Default True (the
    approved behavior); set False to reproduce pre-2026-06 pedagogical scores exactly."""

    PROSE_V2: bool = False
    """Prose overhaul (G4). When True, the course system prompt is post-processed to fix the
    audit's prose defects: Section 6 renamed "Reproducibility (Step 0)" -> "Setup" (D4), the
    "do not assume they have source/writeup/flag" frame line dropped, and an override block
    appended that bans the formulaic per-step template (D3), em-dashes/AI-tells (D2), and
    printing flag values in prose (D5). Separate flag from MANIFEST_GROUNDED_GEN (G2) so the
    leak-fix and prose-fix verify independently. Default False = reproducible baseline prose."""

    MANIFEST_GROUNDED_GEN: bool = False
    """Manifest-grounded course generation (G2). When True and the challenge has a
    ``manifest.json`` (scripts/build_manifests.py), course-gen is grounded ONLY in the
    student materials (``student_prompt`` + ``student_files``) and the author writeup/solver
    are NOT injected into the course prompt — the data-layer fix for the context leak
    (KittyOS ``data.bin`` internals, frame leakage). Solver-gen still reads the author solver
    (correctness). Default False so existing baselines stay reproducible; the legacy path is
    used for any challenge without a manifest regardless of this flag."""

    REDACT_SOURCE_CONTEXT_FLAGS: bool = False
    """Fair-generator mode. When True, CTF flag-format tokens (``PREFIX{...}``)
    are stripped from the author writeup + author solver before they are injected into the
    generation prompt, and the solver-generation prompt gains an explicit "never hardcode
    the flag; implement the technique" clause. Removes the risk that the generator
    hardcodes a visible flag instead of implementing the technique. Default False;
    enable via ``--redact-source-flags``."""

    SOLVER_SELF_IMPROVE_ENABLED: bool = False
    """Multi-pass solver self-improvement loop (plan -> generate -> hybrid-verify -> revise),
    gated. Default False so baselines are unaffected; enable via ``--self-improve``. See
    docs/superpowers/specs/2026-06-03-solver-self-improve-design.md and decision 004."""

    MAX_SOLVER_SELF_IMPROVE_ROUNDS: int = 3
    """Max verify/revise rounds in the solver self-improvement loop before accepting best."""

    SOLVER_CRITIC_PROVIDER: str = "claude-code"
    """Provider for the solver planner + critic calls (plan_solution_stages,
    llm_dry_run_critique). Pinned explicitly (like RANKING_PROVIDER) so a codex
    generation run does NOT pay codex cost for planning/critique."""

    SOLVER_CRITIC_MODEL: str = "claude-haiku-4-5"
    """Model for the solver planner/critic. Haiku is sufficient for structural
    critique and JSON stage decomposition; cheap and no codex quota draw."""

    CODEX_EXEC_TIMEOUT: int = 600
    """Subprocess timeout (seconds) for `codex exec`. Raised from 300 after
    observing repeated timeouts in the self-improve arms. An
    explicit get_chat_model(timeout=...) still overrides this default."""

    STRUCTURAL_VALIDATOR_ENABLED: bool = True
    """If True, validate generated course + solver structure before invoking the ranking node.
    On structural failure (missing section, syntax error, no Extra Resources refs, truncation
    marker), trigger an immediate regen with structured feedback. Saves wasted ranking calls."""

    STRUCTURAL_VALIDATOR_MAX_RETRIES: int = 2
    """Number of structural-validator-driven retries per challenge BEFORE proceeding to the ranking
    node anyway. Separate from MAX_REFINEMENT_ROUNDS (those are judge-driven post-ranking retries)."""

    F3_CATEGORY_GUIDANCE_ENABLED: bool = False
    """F3 (v4.2.1) — inject per-category guidance blocks into the course-gen user prompt.
    Each block lists category-specific requirements (tools, formats), probable ATT&CK/CWE/OWASP IDs,
    and common failure modes. The block lives in the user prompt (not the system prompt) so it does
    NOT invalidate the prompt cache for the stable system rules. Default True. Toggle False via env
    or CLI to run the isolating ablation EXP (F3 contribution = with/without category guidance, all
    other levers equal). Does NOT bump METHODOLOGY_VERSION — this is a content-quality lever, the
    flag serialised in reproducibility.json is the disambiguator."""

    def export_for_reproducibility(
        self,
        *,
        run_started_at: Optional[str] = None,
        cli_overrides: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Export settings that affect generation quality for reproducibility.

        Secrets (API keys) are redacted: "set" if present, "unset" if not.
        Paths are converted to strings for JSON serialization.

        Args:
            run_started_at: ISO timestamp of run start (default: now).
            cli_overrides: Optional dict of CLI flags (e.g. skip_ranking, content_subset).

        Returns:
            Dict suitable for logging and saving to run_config_latest.json.
        """
        ts = run_started_at or datetime.now(timezone.utc).isoformat()
        out: dict[str, Any] = {
            "run_started_at": ts,
            "settings": {
                "LLM_TEMPERATURE": self.LLM_TEMPERATURE,
                "LLM_DEFAULT_PROVIDER": self.LLM_DEFAULT_PROVIDER,
                "LLM_DEFAULT_MODEL": self.LLM_DEFAULT_MODEL,
                "RANKING_PROVIDER": self.RANKING_PROVIDER,
                "RANKING_MODEL": self.RANKING_MODEL,
                "LLM_MAX_TOKENS": self.LLM_MAX_TOKENS,
                "LLM_MAX_CONCURRENT": self.LLM_MAX_CONCURRENT,
                "RAG_CHUNK_SIZE": self.RAG_CHUNK_SIZE,
                "RAG_CHUNK_OVERLAP": self.RAG_CHUNK_OVERLAP,
                "EMBEDDING_PROVIDER": self.EMBEDDING_PROVIDER,
                "EMBEDDING_MODEL": self.EMBEDDING_MODEL,
                "LOCAL_EMBEDDING_MODEL": self.LOCAL_EMBEDDING_MODEL,
                "CONTENT_GENERATION_MAX_TOKENS": self.CONTENT_GENERATION_MAX_TOKENS,
                "CONTENT_GENERATION_SOLVE_MAX_TOKENS": self.CONTENT_GENERATION_SOLVE_MAX_TOKENS,
                "RANKING_MAX_TOKENS": self.RANKING_MAX_TOKENS,
                "METHODOLOGY_VERSION": self.METHODOLOGY_VERSION,
                "V4_ARCHITECTURE_ENABLED": self.V4_ARCHITECTURE_ENABLED,
                "F3_CATEGORY_GUIDANCE_ENABLED": self.F3_CATEGORY_GUIDANCE_ENABLED,
                "RANKING_PASS_THRESHOLD": self.RANKING_PASS_THRESHOLD,
                "MAX_REFINEMENT_ROUNDS": self.MAX_REFINEMENT_ROUNDS,
                "RAG_ENABLED": self.RAG_ENABLED,
                "WRITE_COURSES_TO_DISK": self.WRITE_COURSES_TO_DISK,
                "PROCESSED_DIR": str(self.PROCESSED_DIR),
                "KNOWLEDGE_BASE_DIR": str(self.KNOWLEDGE_BASE_DIR),
                "CHROMA_PERSIST_DIR": (
                    str(self.CHROMA_PERSIST_DIR) if self.CHROMA_PERSIST_DIR else None
                ),
                "LANGCHAIN_TRACING_V2": self.LANGCHAIN_TRACING_V2,
                "LOG_LEVEL": self.LOG_LEVEL,
                "EXPERIMENT_ID": self.EXPERIMENT_ID,
                "OUTPUT_DIR": str(self.OUTPUT_DIR),
            },
            "secrets_redacted": {
                "OPENAI_API_KEY": "set" if self.OPENAI_API_KEY else "unset",
                "ANTHROPIC_API_KEY": "set" if self.ANTHROPIC_API_KEY else "unset",
                "GOOGLE_API_KEY": "set" if self.GOOGLE_API_KEY else "unset",
                "OPENROUTER_API_KEY": "set" if self.OPENROUTER_API_KEY else "unset",
                "DEEPSEEK_API_KEY": "set" if self.DEEPSEEK_API_KEY else "unset",
                "LANGCHAIN_API_KEY": "set" if self.LANGCHAIN_API_KEY else "unset",
            },
        }
        if cli_overrides:
            out["cli_overrides"] = cli_overrides
        return out

    def save_run_config(
        self,
        *,
        run_started_at: Optional[str] = None,
        cli_overrides: Optional[dict[str, Any]] = None,
    ) -> Optional[Path]:
        """Export run config to data/outputs/run_config_latest.json (for reproducibility).

        Called at pipeline start. Overwrites previous file. Secrets are redacted.

        Returns:
            Path to saved file, or None if SAVE_RUN_CONFIG is False or write failed.
        """
        if not self.SAVE_RUN_CONFIG:
            return None
        data = self.export_for_reproducibility(
            run_started_at=run_started_at,
            cli_overrides=cli_overrides,
        )
        out_path = Path(self.OUTPUT_DIR) / "run_config_latest.json"
        try:
            self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            return out_path
        except OSError:
            return None

    def create_directories(self) -> None:
        """Ensure all required data directories exist, creating them if necessary.

        Creates the following directories if they don't exist:
        - INPUT_DIR: For raw challenge files
        - PROCESSED_DIR: For organized challenges
        - KNOWLEDGE_BASE_DIR: For knowledge base documents
        - OUTPUT_DIR: For final outputs
        - OFFICIAL_DOCS_SOURCE: For official documentation (on WSL)

        This method is idempotent and safe to call multiple times.
        Should be called during application initialization.

        Raises:
            PermissionError: If lacking permissions to create directories
            OSError: If directory creation fails for other reasons

        Example:
            >>> from src.config.settings import settings
            >>> settings.create_directories()  # Safe to call at startup
            >>> # All directories now exist and are ready to use
        """
        for path in [
            self.INPUT_DIR,
            self.PROCESSED_DIR,
            self.KNOWLEDGE_BASE_DIR,
            self.OUTPUT_DIR,
            self.BASE_DIR / "output" / "experiments",
        ]:
            path.mkdir(parents=True, exist_ok=True)
        # External source paths (e.g. WSL mounts) may not exist on all machines — best-effort only.
        try:
            self.OFFICIAL_DOCS_SOURCE.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass


settings = Settings()
"""Global settings instance used throughout the application.

Access via:
    from src.config.settings import settings
    input_dir = settings.INPUT_DIR
    settings.create_directories()
"""
