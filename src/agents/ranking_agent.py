"""Ranking Agent: Technical first, then Pedagogical (two-phase).

Evaluates generated courses: (1) Technical review first (correctness, completeness,
technical accuracy; threshold e.g. 9.0). (2) Pedagogical review (structure, CLT,
scaffolding, human language and context as cyber range course; threshold e.g. 9.0).
Output: state.ranking_reports. Criteria: docs/ranking_criterias/technical_criteria.md,
docs/ranking_criterias/pedagogical_criteria.md, objective_ranking.md.
"""

from __future__ import annotations

import ast
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from typing import Any, Dict, List, Optional

from loguru import logger

from src.config.settings import Settings
from src.config.settings import settings as app_settings
from src.core.state import AgentState
from src.models.report_models import RankingReport, RankingScore, ValidationIssue
from src.services.llm_service import (
    LLMServiceError,
    generate_response_with_system,
    reset_challenge_llm_budget,
)
from src.utils.course_content_checker import (
    build_ranking_hint_string,
    check_no_access_framing,
    check_section_presence,
)
from src.utils.log_redaction import redact_sensitive

# LLM settings for ranking (consistent scoring)
_RANKING_TEMPERATURE = 0.3
_RO_RANKING_NOTE = "The course content is in Romanian. Evaluate accordingly; technical correctness is language-independent."
_DEFAULT_SCORE = 5
_VALID_RANKS = ("Beginner", "Intermediate", "Advanced")


def compute_overall_score(
    tech: float,
    ped: float,
    cfg: "Settings",
) -> tuple[float, dict]:
    """Compute overall_score from tech + ped per active SCORING_POLICY.

    Always computes all three views (mean, min, weighted) for sensitivity analysis.

    Args:
        tech: Technical persona score (1-10).
        ped: Pedagogical persona score (1-10).
        cfg: Settings instance (reads SCORING_POLICY, SCORING_TECHNICAL_WEIGHT,
             SCORING_PEDAGOGICAL_WEIGHT).

    Returns:
        Tuple of (active_overall_score, overall_scores_dict).
        overall_scores_dict always contains 'mean', 'min', and a 'weighted_<t>_<p>' key.
    """
    policy = getattr(cfg, "SCORING_POLICY", "mean")
    w_t = float(getattr(cfg, "SCORING_TECHNICAL_WEIGHT", 0.5))
    w_p = float(getattr(cfg, "SCORING_PEDAGOGICAL_WEIGHT", 0.5))

    # Normalise weights so they always sum to 1.0
    total_w = w_t + w_p
    if total_w > 0:
        w_t = w_t / total_w
        w_p = w_p / total_w
    else:
        w_t = w_p = 0.5

    mean_score = round((tech + ped) / 2.0, 1)
    min_score = round(min(tech, ped), 1)
    weighted_score = round(tech * w_t + ped * w_p, 1)

    # Weighted key encodes the normalised weights as integers (e.g. 50_50, 55_45)
    wt_int = round(w_t * 100)
    wp_int = round(w_p * 100)
    weighted_key = f"weighted_{wt_int}_{wp_int}"

    all_views: dict = {
        "mean": mean_score,
        "min": min_score,
        weighted_key: weighted_score,
    }

    if policy == "min":
        active = min_score
    elif policy == "weighted":
        active = weighted_score
    else:
        active = mean_score  # default: "mean"

    return active, all_views


# Expected dimension names for rubric anchoring
_TECHNICAL_DIMENSIONS = {
    "correctness",
    "completeness",
    "technical_accuracy",
    "code_quality",
    "logical_validity",
}
_PEDAGOGICAL_DIMENSIONS = {
    "sections_structure",
    "cognitive_load",
    "scaffolding_reproducibility",
    "relevance_curriculum",
    "skill_level_awareness",
    "human_language_context",
}

# Pedagogical rubric (pedagogical_criteria.md; course-for-cyber-range)
_PEDAGOGICAL_SYSTEM = """You are a Pedagogical Expert Reviewer for cybersecurity **course** content (cyber range training material). Evaluate whether the content reads as a **course** for a cyber range: clear learning progression, human language and context, expression appropriate for training—not a generic writeup.

**Sections (11 sections — match course gen schema exactly):** (1) Title and context, (2) Abstract/TL;DR (~50 words), (3) Objectives, (4) Technical skills, (5) Definitions and concepts, (6) Reproducibility / Step 0 (metadata, resources), (7) Thought process / narrative, (8) Step-by-step resolution, (9) Solution script or code, (10) Conclusion, (11) Extra resources (ATT&CK, CWE, OWASP where relevant). When citing section numbers in improvements, use this 11-section numbering — Conclusion is §10, Extra Resources is §11.

**Cognitive load (CLT):** Low extraneous load; clarity and chunking; "easy to follow" and "explains why."

**Scaffolding / reproducibility:** Step 0 present; steps reproducible; code formatted and commented.

**Human language and context:** Does it express itself as a **course** (training, learning objectives, clear narrative for learners)? Not generic writeup tone; appropriate context and audience for a cyber range.

**Relevance / curriculum:** Reference ATT&CK, CWE, OWASP where relevant. **Skill-level awareness:** Appropriate depth (novice = full worked example; intermediate = completion-style; advanced = core vuln and pointers).

**Anti-patterns that must be penalised** (lower cognitive_load and scaffolding_reproducibility scores):
- Steps that say "run the script" or "execute the exploit" without stating what output to expect and what it means.
- Thought process that attributes reasoning to external sources ("the video shows", "a tool suggested") rather than reasoning from the challenge itself.
- Solution script section that says "see above" or "provided in the previous step" instead of containing the actual script.
- Extra resources citing irrelevant vulnerability IDs (e.g. a web CWE cited for a crypto challenge).

Respond with a single JSON object only, no markdown or preamble. Keep justification to one short sentence so the full JSON fits in one response:
{"score": <1-10>, "justification": "<one short sentence>", "improvements": ["<recommendation 1>", ...], "dimension_scores": {"sections_structure": <1-10>, "cognitive_load": <1-10>, "scaffolding_reproducibility": <1-10>, "relevance_curriculum": <1-10>, "skill_level_awareness": <1-10>, "human_language_context": <1-10>}}
dimension_scores must include all 6 pedagogical dimensions with scores 1-10: sections_structure, cognitive_load, scaffolding_reproducibility, relevance_curriculum, skill_level_awareness, human_language_context.
Give 1-4 concrete improvements that name the specific section and anti-pattern to fix. Score 1-10: 1-3 poor, 4-5 needs work, 6-7 adequate, 8-10 good to excellent (9+ = pass threshold for course quality)."""

# Technical rubric (correctness, completeness, accuracy)
_TECHNICAL_SYSTEM = """You are a Technical Expert Reviewer for cybersecurity CTF write-ups and solution scripts. Evaluate correctness and technical quality.

**Correctness:** Does the solution actually solve the challenge? Are commands and code accurate? Any edge cases or false assumptions?

**Completeness:** Are all steps needed to reproduce the solution present? No implicit "and then you get the flag" without showing how.

**Technical accuracy:** Are vulnerability explanations and exploit steps technically sound? Is the solver script consistent with the write-up and environment? Does the thought process explain WHY the technique applies to this specific challenge (mathematical structure, observable behaviour, vulnerability class) rather than just naming a tool?

**Code quality:** Is the solution script readable, commented where helpful, and following reasonable practices?

**Logical / process validity:** Do steps follow in a feasible order? No logical gaps, impossible sequences, or steps that say "run and see" without specifying expected output.

**Anti-patterns that must be penalised** (lower completeness and logical_validity scores):
- Any step that says only "run the script" or "execute this" without specifying what success looks like.
- Thought process that references external sources ("the video", "a writeup") as the basis for choosing a technique instead of deriving the reasoning from the challenge parameters.
- Extra resources citing vulnerability IDs from a completely different domain (e.g. XSS CWEs for a lattice crypto challenge, binary CWEs for an OSINT challenge).

Respond with a single JSON object only, no markdown or preamble. Keep justification to one short sentence so the full JSON fits in one response:
{"score": <1-10>, "justification": "<one short sentence>", "improvements": ["<recommendation 1>", ...], "technical_rank": "<Beginner|Intermediate|Advanced>", "dimension_scores": {"correctness": <1-10>, "completeness": <1-10>, "technical_accuracy": <1-10>, "code_quality": <1-10>, "logical_validity": <1-10>}}
dimension_scores must include all 5 technical dimensions with scores 1-10: correctness, completeness, technical_accuracy, code_quality, logical_validity.
technical_rank = difficulty: Beginner, Intermediate, or Advanced. Give 1-4 improvements that name the specific step or section that needs fixing. Score 1-10 as above."""


def _resolve_technical_system() -> str:
    """Return the active TECHNICAL_SYSTEM prompt (baseline or variant)."""
    name = app_settings.PROMPT_VARIANT
    if not name or name == "baseline":
        return _TECHNICAL_SYSTEM
    from src.prompts.variants.loader import load_variant

    return load_variant(name)["technical"]


def _resolve_pedagogical_system() -> str:
    """Return the active PEDAGOGICAL_SYSTEM prompt (baseline or variant)."""
    name = app_settings.PROMPT_VARIANT
    if not name or name == "baseline":
        return _PEDAGOGICAL_SYSTEM
    from src.prompts.variants.loader import load_variant

    return load_variant(name)["pedagogical"]


def _aggregate_dimension_scores(
    tech_dims: Optional[Dict[str, int]],
    ped_dims: Optional[Dict[str, int]],
    weights: Optional[Dict[str, float]] = None,
) -> Optional[Dict[str, float]]:
    """Merge technical and pedagogical dimension scores into one flat dict.

    Each set contributes its own dimensions (they don't overlap in practice).
    If both share a key, values are averaged (or weighted-averaged if weights given).
    Returns None if both inputs are None or empty.
    """
    merged: Dict[str, list] = {}
    for src in (tech_dims, ped_dims):
        if not src:
            continue
        for dim, val in src.items():
            merged.setdefault(dim, []).append(float(val))

    if not merged:
        return None

    result: Dict[str, float] = {}
    for dim, vals in merged.items():
        if weights and dim in weights:
            w = max(weights[dim], 0.0)
            result[dim] = round(
                (
                    sum(v * w for v in vals) / (len(vals) * w)
                    if w
                    else sum(vals) / len(vals)
                ),
                2,
            )
        else:
            result[dim] = round(sum(vals) / len(vals), 2)
    return result


def _truncate_for_prompt(text: str, max_chars: int = 100000) -> str:
    """Truncate course/script so prompt stays within reasonable token limits.

    v4.2 (2026-05-30): bumped default 12000 → 100000 after discovering this was THE
    truncation source the judges had been complaining about for 5+ smokes. Every
    judge feedback citing `[... truncated for length ...]` was reading that marker
    in the prompt because OUR CODE appended it. Courses 12-26k chars (typical) were
    getting 40-60% of content hidden — these were framework-side truncations, not LLM
    truncations.

    Claude haiku/sonnet/opus all support 200k+ context. 100k chars (~25k tokens) is
    safely under the input-token budget for any model and accommodates every course
    we've seen (max observed ~26631 bytes).
    """
    if not text or len(text) <= max_chars:
        return text or ""
    logger.warning(
        "Ranking input truncated: text was {} chars, capped to {} chars. "
        "Judge will see incomplete content and may score lower than warranted.",
        len(text),
        max_chars,
    )
    return text[: max_chars - 100].rstrip() + "\n\n[... truncated for length ...]"


# Fenced code block: ```lang\n ... \n``` (non-greedy, multiline).
_FENCED_CODE_RE = re.compile(r"```[^\n]*\n.*?```", re.DOTALL)


def _strip_code_blocks(text: str) -> str:
    """Remove fenced code blocks (```...```) from a course, leaving prose/headings.

    Used for the **pedagogical** judge only (J3): it grades structure, language, and
    learning progression — not code correctness — so the solver and large code blocks
    are noise that bloat the prompt. Stripping them shrinks the haiku input (faster
    calls). Inline single-backtick code is preserved as prose-level context. The
    technical judge is unaffected and still sees full code.
    """
    if not text:
        return text
    return _FENCED_CODE_RE.sub("[code omitted for pedagogical review]", text)


def _parse_review_json(raw: str, persona: str) -> Optional[Dict[str, Any]]:
    """Parse LLM response as JSON; extract score, justification, improvements, dimension_scores (and optional technical_rank).
    If JSON is truncated (e.g. unterminated string), tries to extract "score": N so we don't default to 5.

    Handles:
    - Markdown code fences (```json ... ```)
    - Preamble text before the JSON object (e.g. "Here is my evaluation:\\n{...}")
    - Truncated JSON (regex score fallback)
    - Python-style single-quoted dicts (Gemma-style)
    - Missing or unknown dimension keys (silently ignored; warning logged)
    """
    raw = raw.strip()
    # Strip markdown code block if present
    if raw.startswith("```"):
        raw = re.sub(r"^```\w*\n?", "", raw)
        raw = re.sub(r"\n?```\s*$", "", raw)
        raw = raw.strip()
    # Strip preamble text before the first '{' — some models emit a sentence first
    brace_pos = raw.find("{")
    if brace_pos > 0:
        raw = raw[brace_pos:]
    try:
        data = json.loads(raw)
        if isinstance(data, dict) and "score" in data:
            # Validate and normalize dimension_scores if present
            if "dimension_scores" in data:
                dim_scores = data.get("dimension_scores")
                if isinstance(dim_scores, dict):
                    # Validate dimension names and scores (1-10)
                    expected_dims = (
                        _TECHNICAL_DIMENSIONS
                        if persona == "Technical"
                        else _PEDAGOGICAL_DIMENSIONS
                    )
                    validated = {}
                    for dim, val in dim_scores.items():
                        if dim in expected_dims:
                            try:
                                score_val = int(val)
                                validated[dim] = max(1, min(10, score_val))
                            except (TypeError, ValueError):
                                logger.debug(
                                    "Ranking {}: invalid dimension score for {}: {}",
                                    persona,
                                    dim,
                                    val,
                                )
                    # Only include if we have at least some valid dimensions (backward compatible)
                    if validated:
                        data["dimension_scores"] = validated
                    else:
                        # All dimension keys were unknown — warn so we can detect schema drift
                        logger.warning(
                            "Ranking {}: dimension_scores present in LLM response but "
                            "no keys matched expected dimensions (got: {}); dropping.",
                            persona,
                            list(dim_scores.keys()),
                        )
                        data.pop("dimension_scores", None)
                else:
                    # Remove invalid dimension_scores type
                    logger.warning(
                        "Ranking {}: dimension_scores has unexpected type {}; dropping.",
                        persona,
                        type(dim_scores).__name__,
                    )
                    data.pop("dimension_scores", None)
            else:
                # JSON parsed but dimension_scores key absent — warn for diagnostics
                logger.warning(
                    "Ranking {}: LLM response parsed OK (score={}) but dimension_scores "
                    "key is absent; dim scores will be null.",
                    persona,
                    data.get("score"),
                )
            return data
    except json.JSONDecodeError as e:
        logger.warning(
            "Ranking {} JSON parse failed: {}; attempting fallback. Snippet: {}",
            persona,
            e,
            redact_sensitive(raw[:200]),
        )
        # Fallback 1: some models (e.g. Gemma) emit Python-style single-quoted dicts.
        try:
            data = ast.literal_eval(raw)
            if isinstance(data, dict) and "score" in data:
                # Run the same dimension_scores validation as the happy path.
                if "dimension_scores" in data:
                    dim_scores = data.get("dimension_scores")
                    if isinstance(dim_scores, dict):
                        expected_dims = (
                            _TECHNICAL_DIMENSIONS
                            if persona == "Technical"
                            else _PEDAGOGICAL_DIMENSIONS
                        )
                        validated = {}
                        for dim, val in dim_scores.items():
                            if dim in expected_dims:
                                try:
                                    validated[dim] = max(1, min(10, int(val)))
                                except (TypeError, ValueError):
                                    pass
                        if validated:
                            data["dimension_scores"] = validated
                        else:
                            data.pop("dimension_scores", None)
                    else:
                        data.pop("dimension_scores", None)
                return data
        except (ValueError, SyntaxError):
            pass
        # Fallback 2: truncated response — extract score via regex (single or double quotes).
        match = re.search(r"""['"]score['"]\s*:\s*(\d+)""", raw)
        if match:
            try:
                score = max(1, min(10, int(match.group(1))))
                return {
                    "score": score,
                    "justification": "Response truncated; score extracted from partial JSON.",
                    "improvements": [
                        "Re-run ranking or increase max_tokens if truncation persists."
                    ],
                }
            except (ValueError, TypeError):
                pass
    return None


def _build_pedagogical_score(challenge_id: str, writeup: str, raw: str) -> RankingScore:
    """Build RankingScore for Pedagogical persona from LLM response."""
    data = _parse_review_json(raw, "Pedagogical")
    if not data:
        return RankingScore(
            score=_DEFAULT_SCORE,
            persona="Pedagogical",
            justification="Evaluation could not be parsed; manual review recommended.",
            improvements=["Re-run ranking or provide feedback manually."],
            dimension_scores=None,
        )
    score = data.get("score", _DEFAULT_SCORE)
    if not isinstance(score, int):
        try:
            score = int(float(score))
        except (TypeError, ValueError):
            score = _DEFAULT_SCORE
    score = max(1, min(10, score))
    justification = data.get("justification") or "No justification provided."
    if not isinstance(justification, str):
        justification = str(justification)
    improvements = data.get("improvements") or []
    if not isinstance(improvements, list):
        improvements = [str(improvements)] if improvements else []
    improvements = [str(x).strip() for x in improvements if x][:8]
    # Extract dimension_scores (validated in _parse_review_json)
    dimension_scores = data.get("dimension_scores")
    if not isinstance(dimension_scores, dict) or not dimension_scores:
        logger.warning(
            "Ranking Pedagogical for {}: dimension_scores null after parse "
            "(score={}); check judge response format.",
            challenge_id,
            score,
        )
        dimension_scores = None
    return RankingScore(
        score=score,
        persona="Pedagogical",
        justification=justification.strip(),
        improvements=improvements,
        dimension_scores=dimension_scores,
    )


def _build_technical_score(challenge_id: str, raw: str) -> tuple[RankingScore, str]:
    """Build RankingScore for Technical persona and technical_rank from LLM response."""
    data = _parse_review_json(raw, "Technical")
    rank = "Intermediate"
    if not data:
        return (
            RankingScore(
                score=_DEFAULT_SCORE,
                persona="Technical",
                justification="Evaluation could not be parsed; manual review recommended.",
                improvements=["Re-run ranking or provide feedback manually."],
                dimension_scores=None,
            ),
            rank,
        )
    score = data.get("score", _DEFAULT_SCORE)
    if not isinstance(score, int):
        try:
            score = int(float(score))
        except (TypeError, ValueError):
            score = _DEFAULT_SCORE
    score = max(1, min(10, score))
    justification = data.get("justification") or "No justification provided."
    if not isinstance(justification, str):
        justification = str(justification)
    improvements = data.get("improvements") or []
    if not isinstance(improvements, list):
        improvements = [str(improvements)] if improvements else []
    improvements = [str(x).strip() for x in improvements if x][:8]
    r = data.get("technical_rank") or ""
    if isinstance(r, str) and r.strip() in _VALID_RANKS:
        rank = r.strip()
    # Extract dimension_scores (validated in _parse_review_json)
    dimension_scores = data.get("dimension_scores")
    if not isinstance(dimension_scores, dict) or not dimension_scores:
        logger.warning(
            "Ranking Technical for {}: dimension_scores null after parse "
            "(score={}); check judge response format.",
            challenge_id,
            score,
        )
        dimension_scores = None
    return (
        RankingScore(
            score=score,
            persona="Technical",
            justification=justification.strip(),
            improvements=improvements,
            dimension_scores=dimension_scores,
        ),
        rank,
    )


def _build_contest_note(contest_metadata: Optional[Dict[str, Any]]) -> str:
    """Return a one-line contest context note, or empty string if metadata is absent."""
    if not contest_metadata:
        return ""
    name = contest_metadata.get("name", "")
    ctype = contest_metadata.get("type", "")
    date = contest_metadata.get("date", "")
    parts = [p for p in (ctype, date) if p]
    detail = f" ({', '.join(parts)})" if parts else ""
    return f"Contest: {name}{detail}. Evaluate with this context." if name else ""


def _evaluate_one_challenge_retest(
    challenge_id: str,
    writeup: str,
    solve_script: str,
    run_technical: bool,
    run_pedagogical: bool,
    existing: Optional[RankingReport] = None,
    check_hints: Optional[str] = None,
    output_language: str = "en",
    contest_metadata: Optional[Dict[str, Any]] = None,
) -> RankingReport:
    """Run only the requested persona(s) and merge with existing report. Used when refining only low-scoring persona(s)."""
    tech_review = existing.technical_review if existing and not run_technical else None
    ped_review = (
        existing.pedagogical_review if existing and not run_pedagogical else None
    )
    technical_rank = (
        existing.technical_rank if existing and not run_technical else "Intermediate"
    )
    writeup_trunc = _truncate_for_prompt(writeup)
    script_trunc = _truncate_for_prompt(solve_script, max_chars=50000)
    ranking_max_tokens = getattr(app_settings, "RANKING_MAX_TOKENS", 4096) or 4096
    hints_section = (
        f"\n## Automated Pre-checks (advisory)\n{check_hints}\n" if check_hints else ""
    )
    tech_system = (
        (_RO_RANKING_NOTE + "\n\n" + _resolve_technical_system())
        if output_language == "ro"
        else _resolve_technical_system()
    )
    ped_system = (
        (_RO_RANKING_NOTE + "\n\n" + _resolve_pedagogical_system())
        if output_language == "ro"
        else _resolve_pedagogical_system()
    )
    contest_note = _build_contest_note(contest_metadata)
    if contest_note:
        tech_system = contest_note + "\n\n" + tech_system
        ped_system = contest_note + "\n\n" + ped_system

    if run_technical:
        tech_prompt = f"""## Challenge ID
{challenge_id}
{hints_section}
## Course (Markdown) and solver script
{writeup_trunc}

## Solver script
{script_trunc or "(No solver script.)"}

Evaluate correctness, completeness, and technical accuracy. Respond with JSON only: score (1-10), justification, improvements, technical_rank (Beginner|Intermediate|Advanced)."""
        try:
            tech_raw = generate_response_with_system(
                tech_system,
                tech_prompt,
                provider=app_settings.RANKING_PROVIDER,
                model=app_settings.RANKING_MODEL,
                temperature=_RANKING_TEMPERATURE,
                max_tokens=ranking_max_tokens,
            )
            tech_review, technical_rank = _build_technical_score(challenge_id, tech_raw)
        except LLMServiceError as e:
            # Preserve existing dimension_scores so a transient LLM failure during
            # re-ranking does not silently discard scores from a prior successful pass.
            _existing_tech_dims = (
                existing.technical_review.dimension_scores
                if existing and existing.technical_review
                else None
            )
            if _existing_tech_dims:
                logger.warning(
                    "Ranking technical LLM failed for {} ({}); preserving "
                    "dimension_scores from prior pass.",
                    challenge_id,
                    e,
                )
            else:
                logger.warning(
                    "Ranking technical LLM failed for {}: {}; dimension_scores will be null.",
                    challenge_id,
                    e,
                )
            tech_review = RankingScore(
                score=_DEFAULT_SCORE,
                persona="Technical",
                justification=f"LLM evaluation failed: {e}",
                improvements=[],
                dimension_scores=_existing_tech_dims,
            )
            technical_rank = existing.technical_rank if existing else "Intermediate"
    elif existing:
        tech_review = existing.technical_review

    if run_pedagogical:
        if getattr(app_settings, "RANKING_PEDAGOGICAL_OMIT_CODE", True):
            # J3: pedagogical persona grades teaching quality, not code → strip code
            # blocks and omit the solver section to shrink the haiku input.
            ped_course = _strip_code_blocks(writeup_trunc)
            ped_solver_section = ""
        else:
            ped_course = writeup_trunc
            ped_solver_section = f"\n## Solver script (for context)\n{script_trunc or '(No solver script provided.)'}\n"
        ped_prompt = f"""## Challenge ID
{challenge_id}
{hints_section}
## Course to evaluate (Markdown) — assess as cyber range training material
{ped_course}
{ped_solver_section}
Evaluate whether this reads as a **course** for a cyber range. Respond with JSON only: score (1-10), justification, improvements."""
        try:
            ped_raw = generate_response_with_system(
                ped_system,
                ped_prompt,
                provider=app_settings.RANKING_PROVIDER,
                model=app_settings.RANKING_MODEL,
                temperature=_RANKING_TEMPERATURE,
                max_tokens=ranking_max_tokens,
            )
            ped_review = _build_pedagogical_score(challenge_id, writeup, ped_raw)
        except LLMServiceError as e:
            # Preserve existing dimension_scores so a transient LLM failure during
            # re-ranking does not silently discard scores from a prior successful pass.
            _existing_ped_dims = (
                existing.pedagogical_review.dimension_scores
                if existing and existing.pedagogical_review
                else None
            )
            if _existing_ped_dims:
                logger.warning(
                    "Ranking pedagogical LLM failed for {} ({}); preserving "
                    "dimension_scores from prior pass.",
                    challenge_id,
                    e,
                )
            else:
                logger.warning(
                    "Ranking pedagogical LLM failed for {}: {}; dimension_scores will be null.",
                    challenge_id,
                    e,
                )
            ped_review = RankingScore(
                score=_DEFAULT_SCORE,
                persona="Pedagogical",
                justification=f"LLM evaluation failed: {e}",
                improvements=[],
                dimension_scores=_existing_ped_dims,
            )
    elif existing:
        ped_review = existing.pedagogical_review

    if tech_review is None or ped_review is None:
        tech_review = tech_review or RankingScore(
            score=_DEFAULT_SCORE,
            persona="Technical",
            justification="Not re-run.",
            improvements=[],
            dimension_scores=None,
        )
        ped_review = ped_review or RankingScore(
            score=_DEFAULT_SCORE,
            persona="Pedagogical",
            justification="Not re-run.",
            improvements=[],
            dimension_scores=None,
        )
    overall, all_views = compute_overall_score(
        float(tech_review.score), float(ped_review.score), app_settings
    )
    dim_weights = getattr(app_settings, "DIMENSION_WEIGHTS", None)
    agg_dims = _aggregate_dimension_scores(
        tech_review.dimension_scores,
        ped_review.dimension_scores,
        weights=dim_weights,
    )
    return RankingReport(
        challenge_id=challenge_id,
        overall_score=overall,
        overall_scores=all_views,
        pedagogical_review=ped_review,
        technical_review=tech_review,
        technical_rank=technical_rank,
        dimension_scores=agg_dims,
    )


def _evaluate_one_challenge(
    challenge_id: str,
    writeup: str,
    solve_script: str,
    check_hints: Optional[str] = None,
    output_language: str = "en",
    contest_metadata: Optional[Dict[str, Any]] = None,
) -> RankingReport:
    """Run Technical and Pedagogical reviews concurrently; return RankingReport.

    Both persona judges run via ThreadPoolExecutor(max_workers=2) so wall time is
    ~max(t_tech, t_ped) instead of t_tech + t_ped. Saves ~30s per challenge per
    ranking pass (v2 speedup, Wave 1).

    Thread-safety: generate_response_with_system uses subprocess calls (claude --print)
    or LangChain chat models; both are safe to invoke from multiple threads concurrently.
    ContextVar (_challenge_llm_call_count) is not shared across threads — each thread
    starts with the default (0). This is acceptable: the budget cap acts per-thread,
    preventing runaway within each judge independently.

    Args:
        check_hints: Optional advisory string from automated content checks
            (build_ranking_hint_string). Injected into both prompts so the LLM
            reviewer is aware of pre-detected issues. Advisory only — scores are
            not hard-coded.
        output_language: 'en' or 'ro'. When 'ro', prepends a language note to both
            system prompts so the judge evaluates Romanian content appropriately.
        contest_metadata: Optional contest context dict (id, name, type, date).
            When non-empty, a one-line note is prepended to both system prompts
            after the Romanian note.
    """
    writeup_trunc = _truncate_for_prompt(writeup)
    script_trunc = _truncate_for_prompt(solve_script, max_chars=50000)
    hints_section = (
        f"\n## Automated Pre-checks (advisory)\n{check_hints}\n" if check_hints else ""
    )
    tech_system = (
        (_RO_RANKING_NOTE + "\n\n" + _resolve_technical_system())
        if output_language == "ro"
        else _resolve_technical_system()
    )
    ped_system = (
        (_RO_RANKING_NOTE + "\n\n" + _resolve_pedagogical_system())
        if output_language == "ro"
        else _resolve_pedagogical_system()
    )
    contest_note = _build_contest_note(contest_metadata)
    if contest_note:
        tech_system = contest_note + "\n\n" + tech_system
        ped_system = contest_note + "\n\n" + ped_system

    tech_prompt = f"""## Challenge ID
{challenge_id}
{hints_section}
## Course (Markdown) and solver script
{writeup_trunc}

## Solver script
{script_trunc or "(No solver script.)"}

Evaluate correctness, completeness, and technical accuracy (see technical_criteria.md). Respond with JSON only: score (1-10), justification, improvements, technical_rank (Beginner|Intermediate|Advanced)."""

    ped_prompt = f"""## Challenge ID
{challenge_id}
{hints_section}
## Course to evaluate (Markdown) — assess as cyber range training material
{writeup_trunc}

## Solver script (for context; evaluate course pedagogy, not script syntax)
{script_trunc or "(No solver script provided.)"}

Evaluate whether this reads as a **course** for a cyber range (objectives, narrative, human language and context). Respond with JSON only: score (1-10), justification, improvements."""

    ranking_max_tokens = getattr(app_settings, "RANKING_MAX_TOKENS", 4096) or 4096

    def _run_technical() -> tuple[str, str]:
        """Return (tech_raw, "Intermediate") on success, ("", "") on error."""
        try:
            raw = generate_response_with_system(
                tech_system,
                tech_prompt,
                provider=app_settings.RANKING_PROVIDER,
                model=app_settings.RANKING_MODEL,
                temperature=_RANKING_TEMPERATURE,
                max_tokens=ranking_max_tokens,
            )
            return raw, ""
        except LLMServiceError as e:
            logger.warning("Ranking technical LLM failed for {}: {}", challenge_id, e)
            return "", str(e)

    def _run_pedagogical() -> tuple[str, str]:
        """Return (ped_raw, "") on success, ("", err) on error."""
        try:
            raw = generate_response_with_system(
                ped_system,
                ped_prompt,
                provider=app_settings.RANKING_PROVIDER,
                model=app_settings.RANKING_MODEL,
                temperature=_RANKING_TEMPERATURE,
                max_tokens=ranking_max_tokens,
            )
            return raw, ""
        except LLMServiceError as e:
            logger.warning("Ranking pedagogical LLM failed for {}: {}", challenge_id, e)
            return "", str(e)

    # Run both judges concurrently (S1 — saves ~30s per challenge per ranking pass)
    with ThreadPoolExecutor(max_workers=2) as _pool:
        tech_future = _pool.submit(_run_technical)
        ped_future = _pool.submit(_run_pedagogical)
        tech_raw, tech_err = tech_future.result()
        ped_raw, ped_err = ped_future.result()

    if tech_raw:
        tech_review, technical_rank = _build_technical_score(challenge_id, tech_raw)
    else:
        tech_review = RankingScore(
            score=_DEFAULT_SCORE,
            persona="Technical",
            justification=f"LLM evaluation failed: {tech_err}",
            improvements=["Check LLM configuration and retry."],
            dimension_scores=None,
        )
        technical_rank = "Intermediate"

    if ped_raw:
        ped_review = _build_pedagogical_score(challenge_id, writeup, ped_raw)
    else:
        ped_review = RankingScore(
            score=_DEFAULT_SCORE,
            persona="Pedagogical",
            justification=f"LLM evaluation failed: {ped_err}",
            improvements=["Check LLM configuration and retry."],
            dimension_scores=None,
        )

    overall, all_views = compute_overall_score(
        float(tech_review.score), float(ped_review.score), app_settings
    )
    dim_weights = getattr(app_settings, "DIMENSION_WEIGHTS", None)
    agg_dims = _aggregate_dimension_scores(
        tech_review.dimension_scores,
        ped_review.dimension_scores,
        weights=dim_weights,
    )
    return RankingReport(
        challenge_id=challenge_id,
        overall_score=overall,
        overall_scores=all_views,
        pedagogical_review=ped_review,
        technical_review=tech_review,
        technical_rank=technical_rank,
        dimension_scores=agg_dims,
    )


def _run_batch_ranking(
    items: list,
    hints_by_cid: Dict[str, str],
    output_language: str,
    contest_metadata: Optional[Dict[str, Any]],
) -> Dict[str, "RankingReport"]:
    """Submit all ranking calls for items as a single Anthropic batch and return reports.

    Uses custom_ids of the form {md5(challenge_id)_tech} / {md5(challenge_id)_ped}
    to accommodate challenge IDs that contain characters disallowed in batch IDs.
    """
    import hashlib

    from src.utils.anthropic_batch import poll_batch, submit_ranking_batch

    ranking_max_tokens = getattr(app_settings, "RANKING_MAX_TOKENS", 4096) or 4096
    model = app_settings.RANKING_MODEL or "claude-haiku-4-5-20251001"

    id_map: Dict[str, tuple] = {}  # batch_custom_id -> (challenge_id, persona)
    requests: list = []

    for cid, writeup, solve_script in items:
        writeup_trunc = _truncate_for_prompt(writeup)
        script_trunc = _truncate_for_prompt(solve_script, max_chars=50000)
        hint = hints_by_cid.get(cid, "")
        hints_section = (
            f"\n## Automated Pre-checks (advisory)\n{hint}\n" if hint else ""
        )

        tech_system = (
            (_RO_RANKING_NOTE + "\n\n" + _resolve_technical_system())
            if output_language == "ro"
            else _resolve_technical_system()
        )
        ped_system = (
            (_RO_RANKING_NOTE + "\n\n" + _resolve_pedagogical_system())
            if output_language == "ro"
            else _resolve_pedagogical_system()
        )
        contest_note = _build_contest_note(contest_metadata)
        if contest_note:
            tech_system = contest_note + "\n\n" + tech_system
            ped_system = contest_note + "\n\n" + ped_system

        tech_prompt = (
            f"## Challenge ID\n{cid}{hints_section}\n"
            f"## Course (Markdown) and solver script\n{writeup_trunc}\n\n"
            f"## Solver script\n{script_trunc or '(No solver script.)'}\n\n"
            "Evaluate correctness, completeness, and technical accuracy. "
            "Respond with JSON only: score (1-10), justification, improvements, "
            "technical_rank (Beginner|Intermediate|Advanced)."
        )
        ped_prompt = (
            f"## Challenge ID\n{cid}{hints_section}\n"
            f"## Course to evaluate (Markdown) — assess as cyber range training material\n{writeup_trunc}\n\n"
            f"## Solver script (for context)\n{script_trunc or '(No solver script provided.)'}\n\n"
            "Evaluate whether this reads as a **course** for a cyber range. "
            "Respond with JSON only: score (1-10), justification, improvements."
        )

        cid_hash = hashlib.md5(cid.encode()).hexdigest()[:16]
        tech_id = f"{cid_hash}_tech"
        ped_id = f"{cid_hash}_ped"
        id_map[tech_id] = (cid, "tech")
        id_map[ped_id] = (cid, "ped")

        requests.append(
            {
                "custom_id": tech_id,
                "model": model,
                "system": tech_system,
                "user": tech_prompt,
                "temperature": _RANKING_TEMPERATURE,
                "max_tokens": ranking_max_tokens,
            }
        )
        requests.append(
            {
                "custom_id": ped_id,
                "model": model,
                "system": ped_system,
                "user": ped_prompt,
                "temperature": _RANKING_TEMPERATURE,
                "max_tokens": ranking_max_tokens,
            }
        )

    batch_id = submit_ranking_batch(requests)
    raw_results = poll_batch(batch_id)

    results_by_cid: Dict[str, Dict[str, str]] = {}
    for r in raw_results:
        if r.get("error"):
            logger.warning("Batch result error for {}: {}", r["custom_id"], r["error"])
        original_cid, persona = id_map.get(r["custom_id"], (None, None))
        if original_cid and persona:
            results_by_cid.setdefault(original_cid, {})[persona] = r.get("content", "")

    reports: Dict[str, "RankingReport"] = {}
    for cid, writeup, _ in items:
        cid_results = results_by_cid.get(cid, {})
        tech_raw = cid_results.get("tech", "")
        ped_raw = cid_results.get("ped", "")

        if tech_raw:
            tech_review, technical_rank = _build_technical_score(cid, tech_raw)
        else:
            tech_review = RankingScore(
                score=_DEFAULT_SCORE,
                persona="Technical",
                justification="Batch result missing; manual review needed.",
                improvements=[],
                dimension_scores=None,
            )
            technical_rank = "Intermediate"

        if ped_raw:
            ped_review = _build_pedagogical_score(cid, writeup, ped_raw)
        else:
            ped_review = RankingScore(
                score=_DEFAULT_SCORE,
                persona="Pedagogical",
                justification="Batch result missing; manual review needed.",
                improvements=[],
                dimension_scores=None,
            )

        overall, all_views = compute_overall_score(
            float(tech_review.score), float(ped_review.score), app_settings
        )
        dim_weights = getattr(app_settings, "DIMENSION_WEIGHTS", None)
        agg_dims = _aggregate_dimension_scores(
            tech_review.dimension_scores,
            ped_review.dimension_scores,
            weights=dim_weights,
        )
        reports[cid] = RankingReport(
            challenge_id=cid,
            overall_score=overall,
            overall_scores=all_views,
            pedagogical_review=ped_review,
            technical_review=tech_review,
            technical_rank=technical_rank,
            dimension_scores=agg_dims,
        )
        logger.info(
            "Ranking (batch) {}: overall={}, ped={}, tech={}, rank={}",
            cid,
            overall,
            ped_review.score,
            tech_review.score,
            technical_rank,
        )
    return reports


def run_ranking_agent(state: AgentState) -> AgentState:
    """Evaluate all generated courses with Pedagogical and Technical personas.

    For each challenge_id in state.generated_courses, runs two LLM reviews
    (Pedagogical rubric + Technical correctness), then appends a RankingReport
    to state.ranking_reports. Respects state.skip_ranking (skip LLM) and
    state.ranking_subset_ids (rank only those challenges) to reduce token usage.

    Args:
        state: Current agent state with generated_courses (and optionally
            generated_solve_scripts).

    Returns:
        Updated state with ranking_reports populated.
    """
    # Heartbeat: mark phase as ranking when node starts.
    try:
        from src.services.heartbeat import get_active_state as _hb_get

        _hb_ranking = _hb_get()
        if _hb_ranking is not None:
            _hb_ranking.current_phase = "ranking"
    except Exception:
        _hb_ranking = None  # heartbeat is non-critical

    if state.skip_ranking:
        logger.info("Ranking: skip_ranking=True; skipping LLM calls.")
        return replace(state, current_agent="ranking_agent")

    logger.info(
        "Ranking: entering with {} generated_courses, provider={}, model={}",
        len(state.generated_courses or {}),
        getattr(app_settings, "RANKING_PROVIDER", None)
        or getattr(app_settings, "LLM_DEFAULT_PROVIDER", "unknown"),
        getattr(app_settings, "RANKING_MODEL", None)
        or getattr(app_settings, "LLM_DEFAULT_MODEL", "unknown"),
    )

    writeups: Dict[str, str] = dict(state.generated_courses or {})
    scripts: Dict[str, str] = dict(state.generated_solve_scripts or {})

    if not writeups:
        logger.warning("Ranking: no generated_courses in state; skipping.")
        return replace(state, current_agent="ranking_agent")

    if state.ranking_subset_ids:
        subset = set(state.ranking_subset_ids)
        writeups = {k: v for k, v in writeups.items() if k in subset}
        logger.info("Ranking: running on subset of {} challenges.", len(writeups))
        if not writeups:
            return replace(state, current_agent="ranking_agent")

    # Upsert by challenge_id: keep one report per challenge (latest) so routing sees current scores
    reports_by_id: Dict[str, RankingReport] = {
        r.challenge_id: r for r in (state.ranking_reports or [])
    }
    items = [
        (cid, w, scripts.get(cid, "")) for cid, w in writeups.items() if w and w.strip()
    ]
    if not items:
        return replace(
            state,
            ranking_reports=list(reports_by_id.values()),
            current_agent="ranking_agent",
        )

    try:
        retest_tech = set(state.ranking_retest_technical_ids or [])
        retest_ped = set(state.ranking_retest_pedagogical_ids or [])
        only_retest_personas = bool(retest_tech or retest_ped)
        if only_retest_personas:
            items = [
                (cid, w, s)
                for cid, w, s in items
                if cid in retest_tech or cid in retest_ped
            ]
            if not items:
                return replace(
                    state,
                    ranking_reports=list(reports_by_id.values()),
                    current_agent="ranking_agent",
                )

        existing_term_issues: Dict[str, List[ValidationIssue]] = dict(
            state.course_terminology_issues or {}
        )
        hints_by_cid: Dict[str, str] = {}
        for cid, writeup, _ in items:
            check_issues: List[ValidationIssue] = []
            check_issues.extend(check_no_access_framing(writeup, challenge_id=cid))
            check_issues.extend(check_section_presence(writeup, challenge_id=cid))
            seen_msgs: set[str] = {i.message for i in check_issues}
            for iss in existing_term_issues.get(cid, []):
                if iss.message not in seen_msgs:
                    check_issues.append(iss)
                    seen_msgs.add(iss.message)
            hint = build_ranking_hint_string(check_issues)
            if hint:
                hints_by_cid[cid] = hint
                logger.debug("Ranking {}: check hints → {}", cid, hint)

        output_language = getattr(state, "output_language", "en")
        contest_metadata = dict(getattr(state, "contest_metadata", None) or {})

        use_batch = (
            getattr(app_settings, "RANKING_USE_BATCH_API", False)
            and not only_retest_personas
        )
        if use_batch:
            resolved_ranking_provider = (
                getattr(app_settings, "RANKING_PROVIDER", None)
                or getattr(app_settings, "LLM_DEFAULT_PROVIDER", "")
            ).lower()
            if resolved_ranking_provider != "anthropic":
                logger.warning(
                    "RANKING_USE_BATCH_API=True but RANKING_PROVIDER is not 'anthropic' (got %r); "
                    "falling back to sequential ranking.",
                    resolved_ranking_provider,
                )
                use_batch = False
            elif not getattr(app_settings, "ANTHROPIC_API_KEY", None):
                raise ValueError(
                    "ANTHROPIC_API_KEY is required when RANKING_USE_BATCH_API=True"
                )

        if use_batch:
            logger.info(
                "Ranking: submitting {} challenges via Anthropic Batch API", len(items)
            )
            try:
                batch_reports = _run_batch_ranking(
                    items, hints_by_cid, output_language, contest_metadata
                )
                reports_by_id.update(batch_reports)
            except Exception as e:
                logger.warning(
                    "Batch ranking failed ({}); falling back to sequential.", e
                )
                use_batch = False
    except Exception as exc:
        logger.exception("Ranking: pre-loop setup failed: {}", exc)
        state.add_error("ranking_agent", "setup", str(exc))
        return replace(
            state,
            ranking_reports=list(reports_by_id.values()),
            current_agent="ranking_agent",
        )

    max_concurrent = getattr(app_settings, "LLM_MAX_CONCURRENT", 1) or 1
    if use_batch:
        pass  # already done above
    elif max_concurrent <= 1:
        for challenge_id, writeup, solve_script in items:
            # Reset LLM budget per-challenge so gen-phase calls don't count against
            # the ranking budget.  Each challenge gets a fresh cap=20 for ranking.
            reset_challenge_llm_budget(challenge_id)
            try:
                hint = hints_by_cid.get(challenge_id)
                if only_retest_personas:
                    report = _evaluate_one_challenge_retest(
                        challenge_id,
                        writeup,
                        solve_script or "",
                        run_technical=challenge_id in retest_tech,
                        run_pedagogical=challenge_id in retest_ped,
                        existing=reports_by_id.get(challenge_id),
                        check_hints=hint,
                        output_language=output_language,
                        contest_metadata=contest_metadata or None,
                    )
                else:
                    report = _evaluate_one_challenge(
                        challenge_id,
                        writeup,
                        solve_script or "",
                        check_hints=hint,
                        output_language=output_language,
                        contest_metadata=contest_metadata or None,
                    )
                reports_by_id[challenge_id] = report
                logger.info(
                    "Ranking {}: overall={}, ped={}, tech={}, rank={}",
                    challenge_id,
                    report.overall_score,
                    report.pedagogical_review.score,
                    report.technical_review.score,
                    report.technical_rank,
                )
                # Heartbeat: increment completed_challenges after each dual-rubric finishes.
                try:
                    if _hb_ranking is not None:
                        _hb_ranking.completed_challenges += 1
                except Exception:
                    pass
            except Exception as e:
                logger.exception("Ranking failed for {}: {}", challenge_id, e)
                state.add_error("ranking_agent", challenge_id, str(e))
                reports_by_id[challenge_id] = RankingReport(
                    challenge_id=challenge_id,
                    overall_score=float(_DEFAULT_SCORE),
                    pedagogical_review=RankingScore(
                        score=_DEFAULT_SCORE,
                        persona="Pedagogical",
                        justification=f"Error during evaluation: {e}",
                        improvements=[],
                        dimension_scores=None,
                    ),
                    technical_review=RankingScore(
                        score=_DEFAULT_SCORE,
                        persona="Technical",
                        justification=f"Error during evaluation: {e}",
                        improvements=[],
                        dimension_scores=None,
                    ),
                    technical_rank="Intermediate",
                )
                # Heartbeat: count failed rankings too (challenge was "processed")
                try:
                    if _hb_ranking is not None:
                        _hb_ranking.completed_challenges += 1
                except Exception:
                    pass
    else:
        workers = min(max_concurrent, len(items) or 1)
        logger.info(
            "Ranking: evaluating {} courses with {} concurrent workers",
            len(items),
            workers,
        )

        def _eval_item(
            args: tuple,
        ) -> tuple[str, Optional[RankingReport], Optional[Exception]]:
            cid, wu, sc = args
            # Reset LLM budget per-challenge in each worker thread.
            reset_challenge_llm_budget(cid)
            hint = hints_by_cid.get(cid)
            try:
                if only_retest_personas:
                    r = _evaluate_one_challenge_retest(
                        cid,
                        wu,
                        sc or "",
                        run_technical=cid in retest_tech,
                        run_pedagogical=cid in retest_ped,
                        existing=reports_by_id.get(cid),
                        check_hints=hint,
                        output_language=output_language,
                        contest_metadata=contest_metadata or None,
                    )
                else:
                    r = _evaluate_one_challenge(
                        cid,
                        wu,
                        sc or "",
                        check_hints=hint,
                        output_language=output_language,
                        contest_metadata=contest_metadata or None,
                    )
                return (cid, r, None)
            except Exception as e:
                return (cid, None, e)

        result_by_cid: Dict[
            str, tuple[Optional[RankingReport], Optional[Exception]]
        ] = {}
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_cid = {
                executor.submit(_eval_item, (cid, wu, sc)): cid for cid, wu, sc in items
            }
            for future in as_completed(future_to_cid):
                challenge_id, report, err = future.result()
                result_by_cid[challenge_id] = (report, err)

        for challenge_id, _, _ in items:
            report, err = result_by_cid.get(challenge_id, (None, None))
            if err:
                logger.exception("Ranking failed for {}: {}", challenge_id, err)
                state.add_error("ranking_agent", challenge_id, str(err))
                reports_by_id[challenge_id] = RankingReport(
                    challenge_id=challenge_id,
                    overall_score=float(_DEFAULT_SCORE),
                    pedagogical_review=RankingScore(
                        score=_DEFAULT_SCORE,
                        persona="Pedagogical",
                        justification=f"Error during evaluation: {err}",
                        improvements=[],
                        dimension_scores=None,
                    ),
                    technical_review=RankingScore(
                        score=_DEFAULT_SCORE,
                        persona="Technical",
                        justification=f"Error during evaluation: {err}",
                        improvements=[],
                        dimension_scores=None,
                    ),
                    technical_rank="Intermediate",
                )
            else:
                assert report is not None
                reports_by_id[challenge_id] = report
                logger.info(
                    "Ranking {}: overall={}, ped={}, tech={}, rank={}",
                    challenge_id,
                    report.overall_score,
                    report.pedagogical_review.score,
                    report.technical_review.score,
                    report.technical_rank,
                )
            # Heartbeat: increment completed_challenges after each challenge (success or error).
            try:
                if _hb_ranking is not None:
                    _hb_ranking.completed_challenges += 1
            except Exception:
                pass

    # Build challenge_dimension_scores for downstream use (refinement, HITL)
    challenge_dimension_scores: Dict[str, Dict[str, float]] = {}
    for report in reports_by_id.values():
        if report.dimension_scores:
            challenge_dimension_scores[report.challenge_id] = report.dimension_scores
            logger.debug(
                "Ranking {}: dimension_scores={}",
                report.challenge_id,
                report.dimension_scores,
            )

    return replace(
        state,
        ranking_reports=list(reports_by_id.values()),
        challenge_dimension_scores=challenge_dimension_scores or None,
        current_agent="ranking_agent",
    )


async def ranking_agent(state: AgentState) -> AgentState:
    """Evaluate quality of generated courses (Pedagogical + Technical rubric).

    Uses Pedagogical and Technical expert personas to score courses against
    the course writeup guidelines and ranking rubrics; output in state.ranking_reports.

    Args:
        state: Pipeline state with generated_courses (and generated_solve_scripts).

    Returns:
        AgentState: Updated state with ranking_reports (RankingReport per challenge).
    """
    return run_ranking_agent(state)
