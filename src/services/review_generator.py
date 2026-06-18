"""Auto-generate REVIEW.md for experiment runs.

Reads manifest.json + ranking_reports.json from an experiment directory,
computes metrics, detects anomalies, and renders a human-readable REVIEW.md.
"""

from __future__ import annotations

import json
import random
import statistics
from pathlib import Path
from typing import Optional

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from src.config.settings import settings

# ---------------------------------------------------------------------------
# Pydantic V2 models
# ---------------------------------------------------------------------------


class ReviewMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    pass_rate: float = Field(description="Fraction of challenges passing threshold")
    challenges_passed: int = 0
    challenges_total: int = 0
    mean_overall: float = 0.0
    mean_technical: float = 0.0
    mean_pedagogical: float = 0.0
    refinement_rounds_avg: float = 0.0
    terminology_issues_total: int = 0
    delta_pass_rate: Optional[float] = None
    delta_mean_overall: Optional[float] = None


class ScoreDistribution(BaseModel):
    model_config = ConfigDict(frozen=True)

    below_5: int = 0
    range_5_7: int = 0
    range_7_9: int = 0
    above_9: int = 0


class ReviewAnomalies(BaseModel):
    model_config = ConfigDict(frozen=True)

    mid_graph_halt: bool = False
    empty_challenge_ids: bool = False
    courses_missing: bool = False
    ranking_missing: bool = False
    llm_log_empty: bool = False
    errors_in_log: bool = False
    repro_violations: list[str] = Field(default_factory=list)


class ReproAudit(BaseModel):
    model_config = ConfigDict(frozen=True)

    temperature: Optional[float] = None
    embedding_provider: str = "unknown"
    judge_model_pinned: bool = False
    run_config_captured: bool = False
    langsmith_trace: bool = False


class RefinementHistogram(BaseModel):
    model_config = ConfigDict(frozen=True)

    round_counts: dict[str, int] = Field(default_factory=dict)
    """Maps round number (as str "1", "2", ...) to challenge count."""


class Review(BaseModel):
    model_config = ConfigDict(frozen=True)

    exp_id: str
    status: str
    provider: str = "unknown"
    model: str = "unknown"
    judge_model: str = "unknown"
    challenges_completed: int = 0
    challenges_total: int = 0
    wall_time: float = 0.0
    cost_estimate: str = "--"
    metrics: ReviewMetrics
    distribution: ScoreDistribution
    anomalies: ReviewAnomalies
    repro: ReproAudit
    refinement_histogram: RefinementHistogram = Field(
        default_factory=RefinementHistogram
    )
    best_course: str = "--"
    worst_course: str = "--"
    random_sample: str = "--"


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

_EXPECTED_NODES = [
    "coordinator",
    "validator",
    "content_generation",
    "course_terminology_checker",
    "mapping",
    "ranking",
    "refinement_step",
    "hitl",
]


def _parse_manifest(exp_dir: Path) -> Optional[dict]:
    manifest_path = exp_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning(f"Could not parse manifest.json in {exp_dir}: {exc}")
        return None


def _parse_ranking_reports(exp_dir: Path) -> list[dict]:
    """Parse ranking_reports.json, handling both modern (flat list) and legacy (per_challenge) formats."""
    rr_path = exp_dir / "ranking_reports.json"
    if not rr_path.exists():
        return []
    try:
        raw = json.loads(rr_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning(f"Could not parse ranking_reports.json in {exp_dir}: {exc}")
        return []

    if isinstance(raw, list):
        return raw

    if isinstance(raw, dict) and "per_challenge" in raw:
        entries = []
        for ch in raw["per_challenge"]:
            ranking = ch.get("ranking", {})
            ped = ranking.get("pedagogical", {})
            tech = ranking.get("technical", {})
            ped_score = ped.get("score", 0)
            tech_score = tech.get("score", 0)
            overall = (ped_score + tech_score) / 2 if (ped_score or tech_score) else 0.0
            entries.append(
                {
                    "challenge_id": ch.get("challenge_id", "unknown"),
                    "overall_score": overall,
                    "pedagogical_review": {
                        "score": ped_score,
                        "persona": "Pedagogical",
                        "justification": ped.get("justification", ""),
                        "dimension_scores": ped.get("dimension_scores"),
                    },
                    "technical_review": {
                        "score": tech_score,
                        "persona": "Technical",
                        "justification": tech.get("justification", ""),
                        "dimension_scores": tech.get("dimension_scores"),
                    },
                    "technical_rank": tech.get("technical_rank", "Unknown"),
                    "_refinement_rounds": ch.get("refinement_rounds", 0),
                }
            )
        return entries

    return []


def _extract_scores(
    rankings: list[dict],
) -> tuple[list[float], list[float], list[float]]:
    """Extract (overall, technical, pedagogical) score lists from parsed rankings."""
    overalls, techs, peds = [], [], []
    for r in rankings:
        overalls.append(float(r.get("overall_score", 0)))
        tr = r.get("technical_review", {})
        pr = r.get("pedagogical_review", {})
        techs.append(float(tr.get("score", 0)))
        peds.append(float(pr.get("score", 0)))
    return overalls, techs, peds


def _compute_metrics(
    rankings: list[dict],
    manifest: Optional[dict],
    prev_review_path: Optional[Path],
) -> ReviewMetrics:
    threshold = settings.RANKING_PASS_THRESHOLD
    overalls, techs, peds = _extract_scores(rankings)
    total = len(overalls)
    passed = sum(1 for s in overalls if s >= threshold) if total else 0
    pass_rate = passed / total if total else 0.0

    rounds = [r.get("_refinement_rounds", 0) for r in rankings]
    rounds_avg = statistics.mean(rounds) if rounds else 0.0

    delta_pass_rate = None
    delta_mean_overall = None
    if prev_review_path and prev_review_path.exists():
        try:
            prev_text = prev_review_path.read_text(encoding="utf-8")
            prev_data = json.loads(prev_text)
            prev_metrics = prev_data.get("metrics", {})
            prev_pr = prev_metrics.get("pass_rate")
            prev_mo = prev_metrics.get("mean_overall")
            if prev_pr is not None:
                delta_pass_rate = pass_rate - prev_pr
            if prev_mo is not None and overalls:
                delta_mean_overall = statistics.mean(overalls) - prev_mo
        except (json.JSONDecodeError, OSError):
            pass

    return ReviewMetrics(
        pass_rate=pass_rate,
        challenges_passed=passed,
        challenges_total=total,
        mean_overall=statistics.mean(overalls) if overalls else 0.0,
        mean_technical=statistics.mean(techs) if techs else 0.0,
        mean_pedagogical=statistics.mean(peds) if peds else 0.0,
        refinement_rounds_avg=rounds_avg,
        delta_pass_rate=delta_pass_rate,
        delta_mean_overall=delta_mean_overall,
    )


def _compute_distribution(overalls: list[float]) -> ScoreDistribution:
    below_5 = sum(1 for s in overalls if s < 5.0)
    range_5_7 = sum(1 for s in overalls if 5.0 <= s < 7.0)
    range_7_9 = sum(1 for s in overalls if 7.0 <= s < 9.0)
    above_9 = sum(1 for s in overalls if s >= 9.0)
    return ScoreDistribution(
        below_5=below_5, range_5_7=range_5_7, range_7_9=range_7_9, above_9=above_9
    )


def _detect_anomalies(
    exp_dir: Path, manifest: Optional[dict], rankings: list[dict]
) -> ReviewAnomalies:
    node_timings = (manifest or {}).get("node_timings", {})
    present_nodes = set(node_timings.keys())
    mid_graph_halt = bool(present_nodes) and present_nodes != set(_EXPECTED_NODES)

    challenge_ids = (manifest or {}).get("challenge_ids", [])
    empty_challenge_ids = len(challenge_ids) == 0

    courses_dir = exp_dir / "courses"
    courses_missing = not courses_dir.exists() or not any(
        courses_dir.rglob("course.md")
    )

    ranking_missing = not (exp_dir / "ranking_reports.json").exists()

    llm_log_path = exp_dir / "llm_calls.jsonl"
    llm_log_empty = not llm_log_path.exists() or llm_log_path.stat().st_size == 0

    errors_in_log = False
    if llm_log_path.exists():
        try:
            text = llm_log_path.read_text(encoding="utf-8")
            errors_in_log = "error" in text.lower()
        except OSError:
            pass

    repro_violations: list[str] = []
    if manifest:
        snap = manifest.get("settings_snapshot", {})
        temp = snap.get("temperature")
        if temp is not None and temp != 0.0:
            repro_violations.append(
                f"temperature={temp} (expected 0.0 for reproducibility)"
            )

    return ReviewAnomalies(
        mid_graph_halt=mid_graph_halt,
        empty_challenge_ids=empty_challenge_ids,
        courses_missing=courses_missing,
        ranking_missing=ranking_missing,
        llm_log_empty=llm_log_empty,
        errors_in_log=errors_in_log,
        repro_violations=repro_violations,
    )


def _repro_audit(manifest: Optional[dict], exp_dir: Path) -> ReproAudit:
    snap = (manifest or {}).get("settings_snapshot", {})
    temperature = snap.get("temperature")

    run_config_path = exp_dir / "run_config.json"
    run_config_captured = run_config_path.exists()
    if temperature is None and run_config_captured:
        try:
            rc = json.loads(run_config_path.read_text(encoding="utf-8"))
            temperature = rc.get("temperature")
        except (json.JSONDecodeError, OSError):
            pass

    embedding_provider = (
        str(settings.EMBEDDING_PROVIDER) if settings.EMBEDDING_PROVIDER else "unknown"
    )

    judge_model = snap.get("model", "")
    ranking_model_from_rc = ""
    if run_config_captured:
        try:
            rc = json.loads(run_config_path.read_text(encoding="utf-8"))
            ranking_model_from_rc = rc.get("ranking_model", "")
        except (json.JSONDecodeError, OSError):
            pass
    actual_judge = ranking_model_from_rc or judge_model
    judge_model_pinned = actual_judge == settings.FEEDBACK_JUDGE_MODEL

    langsmith_trace = (exp_dir / "langsmith_trace.json").exists()

    return ReproAudit(
        temperature=temperature,
        embedding_provider=embedding_provider,
        judge_model_pinned=judge_model_pinned,
        run_config_captured=run_config_captured,
        langsmith_trace=langsmith_trace,
    )


def _find_last_same_model(exp_dir: Path, provider: str, model: str) -> Optional[Path]:
    experiments_root = exp_dir.parent
    candidates: list[tuple[str, Path]] = []
    for mpath in sorted(experiments_root.glob("EXP-*/manifest.json")):
        if mpath.parent == exp_dir:
            continue
        try:
            m = json.loads(mpath.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        snap = m.get("settings_snapshot", {})
        if snap.get("provider") == provider and snap.get("model") == model:
            started = m.get("started_at", "")
            candidates.append((started, mpath.parent))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    current_started = ""
    try:
        cm = json.loads((exp_dir / "manifest.json").read_text(encoding="utf-8"))
        current_started = cm.get("started_at", "")
    except (json.JSONDecodeError, OSError):
        pass

    before = (
        [c for c in candidates if c[0] < current_started]
        if current_started
        else candidates
    )
    if not before:
        return None
    return before[-1][1]


def _select_samples(rankings: list[dict], exp_id: str) -> tuple[str, str, str]:
    if not rankings:
        return "--", "--", "--"
    best = max(rankings, key=lambda r: r.get("overall_score", 0))
    worst = min(rankings, key=lambda r: r.get("overall_score", 0))
    rng = random.Random(exp_id)
    sample = rng.choice(rankings)
    return (
        best.get("challenge_id", "--"),
        worst.get("challenge_id", "--"),
        sample.get("challenge_id", "--"),
    )


def _compute_refinement_histogram(rankings: list[dict]) -> "RefinementHistogram":
    """Count challenges by refinement round from _refinement_rounds field."""
    counts: dict[str, int] = {}
    for r in rankings:
        rounds = int(r.get("_refinement_rounds", 0))
        key = str(rounds)
        counts[key] = counts.get(key, 0) + 1
    return RefinementHistogram(round_counts=counts)


def _render_refinement_histogram(hist: "RefinementHistogram") -> str:
    """Render refinement-round histogram as a Markdown table."""
    if not hist.round_counts:
        return "_No refinement data available._"
    lines = ["| Round | Challenges |", "|-------|-----------|"]
    for key in sorted(hist.round_counts, key=lambda k: int(k)):
        count = hist.round_counts[key]
        lines.append(f"| Round {key} | {count} |")
    return "\n".join(lines)


def _histogram_ascii(distribution: ScoreDistribution) -> str:
    buckets = [
        ("<5.0", distribution.below_5),
        ("5-7", distribution.range_5_7),
        ("7-9", distribution.range_7_9),
        (">=9.0", distribution.above_9),
    ]
    max_count = max(c for _, c in buckets) if any(c for _, c in buckets) else 1
    max_bar = 30
    lines = []
    for label, count in buckets:
        bar_len = round(count / max_count * max_bar) if max_count > 0 else 0
        bar = "\u2588" * bar_len + "\u2591" * (max_bar - bar_len)
        lines.append(f"  {label:>5} |{bar}| {count}")
    return "\n".join(lines)


def _render_markdown(review: Review) -> str:
    m = review.metrics
    d = review.distribution
    a = review.anomalies
    r = review.repro
    ref_hist = review.refinement_histogram
    threshold = settings.RANKING_PASS_THRESHOLD

    delta_pr = (
        f" (delta: {m.delta_pass_rate:+.1%})" if m.delta_pass_rate is not None else ""
    )
    delta_mo = (
        f" (delta: {m.delta_mean_overall:+.2f})"
        if m.delta_mean_overall is not None
        else ""
    )

    anomaly_lines = []
    if a.mid_graph_halt:
        anomaly_lines.append("- Mid-graph halt detected (not all 8 nodes completed)")
    if a.empty_challenge_ids:
        anomaly_lines.append("- Empty challenge_ids in manifest")
    if a.courses_missing:
        anomaly_lines.append("- No course.md files found in courses/")
    if a.ranking_missing:
        anomaly_lines.append("- ranking_reports.json missing")
    if a.llm_log_empty:
        anomaly_lines.append("- llm_calls.jsonl empty or missing")
    if a.errors_in_log:
        anomaly_lines.append("- Errors detected in llm_calls.jsonl")
    for v in a.repro_violations:
        anomaly_lines.append(f"- Repro violation: {v}")
    anomaly_section = "\n".join(anomaly_lines) if anomaly_lines else "_None detected._"

    repro_lines = [
        f"- Temperature: {r.temperature if r.temperature is not None else 'unknown'}",
        f"- Embedding provider: {r.embedding_provider}",
        f"- Judge model pinned: {'yes' if r.judge_model_pinned else 'no'}",
        f"- run_config.json captured: {'yes' if r.run_config_captured else 'no'}",
        f"- LangSmith trace: {'yes' if r.langsmith_trace else 'no'}",
    ]

    histogram = _histogram_ascii(d)
    refinement_histogram_md = _render_refinement_histogram(ref_hist)

    return f"""# REVIEW - {review.exp_id}

**Status**: {review.status}
**Provider**: {review.provider} | **Model**: {review.model}
**Judge**: {review.judge_model}
**Challenges**: {review.challenges_completed}/{review.challenges_total}
**Wall time**: {review.wall_time:.0f}s
**Cost**: {review.cost_estimate}

---

## 1. Metrics

| Metric | Value |
|--------|-------|
| Pass rate (>={threshold}) | {m.pass_rate:.1%}{delta_pr} |
| Passed / Total | {m.challenges_passed} / {m.challenges_total} |
| Mean overall | {m.mean_overall:.2f}{delta_mo} |
| Mean technical | {m.mean_technical:.2f} |
| Mean pedagogical | {m.mean_pedagogical:.2f} |
| Refinement rounds (avg) | {m.refinement_rounds_avg:.1f} |
| Terminology issues | {m.terminology_issues_total} |

## 2. Score distribution

```
{histogram}
```

## 3. Refinement round distribution

{refinement_histogram_md}

## 4. Anomalies

{anomaly_section}

## 5. Reproducibility audit

{chr(10).join(repro_lines)}

## 6. Samples

| Role | Challenge |
|------|-----------|
| Best | {review.best_course} |
| Worst | {review.worst_course} |
| Random | {review.random_sample} |

## 7. Checklist (user)

- [ ] Scores align with manual spot-check
- [ ] Anomalies investigated
- [ ] Reproducibility requirements met
- [ ] Ready to cite

---

_Auto-generated by review_generator.py_
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_review(exp_dir: Path) -> Path:
    """Generate REVIEW.md for the given experiment directory.

    Returns the path to the written REVIEW.md file.
    """
    exp_id = exp_dir.name
    logger.info(f"Generating REVIEW.md for {exp_id}")

    manifest = _parse_manifest(exp_dir)
    rankings = _parse_ranking_reports(exp_dir)
    overalls, _, _ = _extract_scores(rankings)

    snap = (manifest or {}).get("settings_snapshot", {})
    provider = snap.get("provider", "unknown")
    model_name = snap.get("model", "unknown")

    # For legacy experiments without manifest, try run_config.json
    if manifest is None:
        rc_path = exp_dir / "run_config.json"
        if rc_path.exists():
            try:
                rc = json.loads(rc_path.read_text(encoding="utf-8"))
                if provider == "unknown":
                    provider = rc.get("generator_model", "unknown")
                if model_name == "unknown":
                    model_name = rc.get("model", rc.get("generator_model", "unknown"))
            except (json.JSONDecodeError, OSError):
                pass

    # Resolve judge model from canonical sources (fix D13: was incorrectly reading
    # snap.get("model") which is the GENERATOR model, not the judge model).
    # Fallback chain (most authoritative first):
    #   1. reproducibility.json:settings.RANKING_MODEL
    #   2. reproducibility.json:cli_overrides.judge_model
    #   3. run_config.json:ranking_model (written by some experiment runners)
    #   4. manifest.settings_snapshot.judge_model (not written by ArtifactWriter, but
    #      may exist in manually constructed manifests)
    #   5. "unknown"
    judge_model = snap.get("judge_model", "unknown")
    # Try reproducibility.json first (most reliable post-7ae70f9+bdc4f80)
    repro_path = exp_dir / "reproducibility.json"
    if repro_path.exists():
        try:
            repro = json.loads(repro_path.read_text(encoding="utf-8"))
            ranking_model = repro.get("settings", {}).get("RANKING_MODEL", "")
            if ranking_model:
                judge_model = ranking_model
            else:
                cli_judge = repro.get("cli_overrides", {}).get("judge_model", "")
                if cli_judge:
                    judge_model = cli_judge
        except (json.JSONDecodeError, OSError):
            pass
    # Fallback to run_config.json:ranking_model if still unknown
    if judge_model == "unknown":
        rc_path = exp_dir / "run_config.json"
        if rc_path.exists():
            try:
                rc = json.loads(rc_path.read_text(encoding="utf-8"))
                judge_model = rc.get("ranking_model", judge_model)
            except (json.JSONDecodeError, OSError):
                pass

    prev_dir = _find_last_same_model(exp_dir, provider, model_name)
    prev_review_path = (prev_dir / "REVIEW.json") if prev_dir else None

    metrics = _compute_metrics(rankings, manifest, prev_review_path)
    distribution = _compute_distribution(overalls)
    anomalies = _detect_anomalies(exp_dir, manifest, rankings)
    repro = _repro_audit(manifest, exp_dir)
    ref_histogram = _compute_refinement_histogram(rankings)
    best, worst, sample = _select_samples(rankings, exp_id)

    status = (manifest or {}).get("status", "unknown")
    if status == "unknown" and rankings:
        status = "complete (inferred)"

    # F — wall_time: prefer (finished_at - started_at) delta; fall back to sum(node_timings)
    wall_time = 0.0
    if manifest and manifest.get("started_at") and manifest.get("finished_at"):
        from datetime import datetime

        try:
            start_dt = datetime.fromisoformat(manifest["started_at"])
            end_dt = datetime.fromisoformat(manifest["finished_at"])
            wall_time = (end_dt - start_dt).total_seconds()
        except (ValueError, TypeError):
            pass
    if wall_time == 0.0:
        node_timings = (manifest or {}).get("node_timings", {})
        if node_timings:
            wall_time = sum(node_timings.values())

    cost_estimate = "--"
    if provider == "claude-code":
        cost_estimate = "claude-code (Pro plan)"

    ch_completed = len(rankings)
    ch_total = len((manifest or {}).get("challenge_ids", [])) or ch_completed

    review = Review(
        exp_id=exp_id,
        status=status,
        provider=provider,
        model=model_name,
        judge_model=judge_model,
        challenges_completed=ch_completed,
        challenges_total=ch_total,
        wall_time=wall_time,
        cost_estimate=cost_estimate,
        metrics=metrics,
        distribution=distribution,
        anomalies=anomalies,
        repro=repro,
        refinement_histogram=ref_histogram,
        best_course=best,
        worst_course=worst,
        random_sample=sample,
    )

    md_content = _render_markdown(review)
    review_md_path = exp_dir / "REVIEW.md"
    review_md_path.write_text(md_content, encoding="utf-8")

    review_json_path = exp_dir / "REVIEW.json"
    review_json_path.write_text(review.model_dump_json(indent=2), encoding="utf-8")

    logger.info(f"REVIEW.md written to {review_md_path}")
    return review_md_path
