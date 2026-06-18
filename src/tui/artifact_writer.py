"""ArtifactWriter — incremental output to output/experiments/{exp_id}/.

Writes all experiment artifacts during a pipeline run:
  manifest.json         — written on start_run (status=running), updated on finish_run
  run_config.json       — written on start_run from cfg
  courses/{cat}/{name}/course.md — written via write_course
  llm_calls.jsonl       — appended via append_llm_call (one JSON line per call)
  ranking_reports.json  — written via write_ranking (overwrites)
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from src.tui.run_config import RunConfig


@dataclass
class ManifestData:
    """Schema for manifest.json written alongside each experiment run."""

    exp_id: str
    status: str  # "running" | "complete" | "failed" | "aborted"
    started_at: str
    finished_at: str | None
    challenge_ids: list[str] = field(default_factory=list)
    node_timings: dict[str, float] = field(default_factory=dict)
    pass_count: int = 0
    mean_overall_score: float = 0.0
    settings_snapshot: dict = field(default_factory=dict)


class ArtifactWriter:
    """Writes experiment artifacts incrementally during a pipeline run."""

    def __init__(self, base_dir: Path, cfg: RunConfig) -> None:
        self._base = base_dir / cfg.exp_id
        self._cfg = cfg
        self._manifest: ManifestData | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_run(self, cfg: RunConfig) -> None:
        """Create output directory structure and write initial manifest.json and run_config.json."""
        self._base.mkdir(parents=True, exist_ok=True)

        # Capture launch-time human intent so closeout recording is fully automatic.
        purpose = getattr(cfg, "purpose", None)
        note = getattr(cfg, "note", None)
        if purpose or note:
            (self._base / "intent.json").write_text(
                json.dumps({"purpose": purpose, "note": note}, indent=2),
                encoding="utf-8",
            )

        now = datetime.now(timezone.utc).isoformat()
        self._manifest = ManifestData(
            exp_id=cfg.exp_id,
            status="running",
            started_at=now,
            finished_at=None,
            challenge_ids=list(cfg.challenge_ids),
            settings_snapshot={
                "provider": cfg.provider,
                "model": cfg.model,
                "temperature": cfg.temperature,
                "threshold": cfg.threshold,
                "max_refinements": cfg.max_refinements,
                "skip_ranking": cfg.skip_ranking,
                "source": cfg.source,
                "categories": list(cfg.categories),
            },
        )
        self._write_manifest()
        self._write_run_config(cfg)
        logger.info(f"ArtifactWriter: started run {cfg.exp_id} at {self._base}")

    def update_challenge_ids(self, ids: list[str]) -> None:
        """Replace manifest challenge_ids with the resolved (discovered) IDs and re-write."""
        if self._manifest is not None:
            self._manifest.challenge_ids = list(ids)
            self._write_manifest()
            logger.debug(
                f"ArtifactWriter: updated manifest challenge_ids → {len(ids)} entries"
            )

    def write_course(self, challenge_id: str, content: str) -> None:
        """Write courses/{category}/{name}/course.md.

        challenge_id format: "category/name" or just "name" (uses "misc" as default category).
        Skips writing if content is None, empty, whitespace-only, or the literal
        string "None". `str(content)` keeps the guard and the all-empty check
        consistent and avoids AttributeError on a non-str value.
        """
        _stripped = str(content).strip() if content is not None else ""
        if not _stripped or _stripped == "None":
            logger.warning(
                "ArtifactWriter: skipping empty/None course content for {} — course.md not written",
                challenge_id,
            )
            return

        if "/" in challenge_id:
            category, name = challenge_id.split("/", 1)
        else:
            category, name = "misc", challenge_id

        course_path = self._base / "courses" / category / name / "course.md"
        from src.services.atomic_write import atomic_write_text

        atomic_write_text(course_path, str(content))
        logger.debug(f"ArtifactWriter: wrote course to {course_path}")

    def append_llm_call(
        self,
        node: str,
        prompt: str,
        tokens: list[str],
        duration_s: float,
    ) -> None:
        """Append one JSON line to llm_calls.jsonl."""
        self._base.mkdir(parents=True, exist_ok=True)
        record = {
            "node": node,
            "prompt": prompt,
            "response_tokens": tokens,
            "duration_s": duration_s,
        }
        jsonl_path = self._base / "llm_calls.jsonl"
        with jsonl_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        logger.debug(f"ArtifactWriter: appended llm_call for node={node!r}")

    def write_ranking(self, reports: list | dict) -> None:
        """Write ranking_reports.json (overwrites on each call).

        Normalises challenge_id values in each report entry to the canonical
        ``category/name`` format used by manifest.challenge_ids (fix D11).

        Historical note: ranking_agent previously produced ``name/source`` IDs
        (e.g. ``0_solves/cyberedu``) while manifest used ``category/name``
        (e.g. ``crypto/0_solves``). Earlier experiment outputs were written
        before this normalisation was in place — do NOT retroactively migrate
        those files.
        """
        self._base.mkdir(parents=True, exist_ok=True)
        normalised = self._normalise_ranking_ids(reports)
        ranking_path = self._base / "ranking_reports.json"
        from src.services.atomic_write import atomic_write_json

        atomic_write_json(ranking_path, normalised)
        logger.debug(f"ArtifactWriter: wrote ranking reports to {ranking_path}")

    def _normalise_ranking_ids(self, reports: list | dict) -> list | dict:
        """Return reports with challenge_id normalised to ``category/name``.

        The ranking agent may emit IDs as ``name/source`` (e.g. ``rsb/cyberedu``).
        Manifest challenge_ids use ``category/name`` (e.g. ``crypto/rsb``).
        This method builds an inverse-lookup from challenge name → canonical id
        using the current manifest and rewrites each report's challenge_id.
        If the manifest is unavailable or a report's id is already canonical,
        the original value is preserved.
        """
        if not isinstance(reports, list):
            return reports

        canonical_ids: list[str] = []
        if self._manifest is not None:
            canonical_ids = self._manifest.challenge_ids

        # Build lookup: bare challenge name → canonical "category/name"
        # Only entries in "category/name" form are trusted as canonical.
        name_to_canonical: dict[str, str] = {}
        for cid in canonical_ids:
            parts = cid.split("/")
            if len(parts) == 2:
                _cat, name = parts
                name_to_canonical[name] = cid

        normalised_reports = []
        for report in reports:
            if not isinstance(report, dict):
                normalised_reports.append(report)
                continue
            raw_id = report.get("challenge_id", "")
            # Already canonical ("category/name" where category is known)?
            if raw_id in canonical_ids:
                normalised_reports.append(report)
                continue
            # Try to resolve via bare name lookup (handles "name/source" format)
            bare_name = raw_id.split("/")[0] if "/" in raw_id else raw_id
            if bare_name in name_to_canonical:
                report = dict(report)
                report["challenge_id"] = name_to_canonical[bare_name]
            normalised_reports.append(report)
        return normalised_reports

    def finish_run(
        self, success: bool, node_timings: dict, status: str | None = None
    ) -> None:
        """Update manifest.json status to 'complete' or 'failed'.

        Also computes pass_count and mean_overall_score from ranking_reports.json
        (if present) so the manifest reflects real ranking results.

        Rule: only entries with a numeric overall_score (not None) are counted.
        Empty ranking_reports or missing file → both fields stay at 0.
        Entries where ranking was skipped or errored keep overall_score from the
        error default (5.0), which is below the pass threshold — consistent behavior.
        """
        if self._manifest is None:
            logger.warning(
                "ArtifactWriter.finish_run called before start_run — creating minimal manifest"
            )
            self._base.mkdir(parents=True, exist_ok=True)
            now = datetime.now(timezone.utc).isoformat()
            self._manifest = ManifestData(
                exp_id=self._cfg.exp_id,
                status="running",
                started_at=now,
                finished_at=None,
            )

        self._manifest.status = (
            status if status is not None else ("complete" if success else "failed")
        )
        self._manifest.finished_at = datetime.now(timezone.utc).isoformat()
        self._manifest.node_timings = dict(node_timings)

        # Populate pass_count / mean_overall_score from ranking_reports.json.
        # We read the file rather than relying on in-memory state so this also
        # works if ranking_reports was written in a previous partial run.
        ranking_path = self._base / "ranking_reports.json"
        if ranking_path.exists():
            try:
                reports = json.loads(ranking_path.read_text(encoding="utf-8"))
                if isinstance(reports, list) and reports:
                    from src.config.settings import settings as _settings

                    threshold = float(getattr(_settings, "RANKING_PASS_THRESHOLD", 9.0))
                    # Only use entries with a valid numeric overall_score.
                    valid_scores = [
                        float(r["overall_score"])
                        for r in reports
                        if isinstance(r, dict) and r.get("overall_score") is not None
                    ]
                    if valid_scores:
                        self._manifest.pass_count = sum(
                            1 for s in valid_scores if s >= threshold
                        )
                        self._manifest.mean_overall_score = round(
                            sum(valid_scores) / len(valid_scores), 4
                        )
            except Exception as exc:
                logger.warning(
                    "ArtifactWriter: failed to compute ranking summary for manifest: {}",
                    exc,
                )

        self._write_manifest_atomic()

        try:
            from src.services.review_generator import generate_review
            from src.services.review_queue_updater import append_to_review_queue

            review_path = generate_review(self._base)
            append_to_review_queue(
                exp_id=self._cfg.exp_id,
                review_path=review_path,
                summary=f"{self._manifest.status} | {self._manifest.settings_snapshot.get('model', 'unknown')}",
            )
        except Exception as exc:
            logger.warning(
                f"ArtifactWriter: review generation failed (non-fatal): {exc}"
            )

        logger.info(
            f"ArtifactWriter: finished run {self._cfg.exp_id} "
            f"status={self._manifest.status}"
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _write_manifest(self) -> None:
        """Serialize and overwrite manifest.json (non-atomic, for intermediate writes)."""
        assert self._manifest is not None
        manifest_path = self._base / "manifest.json"
        manifest_path.write_text(
            json.dumps(asdict(self._manifest), indent=2),
            encoding="utf-8",
        )

    def _write_manifest_atomic(self) -> None:
        """Atomically write manifest.json via .tmp + fsync + rename.

        Used for the final write in finish_run so a concurrent reader never
        sees a partially-written manifest.
        """
        assert self._manifest is not None
        manifest_path = self._base / "manifest.json"
        tmp_path = self._base / "manifest.json.tmp"
        payload = json.dumps(asdict(self._manifest), indent=2).encode("utf-8")
        try:
            fd = os.open(str(tmp_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
            try:
                os.write(fd, payload)
                os.fsync(fd)
            finally:
                os.close(fd)
            os.rename(str(tmp_path), str(manifest_path))
        except OSError as exc:
            logger.warning(
                "ArtifactWriter: atomic manifest write failed ({}); falling back to direct write",
                exc,
            )
            manifest_path.write_text(payload.decode("utf-8"), encoding="utf-8")

    def _write_run_config(self, cfg: RunConfig) -> None:
        """Serialize RunConfig to run_config.json."""
        run_config_path = self._base / "run_config.json"
        data = {
            "exp_id": cfg.exp_id,
            "provider": cfg.provider,
            "model": cfg.model,
            "temperature": cfg.temperature,
            "threshold": cfg.threshold,
            "challenge_ids": list(cfg.challenge_ids),
            "categories": list(cfg.categories),
            "source": cfg.source,
            "max_refinements": cfg.max_refinements,
            "skip_ranking": cfg.skip_ranking,
        }
        run_config_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
