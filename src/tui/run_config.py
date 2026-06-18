"""RunConfig — pure data contract for a TUI pipeline run. No UI, no I/O."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class RunConfig:
    """Immutable description of a single pipeline experiment run."""

    exp_id: str
    provider: str
    model: str
    temperature: float
    threshold: float
    challenge_ids: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    source: Literal["local", "processed"] = "local"
    max_refinements: int = 5
    skip_ranking: bool = False
    purpose: str | None = None
    note: str | None = None
    multi_judges: list[tuple[str, str]] | None = None  # parsed --judges list
