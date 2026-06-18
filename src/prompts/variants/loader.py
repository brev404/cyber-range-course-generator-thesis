"""Load a prompt variant by name, falling back to baseline constants."""

from __future__ import annotations

import importlib
from typing import Optional

from src.prompts.variants import VARIANT_NAMES


def load_variant(name: Optional[str] = None) -> dict[str, str]:
    """Return {"writeup": ..., "technical": ..., "pedagogical": ...} for a variant.

    When *name* is None or "baseline", returns the baseline constants from the
    agent modules (single source of truth).
    """
    if name is None or name == "baseline":
        from src.agents.content_generation_agent import _WRITEUP_SYSTEM
        from src.agents.ranking_agent import _PEDAGOGICAL_SYSTEM, _TECHNICAL_SYSTEM

        return {
            "writeup": _WRITEUP_SYSTEM,
            "technical": _TECHNICAL_SYSTEM,
            "pedagogical": _PEDAGOGICAL_SYSTEM,
        }

    if name not in VARIANT_NAMES:
        raise ValueError(
            f"unknown prompt variant: {name!r} "
            f"(available: {VARIANT_NAMES + ('baseline',)})"
        )

    mod = importlib.import_module(f"src.prompts.variants.{name}")
    return {
        "writeup": mod.WRITEUP_SYSTEM,
        "technical": mod.TECHNICAL_SYSTEM,
        "pedagogical": mod.PEDAGOGICAL_SYSTEM,
    }
