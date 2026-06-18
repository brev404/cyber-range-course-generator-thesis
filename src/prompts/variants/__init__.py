"""Prompt variant registry.

Each variant is a sub-package exposing WRITEUP_SYSTEM, TECHNICAL_SYSTEM,
PEDAGOGICAL_SYSTEM as module-level string constants. VARIANT_NAMES lists
all registered variant names (excluding 'baseline' which is implicit).
"""

VARIANT_NAMES: tuple[str, ...] = (
    "duplicate",
    "over_explain",
    "under_explain",
    "cot_scaffolding",
    "terse",
    "normal",
    "verbose",
    "rigor",
    "minimal",
    "calibrated",
    "optimal",
)
