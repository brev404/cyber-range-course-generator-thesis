"""Optimal variant — minimal content prompt + calibrated judge prompts.

Combines the two best-performing variant components from internal evaluation:
- WRITEUP_SYSTEM: 5-rule minimal prompt (from 'minimal' variant) — simpler instructions
  produce better courses than the 13-rule baseline
- TECHNICAL_SYSTEM: Calibrated with scoring exemplars (from 'calibrated' variant) —
  scale anchors reduce intra-judge variance
- PEDAGOGICAL_SYSTEM: Calibrated with scoring exemplars (from 'calibrated' variant) —
  same treatment for pedagogical persona

The hypothesis: combining minimal generation prompt with calibrated judge prompts
should exceed either configuration individually.
"""

from src.prompts.variants.calibrated import PEDAGOGICAL_SYSTEM, TECHNICAL_SYSTEM
from src.prompts.variants.minimal import WRITEUP_SYSTEM

__all__ = ["WRITEUP_SYSTEM", "TECHNICAL_SYSTEM", "PEDAGOGICAL_SYSTEM"]
