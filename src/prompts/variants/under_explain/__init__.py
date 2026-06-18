"""Under-explain variant: minimal generator prompt (~30% of baseline length).

Retains only persona, output structure skeleton, and output-format instruction.
No quality standards, no anti-patterns, no framing guidance.
Judge prompts are baseline (this tests generator behavior only).
"""

from src.agents.ranking_agent import _PEDAGOGICAL_SYSTEM as PEDAGOGICAL_SYSTEM
from src.agents.ranking_agent import _TECHNICAL_SYSTEM as TECHNICAL_SYSTEM

WRITEUP_SYSTEM = """You are a Cybersecurity Expert Educator.

Write a pedagogical course for a CTF challenge using this structure:

1. Title and context
2. Abstract / TL;DR
3. Objectives
4. Technical skills
5. Definitions and concepts
6. Reproducibility (Step 0)
7. Thought process / narrative
8. Step-by-step resolution
9. Solution script
10. Conclusion
11. Extra resources

Output Markdown only, no preamble, no wrapping code fence."""

__all__ = ["WRITEUP_SYSTEM", "TECHNICAL_SYSTEM", "PEDAGOGICAL_SYSTEM"]
