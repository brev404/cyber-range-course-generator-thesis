"""CoT scaffolding variant: baseline + internal reasoning steps before output.

Inserts a 5-step reasoning block between the Structure and Quality standards
sections. The model should perform these steps internally (not as visible output).
Judge prompts are baseline (this tests generator behavior only).
"""

from src.agents.content_generation_agent import _WRITEUP_SYSTEM as _BASELINE
from src.agents.ranking_agent import _PEDAGOGICAL_SYSTEM as PEDAGOGICAL_SYSTEM
from src.agents.ranking_agent import _TECHNICAL_SYSTEM as TECHNICAL_SYSTEM

_COT_BLOCK = """

**Reasoning steps before output** (perform internally, do not include in the course text):

1. Identify the vulnerability class and its underlying mathematical, logical, or architectural mechanism. Name it precisely (e.g. "heap use-after-free via dangling pointer in the tcache", not just "memory bug").
2. List what the student observably has: challenge description text, any public files enumerated in the challenge, and whether a live deployment endpoint is available. Do not assume access to source code, author writeup, or the flag.
3. Sketch a 3-stage discovery narrative for the course: reconnaissance (what the student examines first and why), hypothesis (what vulnerability or weakness the observations suggest), test-and-confirm (the exploit or technique that validates the hypothesis and yields the flag).
4. For each resolution step, map it to one CWE, ATT&CK technique, or OWASP WSTG scenario if (and only if) the mapping is directly and specifically relevant to the observed behavior. Do not force a citation where none fits.
5. Verify that the solution script section will contain the complete, runnable solver script — not a reference to "the code above" or "see previous step". If the script requires external libraries, include the import statements and a brief install note.

After completing these reasoning steps internally, write the course following the Structure and Quality standards below."""

_QUALITY_MARKER = "\n\n**Quality standards — hard rules:**"

_split_idx = _BASELINE.find(_QUALITY_MARKER)
WRITEUP_SYSTEM = _BASELINE[:_split_idx] + _COT_BLOCK + _BASELINE[_split_idx:]

__all__ = ["WRITEUP_SYSTEM", "TECHNICAL_SYSTEM", "PEDAGOGICAL_SYSTEM"]
