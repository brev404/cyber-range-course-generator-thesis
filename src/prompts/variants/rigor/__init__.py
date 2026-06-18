"""Rigor variant: baseline course prompt + a high-salience technical-rigor appendix.

The baseline already contains the right rules but compliance decays on hard multi-step
tasks (salience loss). This appendix hoists the most-violated technical rules into a
short, prominent checklist and adds the two genuinely-missing asks (state-assumptions /
no-vague-references; robustness).

Judge prompts (TECHNICAL_SYSTEM, PEDAGOGICAL_SYSTEM) are byte-for-byte baseline so the
rankings stay comparable; only the generator's course prompt changes.
"""

from src.agents.content_generation_agent import _WRITEUP_SYSTEM as _BASE_WRITEUP
from src.agents.ranking_agent import _PEDAGOGICAL_SYSTEM as PEDAGOGICAL_SYSTEM
from src.agents.ranking_agent import _TECHNICAL_SYSTEM as TECHNICAL_SYSTEM

_RIGOR_APPENDIX = """

---
## TECHNICAL RIGOR CHECKLIST — your course is judged primarily on these (pedagogy is assumed). Self-check every item before finishing:

R1. **Completeness on multi-step challenges.** Cover the FULL solution path end to end — every step needed to go from the challenge inputs to the flag. Do not stop at the interesting part and leave setup, data acquisition, or the final extraction implicit. If the challenge has N stages, all N appear with working detail.

R2. **Define before use; state assumptions.** Define every term, value, file, and check the moment you first reference it. Never use a vague forward-reference ("a secondary validation", "some processing", "the helper"). State explicitly what is GIVEN vs DERIVED vs ASSUMED. If a value's origin is not obvious, say exactly where it comes from.

R3. **Concrete grounding — show, don't describe.** For every observable step, show a concrete example artifact: a real sample line of output, an actual command with its flags, a sample API/JSON response, a real packet/field value. Replace "you should see the result" with the literal thing the student should see. Generic descriptions score low.

R4. **Narrative and solver MUST agree.** The technique described in the Thought-process and Step-by-step sections must be exactly what the Section 9 solver implements — same approach, same order, same key operations. No divergence.

R5. **No stubs, placeholders, or hardcoded answers in the solver.** No empty arrays, no `<paste here>`, no hardcoded flag, no reading an answer file. Unknown values must be obtained programmatically (a `fetch_*()` function) with an inline comment on how.

R6. **Robustness.** The solver should handle the obvious failure mode (service down, value missing) with a clear error, not a silent wrong answer or a fallback that fabricates the flag.
"""

WRITEUP_SYSTEM = _BASE_WRITEUP + _RIGOR_APPENDIX

__all__ = ["WRITEUP_SYSTEM", "TECHNICAL_SYSTEM", "PEDAGOGICAL_SYSTEM"]
