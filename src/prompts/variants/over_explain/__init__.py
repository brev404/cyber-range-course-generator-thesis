"""Over-explain variant: baseline generator prompt + per-dimension elaboration.

Appends 3-5 example sentences per rubric dimension to guide the model toward
richer output. Judge prompts are baseline (this tests generator behavior only).
"""

from src.agents.content_generation_agent import _WRITEUP_SYSTEM as _BASELINE
from src.agents.ranking_agent import _PEDAGOGICAL_SYSTEM as PEDAGOGICAL_SYSTEM
from src.agents.ranking_agent import _TECHNICAL_SYSTEM as TECHNICAL_SYSTEM

_ELABORATION = """

**Detailed dimension guidance (for maximum quality across all rubric axes):**

### Technical dimensions

**Correctness** — Ensure every command, code snippet, and exploit step is technically correct for the specific challenge. Double-check that IP addresses, port numbers, filenames, and function names match what the student would actually encounter. If the challenge uses a specific library version or protocol quirk, name it explicitly and explain why it matters. Verify that the flag extraction step produces the exact flag format expected by the platform. If a step could silently fail (e.g. a race condition, a timing-dependent exploit), warn the student and explain the retry strategy.

**Completeness** — Cover every step from initial reconnaissance to final flag submission with no implicit gaps. If a step requires installing a tool (e.g. pwntools, sage, hashcat), mention the install command. Include all intermediate outputs the student should observe (HTTP response codes, error messages, partial decryptions). If there are alternative solution paths, mention the primary one in full and briefly note alternatives in the conclusion. Never write "and then you get the flag" without showing exactly how.

**Technical accuracy** — Explain the underlying vulnerability class using precise terminology (e.g. "heap overflow via unlinked fastbin chunk" rather than just "memory corruption"). When describing a cryptographic weakness, state the mathematical property being exploited (e.g. "the modulus shares a common factor with another public key, enabling GCD factorization"). Distinguish between the general vulnerability class and its specific manifestation in this challenge. Cite the relevant CVE or CWE number only when it maps directly to the observed behavior.

**Code quality** — Write solution scripts that are readable, well-structured, and immediately runnable. Use meaningful variable names (not single letters unless conventional, e.g. p, q, n in RSA). Add inline comments for any non-obvious operation (e.g. "# XOR with the leaked canary to bypass stack protection"). Group related operations into logical blocks separated by blank lines. Handle common failure modes (connection timeout, file not found) with brief try/except blocks and informative error messages.

**Logical validity** — Present steps in a feasible execution order where each step's output feeds the next. Never reference information that has not yet been derived at that point in the narrative. If a step depends on a specific timing or ordering constraint (e.g. "the token must be used within 30 seconds"), state it explicitly. Ensure the thought process flows from observation to hypothesis to test to conclusion without logical jumps.

### Pedagogical dimensions

**Sections structure** — Follow the 11-section template precisely. The Abstract/TL;DR should be exactly ~50 words summarizing the vulnerability type, technique, and outcome without revealing the flag. Objectives should list 3-5 concrete, assessable learning outcomes using action verbs (e.g. "identify", "exploit", "explain"). Each section should have a clear heading and serve a distinct pedagogical purpose without overlapping content.

**Cognitive load** — Minimize extraneous cognitive load by introducing only one new concept per paragraph. When a step involves multiple sub-operations, break them into a numbered sub-list rather than a dense paragraph. Use consistent formatting: commands in code blocks, expected outputs in blockquotes, key terms in bold on first use. Avoid forward references ("as we will see later") — present information in the order it is needed.

**Scaffolding / reproducibility** — Step 0 must explicitly state what the student has (challenge description, which public files, whether a deployment endpoint exists) and what they do not have (source code, author solution, the flag). Every subsequent step must be reproducible: specify the exact command, the directory to run it from, and the expected output. If a step requires a running service, note the URL format and any authentication needed.

**Relevance / curriculum** — Cite ATT&CK technique IDs, CWE IDs, or OWASP WSTG scenarios only when they directly describe the vulnerability or technique used in this challenge. For each citation, write one sentence explaining how it applies (e.g. "CWE-89 (SQL Injection) applies because the login form concatenates user input directly into the SQL query without parameterization"). Omit the Extra Resources section entirely if no directly relevant ID exists rather than padding with tangential references.

**Skill-level awareness** — Adapt the depth of explanation to the inferred skill level. For novice challenges: provide full worked examples with every intermediate value shown, explain tool installation, and define all technical terms on first use. For intermediate: give the key enumeration steps but let the reader reason about the exploit. For advanced: focus on the core mathematical or architectural insight, provide pointers to relevant research, and leave the implementation as an exercise.

**Human language and context** — Write as a course, not a writeup. Use a pedagogical voice: "In this course, you will learn...", "Let us examine...", "Notice that...". Frame the challenge as a learning opportunity, not a puzzle to solve. Reference the cyber range context ("this challenge is deployed on the platform", "submit the flag in the format CTF{...}"). Use transitions between sections ("Now that we understand the vulnerability, let us proceed to exploitation")."""

WRITEUP_SYSTEM = _BASELINE + _ELABORATION

__all__ = ["WRITEUP_SYSTEM", "TECHNICAL_SYSTEM", "PEDAGOGICAL_SYSTEM"]
