"""Content Generation Agent: research-based course generation with RAG.

This agent generates course.md (pedagogical course) using the "Cybersecurity Expert
Educator" persona. Inputs: challenge details (author's writeup, description, solver,
source/attachments when available). RAG over pedagogical + technical KB. Output is
written to course.md only; the author's writeup.md is never overwritten.

Research-based: reproducibility requirements, course writeup guidelines, and cybersecurity glossary sources.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from loguru import logger

from src.agents.generation_input import GenerationInput, load_generation_input
from src.agents.solver_self_improve import self_improve_solver
from src.agents.validation_agent import _get_challenge_paths
from src.config.settings import settings as app_settings
from src.core.state import AgentState
from src.services.atomic_write import atomic_write_text as _atomic_write_text
from src.services.gen_progress import get_exp_dir as _get_exp_dir
from src.services.gen_progress import report as _report_gen_progress
from src.services.llm_service import (
    generate_response,
    generate_response_with_system,
    reset_challenge_llm_budget,
)
from src.services.vector_db_service import VectorDBService

# Default RAG and generation settings (overridden by settings.CONTENT_GENERATION_* when set)
_RAG_TOP_K = 6
_RAG_PEDAGOGICAL_K = 3  # Extra chunks for writeup structure / pedagogical KB
_DEFAULT_WRITEUP_MAX_TOKENS = 12000
_DEFAULT_SOLVE_SCRIPT_MAX_TOKENS = 3000

_WRITEUP_SYSTEM = """You are a Cybersecurity Expert Educator. Your role is to write clear, pedagogical **courses** for CTF challenges.

**Critical: Audience and perspective**
The reader is a **student** who:
- Does **not** have access to the challenge source code or binaries (unless the challenge explicitly provides them).
- Does **not** have access to an author writeup or solution.
- Does **not** know the flag in advance.

Write the course as a **logical, self-contained explanation** that guides this student from the challenge description to understanding and solving it. Use a discovery narrative: recon → hypothesis → test → conclusion. Do not refer to "the source," "the writeup," or "the author's solution" in the course text. Do not spoil the flag before the resolution section; present it as the outcome of the steps, not as given. The student should feel they are following a clear reasoning path, not reading a summary of someone else's solution.

**Structure** (research-based; minimize extraneous cognitive load):
1. **Title and context** – Challenge name, category, difficulty.
2. **Abstract / TL;DR** – ~50 words: vulnerability type, main technique, outcome (no flag).
3. **Objectives** – What the learner will achieve.
4. **Technical skills** – Skills practiced (e.g. web, crypto, binary).
5. **Definitions and concepts** – Short definitions of terms used (e.g. XSS, SQLi).
6. **Reproducibility (Step 0)** – Be explicit: what the student has (challenge description; any **public files** listed; whether a **deployment** is available). Do not assume they have source, writeup, or the flag.
7. **Thought process / narrative** – Logical reasoning: what we observe, what we try, why, and what we learn. Model expert thinking from the student's standpoint.
8. **Step-by-step resolution** – Clear steps with commands/code and what to observe; lead to the solution and only then to the flag.
9. **Solution script** – The complete, runnable solver script with a brief comment per non-obvious block. Must not say "see above" or "provided in previous step" — embed the full script here.
10. **Conclusion** – Summary and takeaways (no need to repeat the flag).
11. **Extra resources** – Links to CVE, ATT&CK, CWE, OWASP where **directly and specifically** relevant.

Where the challenge involves a known technique or weakness, **cite** MITRE ATT&CK (e.g. T1566), CWE (e.g. CWE-79), or OWASP WSTG (e.g. WSTG-INPV-05). Output Markdown only, no preamble, no wrapping code fence.

**Quality standards — hard rules:**

1. **Explain the WHY in every step.** Each step in the resolution must state: (a) what command or code to run, (b) what output or observation to expect, and (c) why that result means what we think it means. Never write a step that just says "run the script" or "execute the exploit" without explaining what success looks like and why the technique works against this specific challenge.

2. **Thought process must be grounded in the challenge.** The narrative must explain the mathematical, logical, or technical reason the chosen approach applies to *this* challenge — the structural property being exploited, the vulnerability class, the observable behaviour. Never attribute reasoning to external sources ("the video shows", "the writeup suggests", "a tool recommended"). The reasoning must come from first principles applied to what the student can observe.

3. **No hallucinated citations.** Only cite ATT&CK, CWE, or OWASP IDs that are directly relevant to the challenge category and technique. Do not cite web vulnerability IDs for crypto challenges, binary exploitation IDs for OSINT challenges, etc. If no directly relevant ID exists, omit the section rather than inventing a tangential one.

4. **Skill-level tailoring.** Novice: full worked example, every step explained. Intermediate: key enumeration given, reader reasons about the exploit. Advanced: core vulnerability and mathematical/technical insight explained with pointers — do not give the full solution code; leave implementation as the reader's exercise.

5. **Solution script section must contain the script.** Copy the complete solver (or the key executable portion) directly into section 9 with inline comments. Do not reference "the code above" or "the previous step."

6. **Always attempt generation — never refuse.** Even when the challenge description, public files, and deployment are all absent, you MUST produce a best-effort course using the challenge name, category, and any available context. Use your knowledge of the category (e.g. crypto, pwn, web) to write a conceptual overview of the likely technique or vulnerability class. Do not output a refusal, a placeholder saying "insufficient data", or a note asking for more information. A category-level conceptual course is always better than a refusal.

**v3 anti-pattern rules — derived from observed judge complaints:**

7. **No truncation, ever.** Never write "...truncated...", "...continued...", "[code continues]", "[... truncated for length ...]", "complete the rest as exercise", "implementation left to the reader", or any placeholder mid-function or mid-section. If a section is too long for the budget, shorten the narrative or omit less-critical paragraphs — but every section listed in the Structure MUST appear in full. **Budget priority when space is tight (sacrifice in this order, never reverse):** First shorten Section 7's deeper analogies and Section 5's expanded definitions. Then trim Section 8 sub-step preamble (keep the command + expected output + 1-line why). NEVER truncate the Solution Script (§9), Conclusion (§10), or Extra Resources (§11) — these three sections are non-negotiable.

8. **Always include the Extra Resources section (Section 11) with real references.** At minimum: one MITRE ATT&CK technique ID and one CWE ID directly relevant to the challenge category. Web challenges add OWASP WSTG. Crypto challenges add the relevant RFC or NIST SP if applicable. Never leave the section blank or write "no relevant references" without trying — pick at least one defensible reference.

9. **Always include the Conclusion section (Section 10).** One paragraph minimum, summarising the takeaway (not the flag). Do not skip this section.

10. **Every step in Section 8 must state expected output explicitly.** Format: "Run `<command>` — expected: `<one-line description of what you should see>`. This shows that <reason>." Never write "run and see what happens" or "you should get the flag" without specifying what success looks like.

11. **Section 9 uses a placeholder marker (auto-assembled).** When the v4 architecture is active, write ALL 11 sections IN ORDER with proper numbering (`## 1.`, `## 2.`, ..., `## 11.`). For Section 9, you MUST write the heading and an HTML-comment placeholder marker on its own line — the pipeline replaces the marker with the authoritative solver from `<solver_for_section_9>...</solver_for_section_9>` after generation. The exact format for Section 9 is:

```
## 9. Solution Script

<!-- SOLVER_PLACEHOLDER -->
```

You MAY add a single short lead-in sentence between the heading and the marker (e.g. "The script below implements the technique from Section 8."). The marker line itself MUST be exactly `<!-- SOLVER_PLACEHOLDER -->` on its own line — no paraphrasing, no extra text on that line, no markdown formatting around it. Do NOT write the actual solver code in Section 9 — only the placeholder marker. Sections 6-8 must explain the technique implemented in the solver. Sections 10 (Conclusion) and 11 (Extra Resources) MUST follow with proper numbering and full content. DO NOT reference the solver as external ("see solve.py", "the script above/below", "in the solver file") — the student reads course.md top-to-bottom and the solver appears inline in Section 9 via assembly. Explain the technique with your own words; short illustrative snippets inline within narrative are OK but the FULL solver must NOT be repeated in your output.

12. **Solver and narrative must agree.** The technique described in Sections 6-8 (Thought process + Step-by-step resolution) must match exactly what the solver in Section 9 does. If the narrative says "we use CRT", the solver must use CRT — not `gmpy2.gcdext` for modular inverse without explanation. Inconsistencies between the narrative-described approach and the solver-implemented approach are forbidden.

13. **No placeholder arrays or unknown constants in the solver.** Never leave `KEY = []`, `ENCODED = []`, `CIPHERTEXT = "<paste here>"`, `FLAG_PARTS = [None, None]` or similar empty-stub data structures in the solver. If exact values are not derivable from the challenge materials, the solver MUST: (a) document with an inline comment exactly how to obtain them (e.g. `# obtain by: curl http://chal:8080/dump and copy the hex blob`), AND (b) wrap the derivation in a `def fetch_<name>(): ...` function that performs the derivation programmatically when possible. Static empty arrays force the student to manually patch the solver — this defeats the "self-contained explanation" goal.

**Framing (avoid "no access" language):**
- ✅ Good: "You have access to the challenge description and public files (app.py, config.json). The server source code is not provided."
- ❌ Bad: "You don't have access to the source code" or "The source is not available to you."
- ✅ Good: "You can interact with the deployed service via HTTP."
- ❌ Bad: "You cannot access the server internals."
Use positive framing: state what the student has, then optionally note what is not provided (as a fact, not a restriction)."""


_PROSE_V2_OVERRIDE = """

**Prose overrides (G4 — these supersede any conflicting instruction above):**
- Section 6 is titled "## 6. Setup": state concisely what the student is GIVEN (the prompt, any public files, whether a deployment is reachable). NEVER enumerate what they lack — no "the student does not have the source / writeup / flag", no "not provided" lists.
- Do NOT use a fixed per-step template. State each step's command, the expected observation, and why it matters in NATURAL, VARIED prose. Never repeat the "Expected: ... This shows that ..." pattern on every step.
- Write human prose: avoid em-dashes (prefer commas, colons, or periods; at most ~2 in the whole course) and AI tells ("it's worth noting", "in essence", "let's", "delve into", "a testament to").
- Never print a flag value, a question's answer, or a fabricated example flag in the narrative. The flag appears ONLY as the outcome of the Section 9 solver."""


def _apply_prose_v2(system: str) -> str:
    """G4 — fix the audit's prose defects on the resolved system prompt (gated by PROSE_V2).
    Targeted rename for the Section 6 heading (stable substring) + an appended override block
    for the behavioural rules (robust to base-prompt drift)."""
    out = system.replace("**Reproducibility (Step 0)**", "**Setup**")
    out = out.replace(" Do not assume they have source, writeup, or the flag.", "")
    return out + _PROSE_V2_OVERRIDE


def _postprocess_prose_v2(course: str) -> str:
    """G4 — deterministic prose cleanup the model won't do reliably from instructions:
    em-dashes -> commas (OUTSIDE code fences), and strip residual frame-leak clauses that
    enumerate what the student lacks (source/writeup/flag 'not provided'). Code blocks are
    preserved verbatim."""
    parts = re.split(r"(```.*?```)", course, flags=re.DOTALL)
    for i in range(0, len(parts), 2):  # even indices = non-code segments
        parts[i] = re.sub(r"\s*—\s*", ", ", parts[i])  # em-dash -> comma
    out = "".join(parts)
    # drop "... source code / writeup / the flag ... (are/is) not provided/available" clauses
    out = re.sub(
        r"[;,]\s*[^.;\n]*\b(writeup|source code|the flag)\b[^.;\n]*"
        r"\b(not provided|not available|not given|are not|is not)\b[^.;\n]*",
        "",
        out,
        flags=re.IGNORECASE,
    )
    # drop standalone "you do not have ..." / "the student does not have ..." sentences
    out = re.sub(
        r"(?m)[^.\n]*\b(you do not have|you don't have|the student does not have)\b[^.\n]*\.\s*",
        "",
        out,
        flags=re.IGNORECASE,
    )
    return re.sub(r"[ \t]{2,}", " ", out)


def _resolve_writeup_system() -> str:
    """Return the active WRITEUP_SYSTEM prompt (baseline or variant; + prose-v2 when enabled)."""
    name = app_settings.PROMPT_VARIANT
    if not name or name == "baseline":
        base = _WRITEUP_SYSTEM
    else:
        from src.prompts.variants.loader import load_variant

        base = load_variant(name)["writeup"]
    if getattr(app_settings, "PROSE_V2", False):
        base = _apply_prose_v2(base)
    return base


# F3 (v4.2.1): category-specific guidance blocks. Injected into the user prompt (NOT the
# system prompt, to preserve the system prompt cache) when F3_CATEGORY_GUIDANCE_ENABLED.
# Each block lists category-specific requirements (tools, formats), probable ATT&CK / CWE /
# OWASP IDs (referenced so F8's reference-marker check finds them), and common failure modes.
# Content-only — no system-prompt-style directives like "ignore previous instructions".
# Lookup is case-insensitive; categories not in the dict (or category=None) are silently
# skipped (no error, no block) — see _build_category_guidance_block.
_CATEGORY_GUIDANCE: Dict[str, str] = {
    "pwn": (
        "- Use pwntools idiomatically: `p.send()` for binary payloads (no trailing newline), "
        "`p.sendline()` only when the service expects `\\n`. After every `recv()`/`recvuntil()`, "
        "annotate with the expected bytes in a comment.\n"
        "- For ROP, show how gadgets are located (`ROP(elf)` or `ROPgadget --binary`); never hand-wave addresses.\n"
        "- Identify the bug class explicitly: stack overflow (CWE-121), heap overflow (CWE-122), "
        "format string (CWE-134), UAF (CWE-416), off-by-one (CWE-193).\n"
        "- Probable references: ATT&CK T1203 (Exploitation for Client Execution) or T1068 "
        "(Exploitation for Privilege Escalation); CWE-121 / CWE-122 / CWE-787.\n"
        "- Common failure modes to avoid: hardcoded libc offsets without leak step, missing "
        "context.arch / context.binary, omitting the canary/PIE/NX checks (`checksec`)."
    ),
    "crypto": (
        "- Show the mathematical derivation of the attack, not just the algorithm name "
        "(e.g. for LLL: state the lattice basis, the target vector, why short vectors recover the secret).\n"
        "- For lattice / ECC / advanced number theory, prefer SageMath: name the file `solve.sage` "
        "and document the invocation (`sage solve.sage`). For RSA / DLP arithmetic, "
        "`Crypto.Util.number` (pycryptodome) or `gmpy2` is acceptable.\n"
        "- Name the primitive (RSA-CRT, ECDSA nonce reuse, AES-CBC padding oracle, AES-ECB, "
        "stream-cipher key reuse, hash length extension) and the structural property that breaks it.\n"
        "- Probable references: CWE-327 (Broken/Risky Crypto), CWE-330 (Insufficient Randomness), "
        "CWE-326 (Inadequate Encryption Strength); ATT&CK T1600 (Weaken Encryption).\n"
        "- Common failure modes to avoid: claiming an attack without the math, using `pow(c, d, n)` "
        "with placeholder constants, omitting how `c` / `n` / `e` are obtained from the challenge."
    ),
    "web": (
        "- Show each HTTP request/response as a full `curl` invocation (headers, cookies, body) — never "
        "paraphrase. Note the response status and the key header/body bytes that confirm the exploit step.\n"
        "- Identify whether a bot/admin visits a URL; if so, document the trigger endpoint and the "
        "expected cookie/session capture path.\n"
        "- Cite the OWASP Top 10 category (A01-A10) AND the specific CWE: CWE-79 (XSS), CWE-89 (SQLi), "
        "CWE-918 (SSRF), CWE-434 (Unrestricted Upload), CWE-352 (CSRF), CWE-22 (Path Traversal).\n"
        "- Probable references: ATT&CK T1190 (Exploit Public-Facing Application); OWASP WSTG entries "
        "(e.g. WSTG-INPV-05 for SQLi, WSTG-INPV-01 for reflected XSS).\n"
        "- Common failure modes to avoid: `requests.get(...)` with no session/cookie state, omitting "
        "URL-encoding of payloads, claiming SSRF without showing the internal-IP/loopback request."
    ),
    "forensics": (
        "- Show the exact tool invocation with flags for every step: `tshark -r capture.pcap -Y 'http.request'`, "
        "`binwalk -e file.bin`, `foremost -i image.dd`, `volatility -f mem.raw --profile=... pslist`.\n"
        "- For each artifact, show what the output looks like: a hex-dump excerpt, a Wireshark filter "
        "result, a strings/grep hit. Never write 'find the flag in the dump' without showing what found it.\n"
        "- Identify the artifact class: network capture (pcap), disk image, memory dump, file carving, "
        "steganography, log analysis.\n"
        "- Probable references: ATT&CK T1005 (Data from Local System), T1074 (Data Staged), T1056 "
        "(Input Capture); CWE-200 (Exposure of Sensitive Information), CWE-532 (Insertion of Sensitive "
        "Information into Log File).\n"
        "- Common failure modes to avoid: 'open in Wireshark and look around', omitting the exact filter "
        "string, claiming a carve worked without showing the recovered file's `file` / `strings` output."
    ),
    "rev": (
        "- Name the disassembler/decompiler used (Ghidra, IDA, radare2, Binary Ninja) and the specific "
        "workflow (e.g. 'load → auto-analyze → navigate to `main` → decompile').\n"
        "- Quote the key decompiled function snippet that reveals the check or the algorithm (10-30 "
        "lines max) — show the structural property the attack exploits.\n"
        "- Distinguish static analysis (what the disassembler shows) from dynamic analysis (gdb, ltrace, "
        "strace, frida) — say which step needs which.\n"
        "- Probable references: CWE-477 (Use of Obsolete Function), CWE-489 (Active Debug Code), "
        "CWE-798 (Hard-coded Credentials); ATT&CK T1027 (Obfuscated Files or Information), T1480 "
        "(Execution Guardrails) for anti-debug.\n"
        "- Common failure modes to avoid: 'reverse-engineer the binary' without naming the tool, "
        "missing the decompiled snippet, conflating static + dynamic steps in a single bullet."
    ),
    "osint": (
        "- Document every pivot explicitly: what was found, on what platform, what it led to next. No "
        "'run Sherlock' without showing the actual output snippet that produced the next lead.\n"
        "- The solver (where present) must be a documented pivot chain — `requests.get(...)` against a "
        "specific username/handle endpoint — not a generic username scanner.\n"
        "- Cite the public source for each pivot (GitHub user page, Twitter/X profile, WHOIS, certificate "
        "transparency log, Wayback Machine). Real URLs preferred over tool names.\n"
        "- Probable references: ATT&CK T1593 (Search Open Websites/Domains), T1589 (Gather Victim Identity "
        "Information), T1596 (Search Open Technical Databases); OSINT Framework (osintframework.com).\n"
        "- Common failure modes to avoid: tool-name-dumping without outputs ('use Sherlock, Maltego, "
        "theHarvester'), claiming a pivot without naming the platform it pivoted FROM and TO."
    ),
    "misc": (
        "- Identify the vulnerability class explicitly in the abstract: TOCTOU, path traversal, integer "
        "overflow, PRNG weakness, deserialization, command injection, race condition, side channel.\n"
        "- Once the class is named, cite the matching CWE and ATT&CK technique exactly as for a "
        "categorised challenge (do NOT skip the references section just because the category is 'misc').\n"
        "- Show the structural property that makes the bug exploitable (which check is racing, which "
        "input bypasses which filter).\n"
        "- Probable references: CWE-367 (TOCTOU), CWE-22 (Path Traversal), CWE-78 (OS Command Injection), "
        "CWE-502 (Deserialization of Untrusted Data), CWE-338 (Cryptographically Weak PRNG); ATT&CK "
        "selected per identified class (e.g. T1059 for command injection, T1574 for hijack flow).\n"
        "- Common failure modes to avoid: 'misc means anything goes' (no — still pick one vuln class), "
        "omitting the CWE/ATT&CK section because 'no clear category'."
    ),
    "mobile": (
        "- Name the RE tool (jadx, apktool, ghidra for native libs, frida for dynamic) and the specific "
        "class/method where the bug lives — full package path (e.g. `com.example.app.CryptoHelper.decrypt`).\n"
        "- Frida scripts must include the exact `Java.use('full.class.Name')` target with the method "
        "signature; never `Java.use('CryptoHelper')` without the package.\n"
        "- For Android-specific issues, note manifest properties: `android:debuggable`, "
        "`android:allowBackup`, `android:networkSecurityConfig`, exported activities/providers.\n"
        "- Probable references: ATT&CK Mobile T1629 (Impair Defenses), T1577 (Compromise Application "
        "Executable), T1521 (Encrypted Channel); CWE-921 (Storage of Sensitive Data in a Mechanism without "
        "Access Control), CWE-312 (Cleartext Storage of Sensitive Information); OWASP MASVS / MSTG.\n"
        "- Common failure modes to avoid: 'decompile the APK' without naming the tool, omitting the full "
        "class path in frida hooks, skipping the manifest review when relevant."
    ),
    "electron": (
        "- Specify which process the attack targets: renderer (sandboxed UI), main (Node.js privileges), "
        "preload script, or a native module. The attack surface differs by process.\n"
        "- Note `nodeIntegration` (true/false), `contextIsolation` (true/false), `sandbox`, and any "
        "`preload` script — these settings determine whether renderer JS can reach Node APIs.\n"
        "- For IPC abuse, name the `ipcRenderer.send` / `ipcMain.handle` channel and the payload shape.\n"
        "- Probable references: CWE-94 (Code Injection), CWE-1039 (Inadequate Detection of Adversarial "
        "Input), CWE-79 (XSS — relevant when renderer XSS pivots into RCE via nodeIntegration); ATT&CK "
        "T1059.007 (Command and Scripting Interpreter: JavaScript).\n"
        "- Common failure modes to avoid: treating Electron as 'just a browser' (ignores main process), "
        "omitting the nodeIntegration/contextIsolation status, no IPC channel name when IPC is the vector."
    ),
}


def _build_category_guidance_block(category: Optional[str]) -> str:
    """Return the F3 category guidance user-prompt block, or empty string when disabled / unknown.

    Case-insensitive lookup. Returns "" (silently) when:
      - F3_CATEGORY_GUIDANCE_ENABLED is False (ablation),
      - category is None or empty,
      - category is not in _CATEGORY_GUIDANCE.
    Logs a debug-level message when the category is unknown so EXP logs can be audited.
    """
    if not getattr(app_settings, "F3_CATEGORY_GUIDANCE_ENABLED", True):
        return ""
    if not category:
        return ""
    key = category.strip().lower()
    block = _CATEGORY_GUIDANCE.get(key)
    if not block:
        logger.debug(
            "F3: no category guidance for category={!r} (not in _CATEGORY_GUIDANCE); skipping block",
            category,
        )
        return ""
    return f"\n\n## Category-specific requirements ({key})\n{block}"


# Solver script names to look for (same as challenge_parser)
_AUTHOR_SOLVER_NAMES = (
    "solve.py",
    "solve.sage",
    "solve.sh",
    "solver.py",
    "exploit.py",
    "exploit.sh",
)
# Max chars to include for author writeup/solver so prompt stays bounded
_MAX_AUTHOR_WRITEUP_CHARS = 12000
_MAX_AUTHOR_SOLVER_CHARS = 8000

# Fair-generator mode: strip CTF flag-format tokens from author reference
# context before injecting it into the generation prompt, so the generator cannot
# copy the spoiler flag and must implement the technique. Matches PREFIX{...} tokens
# (CTF{...}, flag{...}, PREFIX{...}); requires a word char immediately before '{' so
# python set/dict/f-string braces are preserved.
_FLAG_TOKEN_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9_]{1,20}\{[^}\n]{1,256}\}")


def _redact_flags(text: str) -> str:
    """Replace CTF flag-format tokens (``PREFIX{...}``) with ``[REDACTED_FLAG]``.

    Used only when ``settings.REDACT_SOURCE_CONTEXT_FLAGS`` is enabled. Leaves
    non-flag braces (set/dict literals, f-string interpolations) untouched because
    those have no word character directly preceding ``{``.
    """
    if not text:
        return text
    return _FLAG_TOKEN_RE.sub("[REDACTED_FLAG]", text)


def _read_challenge_description(challenge_path: Path) -> str:
    """Read description.md for a challenge; return empty string if missing or unreadable."""
    path = challenge_path / "cyberedu" / "write-up" / "description.md"
    if not path.is_file():
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError as e:
        logger.warning("Could not read description for {}: {}", challenge_path.name, e)
        return ""


def _read_author_writeup(challenge_path: Path) -> str:
    """Read author's existing writeup.md if present; return empty string otherwise."""
    path = challenge_path / "cyberedu" / "write-up" / "writeup.md"
    if not path.is_file():
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        return text[:_MAX_AUTHOR_WRITEUP_CHARS] + (
            "..." if len(text) > _MAX_AUTHOR_WRITEUP_CHARS else ""
        )
    except OSError as e:
        logger.warning(
            "Could not read author writeup for {}: {}", challenge_path.name, e
        )
        return ""


def _read_author_solver(challenge_path: Path) -> str:
    """Read author's solver script (solve.py, solve.sage, etc.) if present; return empty string otherwise."""
    wu_dir = challenge_path / "cyberedu" / "write-up"
    if not wu_dir.is_dir():
        return ""
    for name in _AUTHOR_SOLVER_NAMES:
        p = wu_dir / name
        if p.is_file():
            try:
                text = p.read_text(encoding="utf-8", errors="replace").strip()
                return text[:_MAX_AUTHOR_SOLVER_CHARS] + (
                    "..." if len(text) > _MAX_AUTHOR_SOLVER_CHARS else ""
                )
            except OSError as e:
                logger.warning(
                    "Could not read author solver {} for {}: {}",
                    name,
                    challenge_path.name,
                    e,
                )
                return ""
    return ""


def _get_student_available_resources(
    challenge_path: Path,
) -> Tuple[List[str], bool, Optional[str]]:
    """Determine what the student has available: public files and deployment.

    Returns:
        (public_file_names, deployment_available, deployment_location).
        public_file_names: names of files under public/ (student-facing).
        deployment_available: True if cyberedu/deploy/ has content or deployment.yaml/yml exists.
        deployment_location: e.g. 'cyberedu/deploy/' or 'cyberedu/src/deployment.yml', or None.
    """
    public_names: List[str] = []
    pub_dir = challenge_path / "public"
    if pub_dir.is_dir():
        try:
            public_names = [f.name for f in pub_dir.iterdir() if f.is_file()]
        except OSError:
            pass

    deployment_available = False
    deployment_location: Optional[str] = None
    deploy_dir = challenge_path / "cyberedu" / "deploy"
    if deploy_dir.is_dir():
        try:
            if any(deploy_dir.iterdir()):
                deployment_available = True
                deployment_location = "cyberedu/deploy/"
        except OSError:
            pass

    # deployment.yaml or deployment.yml anywhere under challenge (e.g. cyberedu/src/)
    for name in ("deployment.yaml", "deployment.yml"):
        for parent in (
            challenge_path,
            challenge_path / "cyberedu",
            challenge_path / "cyberedu" / "src",
            deploy_dir,
        ):
            if not parent.is_dir():
                continue
            p = parent / name
            if p.is_file():
                deployment_available = True
                try:
                    deployment_location = str(p.relative_to(challenge_path)).replace(
                        "\\", "/"
                    )
                except ValueError:
                    deployment_location = p.name
                break
        if deployment_location:
            break

    return (public_names, deployment_available, deployment_location)


def _infer_skill_level(description: str, category: str) -> Optional[str]:
    """Infer target audience (novice/intermediate/advanced) from description and category.

    Used for skill-level awareness: tailor depth (novice=full worked example,
    intermediate=completion-style, advanced=core vuln and pointers).
    """
    if not description:
        return None
    text = (description + " " + category).lower()
    if any(k in text for k in ("advanced", "expert", "hard", "difficult")):
        return "advanced"
    if any(k in text for k in ("intermediate", "medium", "moderate")):
        return "intermediate"
    if any(k in text for k in ("novice", "beginner", "easy", "intro")):
        return "novice"
    return None


def _build_rag_context(
    technical_query: str,
    top_k: int = _RAG_TOP_K,
    pedagogical_k: int = _RAG_PEDAGOGICAL_K,
) -> str:
    """Retrieve pedagogical + technical KB chunks and format as context string.

    Runs two retrievals: (1) technical = challenge-focused (category, name, description);
    (2) pedagogical = writeup structure, guidelines, definitions. Merges and dedupes
    by content so RAG grounds both technical definitions (ATT&CK/CWE/OWASP) and
    pedagogical structure (abstract, thought process, step-by-step).
    """
    try:
        vb = VectorDBService()
        # Technical: challenge-relevant (vulnerability types, techniques, glossary)
        tech_results = vb.similarity_search(technical_query, k=top_k)
        # Pedagogical: writeup structure, guidelines, skill-level (structure guidelines, pedagogical principles)
        pedagogical_query = (
            "writeup structure abstract objectives thought process step-by-step "
            "pedagogical guidelines definitions skills reproducibility"
        )
        ped_results = vb.similarity_search(pedagogical_query, k=pedagogical_k)
        seen: set[str] = set()
        parts: List[str] = []
        for i, r in enumerate(tech_results + ped_results, 1):
            content = r.get("content", "").strip()
            if not content:
                continue
            key = content[:200]  # Dedupe by content prefix
            if key in seen:
                continue
            seen.add(key)
            parts.append(f"[{i}]\n{content}")
        return "\n\n---\n\n".join(parts) if parts else ""
    except Exception as e:
        logger.warning("RAG retrieval failed: {}; continuing without RAG context.", e)
        return ""


def _derive_solver_summary(solver: str) -> str:
    """Extract a one-line summary from the solver's leading docstring or top-level comment.

    Falls back to a generic description when neither is available. Used by the v4 Section 9
    assembler to give the inserted block a brief framing sentence.
    """
    if not solver or not solver.strip():
        return "Implements the technique described in Sections 6-8."
    lines = solver.splitlines()
    # First try a leading triple-quoted docstring
    for i, raw in enumerate(lines[:20]):
        s = raw.strip()
        if s.startswith('"""') or s.startswith("'''"):
            # Single-line docstring
            quote = s[:3]
            if s.endswith(quote) and len(s) > 6:
                inner = s[3:-3].strip()
                if inner:
                    return inner.splitlines()[0][:200]
            # Multi-line: read the next non-empty line
            for raw2 in lines[i + 1 : i + 10]:
                s2 = raw2.strip()
                if s2 and not (s2.startswith('"""') or s2.startswith("'''")):
                    return s2[:200]
            break
    # Then try a leading "# ..." comment block
    for raw in lines[:10]:
        s = raw.strip()
        if s.startswith("#") and len(s) > 1:
            txt = s.lstrip("#").strip()
            # Skip shebangs and encoding declarations
            if txt and not txt.startswith("!") and "coding" not in txt[:20]:
                return txt[:200]
    return "Implements the technique described in Sections 6-8."


SOLVER_PLACEHOLDER_MARKER = "<!-- SOLVER_PLACEHOLDER -->"


def _assemble_course_with_section_9(
    course_without_section_9: str,
    solver: str,
) -> str:
    """Insert the auto-built Section 9 (Solution Script) body into a v4-generated course.

    v4.1 (placeholder approach): the LLM is instructed to write all 11 sections in order
    and use `<!-- SOLVER_PLACEHOLDER -->` as the body of Section 9. This function locates
    the marker and replaces it with the python-fenced solver. The Section 9 heading
    `## 9. Solution Script` and any short lead-in sentence the LLM wrote remain in place.

    Fallback (legacy v4.0 path): if the marker is missing (LLM ignored the rule), fall
    back to the original behaviour — locate the Section 10 boundary and insert a full
    Section 9 block before it. Emits a warning so the fallback is visible in EXP logs.

    Args:
        course_without_section_9: course markdown produced by the v4 LLM call.
        solver: full solver source (the same string passed to course-gen).

    Returns:
        Assembled course.md with Section 9 body inserted.
    """
    if not course_without_section_9:
        return course_without_section_9
    if not solver or not solver.strip():
        # Section 9 still needs to exist so structural validator can find it; emit
        # a placeholder so downstream F8 'solver is empty' issue surfaces visibly.
        solver_body = "# (solver could not be generated)"
    else:
        solver_body = solver.strip()

    fenced_solver = "```python\n" + solver_body + "\n```"

    # Preferred path: replace the placeholder marker the LLM wrote (only first occurrence).
    if SOLVER_PLACEHOLDER_MARKER in course_without_section_9:
        return course_without_section_9.replace(
            SOLVER_PLACEHOLDER_MARKER, fenced_solver, 1
        )

    # Fallback: marker missing — log a warning and use legacy insertion-before-Section-10.
    logger.warning(
        "v4: SOLVER_PLACEHOLDER missing in course; using legacy assembly "
        "(inserting Section 9 before Section 10 heading)"
    )

    # v4.1.1: when the LLM ignored the placeholder AND wrote its own ## 9. ... block
    # (e.g. "## 9. Step-by-Step Resolution"), inserting another Section 9 would
    # produce duplicate (or triple) Section 9 headings. Strip any pre-existing
    # ## 9. block FIRST so the legacy insertion produces exactly one Section 9.
    course_for_insert = _strip_assembled_section_9(course_without_section_9)

    summary = _derive_solver_summary(solver)
    section_9 = (
        "## 9. Solution Script\n\n"
        "The following script implements the technique described in Sections 6-8. "
        f"{summary}\n\n" + fenced_solver + "\n"
    )

    # Find an insertion point: prefer a markdown heading that signals Section 10.
    # Tolerant: matches "## 10.", "## Conclusion", "**10.**", "## 10 -", case-insensitive.
    insertion_patterns = [
        re.compile(r"^##\s*10[.\s]", re.IGNORECASE | re.MULTILINE),
        re.compile(r"^##\s*Conclusion\b", re.IGNORECASE | re.MULTILINE),
        re.compile(r"^#\s*10[.\s]", re.IGNORECASE | re.MULTILINE),
        re.compile(r"^\*\*10[.\s]", re.IGNORECASE | re.MULTILINE),
        re.compile(r"^\*\*Conclusion\*\*", re.IGNORECASE | re.MULTILINE),
    ]
    for pat in insertion_patterns:
        m = pat.search(course_for_insert)
        if m:
            pos = m.start()
            return (
                course_for_insert[:pos].rstrip()
                + "\n\n"
                + section_9
                + "\n"
                + course_for_insert[pos:]
            )
    # No Section 10 heading found — append Section 9 at the end.
    return course_for_insert.rstrip() + "\n\n" + section_9


_FENCE_OPEN_RE = re.compile(r"^```[a-zA-Z0-9_+-]*\s*\n")
_FENCE_CLOSE_RE = re.compile(r"\n```\s*$")


def _strip_markdown_fence(text: str) -> str:
    """Remove a wrapping fence the model adds around its output.

    Handles ``` with any language tag (```python, ```py, ```markdown, ```md, ``` etc.).
    Also handles the truncation case where the opening fence is present but the
    closing fence got cut off — strip the opener anyway so downstream parsers
    (ast.parse for solvers) don't choke on line 1.
    """
    text = text.strip()
    open_match = _FENCE_OPEN_RE.match(text)
    if open_match:
        text = text[open_match.end() :]
        # Strip a trailing fence if present; tolerate truncation if it's missing.
        close_match = _FENCE_CLOSE_RE.search(text)
        if close_match:
            text = text[: close_match.start()]
    return text.strip()


_RO_CONTENT_INSTRUCTION = (
    "\n\n## Language instruction\n"
    "Generate all course content in Romanian. Use the cybersecurity glossary terms from the knowledge base. "
    "Prefer established Romanian technical terms over anglicisms where a standard Romanian term exists; "
    'retain English terms only when no standard Romanian equivalent is in use (e.g. "flag", "exploit", "payload" are acceptable as-is).'
)


def _read_romanian_glossary() -> str:
    """Read the Romanian cybersecurity glossary from the knowledge base directory."""
    glossary_path = app_settings.KNOWLEDGE_BASE_DIR / "romanian_glossary.md"
    try:
        return glossary_path.read_text(encoding="utf-8")
    except OSError as e:
        logger.warning("Could not read Romanian glossary from {}: {}", glossary_path, e)
        return ""


def _get_regen_flags(challenge_id: str, state: "AgentState") -> tuple:
    """Return (regen_course, regen_solver) for selective content regen (R3 — Wave 2).

    On initial generation (no retest IDs set), regenerate both.
    On refinement rounds:
      - tech failed only → regen solver only (course stays)
      - ped failed only  → regen course only (solver stays)
      - both failed      → regen both (existing behaviour)
    """
    retest_tech = set(state.ranking_retest_technical_ids or [])
    retest_ped = set(state.ranking_retest_pedagogical_ids or [])

    # No retest context set → initial generation, always regen both
    if not retest_tech and not retest_ped:
        return (True, True)

    tech_fail = challenge_id in retest_tech
    ped_fail = challenge_id in retest_ped

    if tech_fail and ped_fail:
        return (True, True)
    if tech_fail:
        return (False, True)  # tech only → solver only
    if ped_fail:
        return (True, False)  # ped only → course only
    # Challenge isn't in any retest set — shouldn't regenerate at all; return True/True
    # so the caller's existing skip-already-generated logic handles it
    return (True, True)


def _manifest_student_block(gi: GenerationInput) -> str:
    """Student-facing context from the manifest: the prompt + the actual student file contents.
    No 'the student does NOT have ...' line — that framing leaked the author's existence.
    """
    block = "- **Challenge prompt (exactly what the student is given):**\n" + (
        gi.student_prompt or "(No prompt text; reason from the files and category.)"
    )
    if gi.student_files:
        block += (
            "\n\n- **Files the student has** (these, and only these, are available):"
        )
        for sf in gi.student_files:
            if sf.kind == "text" and sf.content:
                block += f"\n\n`{sf.path}`:\n```\n{sf.content}\n```"
            elif sf.content:
                # Binary file with metadata snapshot (file type, size, strings)
                block += (
                    f"\n\n`{sf.path}` — binary file (metadata snapshot):\n"
                    f"```\n{sf.content}\n```"
                )
            else:
                block += f"\n\n`{sf.path}` — binary file (no metadata available)"
    else:
        block += "\n\n- **Files the student has:** none beyond the prompt."
    return block


def _generate_writeup_for_challenge(
    challenge_id: str,
    category: str,
    challenge_name: str,
    description: str,
    challenge_path: Optional[Path] = None,
    skill_level: Optional[str] = None,
    human_feedback_items: Optional[List[str]] = None,
    author_writeup: str = "",
    author_solver: str = "",
    max_tokens: Optional[int] = None,
    output_language: str = "en",
    prior_improvements: Optional[List[str]] = None,
    prior_dim_scores: Optional[Dict[str, float]] = None,
    solver_for_section_9: str = "",
    generation_input: Optional[GenerationInput] = None,
) -> str:
    """Generate a single course using RAG + LLM for a student with no source, writeup, or flag.

    generation_input: G2 — when provided (manifest-grounded mode), the course is grounded ONLY
        in the student materials it carries (student_prompt + student_files); the author
        writeup/solver are NOT injected into the course prompt (the context-leak fix). When
        None, legacy behaviour (author writeup/solver injected as reference).
    Author writeup/solver are used only as reference for technical correctness; the course text
    must not refer to them and must be a self-contained logical explanation from the student's standpoint.
    challenge_path: used to detect what the student has (public files, deployment); optional.
    max_tokens: override for course output length (default from settings.CONTENT_GENERATION_MAX_TOKENS).
    prior_dim_scores: D2 — per-dimension scores from the prior ranking round (e.g.
        {"technical": 8.5, "pedagogical": 6.5}). When set, a "scores needing 9.0" block is
        prepended to the feedback section so the LLM knows which dimension failed.
    solver_for_section_9: v4 — the authoritative solver that will be auto-assembled into Section 9.
        When non-empty (v4 mode active), the LLM is told NOT to write Section 9 itself; Sections 6-8
        must explain the technique implemented in this solver. Empty string = legacy v3 behaviour
        (LLM writes Section 9 with the full solver).
    """
    technical_query = f"{category} {challenge_name}: {description[:300]}".strip()
    rag_enabled = getattr(app_settings, "RAG_ENABLED", True)
    if rag_enabled:
        rag_context = _build_rag_context(technical_query)
    else:
        logger.info(
            "RAG disabled (RAG_ENABLED=False); skipping retrieval for {}", challenge_id
        )
        rag_context = ""
    if rag_context:
        context_block = (
            "Use the following reference material (pedagogical and technical) to shape structure and definitions:\n\n"
            + rag_context
        )
    else:
        context_block = "(No RAG context retrieved; use standard writeup structure and definitions.)"

    skill_note = ""
    if skill_level:
        skill_note = f"Target audience: {skill_level}. Tailor depth accordingly (novice=full worked example, intermediate=completion-style, advanced=core vuln and pointers)."

    feedback_block = ""
    if human_feedback_items:
        feedback_block = (
            "\n## Human feedback (incorporate these improvements)\n"
            + "\n".join(f"- {item}" for item in human_feedback_items)
        )

    judge_feedback_block = ""
    if prior_improvements:
        dim_scores_intro = ""
        if prior_dim_scores:
            # D2: surface per-dimension scores so the LLM knows which dimension failed
            # and can prioritise the matching subset of improvements.
            dim_lines = []
            for dim_name in ("technical", "pedagogical"):
                if dim_name in prior_dim_scores:
                    score = prior_dim_scores[dim_name]
                    focus_note = (
                        " — already at or above 9.0; preserve what works"
                        if score >= 9.0
                        else " — focus on improvements below"
                    )
                    dim_lines.append(
                        f"- {dim_name}: {score:.1f}/10 (target 9.0){focus_note}"
                    )
            if dim_lines:
                dim_scores_intro = (
                    "\n## Scores from previous round that need to reach 9.0:\n"
                    + "\n".join(dim_lines)
                    + "\n"
                )
        judge_feedback_block = (
            dim_scores_intro + "\n## Judge feedback — MUST address in this revision\n"
            "The previous version of this course was evaluated by expert judges. "
            "You MUST address every item below:\n"
            + "\n".join(f"- {item}" for item in prior_improvements)
        )

    author_writeup_block = ""
    if author_writeup and author_writeup.strip():
        author_writeup_block = (
            "\n## Reference only (do NOT cite in the course): author's writeup\n"
            "Use this only to ensure the solution path and steps are technically correct. "
            "Do not mention 'author writeup' or 'source' in the course; write for a student who does not have it.\n\n"
            + author_writeup.strip()
        )

    author_solver_block = ""
    if author_solver and author_solver.strip():
        author_solver_block = (
            "\n## Reference only (do NOT cite in the course): author's solver\n"
            "Use this only to ensure correctness of the solution script you generate. "
            "Do not refer to it in the course text; the student does not have access to it.\n\n```\n"
            + author_solver.strip()
            + "\n```"
        )

    # Manifest-grounded (G2): course-gen sees ONLY the student materials; the author writeup/
    # solver are removed from context (the leak fix) and the "does NOT have ..." frame line dropped.
    if generation_input is not None:
        author_writeup_block = ""
        author_solver_block = ""
        student_has_block = _manifest_student_block(generation_input)
        student_lacks_sentence = ""
    else:
        # Legacy: explicit "what the student has" (public file NAMES) + the lacks line.
        student_has_block = "- **Challenge description (what the student sees):**\n" + (
            description or "(No description provided.)"
        )
        if challenge_path is not None:
            public_names, deployment_available, deployment_location = (
                _get_student_available_resources(challenge_path)
            )
            student_has_block += (
                "\n\n- **Public files available to the student** (in `public/`): "
            )
            if public_names:
                student_has_block += ", ".join(public_names)
            else:
                student_has_block += "None listed."
            student_has_block += "\n\n- **Deployment available to the student**: "
            if deployment_available:
                student_has_block += f"Yes ({deployment_location or 'cyberedu/deploy/ or deployment file present'})."
            else:
                student_has_block += "No deployment provided."
            student_has_block += "\n\n- **The student does NOT have:** source code, author writeup, or the flag."
        student_lacks_sentence = (
            " The student does NOT have source code, author writeup, or the flag."
        )

    # v4 architecture: when a solver is passed in, it is the authoritative implementation.
    # Course writes Section 9 as a placeholder marker; the pipeline replaces the marker
    # with the actual solver after generation. The placeholder approach keeps the LLM's
    # section numbering rhythm intact (no "skip 9 then resume at 10" mental gymnastics).
    v4_solver_block = ""
    v4_task_note = ""
    if solver_for_section_9 and solver_for_section_9.strip():
        v4_solver_block = (
            "\n## Authoritative solver (the pipeline replaces the Section 9 placeholder marker with THIS code)\n"
            "This solver IS the implementation the course must explain. Sections 6-8 (Thought process / "
            "Step-by-step) MUST faithfully describe the technique this code uses. You MAY quote short "
            "snippets inline within narrative if it aids explanation, but you MUST NOT:\n"
            "  - write 'see the solver', 'as in solve.py', 'the script above/below', 'in the solver file'\n"
            "  - reference Section 9 or the solver as a separate file (the student reads top-to-bottom)\n"
            "  - copy the full solver into Section 9 yourself — write only the placeholder marker\n"
            "Section 9 in YOUR output must be exactly:\n\n"
            "## 9. Solution Script\n\n"
            "<!-- SOLVER_PLACEHOLDER -->\n\n"
            "(An optional one-line lead-in sentence between the heading and the marker is allowed.)\n"
            "The pipeline replaces `<!-- SOLVER_PLACEHOLDER -->` with the python-fenced solver below "
            "after you finish generating. Write all 11 sections in order with proper numbering "
            "(## 1., ## 2., ..., ## 11.); Sections 10 (Conclusion) and 11 (Extra Resources) MUST "
            "follow Section 9 with full content.\n\n"
            "<solver_for_section_9>\n"
            + solver_for_section_9.strip()
            + "\n</solver_for_section_9>"
        )
        v4_task_note = (
            "\n\n**v4 architecture is active.** Write all 11 sections numbered ## 1. through ## 11. "
            "in order. For Section 9, write the heading `## 9. Solution Script` followed by the "
            "literal placeholder line `<!-- SOLVER_PLACEHOLDER -->` (with an optional one-line lead-in). "
            "Do NOT write the solver code yourself — the pipeline substitutes the placeholder with the "
            "authoritative solver shown above. Sections 10 and 11 MUST still appear after Section 9."
        )

    user_prompt = f"""## Challenge
- **ID:** {challenge_id}
- **Category:** {category}
- **Name:** {challenge_name}
{skill_note}

## Reference material (for structure and definitions)
{context_block}

## What the student has (be explicit about this in the course)
{student_has_block}
{author_writeup_block}
{author_solver_block}
{v4_solver_block}
{feedback_block}
{judge_feedback_block}

## Task
Write a complete pedagogical **course** in Markdown. In the course, **state explicitly** what the student has (challenge description, any public files, and whether a deployment is available).{student_lacks_sentence} Use the structure in your instructions. The narrative must be a logical, self-contained explanation from the student's standpoint: discovery, reasoning, steps, then solution/flag. Do not mention author writeup or source code. Output only the Markdown document.{v4_task_note}"""

    # F3 (v4.2.1): append category-specific guidance block to the user prompt (NOT the system
    # prompt, so prompt cache stays warm). Empty string when F3_CATEGORY_GUIDANCE_ENABLED=False
    # or category is unknown — silent no-op in those cases.
    user_prompt += _build_category_guidance_block(category)

    ro_glossary_block = ""
    if output_language == "ro":
        glossary = _read_romanian_glossary()
        if glossary:
            ro_glossary_block = (
                "\n\n## Romanian cybersecurity glossary (use these terms)\n" + glossary
            )

    system_prompt = _resolve_writeup_system() + (
        _RO_CONTENT_INSTRUCTION if output_language == "ro" else ""
    )
    # user_prompt + glossary is volatile (per-challenge); system_prompt is stable → cached
    combined_user = user_prompt + ro_glossary_block

    tok = (
        max_tokens
        if max_tokens is not None
        else getattr(
            app_settings, "CONTENT_GENERATION_MAX_TOKENS", _DEFAULT_WRITEUP_MAX_TOKENS
        )
    )
    try:
        raw = generate_response_with_system(
            system_prompt,
            combined_user,
            temperature=0.5,
            max_tokens=tok,
        )
        course = _strip_markdown_fence(raw)
        if getattr(app_settings, "PROSE_V2", False):
            course = _postprocess_prose_v2(course)
        return course
    except Exception as e:
        logger.exception("LLM course generation failed for {}: {}", challenge_id, e)
        raise


def _generate_solve_script_for_challenge(
    challenge_id: str,
    category: str,
    writeup: str,
    description: str,
    author_solver: str = "",
    max_tokens: Optional[int] = None,
    existing_solver: str = "",
) -> str:
    """Generate a minimal solve script (e.g. solve.py) based on course text, challenge, and optional author solver.
    max_tokens: override for script output length (default from settings.CONTENT_GENERATION_SOLVE_MAX_TOKENS).
    existing_solver: when non-empty (refinement round > 0), instructs the LLM to EDIT this prior version
        addressing feedback, rather than regenerate from scratch.
    """
    # D1: bumped writeup window 2000 -> 8000 to address C3 cluster (solver diverges from course narrative).
    # Solver gen was previously seeing only the first 2000 chars of course → re-derived its own technique
    # which often disagreed with the full narrative. 8000 covers most full courses.
    writeup_snippet = (writeup[:8000] + "...") if len(writeup) > 8000 else writeup
    author_block = ""
    if author_solver and author_solver.strip():
        author_block = (
            "\nAuthor's solver (align with or extend this):\n```\n"
            + author_solver[:4000].strip()
            + "\n```\n"
        )
    # D1 edit-mode: when a prior solver exists (refinement round), instruct the LLM to EDIT it
    # rather than regenerate from scratch. Preserves correct parts; only changes what's flagged.
    edit_mode_block = ""
    if existing_solver and existing_solver.strip():
        edit_mode_block = (
            "\nEdit this version (do not regenerate from scratch — preserve what works, only fix the issues):\n```\n"
            + existing_solver[:8000].strip()
            + "\n```\n"
        )
    anti_hardcode_block = ""
    if getattr(app_settings, "REDACT_SOURCE_CONTEXT_FLAGS", False):
        anti_hardcode_block = (
            "\nIMPORTANT (no shortcuts): The script MUST obtain the flag by actually "
            "implementing and running the technique. Do NOT hardcode the flag value or any "
            "`PREFIX{...}` token. Do NOT read, open, or print `writeup.md` or any author "
            "solution file. A stub that prints a literal flag or echoes a writeup is invalid.\n"
        )

    rigor_block = ""
    if getattr(app_settings, "PROMPT_VARIANT", "") == "rigor":
        # P1: hoist the most-violated technical rules into the solver prompt.
        rigor_block = (
            "\nTECHNICAL RIGOR (the solver is judged on these):\n"
            "- Implement the COMPLETE solution path end to end — every stage from challenge "
            "inputs to flag; do not leave setup, data acquisition, or final extraction implicit.\n"
            "- No stubs/placeholders/empty arrays/`<paste here>` and no hardcoded flag. Obtain "
            "unknown values programmatically (a `fetch_*()` function) with a comment on how.\n"
            "- The script must match the technique described in the course narrative exactly.\n"
            "- Handle the obvious failure mode (service down, value missing) with a clear error, "
            "never a silent wrong answer or a fabricated flag.\n"
        )

    prompt = f"""You are a cybersecurity educator. Output only valid Python code for a solve script (solve.py). No markdown fences or explanation.

Challenge: {challenge_id} ({category}).
Description: {description[:400] if description else "N/A"}
{author_block}{edit_mode_block}{anti_hardcode_block}{rigor_block}
Course excerpt (use to implement the solution):
{writeup_snippet}

Produce a single Python script that implements the solution. Use clear comments. Output only the Python code."""

    tok = (
        max_tokens
        if max_tokens is not None
        else getattr(
            app_settings,
            "CONTENT_GENERATION_SOLVE_MAX_TOKENS",
            _DEFAULT_SOLVE_SCRIPT_MAX_TOKENS,
        )
    )
    try:
        raw = generate_response(
            prompt,
            temperature=0.3,
            max_tokens=tok,
        )
        return _strip_markdown_fence(raw)
    except Exception as e:
        logger.warning("Solve script generation failed for {}: {}", challenge_id, e)
        return ""


def _write_generated_to_disk(
    paths_with_category: List[tuple[Path, str]],
    courses: Dict[str, str],
    scripts: Dict[str, str],
) -> None:
    """Write generated course to course.md (and optional solve script) per challenge. Never overwrites author's writeup.md."""
    processed_dir = getattr(app_settings, "PROCESSED_DIR", None)
    if paths_with_category:
        first_path, _ = paths_with_category[0]
        first_wu = first_path / "cyberedu" / "write-up"
        logger.info(
            "Writing generated courses to {} (PROCESSED_DIR: {})",
            first_wu.resolve(),
            processed_dir.resolve() if processed_dir else "N/A",
        )
    for challenge_path, category in paths_with_category:
        challenge_name = challenge_path.name
        challenge_id = f"{category}/{challenge_name}"
        wu_dir = challenge_path / "cyberedu" / "write-up"
        if not wu_dir.is_dir():
            wu_dir.mkdir(parents=True, exist_ok=True)
        course_text = (
            courses.get(challenge_id, "").strip() if challenge_id in courses else ""
        )
        if course_text:
            if app_settings.EXPERIMENT_ID:
                course_path = (
                    app_settings.OUTPUT_DIR
                    / app_settings.EXPERIMENT_ID
                    / challenge_id
                    / "course.md"
                )
                course_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                course_path = wu_dir / "course.md"
            try:
                _atomic_write_text(course_path, courses[challenge_id])
                logger.info("Wrote course to {}", course_path)
            except OSError as e:
                logger.warning("Could not write course to {}: {}", course_path, e)
        else:
            logger.warning(
                "Skipped writing course for {} (empty content)", challenge_id
            )
        if challenge_id in scripts and scripts[challenge_id]:
            # Mirror the course path decision: when EXPERIMENT_ID is set, write solve_generated.py
            # into the experiment output dir alongside course.md (not into PROCESSED_DIR).
            # Root cause of M3: the old code always wrote to wu_dir regardless of EXPERIMENT_ID,
            # so solve_generated.py landed in data/processed/…/cyberedu/write-up/ and was never
            # visible in output/experiments/<EXP_ID>/courses/<challenge_id>/.
            if app_settings.EXPERIMENT_ID:
                script_path = (
                    app_settings.OUTPUT_DIR
                    / app_settings.EXPERIMENT_ID
                    / challenge_id
                    / "solve_generated.py"
                )
                script_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                script_path = wu_dir / "solve_generated.py"
            try:
                _atomic_write_text(script_path, scripts[challenge_id])
                logger.info("Wrote solve script to {}", script_path)
            except OSError as e:
                logger.warning("Could not write solve script to {}: {}", script_path, e)


def _write_course_to_exp_dir(challenge_id: str, content: str) -> None:
    """Write course immediately to the experiment output dir (enables TUI poll).

    Only fires when the TUI has set a ContextVar via gen_progress.set_exp_dir().
    Silent no-op outside of TUI runs.
    """
    exp_dir = _get_exp_dir()
    if exp_dir is None or not content:
        return
    try:
        cat, name = (
            challenge_id.split("/", 1)
            if "/" in challenge_id
            else ("misc", challenge_id)
        )
        course_path = exp_dir / "courses" / cat / name / "course.md"
        course_path.parent.mkdir(parents=True, exist_ok=True)
        course_path.write_text(content, encoding="utf-8")
    except Exception as exc:
        logger.debug("_write_course_to_exp_dir failed for {}: {}", challenge_id, exc)


def _write_solve_to_exp_dir(challenge_id: str, content: str) -> None:
    """Write solve_generated.py immediately to the experiment output dir alongside course.md.

    Mirrors _write_course_to_exp_dir: same ContextVar mechanism, same path layout.
    Ensures solver scripts land in the canonical output/experiments/<exp>/courses/...
    location (not in data/outputs/ or data/processed/). Silent no-op outside of TUI runs.
    """
    exp_dir = _get_exp_dir()
    if exp_dir is None or not content:
        return
    try:
        cat, name = (
            challenge_id.split("/", 1)
            if "/" in challenge_id
            else ("misc", challenge_id)
        )
        script_path = exp_dir / "courses" / cat / name / "solve_generated.py"
        script_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(script_path, content)
    except Exception as exc:
        logger.debug("_write_solve_to_exp_dir failed for {}: {}", challenge_id, exc)


def _process_one_challenge_content(
    challenge_path: Path,
    category: str,
    human_feedback_items: Optional[List[str]],
    content_max_tokens_override: Optional[Dict[str, int]] = None,
    output_language: str = "en",
    prior_improvements: Optional[List[str]] = None,
    regen_course: bool = True,
    regen_solver: bool = True,
    existing_course: str = "",
    existing_solver: str = "",
    prior_dim_scores: Optional[Dict[str, float]] = None,
) -> Tuple[str, str, str, Optional[Exception]]:
    """Process one challenge: generate course + solve script. For use in parallel or sequential loop.

    content_max_tokens_override: optional challenge_id -> max_tokens for course output (longer challenges).
    prior_improvements: R1 (Wave 2) — judge improvements to inject into the gen prompt.
    regen_course/regen_solver: R3 (Wave 2) — selective regen flags; when False the existing
        content is kept as-is, avoiding an unnecessary LLM call.
    existing_course/existing_solver: current content, used when one regen flag is False.
    prior_dim_scores: D2 — per-dimension scores from prior round, e.g.
        {"technical": 8.5, "pedagogical": 6.5}. Surfaced in the gen prompt as a
        "scores needing 9.0" block alongside the judge improvements.

    v4 architecture (gated by settings.V4_ARCHITECTURE_ENABLED, default True):
        Solver-gen runs FIRST; course-gen receives the solver as authoritative input
        and Sections 6-8 explain it; Section 9 is auto-assembled by the pipeline,
        not written by the LLM. Selective refinement still honoured:
            tech-only fail  -> regen solver, re-assemble course with new solver + existing 1-8/10-11
            ped-only fail   -> regen course with unchanged solver, re-assemble
            both fail       -> regen solver first, then regen course (with new solver), re-assemble
        When V4_ARCHITECTURE_ENABLED=False, falls back to v3 (course-first, LLM writes Section 9).

    Returns:
        (challenge_id, course_text, script, error). error is None on success.
    """
    challenge_name = challenge_path.name
    challenge_id = f"{category}/{challenge_name}"
    # Reset per-challenge LLM call budget (belt-and-suspenders against runaway loops).
    # Each concurrent worker runs in its own thread context so ContextVar isolation is safe.
    reset_challenge_llm_budget(challenge_id)
    max_tokens_course = (
        (content_max_tokens_override or {}).get(challenge_id)
        if content_max_tokens_override
        else None
    )
    v4_enabled = getattr(app_settings, "V4_ARCHITECTURE_ENABLED", True)
    try:
        description = _read_challenge_description(challenge_path)
        author_writeup = _read_author_writeup(challenge_path)
        author_solver = _read_author_solver(challenge_path)
        # G2 — manifest-grounded generation: load the resolved contract when enabled and a
        # manifest exists. Course-gen will use student materials only; solver-gen still reads
        # the author solver (correctness). No manifest -> gen_input stays None -> legacy path.
        gen_input: Optional[GenerationInput] = None
        if getattr(app_settings, "MANIFEST_GROUNDED_GEN", False):
            gen_input = load_generation_input(challenge_path)
            if gen_input is not None:
                description = gen_input.student_prompt or description
                author_writeup = gen_input.author_writeup or ""
                author_solver = gen_input.author_solver or ""
        if getattr(app_settings, "REDACT_SOURCE_CONTEXT_FLAGS", False):
            # Fair-generator mode: strip the spoiler flag from reference context so the
            # generator must implement the technique rather than hardcoding it.
            author_writeup = _redact_flags(author_writeup)
            author_solver = _redact_flags(author_solver)
        skill_level = _infer_skill_level(description, category)

        if v4_enabled:
            # v4 flow: solver-first, then course-explains-solver, then assembled Section 9.
            # Step 1 — solver gen (or keep existing)
            if regen_solver:
                # On first round (no existing_course), pass empty writeup so the solver-gen
                # prompt does not embed stale narrative. On refinement rounds, the prior
                # course IS available and serves as D1's 8000-char context window.
                solver_writeup_context = existing_course or ""

                def _gen_solver(
                    writeup_context: str = solver_writeup_context,
                    existing_solver: str = "",
                ) -> str:
                    try:
                        return _generate_solve_script_for_challenge(
                            challenge_id=challenge_id,
                            category=category,
                            writeup=writeup_context or solver_writeup_context,
                            description=description,
                            author_solver=author_solver,
                            existing_solver=existing_solver,
                        )
                    except Exception as e:  # noqa: BLE001
                        logger.debug(
                            "v4: solver-gen failed for {}: {}", challenge_id, e
                        )
                        return existing_solver or ""

                if (
                    getattr(app_settings, "SOLVER_SELF_IMPROVE_ENABLED", False)
                    and not existing_solver
                ):
                    # Multi-pass self-improvement on the FIRST solver gen (spec 2026-06-03).
                    # Trace summary (rounds + verdict) is logged by the orchestrator;
                    # behavioral metrics read the solver directly (decision 004).
                    script, _ = self_improve_solver(
                        challenge_id=challenge_id,
                        category=category,
                        description=description,
                        author_solver=author_solver,
                        gen_solver=_gen_solver,
                        max_rounds=getattr(
                            app_settings, "MAX_SOLVER_SELF_IMPROVE_ROUNDS", 3
                        ),
                    )
                else:
                    script = _gen_solver(
                        writeup_context=solver_writeup_context,
                        existing_solver=existing_solver,
                    )
                logger.debug("v4: generated solver for {} (first stage)", challenge_id)
            else:
                script = existing_solver
                logger.info(
                    "v4 R3: skipping solver regen for {} "
                    "(technical ok; keeping existing solver)",
                    challenge_id,
                )

            # Step 2 — course gen (or keep existing 1-8/10-11 fragments). The LLM
            # generates only Sections 1-8 + 10-11; Section 9 is auto-assembled below.
            if regen_course:
                writeup_no_s9 = _generate_writeup_for_challenge(
                    challenge_id=challenge_id,
                    category=category,
                    challenge_name=challenge_name,
                    description=description,
                    challenge_path=challenge_path,
                    skill_level=skill_level,
                    human_feedback_items=human_feedback_items,
                    author_writeup=author_writeup,
                    author_solver=author_solver,
                    max_tokens=max_tokens_course,
                    output_language=output_language,
                    prior_improvements=prior_improvements,
                    prior_dim_scores=prior_dim_scores,
                    solver_for_section_9=script,
                    generation_input=gen_input,
                )
                logger.debug("v4: regenerated course for {}", challenge_id)
            else:
                # Keep prior course narrative; assembly will re-insert the (possibly
                # new) solver into Section 9. This preserves the pedagogical-OK course
                # while letting the technical regen update the implementation.
                writeup_no_s9 = existing_course
                logger.info(
                    "v4 R3: skipping course regen for {} "
                    "(pedagogical ok; keeping existing course narrative)",
                    challenge_id,
                )

            # Step 3 — assemble final course.md = sections 1-8 + auto-Section-9 + 10-11
            writeup = _assemble_course_with_section_9(writeup_no_s9, script)
        else:
            # v3 fallback: course-first, LLM writes Section 9 itself.
            if regen_course:
                writeup = _generate_writeup_for_challenge(
                    challenge_id=challenge_id,
                    category=category,
                    challenge_name=challenge_name,
                    description=description,
                    challenge_path=challenge_path,
                    skill_level=skill_level,
                    human_feedback_items=human_feedback_items,
                    author_writeup=author_writeup,
                    author_solver=author_solver,
                    max_tokens=max_tokens_course,
                    output_language=output_language,
                    prior_improvements=prior_improvements,
                    prior_dim_scores=prior_dim_scores,
                )
                logger.debug("v3: regenerated course for {}", challenge_id)
            else:
                writeup = existing_course
                logger.info(
                    "v3 R3: skipping course regen for {} (pedagogical ok)",
                    challenge_id,
                )

            if regen_solver:
                try:
                    script = _generate_solve_script_for_challenge(
                        challenge_id=challenge_id,
                        category=category,
                        writeup=writeup,
                        description=description,
                        author_solver=author_solver,
                        existing_solver=existing_solver,
                    )
                except Exception as e:
                    logger.debug("v3: solver-gen failed for {}: {}", challenge_id, e)
                    script = ""
            else:
                script = existing_solver
                logger.info(
                    "v3 R3: skipping solver regen for {} (technical ok)", challenge_id
                )

        # F8: pre-ranking structural validation. If the course is missing sections,
        # has truncation markers, or the solver doesn't parse, retry with the issues
        # injected as feedback — saves expensive ranking LLM calls on broken output.
        from src.services.structural_validator import (
            format_feedback_for_prompt,
            validate_all,
        )

        validator_enabled = getattr(app_settings, "STRUCTURAL_VALIDATOR_ENABLED", True)
        max_validator_retries = getattr(
            app_settings, "STRUCTURAL_VALIDATOR_MAX_RETRIES", 2
        )
        retry = 0
        while validator_enabled and retry < max_validator_retries:
            report = validate_all(writeup, script)
            if report.is_valid:
                if retry > 0:
                    logger.info(
                        "F8: validator passed for {} after {} retry(s)",
                        challenge_id,
                        retry,
                    )
                break
            logger.info(
                "F8: validator failed for {} (issues={}); retry {}/{}",
                challenge_id,
                len(report.issues),
                retry + 1,
                max_validator_retries,
            )
            validator_feedback = format_feedback_for_prompt(report)
            feedback_items = list(human_feedback_items or []) + [validator_feedback]
            try:
                if v4_enabled:
                    # Solver-first retry: regen solver, then regen course around it, then assemble.
                    if regen_solver:
                        script = _generate_solve_script_for_challenge(
                            challenge_id=challenge_id,
                            category=category,
                            writeup=writeup,
                            description=description,
                            author_solver=author_solver,
                            existing_solver=script,
                        )
                    if regen_course:
                        writeup_no_s9 = _generate_writeup_for_challenge(
                            challenge_id=challenge_id,
                            category=category,
                            challenge_name=challenge_name,
                            description=description,
                            challenge_path=challenge_path,
                            skill_level=skill_level,
                            human_feedback_items=feedback_items,
                            author_writeup=author_writeup,
                            author_solver=author_solver,
                            max_tokens=max_tokens_course,
                            output_language=output_language,
                            prior_improvements=prior_improvements,
                            prior_dim_scores=prior_dim_scores,
                            solver_for_section_9=script,
                        )
                        writeup = _assemble_course_with_section_9(writeup_no_s9, script)
                    else:
                        # Course narrative unchanged but solver updated: re-assemble using
                        # the existing pre-S9 fragment of the course (best-effort: strip the
                        # old assembled Section 9 by re-extracting via heading boundary).
                        writeup = _assemble_course_with_section_9(
                            _strip_assembled_section_9(writeup), script
                        )
                else:
                    if regen_course:
                        writeup = _generate_writeup_for_challenge(
                            challenge_id=challenge_id,
                            category=category,
                            challenge_name=challenge_name,
                            description=description,
                            challenge_path=challenge_path,
                            skill_level=skill_level,
                            human_feedback_items=feedback_items,
                            author_writeup=author_writeup,
                            author_solver=author_solver,
                            max_tokens=max_tokens_course,
                            output_language=output_language,
                            prior_improvements=prior_improvements,
                            prior_dim_scores=prior_dim_scores,
                        )
                    if regen_solver:
                        script = _generate_solve_script_for_challenge(
                            challenge_id=challenge_id,
                            category=category,
                            writeup=writeup,
                            description=description,
                            author_solver=author_solver,
                            existing_solver=script,
                        )
            except Exception as e:
                logger.debug(
                    "F8 retry gen failed for {}: {}; proceeding with current content",
                    challenge_id,
                    e,
                )
                break
            retry += 1

        return (challenge_id, writeup, script, None)
    except Exception as e:
        return (challenge_id, "", "", e)


def _strip_assembled_section_9(course_md: str) -> str:
    """Remove a previously auto-assembled Section 9 block from a course.

    Used in v4 retry when only the solver regenerates and we need to re-insert
    the new solver into a course that already has an assembled Section 9. After
    stripping, the assembly fallback (legacy path: insert before Section 10) can
    run cleanly without producing a duplicate Section 9.

    v4.1 placeholder-mode courses normally do not need this — assembly replaces
    the marker in-place — but the legacy-fallback path still uses it. Also
    re-inserts a fresh `<!-- SOLVER_PLACEHOLDER -->` so a subsequent placeholder-
    based assembly call works idempotently.

    Best-effort: matches "## 9." or "## 9 " heading and strips up to the next
    "## " heading boundary. If no Section 9 is found, returns input unchanged.
    """
    if not course_md:
        return course_md
    # Match the Section 9 heading and the trailing block up to the next ## heading.
    pattern = re.compile(
        r"(?:^|\n)##\s*9[.\s][^\n]*\n.*?(?=\n##\s|\Z)",
        re.DOTALL | re.IGNORECASE,
    )
    m = pattern.search(course_md)
    if not m:
        return course_md
    return (
        course_md[: m.start()].rstrip() + "\n\n" + course_md[m.end() :].lstrip()
    ).strip() + "\n"


def run_content_generation_agent(
    state: AgentState,
    write_to_disk: bool = False,
) -> AgentState:
    """Generate courses and optional solve scripts for all challenges in state.

    Uses RAG over pedagogical + technical knowledge base and LLM with Cybersecurity
    Expert Educator persona. Populates state.generated_courses and
    state.generated_solve_scripts. Run VectorDBService().ingest_knowledge_base()
    before calling so RAG has context.

    Args:
        state: Current agent state (organized_challenges or PROCESSED_DIR scan).
        write_to_disk: If True, write generated course to cyberedu/write-up/course.md
            (and solve_generated.py) per challenge. Author's writeup.md is never overwritten.

    Returns:
        Updated state with generated_courses and generated_solve_scripts.
    """
    # Heartbeat: mark phase as content_generation when node starts.
    try:
        from src.services.heartbeat import get_active_state as _hb_get

        _hb = _hb_get()
        if _hb is not None:
            _hb.current_phase = "content_generation"
    except Exception:
        pass  # heartbeat is non-critical

    paths_with_category = _get_challenge_paths(state)
    if not paths_with_category:
        logger.warning(
            "Content generation: no challenge paths (organized_challenges empty or PROCESSED_DIR missing)"
        )
        return replace(state, current_agent="content_generation_agent")

    # Token-saving: run LLM only for subset IDs when set
    subset_ids: Optional[List[str]] = state.content_generation_subset_ids
    if subset_ids is not None:
        subset_set = set(subset_ids)
        paths_with_category = [
            (p, c) for p, c in paths_with_category if f"{c}/{p.name}" in subset_set
        ]

    # Refinement: when human_feedback is set, re-generate only for those IDs (allow overwrite)
    refinement_ids = set(state.human_feedback.keys()) if state.human_feedback else set()
    if refinement_ids:
        paths_with_category = [
            (p, c) for p, c in paths_with_category if f"{c}/{p.name}" in refinement_ids
        ]
        if not paths_with_category:
            logger.info(
                "Content generation: refinement_ids from human_feedback not in paths; skipping refinement"
            )
            return replace(state, current_agent="content_generation_agent")

    writeups: Dict[str, str] = dict(state.generated_courses)
    scripts: Dict[str, str] = dict(state.generated_solve_scripts)

    # Build work items (skip already-generated unless HITL refinement or auto-refinement subset)
    # When content_generation_subset_ids is set (auto-refinement), we must regenerate those challenges.
    # Each item: (path, cat, hf, prior_improvements, regen_course, regen_solver, prior_dim_scores)
    work_items: List[
        Tuple[
            Path,
            str,
            Optional[List[str]],
            Optional[List[str]],
            bool,
            bool,
            Optional[Dict[str, float]],
        ]
    ] = []
    subset_set = set(subset_ids) if subset_ids else set()
    prior_improvements_map: Dict[str, List[str]] = dict(
        getattr(state, "prior_improvements_per_challenge", None) or {}
    )
    # D2: per-dimension scores from the prior ranking round, keyed by challenge_id.
    # When set, the content_generation prompt surfaces "tech/ped need 9.0" so the LLM
    # knows which dimension failed alongside the improvement list (R1).
    prior_dim_scores_map: Dict[str, Dict[str, float]] = dict(
        getattr(state, "prior_dim_scores_per_challenge", None) or {}
    )
    for challenge_path, category in paths_with_category:
        challenge_id = f"{category}/{challenge_path.name}"
        force_regenerate = (challenge_id in refinement_ids) or (
            subset_ids is not None and challenge_id in subset_set
        )
        if (
            not force_regenerate
            and challenge_id in state.generated_courses
            and state.generated_courses[challenge_id]
        ):
            logger.debug(
                "Content generation: skipping {} (already in generated_courses)",
                challenge_id,
            )
            continue
        hf = (
            state.human_feedback.get(challenge_id) if state.human_feedback else None
        ) or None
        # R1: judge improvements from prior ranking round
        prior_imp = prior_improvements_map.get(challenge_id) or None
        # D2: per-dim scores from prior round (None on first round)
        prior_dim = prior_dim_scores_map.get(challenge_id) or None
        # R3: selective regen — only re-run the aspect(s) that failed
        regen_course, regen_solver = _get_regen_flags(challenge_id, state)
        work_items.append(
            (
                challenge_path,
                category,
                hf,
                prior_imp,
                regen_course,
                regen_solver,
                prior_dim,
            )
        )

    content_max_tokens_override = state.content_max_tokens_override
    output_language = getattr(state, "output_language", "en")
    max_concurrent = getattr(app_settings, "LLM_MAX_CONCURRENT", 1) or 1
    if max_concurrent <= 1:
        for (
            challenge_path,
            category,
            human_feedback_items,
            prior_imp,
            regen_course,
            regen_solver,
            prior_dim,
        ) in work_items:
            challenge_id = f"{category}/{challenge_path.name}"
            _report_gen_progress(challenge_id, "start")
            challenge_id, writeup, script, err = _process_one_challenge_content(
                challenge_path,
                category,
                human_feedback_items,
                content_max_tokens_override,
                output_language,
                prior_improvements=prior_imp,
                regen_course=regen_course,
                regen_solver=regen_solver,
                existing_course=writeups.get(challenge_id, ""),
                existing_solver=scripts.get(challenge_id, ""),
                prior_dim_scores=prior_dim,
            )
            if err:
                logger.exception(
                    "Content generation failed for {}: {}", challenge_id, err
                )
                _report_gen_progress(challenge_id, "failed", str(err))
                state.add_error("content_generation_agent", challenge_id, str(err))
                writeups[challenge_id] = ""
                scripts[challenge_id] = ""
            else:
                writeups[challenge_id] = writeup
                scripts[challenge_id] = script
                _write_course_to_exp_dir(challenge_id, writeup)
                _write_solve_to_exp_dir(challenge_id, script)
                _report_gen_progress(challenge_id, "done")
                logger.info("Generated course for {}", challenge_id)
            # Heartbeat: content_generation does not advance completed_challenges
            # (that counter is owned by ranking_agent to avoid double-counting).
    else:
        workers = min(max_concurrent, len(work_items) or 1)
        logger.info(
            "Content generation: running {} challenges with {} concurrent workers",
            len(work_items),
            workers,
        )
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_item = {
                executor.submit(
                    _process_one_challenge_content,
                    path,
                    cat,
                    hf,
                    content_max_tokens_override,
                    output_language,
                    prior_imp,
                    rc,
                    rs,
                    writeups.get(f"{cat}/{path.name}", ""),
                    scripts.get(f"{cat}/{path.name}", ""),
                    pd,
                ): (path, cat)
                for path, cat, hf, prior_imp, rc, rs, pd in work_items
            }
            for future in as_completed(future_to_item):
                challenge_id, writeup, script, err = future.result()
                if err:
                    logger.exception(
                        "Content generation failed for {}: {}", challenge_id, err
                    )
                    _report_gen_progress(challenge_id, "failed", str(err))
                    state.add_error("content_generation_agent", challenge_id, str(err))
                    writeups[challenge_id] = ""
                    scripts[challenge_id] = ""
                else:
                    writeups[challenge_id] = writeup
                    scripts[challenge_id] = script
                    _write_course_to_exp_dir(challenge_id, writeup)
                    _write_solve_to_exp_dir(challenge_id, script)
                    _report_gen_progress(challenge_id, "done")
                    logger.info("Generated course for {}", challenge_id)

    if write_to_disk:
        _write_generated_to_disk(
            paths_with_category, writeups, scripts
        )  # writeups = generated course text → course.md

    return replace(
        state,
        generated_courses=writeups,
        generated_solve_scripts=scripts,
        current_agent="content_generation_agent",
    )


async def content_generation_agent(state: AgentState) -> AgentState:
    """Generate course (writeup) and solution scripts for challenges (research-based RAG + LLM).

    Persona: Cybersecurity Expert Educator. RAG over pedagogical and technical KB;
    output includes abstract, thought process, step-by-step, and ATT&CK/CWE/OWASP
    citations where relevant.

    Args:
        state: Pipeline state with organized_challenges or validation_reports.

    Returns:
        Updated state with generated_courses and generated_solve_scripts.
    """
    return run_content_generation_agent(state)
