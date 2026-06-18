"""Solver self-improvement loop (plan -> generate -> hybrid-verify -> revise).

Decoupled from the ranking judge. The hybrid verifier combines deterministic static
checks (the ungameable floor) with an LLM dry-run critique. Gated by
settings.SOLVER_SELF_IMPROVE_ENABLED.
See docs/superpowers/specs/2026-06-03-solver-self-improve-design.md and decision 004.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# PREFIX{...} CTF flag token (mirrors content_generation_agent._FLAG_TOKEN_RE).
_FLAG_TOKEN_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9_]{1,20}\{[^}\n]{1,256}\}")
# A flag LITERAL assigned/returned/printed as the answer, where the WHOLE quoted string
# is the flag (no prose like "Example: ..."). Precise stub detector — skips f-strings
# (the quote is preceded by f), concatenation (no closing } inside quotes), and examples.
_HARDCODED_FLAG_RE = re.compile(
    r"""(?:=|return|print\(|yield)\s*(["'])([A-Za-z][A-Za-z0-9_]{1,20}\{[^"'}\n]{1,200}\})\1"""
)
_STUB_PATTERNS = [
    re.compile(r"<paste", re.IGNORECASE),
    re.compile(r"\bTODO\b"),
    re.compile(r"\bFIXME\b"),
    re.compile(r"raise NotImplementedError"),
    re.compile(r"<your[_ ].*here>", re.IGNORECASE),  # <your_value_here> placeholder
    re.compile(r"^\s*\.\.\.\s*$", re.MULTILINE),  # bare ellipsis placeholder
]
_FLAG_PATH_RE = re.compile(
    r"print\(|sys\.stdout|return[^\n]*flag|submit|\.post\(", re.IGNORECASE
)
# Actual writeup-file READ (the s3crets passthrough shortcut), not a mere mention.
_WRITEUP_READ_RE = re.compile(
    r"writeup[^\n]*\.(md|txt)|writeup[^\n]*\.read|open\([^)\n]*writeup|writeup_path",
    re.IGNORECASE,
)


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class VerifyResult:
    verdict: bool
    static: list[CheckResult]
    issues: list[str] = field(default_factory=list)


def _code_lines(solver: str) -> list[str]:
    out = []
    for line in solver.splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            out.append(s)
    return out


def check_no_hardcoded_flag(solver: str, plan: list[dict]) -> CheckResult:
    """Fail ONLY on the actual stub pattern: a complete flag LITERAL assigned/returned/
    printed as the answer (``X = "PREFIX{...}"``, ``return "PREFIX{...}"``,
    ``print("PREFIX{...}")``) where the whole quoted string IS the flag.

    Deliberately precise to avoid false positives that blocked loop convergence:
    illustrative examples (``print("Example: CTF{...}")``), docstring format mentions,
    f-string construction (``f"CTF{{{x}}}"``), and string concatenation
    (``"CTF{" + x + "}"``) are all CORRECT behavior and must pass."""
    for line in solver.splitlines():
        if line.strip().startswith("#"):
            continue  # comment
        m = _HARDCODED_FLAG_RE.search(line)
        if m:
            return CheckResult("no_hardcoded_flag", False, m.group(2))
    return CheckResult("no_hardcoded_flag", True, "")


def check_no_writeup_read(solver: str, plan: list[dict]) -> CheckResult:
    """Fail only on an actual writeup-file READ (the answer-passthrough shortcut),
    not a mere mention in a comment/string."""
    for line in solver.splitlines():
        if line.strip().startswith("#"):
            continue  # comment mention is not a read
        m = _WRITEUP_READ_RE.search(line)
        if m:
            return CheckResult("no_writeup_read", False, m.group(0).strip())
    return CheckResult("no_writeup_read", True, "")


def check_no_stub_placeholders(solver: str, plan: list[dict]) -> CheckResult:
    for pat in _STUB_PATTERNS:
        m = pat.search(solver)
        if m:
            return CheckResult("no_stub_placeholders", False, m.group(0).strip())
    return CheckResult("no_stub_placeholders", True, "")


def check_has_flag_path(solver: str, plan: list[dict]) -> CheckResult:
    ok = _FLAG_PATH_RE.search(solver) is not None
    return CheckResult("has_flag_path", ok, "" if ok else "no print/return/submit path")


def check_nontrivial(solver: str, plan: list[dict]) -> CheckResult:
    n = len(_code_lines(solver))
    return CheckResult("nontrivial", n >= 10, f"{n} code lines")


def check_covers_stages(solver: str, plan: list[dict]) -> CheckResult:
    if not plan:
        return CheckResult("covers_stages", True, "no plan")
    low = solver.lower()
    hits = 0
    for stage in plan:
        kw = str(stage.get("name", "")).lower().split()
        if any(tok for tok in kw if len(tok) >= 4 and tok in low):
            hits += 1
    need = max(1, int(0.6 * len(plan)))
    return CheckResult(
        "covers_stages", hits >= need, f"{hits}/{len(plan)} stages referenced"
    )


_STATIC_CHECKS = [
    check_no_hardcoded_flag,
    check_no_writeup_read,
    check_no_stub_placeholders,
    check_has_flag_path,
    check_nontrivial,
    check_covers_stages,
]


def run_static_checks(solver: str, plan: list[dict]) -> list[CheckResult]:
    return [chk(solver, plan) for chk in _STATIC_CHECKS]


# --- Planner (Task 2) ---
import json  # noqa: E402

from loguru import logger  # noqa: E402

from src.config.settings import settings  # noqa: E402
from src.services.llm_service import generate_response_with_system  # noqa: E402

_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)

_PLANNER_SYSTEM = (
    "You are a CTF solution planner. Given a challenge, output ONLY a JSON array of the "
    "ordered stages needed to go from the materials a student has to the flag. Each element: "
    '{"name": short, "goal": what it achieves, "technique": how, "expected_output": what you observe}. '
    "3-8 stages for complex challenges, 1-2 for trivial ones. No prose, JSON array only."
)


def parse_plan_json(raw: str) -> list[dict]:
    """Tolerant parse of the planner output into a list of stage dicts; [] on failure."""
    if not raw or not raw.strip():
        return []
    m = _JSON_ARRAY_RE.search(raw)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except (json.JSONDecodeError, ValueError):
        return []
    if not isinstance(data, list):
        return []
    out = []
    for item in data:
        if isinstance(item, dict) and item.get("name"):
            out.append(
                {
                    "name": str(item.get("name", "")),
                    "goal": str(item.get("goal", "")),
                    "technique": str(item.get("technique", "")),
                    "expected_output": str(item.get("expected_output", "")),
                }
            )
    return out


def plan_solution_stages(
    challenge_id: str, category: str, description: str, author_solver: str = ""
) -> list[dict]:
    """Decompose the challenge into ordered stages. Returns [] on any failure (caller falls back)."""
    ref = (
        f"\nReference solver (for technique only, do not copy):\n{author_solver[:3000]}\n"
        if author_solver
        else ""
    )
    prompt = (
        f"Challenge: {challenge_id} ({category}).\n"
        f"Description: {description[:1500] or 'N/A'}{ref}\n"
        "Output the JSON array of solution stages."
    )
    try:
        raw = generate_response_with_system(
            _PLANNER_SYSTEM,
            prompt,
            provider=settings.SOLVER_CRITIC_PROVIDER,
            model=settings.SOLVER_CRITIC_MODEL,
            temperature=0.2,
            max_tokens=1500,
        )
    except (
        Exception
    ) as e:  # noqa: BLE001 — planner is best-effort; caller degrades gracefully
        logger.debug("planner failed for {}: {}", challenge_id, e)
        return []
    plan = parse_plan_json(raw)
    logger.debug("planned {} stages for {}", len(plan), challenge_id)
    return plan


# --- Hybrid verifier (Task 3) ---
_CRITIC_SYSTEM = (
    "You are a strict CTF solver reviewer. You are given a solution plan and a Python solver. "
    "Walk the solver against the plan. For EACH stage decide whether it is actually implemented "
    "(not stubbed, not a manual instruction, not hardcoded). Decide whether the chain actually "
    "produces the flag end to end. Respond with JSON ONLY: "
    '{"verdict": true|false, "issues": ["concrete missing/broken item", ...]}. '
    "verdict is true only if every stage is implemented and the solver produces the flag."
)


def llm_dry_run_critique(
    solver: str, plan: list[dict], challenge_id: str, category: str, description: str
) -> dict:
    """Return {"verdict": bool, "issues": [str]}. On failure, verdict True with empty issues
    so an LLM outage cannot block the loop (the static floor still governs)."""
    plan_txt = json.dumps(plan, indent=1) if plan else "(no explicit plan)"
    prompt = (
        f"Challenge: {challenge_id} ({category}). {description[:600]}\n\n"
        f"PLAN:\n{plan_txt}\n\nSOLVER:\n```python\n{solver[:12000]}\n```\n\n"
        "Return the JSON verdict."
    )
    try:
        raw = generate_response_with_system(
            _CRITIC_SYSTEM,
            prompt,
            provider=settings.SOLVER_CRITIC_PROVIDER,
            model=settings.SOLVER_CRITIC_MODEL,
            temperature=0.2,
            max_tokens=1200,
        )
    except Exception as e:  # noqa: BLE001
        logger.debug("critique failed for {}: {}", challenge_id, e)
        return {"verdict": True, "issues": []}
    m = re.search(r"\{.*\}", raw or "", re.DOTALL)
    if not m:
        return {"verdict": True, "issues": []}
    try:
        data = json.loads(m.group(0))
    except (json.JSONDecodeError, ValueError):
        return {"verdict": True, "issues": []}
    issues = [str(i) for i in (data.get("issues") or []) if str(i).strip()]
    return {"verdict": bool(data.get("verdict", True)), "issues": issues}


def verify_solver(
    solver: str, plan: list[dict], challenge_id: str, category: str, description: str
) -> VerifyResult:
    """Hybrid verify: deterministic static floor AND LLM dry-run critique. verdict requires BOTH."""
    static = run_static_checks(solver, plan)
    issues = [
        f"{r.name}: {r.detail}".strip().rstrip(":") for r in static if not r.passed
    ]
    crit = llm_dry_run_critique(solver, plan, challenge_id, category, description)
    issues.extend(crit["issues"])
    verdict = all(r.passed for r in static) and bool(crit["verdict"])
    return VerifyResult(verdict=verdict, static=static, issues=issues)


# --- Orchestrator (Task 4) ---
from typing import Callable  # noqa: E402


def self_improve_solver(
    *,
    challenge_id: str,
    category: str,
    description: str,
    author_solver: str,
    gen_solver: "Callable[..., str]",
    max_rounds: int = 3,
) -> tuple[str, dict]:
    """Plan -> generate -> verify -> (revise -> verify)* until verdict passes or max_rounds.

    gen_solver(writeup_context="", existing_solver="") -> str is injected by the caller
    (content_generation_agent) so this module has no dependency on it. Returns
    (final_solver, trace) where trace feeds the behavioral metrics.
    """
    plan = plan_solution_stages(challenge_id, category, description, author_solver)
    solver = gen_solver(writeup_context="", existing_solver="")
    rounds = []
    final_verdict = False
    prev_issue_count: int | None = None
    for i in range(max(1, max_rounds)):
        result = verify_solver(solver, plan, challenge_id, category, description)
        rounds.append(
            {
                "round": i + 1,
                "solver_lines": len(_code_lines(solver)),
                "static": [{"name": r.name, "passed": r.passed} for r in result.static],
                "verdict": result.verdict,
                "issues": result.issues,
            }
        )
        if result.verdict:
            final_verdict = True
            break
        # Early-exit: the static floor passes but the strict critic rarely says "done",
        # so the loop otherwise runs to the cap on healthy solvers. If static passes AND
        # the critic's issue count is no longer shrinking, accept the statically-healthy
        # solver instead of burning more (codex) rounds.
        static_ok = all(r.passed for r in result.static)
        if (
            static_ok
            and prev_issue_count is not None
            and len(result.issues) >= prev_issue_count
        ):
            final_verdict = True
            break
        prev_issue_count = len(result.issues)
        if i == max_rounds - 1:
            break  # cap reached; accept best
        fix = "Fix these issues in the solver:\n- " + "\n- ".join(result.issues[:12])
        try:
            solver = gen_solver(writeup_context=fix, existing_solver=solver)
        except Exception as e:  # noqa: BLE001
            logger.debug("revise failed for {}: {}", challenge_id, e)
            break
    trace = {
        "challenge_id": challenge_id,
        "plan": plan,
        "rounds": rounds,
        "rounds_used": len(rounds),
        "final_verdict": final_verdict,
    }
    logger.info(
        "self-improve {}: {} round(s), final_verdict={}",
        challenge_id,
        len(rounds),
        final_verdict,
    )
    return solver, trace
