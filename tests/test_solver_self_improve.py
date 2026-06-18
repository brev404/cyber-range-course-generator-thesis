"""Tests for the solver self-improvement loop (spec 2026-06-03, plan 2026-06-04).

Static checks are the ungameable deterministic floor; the loop combines them with an
LLM dry-run critique. Fixtures mirror observed generator failure modes.
"""

from src.agents.solver_self_improve import (
    CheckResult,
    check_has_flag_path,
    check_no_hardcoded_flag,
    check_no_stub_placeholders,
    check_no_writeup_read,
    check_nontrivial,
    run_static_checks,
)

# Observed failure modes (paraphrased minimal forms)
STUB_HARDCODED = """#!/usr/bin/env python3
FLAG = "CTF{4CA9A1_NOS7026}"
def main():
    print(FLAG)
"""
STUB_WRITEUP = """#!/usr/bin/env python3
from pathlib import Path
p = Path(__file__).parent / "writeup.md"
print(p.read_text())
"""
GOOD_SOLVER = """#!/usr/bin/env python3
import requests
def fetch_token(host):
    r = requests.get(f"http://{host}/dump")
    return r.json()["token"]
def solve(host):
    token = fetch_token(host)
    resp = requests.post(f"http://{host}/submit", json={"token": token})
    print(resp.json()["flag"])
def main():
    solve("localhost:5000")
if __name__ == "__main__":
    main()
"""


def test_hardcoded_flag_check_fails_on_stub():
    r = check_no_hardcoded_flag(STUB_HARDCODED, [])
    assert (
        isinstance(r, CheckResult)
        and r.name == "no_hardcoded_flag"
        and r.passed is False
    )


def test_hardcoded_flag_check_passes_on_good():
    assert check_no_hardcoded_flag(GOOD_SOLVER, []).passed is True


def test_writeup_read_check_fails_on_passthrough():
    assert check_no_writeup_read(STUB_WRITEUP, []).passed is False


def test_nontrivial_fails_on_short_stub():
    assert check_nontrivial(STUB_HARDCODED, []).passed is False
    assert check_nontrivial(GOOD_SOLVER, []).passed is True


def test_has_flag_path_passes_on_good():
    assert check_has_flag_path(GOOD_SOLVER, []).passed is True


def test_run_static_checks_good_solver_all_pass():
    results = run_static_checks(GOOD_SOLVER, [])
    assert all(r.passed for r in results), [r.name for r in results if not r.passed]


def test_run_static_checks_stub_has_failures():
    results = run_static_checks(STUB_HARDCODED, [])
    failed = {r.name for r in results if not r.passed}
    assert "no_hardcoded_flag" in failed
    assert "nontrivial" in failed


# --- Task 2: plan parser ---
from src.agents.solver_self_improve import parse_plan_json  # noqa: E402


def test_parse_plan_valid_json():
    raw = '[{"name":"recon","goal":"find endpoint","technique":"nmap","expected_output":"open port"}]'
    plan = parse_plan_json(raw)
    assert len(plan) == 1 and plan[0]["name"] == "recon"


def test_parse_plan_fenced_json():
    raw = (
        '```json\n[{"name":"x","goal":"g","technique":"t","expected_output":"e"}]\n```'
    )
    assert parse_plan_json(raw)[0]["name"] == "x"


def test_parse_plan_malformed_returns_empty():
    assert parse_plan_json("not json at all") == []
    assert parse_plan_json("") == []


# --- Task 3: hybrid verifier ---
from src.agents import solver_self_improve as ssi  # noqa: E402
from src.agents.solver_self_improve import verify_solver  # noqa: E402

_VERIFY_GOOD = (
    "import requests\n"
    "def fetch(h):\n    return requests.get(h).json()['t']\n"
    "def solve(h):\n    print(requests.post(h, json={'t': fetch(h)}).json()['flag'])\n"
    "def main():\n    solve('x')\n    return 0\n"
    "if __name__ == '__main__':\n    main()\n"
)


def test_verify_static_fail_forces_overall_fail(monkeypatch):
    monkeypatch.setattr(
        ssi, "llm_dry_run_critique", lambda *a, **k: {"verdict": True, "issues": []}
    )
    stub = 'FLAG = "CTF{x}"\nprint(FLAG)\n'
    r = verify_solver(stub, [], "misc/x", "misc", "desc")
    assert r.verdict is False
    assert any("no_hardcoded_flag" in i for i in r.issues)


def test_verify_all_pass(monkeypatch):
    monkeypatch.setattr(
        ssi, "llm_dry_run_critique", lambda *a, **k: {"verdict": True, "issues": []}
    )
    r = verify_solver(_VERIFY_GOOD, [], "misc/x", "misc", "desc")
    assert r.verdict is True


def test_verify_llm_fail_propagates_issues(monkeypatch):
    monkeypatch.setattr(
        ssi,
        "llm_dry_run_critique",
        lambda *a, **k: {"verdict": False, "issues": ["stage 2 not implemented"]},
    )
    r = verify_solver(_VERIFY_GOOD, [], "misc/x", "misc", "desc")
    assert r.verdict is False
    assert "stage 2 not implemented" in r.issues


# --- Task 4: orchestrator ---
from src.agents.solver_self_improve import self_improve_solver  # noqa: E402


def test_loop_stops_when_verdict_passes(monkeypatch):
    # stage keyword "fetch" is present in _VERIFY_GOOD (def fetch) so covers_stages passes
    monkeypatch.setattr(
        ssi, "plan_solution_stages", lambda *a, **k: [{"name": "fetch"}]
    )
    monkeypatch.setattr(
        ssi, "llm_dry_run_critique", lambda *a, **k: {"verdict": True, "issues": []}
    )
    calls = {"n": 0}

    def gen(writeup_context="", existing_solver=""):
        calls["n"] += 1
        return _VERIFY_GOOD

    solver, trace = self_improve_solver(
        challenge_id="misc/x",
        category="misc",
        description="d",
        author_solver="",
        gen_solver=gen,
        max_rounds=3,
    )
    assert trace["final_verdict"] is True
    assert calls["n"] == 1
    assert trace["rounds_used"] == 1


def test_loop_revises_until_cap(monkeypatch):
    monkeypatch.setattr(
        ssi, "plan_solution_stages", lambda *a, **k: [{"name": "recon"}]
    )
    monkeypatch.setattr(
        ssi, "llm_dry_run_critique", lambda *a, **k: {"verdict": False, "issues": ["x"]}
    )
    calls = {"n": 0}

    def gen(writeup_context="", existing_solver=""):
        calls["n"] += 1
        return _VERIFY_GOOD

    solver, trace = self_improve_solver(
        challenge_id="misc/x",
        category="misc",
        description="d",
        author_solver="",
        gen_solver=gen,
        max_rounds=3,
    )
    assert trace["final_verdict"] is False
    assert trace["rounds_used"] == 3
    assert calls["n"] == 3


def test_loop_planner_failure_still_runs(monkeypatch):
    monkeypatch.setattr(ssi, "plan_solution_stages", lambda *a, **k: [])
    monkeypatch.setattr(
        ssi, "llm_dry_run_critique", lambda *a, **k: {"verdict": True, "issues": []}
    )
    solver, trace = self_improve_solver(
        challenge_id="misc/x",
        category="misc",
        description="d",
        author_solver="",
        gen_solver=lambda writeup_context="", existing_solver="": _VERIFY_GOOD,
        max_rounds=3,
    )
    assert trace["plan"] == []
    assert trace["final_verdict"] is True


# --- Task 5: settings ---
def test_settings_defaults_off():
    from src.config.settings import settings as s

    assert s.SOLVER_SELF_IMPROVE_ENABLED is False
    assert s.MAX_SOLVER_SELF_IMPROVE_ROUNDS == 3


# --- Cost mitigations ---
def test_critic_settings_defaults():
    from src.config.settings import settings as s

    assert s.SOLVER_CRITIC_PROVIDER == "claude-code"
    assert s.SOLVER_CRITIC_MODEL == "claude-haiku-4-5"
    assert s.CODEX_EXEC_TIMEOUT == 600


def test_planner_pins_critic_provider(monkeypatch):
    # Planner must call the LLM with the pinned cheap provider/model, NOT inherit
    # the generator's run-provider (codex) — else codex pays for planning.
    captured = {}

    def spy(system, user, **kwargs):
        captured.update(kwargs)
        return '[{"name":"recon","goal":"g","technique":"t","expected_output":"e"}]'

    monkeypatch.setattr(ssi, "generate_response_with_system", spy)
    ssi.plan_solution_stages("misc/x", "misc", "desc", author_solver="")
    assert captured["provider"] == "claude-code"
    assert captured["model"] == "claude-haiku-4-5"


def test_critic_pins_critic_provider(monkeypatch):
    captured = {}

    def spy(system, user, **kwargs):
        captured.update(kwargs)
        return '{"verdict": true, "issues": []}'

    monkeypatch.setattr(ssi, "generate_response_with_system", spy)
    ssi.llm_dry_run_critique("solver", [], "misc/x", "misc", "desc")
    assert captured["provider"] == "claude-code"
    assert captured["model"] == "claude-haiku-4-5"


def test_loop_early_exit_when_static_passes_and_issues_stall(monkeypatch):
    # Static-clean solver, critic always verdict=False with a stable issue list:
    # the loop must accept early (round 2), NOT burn all 3 rounds.
    monkeypatch.setattr(
        ssi, "plan_solution_stages", lambda *a, **k: [{"name": "fetch"}]
    )
    monkeypatch.setattr(
        ssi,
        "llm_dry_run_critique",
        lambda *a, **k: {"verdict": False, "issues": ["wants more rigor"]},
    )
    calls = {"n": 0}

    def gen(writeup_context="", existing_solver=""):
        calls["n"] += 1
        return _VERIFY_GOOD  # static-clean

    solver, trace = self_improve_solver(
        challenge_id="misc/x",
        category="misc",
        description="d",
        author_solver="",
        gen_solver=gen,
        max_rounds=3,
    )
    # round1: prev=None -> set prev=1; round2: static ok & issues(1)>=prev(1) -> exit
    assert trace["rounds_used"] == 2
    assert calls["n"] == 2
    assert trace["final_verdict"] is True


def test_loop_no_early_exit_when_static_fails(monkeypatch):
    # Static FAILING solver must keep revising to the cap even if issues are stable —
    # early-exit only applies when the static floor passes.
    monkeypatch.setattr(
        ssi, "plan_solution_stages", lambda *a, **k: [{"name": "recon"}]
    )
    monkeypatch.setattr(
        ssi,
        "llm_dry_run_critique",
        lambda *a, **k: {"verdict": False, "issues": ["x"]},
    )
    stub = 'FLAG = "CTF{x}"\nprint(FLAG)\n'  # fails no_hardcoded_flag + nontrivial

    def gen(writeup_context="", existing_solver=""):
        return stub

    solver, trace = self_improve_solver(
        challenge_id="misc/x",
        category="misc",
        description="d",
        author_solver="",
        gen_solver=gen,
        max_rounds=3,
    )
    assert trace["rounds_used"] == 3
    assert trace["final_verdict"] is False


def test_codex_timeout_uses_setting(monkeypatch):
    import src.services.codex_model as cxm
    import src.services.llm_service as svc
    from src.config.settings import settings as s

    captured = {}

    class FakeCodex:
        def __init__(self, model_name=None, timeout=None):
            captured["timeout"] = timeout

    monkeypatch.setattr(svc, "get_available_providers", lambda: ["codex"])
    monkeypatch.setattr(cxm, "CodexModel", FakeCodex)
    svc.get_chat_model(provider="codex", model="gpt-5.5")
    assert captured["timeout"] == s.CODEX_EXEC_TIMEOUT == 600


# --- Regression: no_hardcoded_flag false positives (smoke finding) ---
def test_hardcoded_flag_allows_fstring_construction():
    # building the flag from computed values is GOOD, not a hardcode
    solver = "icao24='x'\ncallsign='y'\nreturn f'CTF{{{icao24}_{callsign}}}'\n"
    assert check_no_hardcoded_flag(solver, []).passed is True


def test_hardcoded_flag_allows_format_doc():
    solver = '"""Flag format: CTF{ICAO24hex_CallSign}"""\nimport os\n'
    assert check_no_hardcoded_flag(solver, []).passed is True


def test_hardcoded_flag_allows_format_comment():
    solver = "# Flag format: CTF{ICAO_Callsign}\nimport os\n"
    assert check_no_hardcoded_flag(solver, []).passed is True


def test_hardcoded_flag_still_catches_literal():
    assert (
        check_no_hardcoded_flag(
            'FLAG = "CTF{4CA9A1_NOS7026}"\nprint(FLAG)\n', []
        ).passed
        is False
    )


# --- Regression: precise hardcoded-flag detector (suite finding) ---
def test_hardcoded_flag_allows_printed_example():
    # printing the expected FORMAT to the user is good UX, not a hardcode
    assert (
        check_no_hardcoded_flag('print("Example: CTF{a1b2c3_UAL803}")\n', []).passed
        is True
    )


def test_hardcoded_flag_allows_docstring_mention():
    assert (
        check_no_hardcoded_flag(
            '"""Flag 1  CTF{dns,dns}  -- two domains"""\nimport os\n', []
        ).passed
        is True
    )


def test_hardcoded_flag_allows_concatenation():
    assert (
        check_no_hardcoded_flag('flag = "CTF{" + icao + "}"\nprint(flag)\n', []).passed
        is True
    )


def test_hardcoded_flag_catches_assignment_and_return_and_print():
    assert check_no_hardcoded_flag('FLAG = "CTF{4CA9A1_NOS7026}"\n', []).passed is False
    assert check_no_hardcoded_flag('    return "CTF{abc123}"\n', []).passed is False
    assert check_no_hardcoded_flag('print("flag{deadbeef}")\n', []).passed is False


# --- Regression: precise writeup-read + stub checks (suite finding) ---
def test_writeup_read_allows_comment_mention():
    assert (
        check_no_writeup_read(
            "# fallback ground-truth from writeup\nimport os\n", []
        ).passed
        is True
    )


def test_writeup_read_catches_actual_read():
    assert (
        check_no_writeup_read(
            'p = base/"writeup.md"\nprint(p.read_text())\n', []
        ).passed
        is False
    )
    assert check_no_writeup_read("writeup_path = x\n", []).passed is False


def test_stub_allows_normal_empty_accumulator():
    assert (
        check_no_stub_placeholders(
            "payload_candidates = []  # list of frames\n", []
        ).passed
        is True
    )


def test_stub_still_catches_paste_and_ellipsis():
    assert check_no_stub_placeholders("KEY = '<paste here>'\n", []).passed is False
    assert check_no_stub_placeholders("def f():\n    ...\n", []).passed is False
