"""Mapping Agent: tag generated courses with ATT&CK, CWE, OWASP WSTG.

Maps each generated course (state.generated_courses) to (1) MITRE ATT&CK
technique(s) used, (2) CWE(s) involved, (3) OWASP WSTG scenario(s) / skills
applied. Output feeds ranking, reporting, and curriculum tagging. IDs align with
Content Generation. Depends on dictionary files
from the knowledge-base build step: attack_techniques.md, cwe_weaknesses.md, owasp_wstg.md.

Flow optimization: This agent is Python-only (no LLM). It uses only
rule-based ID loading from the knowledge base, regex parsing of course text,
and validation against loaded IDs.
"""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path
from typing import FrozenSet, Set, Tuple

from loguru import logger

from src.config.settings import settings
from src.core.state import AgentState
from src.models.report_models import WriteupMapping

# Regex to extract taxonomy IDs from course text (align with Content Generation citations)
_RE_ATTACK = re.compile(r"\b(T\d{4}(?:\.\d{3})?)\b")
_RE_CWE = re.compile(r"\b(CWE-\d+)\b")
_RE_WSTG = re.compile(r"\b(WSTG-(?:v\d+-)?[A-Z]+-\d+)\b")

# Regex to parse section headers in knowledge base files
_RE_ATTACK_HEADER = re.compile(r"^##\s+(T\d{4}(?:\.\d{3})?)\s+[–\-]")
_RE_CWE_HEADER = re.compile(r"^##\s+(CWE-\d+)\s+[–\-]")
_RE_WSTG_HEADER = re.compile(r"^###\s+(WSTG-[A-Z]+-\d+)\s+[–\-]")


def _load_attack_ids(kb_dir: Path) -> FrozenSet[str]:
    """Load valid MITRE ATT&CK technique IDs from attack_techniques.md."""
    path = kb_dir / "attack_techniques.md"
    if not path.exists():
        logger.warning(
            "attack_techniques.md not found; no ATT&CK validation", path=str(path)
        )
        return frozenset()
    ids: Set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = _RE_ATTACK_HEADER.match(line.strip())
            if m:
                ids.add(m.group(1))
    logger.debug("Loaded {} ATT&CK technique IDs from {}", len(ids), path.name)
    return frozenset(ids)


def _load_cwe_ids(kb_dir: Path) -> FrozenSet[str]:
    """Load valid CWE IDs from cwe_weaknesses.md."""
    path = kb_dir / "cwe_weaknesses.md"
    if not path.exists():
        logger.warning("cwe_weaknesses.md not found; no CWE validation", path=str(path))
        return frozenset()
    ids: Set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = _RE_CWE_HEADER.match(line.strip())
            if m:
                ids.add(m.group(1))
    logger.debug("Loaded {} CWE IDs from {}", len(ids), path.name)
    return frozenset(ids)


def _load_owasp_wstg_ids(kb_dir: Path) -> FrozenSet[str]:
    """Load valid OWASP WSTG scenario IDs from owasp_wstg.md (non-versioned form)."""
    path = kb_dir / "owasp_wstg.md"
    if not path.exists():
        logger.warning("owasp_wstg.md not found; no WSTG validation", path=str(path))
        return frozenset()
    ids: Set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = _RE_WSTG_HEADER.match(line.strip())
            if m:
                ids.add(m.group(1))
    logger.debug("Loaded {} OWASP WSTG IDs from {}", len(ids), path.name)
    return frozenset(ids)


def _normalize_wstg_id(raw: str) -> str:
    """Normalize WSTG ID: strip version prefix (e.g. WSTG-v42-INFO-02 -> WSTG-INFO-02)."""
    if raw.startswith("WSTG-v") and "-" in raw[6:]:
        # WSTG-v42-INFO-02 -> WSTG-INFO-02
        rest = raw[6:]
        idx = rest.find("-")
        if idx != -1:
            return "WSTG-" + rest[idx + 1 :]
    return raw


def _extract_and_validate(
    text: str,
    valid_attack: FrozenSet[str],
    valid_cwe: FrozenSet[str],
    valid_wstg: FrozenSet[str],
) -> Tuple[list[str], list[str], list[str]]:
    """Extract taxonomy IDs from course text and return only valid ones (deduplicated, ordered)."""
    attack_ids: list[str] = []
    if valid_attack:
        seen: Set[str] = set()
        for m in _RE_ATTACK.finditer(text):
            tid = m.group(1)
            if tid in valid_attack and tid not in seen:
                seen.add(tid)
                attack_ids.append(tid)
    else:
        for m in _RE_ATTACK.finditer(text):
            attack_ids.append(m.group(1))
        attack_ids = list(dict.fromkeys(attack_ids))

    cwe_ids: list[str] = []
    if valid_cwe:
        seen = set()
        for m in _RE_CWE.finditer(text):
            cid = m.group(1)
            if cid in valid_cwe and cid not in seen:
                seen.add(cid)
                cwe_ids.append(cid)
    else:
        for m in _RE_CWE.finditer(text):
            cwe_ids.append(m.group(1))
        cwe_ids = list(dict.fromkeys(cwe_ids))

    wstg_ids: list[str] = []
    if valid_wstg:
        seen = set()
        for m in _RE_WSTG.finditer(text):
            raw = m.group(1)
            norm = _normalize_wstg_id(raw)
            if norm in valid_wstg and norm not in seen:
                seen.add(norm)
                wstg_ids.append(norm)
    else:
        for m in _RE_WSTG.finditer(text):
            wstg_ids.append(_normalize_wstg_id(m.group(1)))
        wstg_ids = list(dict.fromkeys(wstg_ids))

    return attack_ids, cwe_ids, wstg_ids


def run_mapping_agent(state: AgentState) -> AgentState:
    """Map each generated course content to ATT&CK, CWE, and OWASP WSTG IDs.

    Reads state.generated_courses; for each challenge, parses generated course
    text for taxonomy IDs (regex), validates against knowledge base IDs, and
    produces a WriteupMapping. Updates state.writeup_mappings (Dict[challenge_id, WriteupMapping]).

    Args:
        state: Current agent state with generated_courses populated.

    Returns:
        AgentState: New state with writeup_mappings set. Other fields unchanged.
    """
    if not state.generated_courses:
        logger.info("No generated courses to map; skipping mapping agent")
        return replace(state, writeup_mappings={}, current_agent="mapping")

    kb_dir: Path = settings.KNOWLEDGE_BASE_DIR
    valid_attack = _load_attack_ids(kb_dir)
    valid_cwe = _load_cwe_ids(kb_dir)
    valid_wstg = _load_owasp_wstg_ids(kb_dir)

    mappings: dict[str, WriteupMapping] = {}
    for challenge_id, writeup_text in state.generated_courses.items():
        attack_ids, cwe_ids, owasp_ids = _extract_and_validate(
            writeup_text or "", valid_attack, valid_cwe, valid_wstg
        )
        mappings[challenge_id] = WriteupMapping(
            challenge_id=challenge_id,
            attack_technique_ids=attack_ids,
            cwe_ids=cwe_ids,
            owasp_wstg_ids=owasp_ids,
        )
        logger.debug(
            "Mapped {} -> ATT&CK {}, CWE {}, WSTG {}",
            challenge_id,
            len(attack_ids),
            len(cwe_ids),
            len(owasp_ids),
        )

    logger.info(
        "Writeup mapping complete for {} challenges",
        len(mappings),
    )
    return replace(
        state,
        writeup_mappings=mappings,
        current_agent="mapping",
    )
