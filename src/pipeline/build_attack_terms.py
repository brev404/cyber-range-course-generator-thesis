"""Build MITRE ATT&CK techniques Markdown from Enterprise ATT&CK STIX for the knowledge base.

Fetches the Enterprise ATT&CK STIX bundle from attack-stix-data, parses tactics and
techniques (attack-pattern), and writes data/knowledge_base/attack_techniques.md with
attribution and version. See data/knowledge_base/ for the generated file.
"""

import json
import sys
from pathlib import Path
from urllib.request import Request, urlopen

from loguru import logger

# Add project root for imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config.settings import settings  # noqa: E402
from src.utils.logging_config import setup_logging  # noqa: E402

ENTERPRISE_ATTACK_URL = (
    "https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master/"
    "enterprise-attack/enterprise-attack.json"
)

# Platform-specific extensions: CyberEDU uses T1573.003 (Encrypted Channel) not in official ATT&CK
# See docs/reference/archive/TAG_DICTIONARY_FINDINGS_REPORT.md
PLATFORM_ATTACK_EXTENSIONS: list[dict] = [
    {
        "id": "T1573.003",
        "name": "Encrypted Channel",
        "description": "Adversaries may use encrypted channels to hide command and control traffic.",
        "tactic_names": ["Command and Control"],
    },
]


def fetch_stix(url: str) -> dict:
    """Fetch STIX bundle JSON from URL.

    Args:
        url: URL to the STIX JSON file.

    Returns:
        Parsed JSON bundle.

    Raises:
        Exception: On fetch or parse failure.
    """
    req = Request(url, headers={"Accept": "application/json"})
    with urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def parse_bundle(bundle: dict) -> tuple[dict[str, dict], dict[str, dict], str]:
    """Parse STIX bundle into tactics map, techniques list, and version string.

    Args:
        bundle: STIX 2.x bundle (type "bundle", objects array).

    Returns:
        (tactics_by_id, techniques_list, version). tactics_by_id maps STIX id to
        {name, shortname}. techniques_list is list of {id, name, description, tactic_names}.
    """
    objects = bundle.get("objects") or []

    version = ""
    tactics_by_id: dict[str, dict] = {}
    techniques: list[dict] = []

    tactics_by_shortname: dict[str, str] = {}
    for obj in objects:
        obj_type = obj.get("type", "")
        if obj_type == "x-mitre-collection":
            version = obj.get("x_mitre_version", "")
            continue
        if obj_type == "x-mitre-tactic":
            tactics_by_id[obj["id"]] = {
                "name": obj.get("name", ""),
                "shortname": obj.get("x_mitre_shortname", ""),
            }
            shortname = obj.get("x_mitre_shortname", "")
            if shortname:
                tactics_by_shortname[shortname] = obj.get("name", "")
            continue
        if obj_type == "attack-pattern":
            # Skip deprecated or revoked if present
            if obj.get("x_mitre_deprecated", False) or obj.get("revoked", False):
                continue
            attack_id = obj.get("x_mitre_id") or ""
            if not attack_id:
                for ref in obj.get("external_references") or []:
                    if ref.get("source_name") == "mitre-attack" and ref.get(
                        "external_id"
                    ):
                        attack_id = ref["external_id"]
                        break
            if not attack_id:
                continue
            name = obj.get("name", "")
            desc = (obj.get("description") or "").strip()
            tech_tactics: list[str] = []
            seen: set[str] = set()
            for phase in obj.get("kill_chain_phases") or []:
                tactic_name = None
                phase_id = phase.get("kill_chain_phase_id")
                if phase_id and phase_id in tactics_by_id:
                    tactic_name = tactics_by_id[phase_id]["name"]
                if not tactic_name:
                    phase_name = phase.get("phase_name")
                    if phase_name and phase_name in tactics_by_shortname:
                        tactic_name = tactics_by_shortname[phase_name]
                if tactic_name and tactic_name not in seen:
                    seen.add(tactic_name)
                    tech_tactics.append(tactic_name)
            techniques.append(
                {
                    "id": attack_id,
                    "name": name,
                    "description": desc,
                    "tactic_names": tech_tactics,
                }
            )

    # Add platform-specific extensions (e.g. T1573.003) not in official ATT&CK
    seen_ids = {t["id"] for t in techniques}
    for ext in PLATFORM_ATTACK_EXTENSIONS:
        if ext["id"] not in seen_ids:
            techniques.append(ext)
            seen_ids.add(ext["id"])

    # Sort by ATT&CK ID (e.g. T1001, T1001.001)
    techniques.sort(key=lambda t: (t["id"].replace(".", "\x00"), t["name"]))
    return tactics_by_id, techniques, version


def build_attack_techniques_md(
    tactics_by_id: dict,
    techniques: list[dict],
    version: str,
) -> str:
    """Build Markdown content for attack_techniques.md.

    Args:
        tactics_by_id: Map of tactic STIX id to {name, shortname}.
        techniques: List of {id, name, description, tactic_names}.
        version: ATT&CK version string.

    Returns:
        Full Markdown string.
    """
    lines = [
        "# MITRE ATT&CK Enterprise – Techniques",
        "",
        "*Technique and tactic names/IDs from MITRE ATT&CK®. "
        "See [attack.mitre.org](https://attack.mitre.org/) and "
        "[ATT&CK Terms of Use](https://attack.mitre.org/resources/terms-of-use/).*",
        "",
        f"**ATT&CK version:** {version or 'unknown'}",
        "",
        "---",
        "",
    ]
    for t in techniques:
        tactics_str = ", ".join(t["tactic_names"]) if t["tactic_names"] else "—"
        lines.append(f"## {t['id']} – {t['name']}")
        lines.append("")
        lines.append(f"**Tactics:** {tactics_str}")
        lines.append("")
        if t["description"]:
            # One-line description for RAG
            desc_line = t["description"].split("\n")[0].strip()[:500]
            if len((t["description"] or "").split("\n")[0]) > 500:
                desc_line += "…"
            lines.append(desc_line)
            lines.append("")
        lines.append("")
    return "\n".join(lines)


def build_attack_terms(
    stix_url: str = ENTERPRISE_ATTACK_URL,
    output_path: Path | None = None,
) -> bool:
    """Fetch ATT&CK STIX, parse, and write attack_techniques.md to the knowledge base.

    Args:
        stix_url: URL to Enterprise ATT&CK STIX JSON.
        output_path: Path for output Markdown. Default: data/knowledge_base/attack_techniques.md.

    Returns:
        True if build and write succeeded, False otherwise.
    """
    output_path = output_path or settings.KNOWLEDGE_BASE_DIR / "attack_techniques.md"
    try:
        logger.info("Fetching ATT&CK STIX from {}", stix_url)
        bundle = fetch_stix(stix_url)
        tactics_by_id, techniques, version = parse_bundle(bundle)
        logger.info(
            "Parsed {} techniques, ATT&CK version {}", len(techniques), version or "?"
        )
        content = build_attack_techniques_md(tactics_by_id, techniques, version)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content, encoding="utf-8")
        logger.info("Wrote {} ({} chars)", output_path, len(content))
        return True
    except Exception as e:
        logger.error("Failed to build ATT&CK terms: {}", e)
        return False


def main() -> int:
    """CLI entry point. Returns 0 on success, 1 on failure."""
    setup_logging()
    success = build_attack_terms()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
