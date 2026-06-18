"""Build CWE weaknesses Markdown from CWE XML for the knowledge base.

Fetches the CWE catalog XML (cwec_latest.xml.zip) from cwe.mitre.org, parses
Weakness elements, and writes data/knowledge_base/cwe_weaknesses.md with
attribution and version. See data/knowledge_base/ for the generated file.
"""

import sys
import xml.etree.ElementTree as ET
import zipfile
from io import BytesIO
from pathlib import Path
from urllib.request import Request, urlopen

from loguru import logger

# Add project root for imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config.settings import settings  # noqa: E402
from src.utils.logging_config import setup_logging  # noqa: E402

CWE_XML_ZIP_URL = "https://cwe.mitre.org/data/xml/cwec_latest.xml.zip"
CWE_NS = "http://cwe.mitre.org/cwe-7"
MAX_DESC_LEN = 500


def fetch_cwe_zip(url: str = CWE_XML_ZIP_URL) -> bytes:
    """Fetch CWE XML zip from URL.

    Args:
        url: URL to the CWE XML zip file.

    Returns:
        Raw bytes of the zip file.

    Raises:
        Exception: On fetch failure.
    """
    req = Request(url, headers={"Accept": "application/zip"})
    with urlopen(req, timeout=120) as resp:
        return resp.read()


def _local_tag(elem: ET.Element) -> str:
    """Return local tag name without namespace."""
    return elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag


def parse_cwe_xml(xml_bytes: bytes) -> tuple[list[dict], str, str]:
    """Parse CWE catalog XML and extract weaknesses.

    Args:
        xml_bytes: Raw XML bytes (single file content, no zip).

    Returns:
        (weaknesses_list, version, date). weaknesses_list is list of
        {id, name, description}. Skip Deprecated/Obsolete.
    """
    root = ET.fromstring(xml_bytes)
    version = root.get("Version", "")
    date = root.get("Date", "")

    weaknesses: list[dict] = []
    for elem in root.iter():
        if _local_tag(elem) != "Weakness":
            continue
        status = elem.get("Status", "")
        if status in ("Deprecated", "Obsolete"):
            continue
        wid = elem.get("ID", "")
        name = elem.get("Name", "")
        desc_el = None
        for child in elem:
            if _local_tag(child) == "Description":
                desc_el = child
                break
        desc = (desc_el.text or "").strip() if desc_el is not None else ""
        if not wid:
            continue
        weaknesses.append(
            {
                "id": wid,
                "name": name,
                "description": desc,
            }
        )

    # Sort by numeric ID
    def sort_key(item: dict) -> tuple[int, str]:
        try:
            return (int(item["id"]), item["name"])
        except ValueError:
            return (0, item["name"])

    weaknesses.sort(key=sort_key)
    return weaknesses, version, date


def build_cwe_weaknesses_md(
    weaknesses: list[dict],
    version: str,
    date: str,
) -> str:
    """Build Markdown content for cwe_weaknesses.md.

    Args:
        weaknesses: List of {id, name, description}.
        version: CWE catalog version.
        date: CWE catalog date.

    Returns:
        Full Markdown string.
    """
    lines = [
        "# CWE – Common Weakness Enumeration",
        "",
        "*Weakness names/IDs from CWE (Common Weakness Enumeration). "
        "See [cwe.mitre.org](https://cwe.mitre.org/index.html) and "
        "[CWE Terms of Use](https://cwe.mitre.org/about/termsofuse.html).*",
        "",
        f"**CWE version:** {version}  ",
        f"**Date:** {date}",
        "",
        "---",
        "",
    ]
    for w in weaknesses:
        cwe_id = f"CWE-{w['id']}"
        lines.append(f"## {cwe_id} – {w['name']}")
        lines.append("")
        if w["description"]:
            desc = w["description"].replace("\n", " ").strip()
            if len(desc) > MAX_DESC_LEN:
                desc = desc[:MAX_DESC_LEN].rsplit(" ", 1)[0] + "…"
            lines.append(desc)
            lines.append("")
        lines.append("")
    return "\n".join(lines)


def build_cwe_terms(
    zip_url: str = CWE_XML_ZIP_URL,
    output_path: Path | None = None,
) -> bool:
    """Fetch CWE XML zip, parse, and write cwe_weaknesses.md to the knowledge base.

    Args:
        zip_url: URL to CWE XML zip.
        output_path: Path for output Markdown. Default: data/knowledge_base/cwe_weaknesses.md.

    Returns:
        True if build and write succeeded, False otherwise.
    """
    output_path = output_path or settings.KNOWLEDGE_BASE_DIR / "cwe_weaknesses.md"
    try:
        logger.info("Fetching CWE XML from {}", zip_url)
        zip_bytes = fetch_cwe_zip(zip_url)
        with zipfile.ZipFile(BytesIO(zip_bytes), "r") as z:
            names = z.namelist()
            xml_name = next((n for n in names if n.endswith(".xml")), names[0])
            with z.open(xml_name) as f:
                xml_bytes = f.read()
        logger.info("Parsing CWE XML ({} bytes)", len(xml_bytes))
        weaknesses, version, date = parse_cwe_xml(xml_bytes)
        logger.info(
            "Parsed {} weaknesses, CWE version {} ({})", len(weaknesses), version, date
        )
        content = build_cwe_weaknesses_md(weaknesses, version, date)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content, encoding="utf-8")
        logger.info("Wrote {} ({} chars)", output_path, len(content))
        return True
    except Exception as e:
        logger.error("Failed to build CWE terms: {}", e)
        return False


def main() -> int:
    """CLI entry point. Returns 0 on success, 1 on failure."""
    setup_logging()
    success = build_cwe_terms()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
