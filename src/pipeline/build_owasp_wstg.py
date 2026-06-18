"""Build OWASP WSTG scenarios Markdown for the knowledge base.

Fetches the OWASP WSTG document README from GitHub, parses the table of contents
to extract scenario identifiers (WSTG-<category>-<number>) and titles, fetches
each scenario's full content to extract descriptions, and writes
data/knowledge_base/owasp_wstg.md. See data/knowledge_base/ for the generated file.
"""

import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import Request, urlopen

from loguru import logger

# Add project root for imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config.settings import settings  # noqa: E402
from src.utils.logging_config import setup_logging  # noqa: E402

WSTG_BASE_URL = "https://raw.githubusercontent.com/OWASP/wstg/master/document/"
WSTG_DOC_README_URL = f"{WSTG_BASE_URL}README.md"

# Section 4.x category prefix -> WSTG 4-char code (platform uses IDNT/ATHN; see docs/reference/archive/TAG_DICTIONARY_FINDINGS_REPORT.md)
SECTION_TO_CAT: dict[str, str] = {
    "4.1": "INFO",  # Information Gathering
    "4.2": "CONF",  # Configuration and Deployment Management
    "4.3": "IDNT",  # Identity Management (platform uses IDNT; legacy IDEN)
    "4.4": "ATHN",  # Authentication (platform uses AUTH; legacy AUTH)
    "4.5": "ATHZ",  # Authorization
    "4.6": "SESS",  # Session Management
    "4.7": "INPV",  # Input Validation
    "4.8": "ERRH",  # Error Handling
    "4.9": "CRYP",  # Weak Cryptography
    "4.10": "BUSL",  # Business Logic
    "4.11": "CLNT",  # Client-side
    "4.12": "API",  # API Testing
}
WSTG_VERSION = (
    "4.2"  # Document version for attribution; update when WSTG releases change
)

# Max chars for scenario description in KB (keeps RAG chunks focused)
MAX_DESCRIPTION_CHARS = 600

# Concurrent fetches for scenario content
FETCH_MAX_WORKERS = 5
FETCH_TIMEOUT = 30


def fetch_url(url: str) -> str:
    """Fetch URL content. Raises on failure."""
    req = Request(url, headers={"Accept": "text/plain"})
    with urlopen(req, timeout=FETCH_TIMEOUT) as resp:
        return resp.read().decode("utf-8")


def fetch_wstg_readme(url: str = WSTG_DOC_README_URL) -> str:
    """Fetch WSTG document README (table of contents) from URL."""
    return fetch_url(url)


def parse_wstg_toc(md_content: str) -> list[dict]:
    """Parse WSTG README markdown to extract scenario IDs, titles, and file paths.

    Expects lines like:
    #### 4.1.1 [Title](path/to/file.md)

    Returns:
        List of {id, title, section, category, rel_path} where rel_path is the
        path relative to document/ for fetching scenario content.
    """
    scenarios: list[dict] = []
    # Match #### 4.x.y [Title](path) or ##### 4.x.y.z [Title](path)
    pattern = re.compile(
        r"^#{4,6}\s+(4\.\d+(?:\.\d+)*(?:\.\d+)?)\s+\[([^\]]+)\]\(([^)]+)\)",
        re.MULTILINE,
    )
    cat_count: dict[str, int] = {}

    for m in pattern.finditer(md_content):
        section = m.group(1)
        title = m.group(2).strip()
        rel_path = m.group(3).strip()
        # Map 4.1, 4.1.1 -> INFO; 4.2.x -> CONF; etc.
        prefix = section.split(".")[:2]
        key = ".".join(prefix)
        cat_code = SECTION_TO_CAT.get(key)
        if not cat_code:
            continue
        cat_count[cat_code] = cat_count.get(cat_code, 0) + 1
        num = cat_count[cat_code]
        scenario_id = f"WSTG-{cat_code}-{num:02d}"
        scenarios.append(
            {
                "id": scenario_id,
                "title": title,
                "section": section,
                "category": cat_code,
                "rel_path": rel_path,
            }
        )
    return scenarios


def extract_summary_from_md(md_content: str) -> str:
    """Extract and clean the Summary section from a WSTG scenario markdown.

    Takes content between ## Summary and the next ## header. Cleans markdown,
    strips to first 1-2 paragraphs, and truncates to MAX_DESCRIPTION_CHARS.
    """
    match = re.search(
        r"## Summary\s*\n(.*?)(?=\n## |\Z)",
        md_content,
        re.DOTALL,
    )
    if not match:
        return ""

    raw = match.group(1).strip()

    # Remove code blocks (```...```)
    raw = re.sub(r"```[\s\S]*?```", "", raw)
    # Remove inline code backticks but keep text
    raw = re.sub(r"`([^`]+)`", r"\1", raw)
    # Remove blockquotes
    raw = re.sub(r"^>\s*", "", raw, flags=re.MULTILINE)
    # Remove links but keep text: [text](url) -> text
    raw = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", raw)
    # Collapse multiple newlines
    raw = re.sub(r"\n{2,}", "\n\n", raw)
    # Split into paragraphs
    paragraphs = [p.strip() for p in raw.split("\n\n") if p.strip()]

    if not paragraphs:
        return ""

    # Take first 1-2 paragraphs, cap total length
    result = paragraphs[0]
    if len(paragraphs) > 1 and len(result) < MAX_DESCRIPTION_CHARS - 100:
        second = paragraphs[1]
        # Skip very short fragments (e.g. "Example:")
        if len(second) > 30 and not second.startswith("Example"):
            result = result + " " + second

    # Truncate
    if len(result) > MAX_DESCRIPTION_CHARS:
        result = result[: MAX_DESCRIPTION_CHARS - 3].rsplit(" ", 1)[0] + "..."

    return result.strip()


def fetch_scenario_description(rel_path: str) -> str:
    """Fetch a scenario file and extract its Summary as description."""
    url = WSTG_BASE_URL + rel_path
    try:
        content = fetch_url(url)
        return extract_summary_from_md(content)
    except Exception as e:
        logger.warning("Could not fetch {}: {}", rel_path, e)
        return ""


def enrich_scenarios_with_descriptions(scenarios: list[dict]) -> None:
    """Fetch each scenario's content and add 'description' to each dict."""
    with ThreadPoolExecutor(max_workers=FETCH_MAX_WORKERS) as executor:
        future_to_idx = {
            executor.submit(fetch_scenario_description, s["rel_path"]): i
            for i, s in enumerate(scenarios)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                desc = future.result()
                scenarios[idx]["description"] = desc
            except Exception as e:
                logger.warning(
                    "Failed to enrich scenario {}: {}", scenarios[idx]["id"], e
                )
                scenarios[idx]["description"] = ""


def build_owasp_wstg_md(scenarios: list[dict], version: str = WSTG_VERSION) -> str:
    """Build Markdown content for owasp_wstg.md with descriptions."""
    lines = [
        "# OWASP Web Security Testing Guide (WSTG) – Scenarios",
        "",
        "*Scenario identifiers and content from OWASP Web Security Testing Guide. "
        "See [OWASP WSTG](https://owasp.org/www-project-web-security-testing-guide/). "
        "License: CC BY-SA 4.0.*",
        "",
        f"**WSTG version:** {version}",
        "",
        "Identifiers follow the format `WSTG-<category>-<number>` (e.g. WSTG-INFO-02). "
        "Use versioned identifiers (e.g. WSTG-v42-INFO-02) when referencing a specific release.",
        "",
        "---",
        "",
    ]
    current_cat = None
    for s in scenarios:
        if s["category"] != current_cat:
            current_cat = s["category"]
            lines.append(f"## Category: {current_cat}")
            lines.append("")
        lines.append(f"### {s['id']} – {s['title']}")
        lines.append("")
        lines.append(f"**Section:** {s['section']}  ")
        lines.append("")
        desc = s.get("description", "")
        if desc:
            lines.append(desc)
            lines.append("")
        lines.append("")
    return "\n".join(lines)


def build_owasp_wstg(
    readme_url: str = WSTG_DOC_README_URL,
    output_path: Path | None = None,
    version: str = WSTG_VERSION,
) -> bool:
    """Fetch WSTG README, parse TOC, enrich with scenario descriptions, write owasp_wstg.md."""
    output_path = output_path or settings.KNOWLEDGE_BASE_DIR / "owasp_wstg.md"
    try:
        logger.info("Fetching WSTG document README from {}", readme_url)
        content = fetch_wstg_readme(readme_url)
        scenarios = parse_wstg_toc(content)
        logger.info("Parsed {} WSTG scenarios", len(scenarios))

        logger.info("Fetching scenario content for descriptions...")
        enrich_scenarios_with_descriptions(scenarios)

        md_content = build_owasp_wstg_md(scenarios, version)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(md_content, encoding="utf-8")
        logger.info("Wrote {} ({} chars)", output_path, len(md_content))
        return True
    except Exception as e:
        logger.error("Failed to build OWASP WSTG terms: {}", e)
        return False


def main() -> int:
    """CLI entry point. Returns 0 on success, 1 on failure."""
    setup_logging()
    success = build_owasp_wstg()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
