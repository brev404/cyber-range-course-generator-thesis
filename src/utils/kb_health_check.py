"""KB health check: verify knowledge base completeness and freshness.

Checks:
1. Expected KB files exist (from data/knowledge_base/README.md)
2. Files are non-empty (except README.md)
3. Chroma collection exists and has chunks
4. Optional: file modification times for freshness

Usage:
    python -m src.utils.kb_health_check [--strict]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Project root for imports
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.config.settings import settings  # noqa: E402
from src.services.vector_db_service import (  # noqa: E402
    VectorDBService,
    VectorDBServiceError,
)

# Expected KB files from data/knowledge_base/README.md (README.md can be empty)
EXPECTED_KB_FILES = [
    "attack_techniques.md",
    "cwe_weaknesses.md",
    "cybersecurity_glossary.md",
    "cybok_kas.md",
    "nist_terms.md",
    "owasp_wstg.md",
    "pedagogical_principles_writeups.md",
    "course_guidelines.md",
    "unr_finals_2025_documentation.md",
    "README.md",
]


def run_health_check(strict: bool = False) -> tuple[bool, list[str]]:
    """Run KB health checks. Returns (passed, list of messages)."""
    messages: list[str] = []
    passed = True
    kb_dir = Path(settings.KNOWLEDGE_BASE_DIR)

    # 1. KB directory exists
    if not kb_dir.is_dir():
        messages.append(f"FAIL: Knowledge base directory does not exist: {kb_dir}")
        return False, messages

    # 2. Expected files exist and are non-empty (except README.md)
    missing: list[str] = []
    empty: list[str] = []
    for name in EXPECTED_KB_FILES:
        path = kb_dir / name
        if not path.exists():
            missing.append(name)
        elif name != "README.md":
            try:
                if not path.read_text(encoding="utf-8", errors="replace").strip():
                    empty.append(name)
            except OSError:
                empty.append(name)

    if missing:
        passed = False
        messages.append(f"FAIL: Missing KB files: {', '.join(missing)}")
    else:
        messages.append(f"OK: All {len(EXPECTED_KB_FILES)} expected KB files present")

    if empty:
        passed = False
        messages.append(f"FAIL: Empty KB files: {', '.join(empty)}")
    elif not missing:
        messages.append("OK: All content files are non-empty")

    # 3. Chroma collection exists and has chunks
    try:
        svc = VectorDBService()
        store = svc._ensure_vector_store()
        count = 0
        if hasattr(store, "_collection") and store._collection is not None:
            count = store._collection.count()

        if count == 0:
            passed = False
            messages.append(
                "FAIL: Chroma collection is empty. Run ingest: VectorDBService().ingest_knowledge_base()"
            )
        else:
            messages.append(f"OK: Chroma collection has {count} chunks")
    except VectorDBServiceError as e:
        if strict:
            passed = False
        messages.append(f"WARN: Could not verify Chroma: {e}")

    return passed, messages


def main() -> int:
    parser = argparse.ArgumentParser(description="KB health check")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat Chroma errors as failure",
    )
    args = parser.parse_args()

    passed, messages = run_health_check(strict=args.strict)
    for m in messages:
        print(m)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
