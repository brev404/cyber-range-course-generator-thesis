"""Extract NIST IR 7511 Revision 4 (Cybersecurity Lexicon) to Markdown for the knowledge base.

Reads the NIST PDF from docs/cybersecurity_dictionary/, extracts text from all pages
(preserving case), and writes data/knowledge_base/nist_terms.md with attribution.
Used to populate the cybersecurity dictionary for RAG (see data/knowledge_base/ for the generated file).
"""

import sys
from pathlib import Path

from loguru import logger

# Add project root for imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config.settings import settings  # noqa: E402
from src.utils.logging_config import setup_logging  # noqa: E402

try:
    import pdfplumber
except ImportError:
    logger.error("pdfplumber not found. Install with: pip install pdfplumber")
    sys.exit(1)


def extract_text_from_pdf_full(pdf_path: Path) -> str:
    """Extract text from all pages of a PDF, preserving case.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Concatenated text from all pages, or empty string on failure.
    """
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            logger.info(
                "Extracting text from {} pages of {}", len(pdf.pages), pdf_path.name
            )
            for i, page in enumerate(pdf.pages):
                try:
                    page_text = page.extract_text() or ""
                    text += page_text
                    if (i + 1) % 20 == 0:
                        logger.debug("  Processed {} pages", i + 1)
                except Exception as e:
                    logger.warning("  Page {} extraction failed: {}", i + 1, e)
        return text.strip()
    except Exception as e:
        logger.error("Failed to open PDF {}: {}", pdf_path, e)
        return ""


def extract_nist_lexicon(
    pdf_path: Path | None = None,
    output_path: Path | None = None,
) -> bool:
    """Extract NIST IR 7511 r4 lexicon to a Markdown file in the knowledge base.

    Args:
        pdf_path: Path to NIST.IR.7511r4.pdf. Default: docs/cybersecurity_dictionary/NIST.IR.7511r4.pdf.
        output_path: Path for output Markdown. Default: data/knowledge_base/nist_terms.md.

    Returns:
        True if extraction and write succeeded, False otherwise.
    """
    base = settings.BASE_DIR
    pdf_path = (
        pdf_path or base / "docs" / "cybersecurity_dictionary" / "NIST.IR.7511r4.pdf"
    )
    output_path = output_path or settings.KNOWLEDGE_BASE_DIR / "nist_terms.md"

    if not pdf_path.is_file():
        logger.error("NIST PDF not found: {}", pdf_path)
        return False

    text = extract_text_from_pdf_full(pdf_path)
    if not text:
        logger.warning("No text extracted from {}", pdf_path.name)
        # Still write header so file exists and attribution is clear
        text = "(No text could be extracted from the PDF. Ensure pdfplumber is installed and the file is readable.)"

    attribution = (
        "*Source: NIST Internal Report 7511 Revision 4 (Cybersecurity Lexicon). "
        "See https://csrc.nist.gov/publications/detail/ir/7511/rev-4/final.*\n\n"
        "---\n\n"
    )
    content = "# NIST Cybersecurity Lexicon (IR 7511 Rev. 4)\n\n" + attribution + text

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    logger.info("Wrote {} ({} chars)", output_path, len(content))
    return True


def main() -> int:
    """CLI entry point. Returns 0 on success, 1 on failure."""
    setup_logging()
    success = extract_nist_lexicon()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
