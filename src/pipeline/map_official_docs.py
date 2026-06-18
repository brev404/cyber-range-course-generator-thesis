import logging
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict

from loguru import logger

# Add project root to Python path for imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config.settings import settings  # noqa: E402
from src.utils.logging_config import setup_logging  # noqa: E402

try:
    import pdfplumber
except ImportError:
    print("ERROR: pdfplumber not found. Please install it with: pip install pdfplumber")
    sys.exit(1)

try:
    import pdf2image
    import pytesseract
    from PIL import Image  # noqa: F401

    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("WARNING: OCR libraries not found. Text extraction will be limited.")
    print("For full functionality, install: pip install pytesseract pdf2image pillow")
    print("Also ensure Tesseract is installed on your system.")

# --- Configuration ---
OFFICIAL_DOCS_DIR = settings.OFFICIAL_DOCS_SOURCE
PROCESSED_CHALLENGES_ROOT = settings.PROCESSED_DIR / settings.RAW_CHALLENGES_SOURCE.name


def extract_text_from_pdf(
    pdf_path: Path,
    max_pages: int = 10,
    ocr_only: bool = False,
    max_ocr_pages: int = 2,
    ocr_dpi: int = 150,
) -> str:
    """Extracts text from a PDF. When ocr_only=True or pdfplumber yields no text, uses OCR (slower but works on image-only PDFs)."""
    pdf_path = Path(pdf_path)
    try:
        text = ""

        if not ocr_only:
            # Try pdfplumber on first page only; if empty, skip rest and go to OCR
            with pdfplumber.open(pdf_path) as pdf:
                num_pages = min(len(pdf.pages), max_pages)
                for i in range(num_pages):
                    try:
                        page_text = pdf.pages[i].extract_text() or ""
                        text += page_text
                        if text.strip():
                            break
                    except Exception:
                        pass
                if not text.strip():
                    text = ""

        if (not text or len(text.strip()) == 0) and TESSERACT_AVAILABLE:
            try:
                n_ocr = min(max_ocr_pages, max_pages)
                images = pdf2image.convert_from_path(
                    str(pdf_path), first_page=1, last_page=n_ocr, dpi=ocr_dpi
                )
                ocr_text = ""
                for img in images[:n_ocr]:
                    try:
                        ocr_text += pytesseract.image_to_string(img, lang="eng")
                    except Exception:
                        pass
                text = ocr_text
            except Exception as ocr_e:
                logger.warning(f"OCR failed for {pdf_path.name}: {ocr_e}")

        return text.lower() if text else ""
    except Exception as e:
        logger.error(f"Failed to extract text from {pdf_path.name}: {e}")
        return ""


def map_official_docs(max_workers: int = 4):
    """
    Scans the official documentation directory and maps PDFs to processed challenges
    based on keyword matching in the PDF content and filename.

    Args:
        max_workers: Number of parallel workers for PDF text extraction (default 4).
            Set to 1 for sequential processing; increase to speed up on multi-core machines.
    """
    if not OFFICIAL_DOCS_DIR.exists():
        logger.error(
            f"Official documentation source directory does not exist: {OFFICIAL_DOCS_DIR}"
        )
        return

    if not PROCESSED_CHALLENGES_ROOT.exists():
        logger.error(
            f"Processed challenges root does not exist: {PROCESSED_CHALLENGES_ROOT}"
        )
        return

    # 1. Get list of all processed challenges
    challenges = []
    for cat_dir in PROCESSED_CHALLENGES_ROOT.iterdir():
        if not cat_dir.is_dir():
            continue
        for chal_dir in cat_dir.iterdir():
            if not chal_dir.is_dir():
                continue
            challenges.append(
                {
                    "name": chal_dir.name,
                    "id": f"{cat_dir.name}/{chal_dir.name}",
                    "path": chal_dir,
                }
            )

    logger.info(f"Found {len(challenges)} processed challenges to map against.")
    if logger.level == logging.DEBUG:
        for chal in challenges:
            logger.debug(f"  Challenge: {chal['id']}")

    # 2. Get PDFs and extract text in parallel
    pdf_files = list(OFFICIAL_DOCS_DIR.glob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files in {OFFICIAL_DOCS_DIR}")

    workers = max(1, min(max_workers, len(pdf_files) or 1))
    pdf_text_by_path: Dict[Path, str] = {}
    if workers <= 1:
        for pdf_path in pdf_files:
            logger.info(f"Analyzing PDF: {pdf_path.name}")
            pdf_text_by_path[pdf_path] = extract_text_from_pdf(pdf_path)
    else:
        logger.info(
            f"Extracting text from {len(pdf_files)} PDFs using {workers} workers..."
        )
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_pdf = {
                executor.submit(extract_text_from_pdf, p): p for p in pdf_files
            }
            for future in as_completed(future_to_pdf):
                pdf_path = future_to_pdf[future]
                try:
                    pdf_text_by_path[pdf_path] = future.result()
                    logger.debug(f"Extracted: {pdf_path.name}")
                except Exception as e:
                    logger.error(f"Failed to extract {pdf_path.name}: {e}")
                    pdf_text_by_path[pdf_path] = ""

    mapped_challenge_ids = set()
    mapped_count = 0

    for pdf_path in pdf_files:
        logger.info(f"Matching PDF: {pdf_path.name}")
        pdf_text = pdf_text_by_path.get(pdf_path, "")

        # If text extraction returns empty, we'll use filename-based matching as fallback
        text_extraction_empty = not pdf_text or len(pdf_text.strip()) == 0

        if text_extraction_empty:
            logger.info(
                f"Using filename-based matching for '{pdf_path.name}' (no extractable text)"
            )

        pdf_stem_lower = pdf_path.stem.lower()
        pdf_matched_to_any_challenge = False

        # Log metadata only (no PDF content) to avoid sensitive data in logs
        if logger.level == logging.DEBUG and pdf_text:
            logger.debug(f"PDF '{pdf_path.name}': extracted {len(pdf_text)} chars")

        for chal in challenges:
            challenge_name_lower = chal["name"].lower()

            # Extract just the base name without brackets and metadata
            # e.g., "osint_CV-din-trecut [easy or medium] [STATIC]" -> "osint_CV-din-trecut"
            base_challenge_name = challenge_name_lower.split("[")[0].strip()

            challenge_id_lower = chal["id"].lower()  # e.g., "category/challenge_name"

            # Enhanced Matching Heuristics:
            # 1. Match challenge name directly in PDF text
            # 2. Match challenge name in PDF filename
            # 3. Match challenge ID in PDF text
            # 4. Match parts of challenge ID in PDF filename

            is_match = False
            match_reason = ""

            challenge_name_no_hyphens = base_challenge_name.replace(
                "-", " "
            )  # For flexible text matching
            challenge_name_underscores_to_spaces = base_challenge_name.replace("_", " ")

            # Test all matching patterns
            test_patterns = [
                (base_challenge_name, "base name (exact)"),
                (challenge_name_no_hyphens, "base name (no hyphens)"),
                (
                    challenge_name_underscores_to_spaces,
                    "base name (underscores as spaces)",
                ),
            ]

            if logger.level == logging.DEBUG:
                logger.debug(f"  Testing '{chal['id']}' against PDF '{pdf_path.name}'")
                logger.debug(
                    f"    Base name: '{base_challenge_name}' | No hyphens: '{challenge_name_no_hyphens}' | Underscores: '{challenge_name_underscores_to_spaces}'"
                )

            for pattern, pattern_name in test_patterns:
                if pattern and pdf_text and pattern in pdf_text:
                    is_match = True
                    match_reason = f"challenge {pattern_name} in PDF text"
                    if logger.level == logging.DEBUG:
                        logger.debug(f"    ✓ MATCH FOUND: {match_reason}")
                    break

            if not is_match and base_challenge_name in pdf_stem_lower:
                is_match = True
                match_reason = "challenge name in PDF filename"
                if logger.level == logging.DEBUG:
                    logger.debug(f"    ✓ MATCH FOUND: {match_reason}")

            if not is_match and pdf_text and challenge_id_lower in pdf_text:
                is_match = True
                match_reason = "challenge ID in PDF text"
                if logger.level == logging.DEBUG:
                    logger.debug(f"    ✓ MATCH FOUND: {match_reason}")

            if not is_match and challenge_id_lower.split("/")[-1] in pdf_stem_lower:
                is_match = True
                match_reason = "challenge name from ID in PDF filename"
                if logger.level == logging.DEBUG:
                    logger.debug(f"    ✓ MATCH FOUND: {match_reason}")

            if is_match:
                target_dir = chal["path"] / "cyberedu" / "official-docs"
                target_dir.mkdir(parents=True, exist_ok=True)

                dest_path = target_dir / pdf_path.name
                if dest_path.exists():
                    logger.debug(
                        f"  [SKIPPED] PDF '{pdf_path.name}' already mapped to '{chal['id']}'"
                    )
                else:
                    try:
                        shutil.copy2(pdf_path, dest_path)
                        logger.info(
                            f"  [MATCH] Mapped {pdf_path.name} to challenge: {chal['id']} (Reason: {match_reason})"
                        )
                        pdf_matched_to_any_challenge = True
                        mapped_count += 1
                        mapped_challenge_ids.add(chal["id"])
                        # Keep iterating to see if one PDF matches multiple challenges
                    except Exception as e:
                        logger.error(
                            f"  [ERROR] Failed to copy {pdf_path.name} to {chal['id']}: {e}"
                        )

        if not pdf_matched_to_any_challenge:
            logger.warning(
                f"  [PDF UNMATCHED] Could not identify any challenge for PDF: {pdf_path.name}"
            )

    logger.info(f"Mapping complete. Successfully mapped {mapped_count} documents.")

    # Report on challenges that did not receive any official documentation
    unmapped_challenges = [
        chal for chal in challenges if chal["id"] not in mapped_challenge_ids
    ]
    if unmapped_challenges:
        logger.info(
            f"Found {len(unmapped_challenges)} challenges without official documentation mapped:"
        )
        for chal in unmapped_challenges:
            logger.info(f"  - Challenge: {chal['id']}")
    else:
        logger.info(
            "All processed challenges have at least one official document mapped."
        )


def run_mapping(debug: bool = False, max_workers: int = 4):
    """Main function to run the official documentation mapping.

    Args:
        debug: If True, set log level to DEBUG.
        max_workers: Number of parallel workers for PDF text extraction (default 4).
    """
    global logger
    logger = setup_logging("map_official_docs")

    if debug:
        # Set DEBUG level for more detailed output
        for handler in logger.handlers:
            handler.setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        logger.info("Debug mode enabled - showing detailed matching information")

    map_official_docs(max_workers=max_workers)


if __name__ == "__main__":
    import sys

    debug_mode = "--debug" in sys.argv
    run_mapping(debug=debug_mode)
