import os
import shutil
import sys
from pathlib import Path
from typing import Tuple

from loguru import logger

# Add project root to Python path for imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config.settings import settings  # noqa: E402
from src.utils.logging_config import setup_logging  # noqa: E402

# --- Configuration ---
SOURCE_ROOT_DIR = settings.RAW_CHALLENGES_SOURCE
DESTINATION_ROOT_DIR = settings.PROCESSED_DIR / settings.RAW_CHALLENGES_SOURCE.name

# Directories to ignore during recursive walk
IGNORE_DIRS = {
    ".git",
    ".github",
    ".vscode",
    ".idea",
    "__pycache__",
    "node_modules",
    "venv",
    ".venv",
    "env",
    "target",
    "build",
    "dist",
    "dev",
}

# Naming Heuristics for mapping to standard CyberEDU names
SOLVER_NAMES = {
    "solve.py",
    "solver.py",
    "exploit.py",
    "exp.py",  # common CTF abbreviation for exploit script
    "solution.py",
    "solve.sh",
    "exploit.sh",
    "exp.sh",
    "solve.sage",
    "exploit.sage",
    "exp.sage",
}
# Solver stem prefixes: filenames like solve_stage1.py, exploit_v2.py, exp_rsa.py
_SOLVER_STEMS = {"solve", "solver", "exploit", "exp", "solution", "verify", "evaluate"}
_SOLVER_STEM_PREFIXES = ("solve_", "exploit_", "exp_")
# Parent directory names that indicate writeup/solution context (.py/.sh/.sage only)
_SOLVER_PARENT_DIRS = {
    "solver",
    "exploit",
    "solution",
    "writeup",
    "write-up",
    "evaluation",
}
_SOLVER_PARENT_EXTS = {".py", ".sh", ".sage", ".rb"}
# Directories that indicate challenge infrastructure — solver detection excluded here
_DEPLOY_CONTEXT_DIRS = {"deploy", "docker", "app"}

FLAG_NAMES = {"challenge-flags.txt", "flag.txt", "flags.txt", "flag"}
DESCRIPTION_NAMES = {
    "description.md",
    "description.txt",  # plain-text variant common in contest repos
    "desc.md",
    "desc.txt",
    "descriere.txt",
    "readme.md",
    "readme",
    "descriere.md",
    "task.md",
    "note.txt",
    "notes.txt",
}
WRITEUP_NAMES = {"writeup.md", "write-up.md", "solution.md", "writeup.txt"}

# File extensions mapping
SOURCE_EXTENSIONS = {
    ".py",
    ".c",
    ".cpp",
    ".java",
    ".js",
    ".sh",
    ".rb",
    ".go",
    ".rs",
    ".asm",
    ".s",
    ".sage",
}
BINARY_EXTENSIONS = {".exe", ".elf", ".dll", ".so", ".bin", ".out", ".o"}
MARKDOWN_EXTENSIONS = {".md"}
PUBLIC_ATTACHMENT_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".pdf",
    ".zip",
    ".tar",
    ".gz",
    ".7z",
    ".rar",
}


def get_target_info(file_path: Path, rel_path: Path) -> Tuple[str, str]:
    """
    Determines the target subdirectory and standardized filename.
    Returns (subdir_key, target_filename)
    """
    name_lower = file_path.name.lower()
    ext = file_path.suffix.lower()
    stem_lower = file_path.stem.lower()
    parent_names = [p.lower() for p in rel_path.parts[:-1]]

    # 1. Check for Core Write-up Files (Renaming to standard)
    is_in_deploy_context = any(p in _DEPLOY_CONTEXT_DIRS for p in parent_names)
    is_solver_stem = stem_lower in _SOLVER_STEMS or any(
        stem_lower.startswith(p) for p in _SOLVER_STEM_PREFIXES
    )
    parent_solver_match = (
        any(p in _SOLVER_PARENT_DIRS for p in parent_names)
        and ext in _SOLVER_PARENT_EXTS
    )
    if (
        name_lower in SOLVER_NAMES or is_solver_stem or parent_solver_match
    ) and not is_in_deploy_context:
        if ext in SOURCE_EXTENSIONS or ext == "":
            # Standardize solver name but keep extension (e.g., solve.py, solve.sage)
            target_ext = ext if ext else ".py"
            return "write-up", f"solve{target_ext}"

    if name_lower in FLAG_NAMES:
        return "write-up", "challenge-flags.txt"

    if name_lower in DESCRIPTION_NAMES:
        # If it's a markdown file, we prefer it as description.md
        if ext == ".md":
            return "write-up", "description.md"
        return (
            "write-up",
            file_path.name,
        )  # Keep original name (e.g. note.txt) for bootstrapping later

    if name_lower in WRITEUP_NAMES:
        return "write-up", "writeup.md"

    # 2. Check for Deployment (Docker)
    if any(p in ["deploy", "docker", "app"] for p in parent_names) or name_lower in [
        "dockerfile",
        "docker-compose.yml",
    ]:
        return "deploy", file_path.name

    # 3. Check for Public Attachments
    if any(p in ["public", "attachments", "files"] for p in parent_names):
        return "public", file_path.name

    # 4. Check for Markdown (if not already caught)
    if ext in MARKDOWN_EXTENSIONS:
        # .md files in a writeup dir, or with a challenge-named writeup stem
        # (e.g. boogie_woogie.md in writeup/, eu_strivesc_writeup.md) → writeup.md
        is_writeup_context = (
            any(p in {"writeup", "write-up"} for p in parent_names)
            or stem_lower.endswith("_writeup")
            or stem_lower.endswith("_wr")
        )
        if is_writeup_context:
            return "write-up", "writeup.md"
        return "write-up", file_path.name

    # 5. Check for Source Code / Binaries
    if ext in SOURCE_EXTENSIONS or ext in BINARY_EXTENSIONS:
        return "src", file_path.name

    # 5.5. Route writeup-named artifacts (*_wr.*, *_writeup.*) to write-up/ regardless of type.
    # Covers PDFs, ZIPs, plain-text writeups submitted as contest deliverables.
    is_writeup_artifact = (
        stem_lower.endswith("_wr") or stem_lower.endswith("_writeup")
    ) and ext not in MARKDOWN_EXTENSIONS
    if is_writeup_artifact:
        if ext == ".txt":
            return "write-up", "writeup.txt"
        return "write-up", file_path.name

    # 6. Heuristic for public files if they are at the root or in a common attachment format
    if ext in PUBLIC_ATTACHMENT_EXTENSIONS and len(rel_path.parts) <= 2:
        return "public", file_path.name

    # Default
    return "src", file_path.name


def bootstrap_missing_files(wu_dir: Path):
    """
    If standard files are missing, try to create them from other available info files.
    Falls back to src/ when write-up/ has no usable text (e.g. bare pwn challenges).
    Creates a stub solve.sh for manually-solved challenges that have a writeup but no script.
    """
    existing_files = {f.name.lower(): f for f in wu_dir.iterdir() if f.is_file()}

    # Candidates for bootstrapping in order of preference
    info_candidates = [
        "description.md",
        "writeup.md",
        "writeup.txt",
        "description.txt",
        "desc.md",
        "desc.txt",
        "descriere.txt",
        "readme.md",
        "readme",
        "notes.txt",
        "note.txt",
        "task.md",
    ]

    # Find the best available info file in write-up/
    best_info_file = None
    for cand in info_candidates:
        if cand in existing_files:
            best_info_file = existing_files[cand]
            break

    # Fallback: scan cyberedu/src/ for readable material when write-up/ has nothing useful.
    # Covers bare pwn/rev challenges where the source .c file IS the description.
    if not best_info_file:
        src_dir = wu_dir.parent / "src"
        if src_dir.exists():
            readable_exts = {".txt", ".md", ".c", ".py", ".go", ".rs"}
            src_files = [
                f
                for f in src_dir.iterdir()
                if f.is_file() and f.suffix.lower() in readable_exts
            ]
            if src_files:
                # prefer .txt/.md (human notes) over source code, then smallest size
                src_files.sort(
                    key=lambda f: (
                        0 if f.suffix.lower() in {".txt", ".md"} else 1,
                        f.stat().st_size,
                    )
                )
                best_info_file = src_files[0]
                logger.info(
                    f"  [Bootstrap] Using src/{best_info_file.name} as description source (no write-up text found)"
                )

    if not best_info_file:
        return

    # Bootstrap missing description.md, writeup.md, challenge-flags.txt
    targets = ["description.md", "writeup.md", "challenge-flags.txt"]
    for target in targets:
        if target not in existing_files:
            try:
                shutil.copy2(best_info_file, wu_dir / target)
                logger.info(
                    f"  [Bootstrapped] Created {target} from {best_info_file.name}"
                )
            except Exception as e:
                logger.error(f"  [Bootstrapped] Failed to create {target}: {e}")

    # If no solver script exists but we have writeup content, create a stub solve.sh.
    # Covers OSINT, forensics, and other manually-solved challenges.
    existing_files = {f.name.lower(): f for f in wu_dir.iterdir() if f.is_file()}
    solver_names = {"solve.py", "solve.sh", "solve.sage"}
    if not any(name in existing_files for name in solver_names):
        stub = wu_dir / "solve.sh"
        try:
            stub.write_text(
                "#!/bin/bash\n"
                "# Manual challenge — solution documented in writeup.md.\n"
                'cat "$(dirname "$0")/writeup.md" 2>/dev/null || echo "See writeup.md"\n'
            )
            logger.info("  [Bootstrapped] Created solve.sh stub (manual challenge)")
        except Exception as e:
            logger.error(f"  [Bootstrapped] Failed to create solve.sh stub: {e}")


def copy_challenge_files(source_challenge_dir: Path, dest_challenge_dir: Path):
    """
    Copies relevant files from the source challenge directory to the destination,
    organizing them into the structure defined in Cerinte.pdf.
    """
    subdirs = {
        "src": dest_challenge_dir / "cyberedu" / "src",
        "write-up": dest_challenge_dir / "cyberedu" / "write-up",
        "deploy": dest_challenge_dir / "cyberedu" / "deploy",
        "public": dest_challenge_dir / "public",
    }

    # Clean and create destination subdirectories
    for sd in subdirs.values():
        if sd.exists():
            shutil.rmtree(sd)
        sd.mkdir(parents=True, exist_ok=True)

    # Walk through the source challenge directory
    for root, dirs, files in os.walk(source_challenge_dir):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS and not d.startswith(".")]

        current_source_dir = Path(root)
        for file in files:
            source_file_path = current_source_dir / file
            rel_path = source_file_path.relative_to(source_challenge_dir)

            subdir_key, target_name = get_target_info(source_file_path, rel_path)
            dest_path = subdirs[subdir_key] / target_name

            try:
                # Handle collisions for non-standardized names
                if dest_path.exists() and target_name not in [
                    "solve.py",
                    "solve.sage",
                    "challenge-flags.txt",
                    "description.md",
                    "writeup.md",
                ]:
                    parent_prefix = "_".join(rel_path.parts[:-1])
                    if parent_prefix:
                        dest_path = dest_path.parent / f"{parent_prefix}_{target_name}"
                    else:
                        dest_path = dest_path.parent / f"dup_{target_name}"

                shutil.copy2(source_file_path, dest_path)
                logger.debug(
                    f"Copied: {rel_path} -> {dest_path.relative_to(dest_challenge_dir)}"
                )
            except Exception as e:
                logger.error(f"Failed to copy file '{source_file_path}': {e}")

    # After copying, bootstrap missing files in write-up
    bootstrap_missing_files(subdirs["write-up"])


def organize_challenges():
    """
    Organizes challenge folders from the source directory into a standardized structure.
    """
    if not SOURCE_ROOT_DIR.exists():
        logger.error(f"Source directory does not exist: {SOURCE_ROOT_DIR}")
        return

    logger.info(f"Starting organization of challenges from: {SOURCE_ROOT_DIR}")
    processed_count = 0

    categories = [
        d
        for d in SOURCE_ROOT_DIR.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ]

    for category_dir in categories:
        logger.info(f"Processing category: {category_dir.name}")
        challenges = [
            d
            for d in category_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ]

        for challenge_dir in challenges:
            category_name = category_dir.name
            challenge_name = challenge_dir.name
            standardized_challenge_path = (
                DESTINATION_ROOT_DIR / category_name / challenge_name
            )

            logger.info(f"Organizing: {category_name}/{challenge_name}")

            try:
                standardized_challenge_path.mkdir(parents=True, exist_ok=True)
                copy_challenge_files(challenge_dir, standardized_challenge_path)

                # Check if the standardized challenge path is empty of actual files after processing
                # This ensures we don't leave empty folders if no relevant files were found/copied
                if not any(f.is_file() for f in standardized_challenge_path.rglob("*")):
                    logger.warning(
                        f"  [CLEANUP] No files were organized for {category_name}/{challenge_name}. Removing empty directory."
                    )
                    shutil.rmtree(standardized_challenge_path)
                else:
                    processed_count += 1

            except Exception as e:
                logger.error(
                    f"Failed to organize challenge {category_name}/{challenge_name}: {e}"
                )

    logger.info(f"Organization complete. Processed {processed_count} challenges.")


def run_organizer():
    """Main function to run the challenge organization."""
    DESTINATION_ROOT_DIR.mkdir(parents=True, exist_ok=True)
    organize_challenges()


if __name__ == "__main__":
    setup_logging("organize_challenges")
    run_organizer()
