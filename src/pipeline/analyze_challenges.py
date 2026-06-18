import os
import sys
from pathlib import Path
from typing import Any, Dict, List

from loguru import logger

# Add project root to Python path for imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config.settings import settings  # noqa: E402
from src.utils.logging_config import setup_logging  # noqa: E402

# Define the root directory for challenge folders
CHALLENGE_FOLDERS_ROOT = settings.PROCESSED_DIR / settings.RAW_CHALLENGES_SOURCE.name

# Define expected directory structure components
EXPECTED_STRUCTURE_COMPONENTS = ["category", "challenge_name"]


def analyze_challenge_folder_structure(folder_path: Path) -> Dict[str, Any]:
    """
    Analyzes the directory structure within a single challenge folder.

    Args:
        folder_path: Path to the challenge folder.

    Returns:
        A dictionary containing analysis results, including:
        - 'folder_name': Name of the challenge folder.
        - 'total_files': Total number of files in the folder and its subdirectories.
        - 'root_dirs': List of top-level directories found within the challenge folder.
        - 'potential_challenge_dirs': List of directories that might represent challenges
                                      based on common naming patterns.
        - 'inconsistencies': List of observed structural anomalies.
        - 'file_types': Count of different file extensions.
    """
    analysis_results = {
        "folder_name": folder_path.name,
        "total_files": 0,
        "root_dirs": set(),
        "potential_challenge_dirs": [],
        "inconsistencies": [],
        "file_types": {},
        "found_expected_subdirs": {
            "cyberedu": False,
            "src": False,
            "write-up": False,
            "deploy": False,
            "public": False,
        },
        "found_expected_files": {
            "description.md": False,
            "writeup.md": False,
            "solve_script": False,
            "challenge-flags.txt": False,
        },
    }

    if not folder_path.is_dir():
        analysis_results["inconsistencies"].append(
            f"Path is not a directory: {folder_path}"
        )
        return analysis_results

    logger.debug(f"Analyzing folder: {folder_path.name}")

    # Walk through the directory to find all files and directories
    try:
        for root, dirs, files in os.walk(folder_path):
            try:
                current_dir = Path(root)
                relative_path_from_challenge_root = current_dir.relative_to(folder_path)

                # Track top-level directories within the challenge folder
                if relative_path_from_challenge_root.parts:
                    top_level_dir = relative_path_from_challenge_root.parts[0]
                    analysis_results["root_dirs"].add(top_level_dir)

                    # Check for expected subdirectories
                    if top_level_dir.lower() == "cyberedu":
                        analysis_results["found_expected_subdirs"]["cyberedu"] = True
                    elif top_level_dir.lower() == "public":
                        analysis_results["found_expected_subdirs"]["public"] = True

                # Check for specific expected subdirectories deeper within 'cyberedu'
                if "cyberedu" in relative_path_from_challenge_root.parts:
                    if "src" in relative_path_from_challenge_root.parts:
                        analysis_results["found_expected_subdirs"]["src"] = True
                    if "write-up" in relative_path_from_challenge_root.parts:
                        analysis_results["found_expected_subdirs"]["write-up"] = True
                    if "deploy" in relative_path_from_challenge_root.parts:
                        analysis_results["found_expected_subdirs"]["deploy"] = True

                # Analyze files
                for file in files:
                    file_path = current_dir / file
                    analysis_results["total_files"] += 1

                    # Analyze file types
                    if file_path.suffix:
                        file_extension = file_path.suffix.lower()
                        analysis_results["file_types"][file_extension] = (
                            analysis_results["file_types"].get(file_extension, 0) + 1
                        )

                    file_name_lower = file.lower()

                    # Check for key files at any depth
                    if file_name_lower == "description.md":
                        analysis_results["found_expected_files"][
                            "description.md"
                        ] = True
                    if file_name_lower == "writeup.md":
                        analysis_results["found_expected_files"]["writeup.md"] = True
                    if file_name_lower.startswith("solve.") and file_name_lower.split(
                        "."
                    )[-1] in ["py", "sage", "sh"]:
                        analysis_results["found_expected_files"]["solve_script"] = True
                    if file_name_lower == "challenge-flags.txt":
                        analysis_results["found_expected_files"][
                            "challenge-flags.txt"
                        ] = True

                    # Heuristic for identifying potential challenge directories (directory containing key challenge files)
                    if file_name_lower in [
                        "description.md",
                        "writeup.md",
                        "solve.py",
                        "challenge-flags.txt",
                        "flag.txt",
                    ]:
                        dir_containing_key_file = file_path.parent
                        relative_dir_path = dir_containing_key_file.relative_to(
                            folder_path
                        )
                        depth = len(relative_dir_path.parts)

                        if (
                            dir_containing_key_file.is_dir()
                            and dir_containing_key_file.is_relative_to(folder_path)
                        ):
                            path_str = (
                                str(relative_dir_path) if relative_dir_path else "."
                            )

                            # Check if this directory is already recorded or if it's a parent of an already recorded one
                            is_new_potential_dir = True
                            dirs_to_remove = []
                            for existing_p_dir in analysis_results[
                                "potential_challenge_dirs"
                            ]:
                                existing_path = Path(existing_p_dir["path"])
                                if existing_path == relative_dir_path:
                                    is_new_potential_dir = False
                                    break
                                # If existing is a child of new (new is more general), replace existing
                                if (
                                    existing_path.is_relative_to(relative_dir_path)
                                    and existing_path != relative_dir_path
                                ):
                                    dirs_to_remove.append(existing_p_dir)
                                # If new is a child of existing (existing is more general), skip new
                                if (
                                    relative_dir_path.is_relative_to(existing_path)
                                    and existing_path != relative_dir_path
                                ):
                                    is_new_potential_dir = False
                                    break

                            for item_to_remove in dirs_to_remove:
                                analysis_results["potential_challenge_dirs"].remove(
                                    item_to_remove
                                )

                            if is_new_potential_dir:
                                analysis_results["potential_challenge_dirs"].append(
                                    {"path": path_str, "depth": depth}
                                )

            except (OSError, ValueError) as e:
                logger.warning(f"Error processing directory {root}: {e}")
                continue

        # Identify inconsistencies based on expected subdirs and files
        # The raw challenges might not perfectly match the "processed" structure
        # but we can look for *some* indication of structure.
        # 1. Check for expected core directories
        if not analysis_results["found_expected_subdirs"]["cyberedu"]:
            analysis_results["inconsistencies"].append(
                "Missing 'cyberedu' top-level directory (or similarly structured content)."
            )
        if not analysis_results["found_expected_subdirs"]["write-up"]:
            analysis_results["inconsistencies"].append(
                "Missing 'write-up' directory (expected under 'cyberedu')."
            )
        if not analysis_results["found_expected_subdirs"]["public"]:
            analysis_results["inconsistencies"].append(
                "Missing 'public' top-level directory (for attachments/files)."
            )

        # 2. Check for key files
        if not analysis_results["found_expected_files"]["description.md"]:
            analysis_results["inconsistencies"].append(
                "Missing 'description.md' (challenge description)."
            )
        if not analysis_results["found_expected_files"]["writeup.md"]:
            analysis_results["inconsistencies"].append(
                "Missing 'writeup.md' (challenge write-up)."
            )
        if not analysis_results["found_expected_files"]["solve_script"]:
            analysis_results["inconsistencies"].append(
                "Missing 'solve' script (e.g., solve.py/sage/sh)."
            )
        if not analysis_results["found_expected_files"]["challenge-flags.txt"]:
            analysis_results["inconsistencies"].append("Missing 'challenge-flags.txt'.")

    except PermissionError as e:
        logger.error(f"Permission denied accessing directory {folder_path}: {e}")
        analysis_results["inconsistencies"].append(f"Permission denied: {e}")
    except OSError as e:
        logger.error(f"OS error while analyzing {folder_path}: {e}")
        analysis_results["inconsistencies"].append(f"OS error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error analyzing {folder_path}: {e}")
        analysis_results["inconsistencies"].append(f"Unexpected error: {e}")

    return analysis_results


def process_challenge_folders_in_directory(root_dir: Path) -> List[Dict[str, Any]]:
    """
    Iterates through all challenge folders in a given directory and analyzes their structure.

    Args:
        root_dir: The root directory containing the challenge folders.

    Returns:
        A list of analysis results, one for each challenge folder.
    """
    all_analyses = []
    if not root_dir.exists():
        logger.error(f"Root directory for challenge folders does not exist: {root_dir}")
        return all_analyses

    logger.info(f"Starting analysis of challenge folders in: {root_dir}")

    # Iterate through items in the root directory
    for item in root_dir.iterdir():
        if item.is_dir():
            # Assume each directory here is a potential challenge folder or a category folder
            # If it's a category folder, we'll recursively look for challenge folders inside it.
            # For now, we'll treat each directory as a potential challenge folder to analyze.
            # A more refined approach might distinguish between category folders and challenge folders.

            # Let's assume for now that each directory directly under root_dir is a challenge folder
            # or a category folder that contains challenge folders.
            # We will analyze each directory found.

            # If the directory itself contains key files, it's a challenge.
            # If it contains other directories that contain key files, those are challenges.
            # For simplicity, we'll analyze the structure of each directory found.

            # A more robust approach would be to identify the actual challenge folder
            # (e.g., the one containing description.md, writeup.md, etc.)
            # For now, we'll analyze the structure of each directory encountered.

            # Let's refine this: we want to analyze the structure of what *looks like* a challenge.
            # A challenge folder is typically a directory that contains key files like description.md, writeup.md, etc.
            # Or it contains subdirectories that contain these key files.

            # We will analyze each directory found. If it's a category, it will be analyzed.
            # If it's a challenge, it will be analyzed.
            # The analysis function itself will try to find the "challenge structure" within it.

            analysis = analyze_challenge_folder_structure(item)
            all_analyses.append(analysis)
            if analysis["inconsistencies"]:
                logger.warning(
                    f"Inconsistencies found in {item.name}: {analysis['inconsistencies']}"
                )
            else:
                logger.info(f"Folder {item.name} structure appears consistent.")
        else:
            logger.warning(f"Skipping non-directory item: {item.name}")

    logger.info(f"Finished analysis of {len(all_analyses)} directories in {root_dir}.")
    return all_analyses


def summarize_analysis(all_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Summarizes the results from analyzing multiple challenge folders.

    Args:
        all_analyses: A list of analysis dictionaries.

    Returns:
        A summary dictionary.
    """
    summary = {
        "total_folders_analyzed": len(all_analyses),
        "folders_with_inconsistencies": 0,
        "common_inconsistencies": {},
        "file_type_distribution": {},
        "folder_root_dir_distribution": {},
        "potential_challenge_dir_depth_distribution": {},
        "missing_expected_subdirs": {},
        "missing_expected_files": {},
    }

    if not all_analyses:
        return summary

    for analysis in all_analyses:
        if analysis["inconsistencies"]:
            summary["folders_with_inconsistencies"] += 1
            for inconsistency in analysis["inconsistencies"]:
                summary["common_inconsistencies"][inconsistency] = (
                    summary["common_inconsistencies"].get(inconsistency, 0) + 1
                )

        for file_type, count in analysis["file_types"].items():
            summary["file_type_distribution"][file_type] = (
                summary["file_type_distribution"].get(file_type, 0) + count
            )

        for root_dir in analysis["root_dirs"]:
            summary["folder_root_dir_distribution"][root_dir] = (
                summary["folder_root_dir_distribution"].get(root_dir, 0) + 1
            )

        for p_dir in analysis["potential_challenge_dirs"]:
            depth = p_dir["depth"]
            summary["potential_challenge_dir_depth_distribution"][depth] = (
                summary["potential_challenge_dir_depth_distribution"].get(depth, 0) + 1
            )

        # Aggregate missing subdirs and files
        for subdir_name, found in analysis["found_expected_subdirs"].items():
            if not found:
                summary["missing_expected_subdirs"][subdir_name] = (
                    summary["missing_expected_subdirs"].get(subdir_name, 0) + 1
                )

        for file_name, found in analysis["found_expected_files"].items():
            if not found:
                summary["missing_expected_files"][file_name] = (
                    summary["missing_expected_files"].get(file_name, 0) + 1
                )

    return summary


def run_analysis():
    """Main function to run the challenge folder analysis."""
    if not CHALLENGE_FOLDERS_ROOT.exists():
        logger.error(
            f"The specified challenge folders root directory does not exist: {CHALLENGE_FOLDERS_ROOT}"
        )
        logger.error(
            "Please ensure the path is correct and the directory contains your challenge folders."
        )
    else:
        logger.info(
            f"Starting analysis of challenge folders located at: {CHALLENGE_FOLDERS_ROOT}"
        )
        analysis_results = process_challenge_folders_in_directory(
            CHALLENGE_FOLDERS_ROOT
        )

        if analysis_results:
            summary = summarize_analysis(analysis_results)

            logger.info("\n" + "=" * 50)
            logger.info("          CHALLENGE FOLDER ANALYSIS SUMMARY")
            logger.info("=" * 50)
            logger.info(f"Total folders analyzed: {summary['total_folders_analyzed']}")
            logger.info(
                f"Folders with inconsistencies: {summary['folders_with_inconsistencies']}"
            )

            if summary["common_inconsistencies"]:
                logger.info("\n--- Common Inconsistencies Found ---")
                for inconsistency, count in summary["common_inconsistencies"].items():
                    logger.info(f"- {inconsistency}: {count} times")

            if summary["file_type_distribution"]:
                logger.info("\n--- Overall File Type Distribution ---")
                # Sort by count for better readability
                sorted_file_types = sorted(
                    summary["file_type_distribution"].items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
                for file_type, count in sorted_file_types:
                    logger.info(f"- {file_type}: {count} files")

            if summary["folder_root_dir_distribution"]:
                logger.info(
                    "\n--- Distribution of Top-Level Directories in Folders ---"
                )
                sorted_root_dirs = sorted(
                    summary["folder_root_dir_distribution"].items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
                for root_dir, count in sorted_root_dirs:
                    logger.info(f"- {root_dir}: {count} folders")

            if summary["potential_challenge_dir_depth_distribution"]:
                logger.info(
                    "\n--- Distribution of Identified Challenge Directory Depths ---"
                )
                sorted_depths = sorted(
                    summary["potential_challenge_dir_depth_distribution"].items()
                )
                for depth, count in sorted_depths:
                    logger.info(f"- Depth {depth}: {count} directories")

            if summary["missing_expected_subdirs"]:
                logger.info("\n--- Missing Expected Subdirectories Summary ---")
                for subdir, count in summary["missing_expected_subdirs"].items():
                    logger.info(f"- '{subdir}' missing in {count} folders")

            if summary["missing_expected_files"]:
                logger.info("\n--- Missing Expected Key Files Summary ---")
                for file_name, count in summary["missing_expected_files"].items():
                    logger.info(f"- '{file_name}' missing in {count} folders")

            logger.info("\n" + "=" * 50)
            logger.info(
                "Analysis complete. Check logs for detailed folder-by-folder logs."
            )
        else:
            logger.warning(
                "No challenge folders were analyzed. Please check the root directory path and ensure it contains challenge folders."
            )


if __name__ == "__main__":
    setup_logging("analyze_challenges")
    run_analysis()
