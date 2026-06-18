"""Logging configuration and management utilities.

This module provides a centralized logging setup with automatic log file rotation
based on runs. Each time the application starts, it archives the previous log file
and starts a new one, keeping a history of the last N runs.

The module uses loguru for robust logging with:
- Automatic file rotation after each run
- Separate console (INFO level) and file (DEBUG level) outputs
- Consistent timestamp formatting across all logs
- Configurable backup history (default: 5 previous runs)

Functions:
    rotate_log_files: Archive previous log files before starting new run
    setup_logging: Configure loguru logger with rotation and handlers

Usage:
    from src.utils.logging_config import setup_logging

    # Configure logging at application startup
    logger = setup_logging("my_agent")

    # Use logger throughout application
    logger.info("Processing started")
    logger.debug(f"Processing {item}")
    logger.error("An error occurred", exc_info=True)

    # Log files are automatically archived
    # Previous runs available as: logs/my_agent.log.1 through .log.5

Logging Levels:
    DEBUG: Detailed diagnostic information, variable values, loop iterations
    INFO: General informational messages about application flow
    WARNING: Warning messages about potential issues (not critical)
    ERROR: Error messages for exceptional conditions
    CRITICAL: Critical errors requiring immediate attention

File Structure:
    logs/
        {logger_name}.log      - Current run's logs (created fresh each time)
        {logger_name}.log.1    - Previous run (most recent backup)
        {logger_name}.log.2    - 2 runs ago
        {logger_name}.log.3    - 3 runs ago
        {logger_name}.log.4    - 4 runs ago
        {logger_name}.log.5    - 5 runs ago (oldest, deleted when new rotation occurs)

Example .log file output:
    2024-01-15 14:23:45 - my_agent - INFO - Logging initialized...
    2024-01-15 14:23:46 - my_agent - DEBUG - Processing challenge: crypto_001
    2024-01-15 14:23:47 - my_agent - INFO - Successfully validated 42 challenges
    2024-01-15 14:23:48 - my_agent - ERROR - Failed to read file: permission denied
"""

import logging
import shutil
from pathlib import Path


def rotate_log_files(log_file: Path, max_backups: int = 5) -> None:
    """Archive previous log file and prepare for new session.

    Implements a simple run-based log rotation scheme:
    1. Shifts existing backup files (log.1 → log.2, log.2 → log.3, etc.)
    2. Deletes the oldest backup when max_backups limit is reached
    3. Archives the current log file as log.1
    4. Prepares for a fresh log file in the next session

    This ensures you always have history of the last N runs without
    eating unlimited disk space from ever-growing log files.

    Args:
        log_file (Path): Path to the main log file to be archived
            Example: Path("logs/analyzer.log")

        max_backups (int): Maximum number of backup files to keep
            Default: 5 (keeps current + 5 previous runs)
            Valid range: 1-100 (enforced by pipeline scripts)

    Returns:
        None

    Side Effects:
        - Renames existing backup files (.log.1 → .log.2, etc.)
        - Deletes oldest backup if max_backups would be exceeded
        - Creates .log.1 backup from current log file
        - Removes the current log file

    Example:
        >>> from pathlib import Path
        >>> from src.utils.logging_config import rotate_log_files
        >>>
        >>> log_path = Path("logs/analyzer.log")
        >>> rotate_log_files(log_path, max_backups=5)
        >>> # Previous run's logs are now at logs/analyzer.log.1
        >>> # logs/analyzer.log.2 through .log.5 contain older runs
        >>> # logs/analyzer.log is ready to be created for this run

    Notes:
        - Safe to call multiple times without side effects if log file doesn't exist
        - Thread-safe for single-process applications
        - Not thread-safe for multi-process scenarios (use process locking if needed)
    """
    # Early return if log file doesn't exist yet (first run)
    if not log_file.exists():
        return

    # Shift existing backups backward (.log.N → .log.N+1)
    # Process in reverse order to avoid overwriting files
    for i in range(max_backups - 1, 0, -1):
        old_backup = log_file.with_suffix(f".log.{i}")
        new_backup = log_file.with_suffix(f".log.{i + 1}")

        if old_backup.exists():
            if i == max_backups - 1:
                # Delete the oldest backup to maintain max_backups limit
                old_backup.unlink()
            else:
                # Rename this backup to next position in the sequence
                old_backup.rename(new_backup)

    # Copy current log to .log.1 backup before clearing it
    # Uses copy2 to preserve metadata (timestamp, permissions)
    if log_file.exists():
        backup = log_file.with_suffix(".log.1")
        shutil.copy2(log_file, backup)  # copy2 preserves metadata
        log_file.unlink()  # Remove original, ready for new session


def setup_logging(logger_name: str = "root", max_backups: int = 5) -> logging.Logger:
    """Configure and initialize logger with automatic run-based log rotation.

    Sets up a logger with dual output:
    - Console: INFO and above (user-facing, concise)
    - File: DEBUG and above (detailed diagnostic logs)

    Automatically archives the previous session's log before starting,
    keeping a rolling history of the last max_backups sessions.

    Args:
        logger_name (str): Name for the logger (usually script/module name)
            Used in log output and as filename prefix
            Examples: "analyzer", "validator", "content_generator"
            Default: "root" (global application logger)

        max_backups (int): Number of previous run logs to retain
            Default: 5 (keeps .log.1 through .log.5)
            Higher values use more disk space but preserve more history

    Returns:
        logging.Logger: Configured logger ready for use

    Raises:
        OSError: If logs directory or log file cannot be created
        PermissionError: If lacking permissions to write to logs directory

    Side Effects:
        - Creates logs/ directory if it doesn't exist
        - Archives previous log to .log.1, .log.2, etc.
        - Creates new empty log file for this session
        - Emits startup messages to indicate logging is ready

    Example:
        >>> from src.utils.logging_config import setup_logging
        >>>
        >>> # Initialize logger in your script
        >>> logger = setup_logging("analyze_challenges")
        >>>
        >>> # Use throughout your script
        >>> logger.info("Starting analysis of challenges")
        >>> logger.debug(f"Found {count} challenges")
        >>> logger.error(f"Failed to process {name}", exc_info=True)
        >>>
        >>> # Logs are automatically saved to logs/analyze_challenges.log
        >>> # Previous runs available as .log.1 through .log.5

    Log Format:
        The default format is:
        {timestamp} - {logger_name} - {level} - {message}

        Example output:
        2024-01-15 14:23:45 - analyze_challenges - INFO - Starting analysis
        2024-01-15 14:23:46 - analyze_challenges - DEBUG - Found challenge: crypto_001
        2024-01-15 14:23:47 - analyze_challenges - ERROR - Permission denied on file

    Notes:
        - Each call to setup_logging() creates a new logger instance
        - Passing same logger_name to setup_logging() multiple times returns
          the same logger with updated configuration
        - Safe to call multiple times at application startup
        - Log files use 'w' mode (write), starting fresh each run
          (previous run is archived to .log.1 first via rotate_log_files)
        - Previous function logged to loguru, this now uses logging module
          for consistency with existing handlers
    """
    # Ensure logs directory exists before writing
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    # Define log file path using logger name
    log_file = logs_dir / f"{logger_name}.log"

    # Archive previous log before starting new session
    # This preserves history while keeping log files reasonably sized
    rotate_log_files(log_file, max_backups)

    # Get or create logger instance
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)  # Capture all levels, handlers filter by level

    # Clear existing handlers to avoid duplicates if logger already configured
    logger.handlers.clear()

    # Create console handler for user-facing output
    # Only shows INFO and above (not DEBUG noise on console)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create file handler for detailed logs
    # Starts fresh each run, previous run archived by rotate_log_files()
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.DEBUG)

    # Define consistent format for all log messages
    # Includes timestamp, logger name, level, and message
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Attach both handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # Log initialization messages to confirm everything is working
    logger.info(f"Logging initialized. Current log: {log_file}")
    logger.info(
        f"Previous runs archived as: {log_file}.1 through {log_file}.{max_backups}"
    )

    return logger


def get_log_file_path(logger_name: str = "root") -> Path:
    """Return the path to the current log file for a logger (same rotation logic as setup_logging).

    The file is logs/{logger_name}.log. Rotation (e.g. .log.1–.log.5) is done by
    setup_logging/rotate_log_files at startup.

    Args:
        logger_name: Name of the logger (e.g. "main_runner").

    Returns:
        Path to the current run's log file.
    """
    logs_dir = Path("logs")
    return logs_dir / f"{logger_name}.log"
