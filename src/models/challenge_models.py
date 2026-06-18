"""Data models for representing CyberEdu challenges and their components.

This module contains Pydantic models for structured representation of cybersecurity
challenges in the CyberEdu platform. It defines:
- Individual files within challenges
- Challenge metadata and descriptive information
- Complete challenge representation with all components

These models are used throughout the pipeline for validation, organization, and
processing of challenge data.

Models:
    ChallengeFile: A single file within a challenge archive
    ChallengeMetadata: Descriptive metadata about a challenge
    Challenge: Complete representation of a challenge with all its components

Example:
    >>> metadata = ChallengeMetadata(
    ...     title="RSA Decryption",
    ...     category="Cryptography",
    ...     difficulty="Intermediate",
    ...     author="Security Team",
    ...     tags=["rsa", "crypto", "math"],
    ...     description="Break RSA encryption with weak parameters"
    ... )
    >>> challenge = Challenge(
    ...     id="crypto_rsa_001",
    ...     metadata=metadata,
    ...     root_path=Path("/path/to/challenge"),
    ...     files=[...],
    ...     raw_writeup_content="... solution writeup ..."
    ... )
    >>> solver_files = challenge.get_file_by_type("script")
"""

from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field


class ChallengeFile(BaseModel):
    """Represents a single file discovered within a challenge archive.

    Attributes:
        rel_path (Path): Relative path from challenge root directory
            Example: Path("cyberedu/write-up/writeup.md")
        file_type (str): Categorization of the file's purpose
            Valid types:
                - "source": Source code (.py, .java, .cpp, etc.)
                - "binary": Compiled binaries or executables
                - "script": Executable scripts (.sh, .sage, etc.)
                - "markdown": Documentation files (.md)
                - "configuration": Config files (.txt, .json, .yaml)
                - "attachment": Files for challenge participants
                - "other": Uncategorized files
        content_preview (Optional[str]): First few lines/bytes of file content
            Useful for identifying file purpose without full parse
            None for binary files or large documents
        size_bytes (int): File size in bytes for tracking archive size

    Example:
        >>> writeup_file = ChallengeFile(
        ...     rel_path=Path("cyberedu/write-up/writeup.md"),
        ...     file_type="markdown",
        ...     content_preview="# RSA Decryption Challenge\\n\\nThis challenge...",
        ...     size_bytes=2048
        ... )
        >>> solver_file = ChallengeFile(
        ...     rel_path=Path("cyberedu/write-up/solve.py"),
        ...     file_type="script",
        ...     content_preview="#!/usr/bin/env python3\\nimport...",
        ...     size_bytes=1024
        ... )
    """

    rel_path: Path
    file_type: str  # e.g., 'source', 'binary', 'script', 'markdown'
    content_preview: Optional[str] = None
    size_bytes: int


class ChallengeMetadata(BaseModel):
    """Descriptive metadata about a CyberEdu challenge.

    This model captures the educational and categorization aspects of a challenge,
    separate from its file structure. Used for indexing, searching, and categorizing
    challenges within the system.

    Attributes:
        title (str): Human-readable challenge name
            Example: "RSA Decryption Challenge"
        category (str): Domain classification for the challenge
            Valid categories:
                - "Cryptography": Encryption, hashing, digital signatures
                - "Web Security": SQL injection, XSS, CSRF
                - "Reverse Engineering": Binary analysis, decompilation
                - "Network": Packet analysis, protocols, network penetration
                - "Forensics": Data recovery, artifact analysis
                - "Steganography": Hidden data extraction
                - "Scripting": Python/Bash automation, payload development
                - "Privilege Escalation": Elevation of privilege attacks
        difficulty (str): Estimated difficulty level for learners
            Valid values: "Beginner", "Intermediate", "Advanced", "Expert"
            Default: "Medium"
        author (Optional[str]): Creator/maintainer of the challenge
            Default: "Unknown"
        tags (List[str]): Additional searchable keywords
            Examples: ["rsa", "crypto", "math", "educational"]
        description (Optional[str]): Detailed problem statement
            May contain learning objectives and background context

    Example:
        >>> metadata = ChallengeMetadata(
        ...     title="RSA with Weak Primes",
        ...     category="Cryptography",
        ...     difficulty="Intermediate",
        ...     author="Security Team",
        ...     tags=["rsa", "factorization", "weak-primes"],
        ...     description="Recover a plaintext encrypted with RSA..."
        ... )
    """

    title: str
    category: str
    difficulty: str = "Medium"
    author: Optional[str] = "Unknown"
    tags: List[str] = Field(default_factory=list)
    description: Optional[str] = None


class Challenge(BaseModel):
    """Complete representation of a challenge being processed in the pipeline.

    This is the main data model used throughout the content creation pipeline.
    It combines metadata, file inventory, content, and references to official
    documentation into a single validated structure.

    The Challenge model is used in:
    - Validation Agent: Checks structural completeness
    - Content Generation Agent: Generates writeups and solution scripts
    - Ranking Agent: Evaluates quality of generated content
    - Vector DB Service: Storing challenge embeddings for retrieval

    Attributes:
        id (str): Unique identifier for the challenge
            Convention: "{category}_{number}" or "{category}_{descriptive_name}"
            Example: "crypto_001", "web_sql_injection"
        metadata (ChallengeMetadata): Challenge description and categorization
        root_path (Path): Filesystem path to challenge root directory
            Must contain cyberedu/ subdirectory
        files (List[ChallengeFile]): Discovered files in challenge archive
            Populated by Analyzer, used by Organizer and Validator
        raw_writeup_content (Optional[str]): Extracted or generated solution writeup
            Populated by Content Generation Agent
            Contains step-by-step solution explanation
        raw_solve_script (Optional[str]): Source code for solution/exploit
            Populated by Content Generation Agent or extracted from archive
            Examples: Python exploit, bash automation, reverse engineering script
        official_docs_paths (List[Path]): References to official PDF documentation
            Populated by Official Docs Mapping Agent
            Used for validating technical accuracy of generated content

    Methods:
        get_file_by_type(file_type): Filter files by their type

    Example:
        >>> challenge = Challenge(
        ...     id="crypto_rsa_001",
        ...     metadata=ChallengeMetadata(
        ...         title="RSA Decryption",
        ...         category="Cryptography",
        ...         difficulty="Intermediate"
        ...     ),
        ...     root_path=Path("/data/challenges/crypto_rsa_001"),
        ...     files=[
        ...         ChallengeFile(
        ...             rel_path=Path("cyberedu/write-up/writeup.md"),
        ...             file_type="markdown",
        ...             size_bytes=2048
        ...         ),
        ...         ChallengeFile(
        ...             rel_path=Path("cyberedu/write-up/solve.py"),
        ...             file_type="script",
        ...             size_bytes=1024
        ...         )
        ...     ],
        ...     raw_writeup_content="Solution: Factor the modulus using...",
        ...     official_docs_paths=[Path("/docs/rsa_fundamentals.pdf")]
        ... )
    """

    id: str
    metadata: ChallengeMetadata
    root_path: Path
    files: List[ChallengeFile] = Field(default_factory=list)
    raw_writeup_content: Optional[str] = None
    raw_solve_script: Optional[str] = None
    official_docs_paths: List[Path] = Field(
        default_factory=list
    )  # Paths to official PDF documentation

    def get_file_by_type(self, file_type: str) -> List[ChallengeFile]:
        """Retrieve all files of a specific type from this challenge.

        Filters the challenge's file list by the given file type. Useful for
        accessing specific categories of files (e.g., all scripts, all markdown docs).

        Args:
            file_type (str): Type to filter by. Common types:
                - "script": Solution or deployment scripts (.py, .sh, .sage)
                - "markdown": Documentation files (.md)
                - "source": Source code files (.py, .java, .cpp)
                - "binary": Compiled binaries or executables

        Returns:
            List[ChallengeFile]: All files matching the specified type.
                Returns empty list if no files of that type exist.

        Example:
            >>> challenge = Challenge(...)
            >>> scripts = challenge.get_file_by_type("script")
            >>> for script in scripts:
            ...     print(f"Found solver: {script.rel_path}")
        """
        return [f for f in self.files if f.file_type == file_type]
