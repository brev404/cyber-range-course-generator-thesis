"""Data models for validation and ranking reports.

This module defines Pydantic models for representing the output of various
validation and ranking agents in the content creation pipeline.

Models:
    IssueSeverity: Enumeration of issue severity levels.
    ValidationIssue: A single issue discovered during validation.
    ChallengeChecklist: Requirements checklist for a challenge.
    ValidationReport: Complete validation results for a challenge.
    RankingScore: Score assigned by a ranking persona.
    RankingReport: Final evaluation report for generated content.

Example:
    >>> issue = ValidationIssue(
    ...     code="MISSING_FILE",
    ...     message="Missing solver script",
    ...     severity=IssueSeverity.CRITICAL,
    ...     file_path="cyberedu/write-up/solve.py",
    ...     suggestion="Add a solver script demonstrating the exploit"
    ... )
    >>> report = ValidationReport(
    ...     challenge_id="challenge_001",
    ...     is_valid=False,
    ...     issues=[issue],
    ...     structure_score=0.65
    ... )
"""

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class IssueSeverity(str, Enum):
    """Severity levels for validation issues.

    Attributes:
        LOW: Minor issue that can be addressed later (e.g., typo)
        MEDIUM: Issue affecting quality but not blocking (e.g., missing optional file)
        HIGH: Issue affecting functionality (e.g., incomplete writeup)
        CRITICAL: Issue blocking completion (e.g., missing solver script)
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ValidationIssue(BaseModel):
    """A specific issue found during structural or content validation.

    Attributes:
        code (str): Issue identifier code (e.g., "MISSING_FILE", "INVALID_FORMAT")
        message (str): Human-readable description of the issue
        severity (IssueSeverity): How serious this issue is
        file_path (Optional[str]): Relative path to the affected file, if applicable
        suggestion (Optional[str]): Recommended action to fix this issue

    Example:
        >>> issue = ValidationIssue(
        ...     code="MISSING_WRITEUP",
        ...     message="Missing cyberedu/write-up/description.md",
        ...     severity=IssueSeverity.HIGH,
        ...     file_path="cyberedu/write-up/description.md",
        ...     suggestion="Create description.md with problem statement"
        ... )
    """

    code: str
    message: str
    severity: IssueSeverity
    file_path: Optional[str] = None
    suggestion: Optional[str] = None
    # Advisory hint for ranking: exact dimension key from RankingReport.dimension_scores most
    # affected by this issue. Used by ranking_agent to provide context hints to the LLM reviewer.
    dimension_hint: Optional[str] = None


class ChallengeChecklist(BaseModel):
    """Checklist tracking which required and optional files exist for a challenge.

    Based on the CyberEdu structure defined in requirements (Cerinte.pdf):
    - cyberedu/ directory is the root for all challenge materials
    - write-up/ contains documentation and metadata
    - src/ contains solution code
    - public/ contains files accessible to challenge participants

    Attributes:
        has_cyberedu_dir (bool): Root cyberedu directory exists
        has_src_dir (bool): Source directory for solution code exists
        has_writeup_dir (bool): Write-up documentation directory exists
        has_public_dir (bool): Public files directory exists

        # Core write-up files
        has_writeup_md (bool): Main problem description (writeup.md) exists
        has_solver (bool): Solver/solution script (.py, .sage, .sh) exists
        has_description_md (bool): Challenge description (description.md) exists
        has_flags_txt (bool): Flag file (flags.txt) exists

        # Optional enhancements
        has_deploy_dir (bool): Deployment/docker files exist
        has_attachments (bool): Attachment files exist for participants
        has_official_docs (bool): Official PDF documentation mapped to challenge

    Example:
        >>> checklist = ChallengeChecklist(
        ...     has_cyberedu_dir=True,
        ...     has_writeup_dir=True,
        ...     has_writeup_md=True,
        ...     has_solver=True,
        ...     has_description_md=True,
        ...     has_flags_txt=True,
        ...     has_public_dir=True
        ... )
        >>> checklist.is_complete()
        True
    """

    has_cyberedu_dir: bool = False
    has_src_dir: bool = False
    has_writeup_dir: bool = False
    has_public_dir: bool = False

    # Core files in cyberedu/write-up/
    has_writeup_md: bool = False
    has_solver: bool = (
        False  # Renamed from has_solve_py to be more generic (.py, .sage, .sh)
    )
    has_description_md: bool = False
    has_flags_txt: bool = False

    # Optional / Enrichment
    has_deploy_dir: bool = False
    has_attachments: bool = False
    has_official_docs: bool = False  # Whether official PDF documentation was mapped

    def is_complete(self) -> bool:
        """Check if challenge meets all basic requirements.

        A challenge is considered complete if it has:
        - cyberedu root directory
        - write-up subdirectory with all core files
        - description.md (problem statement)
        - writeup.md (solution writeup)
        - solver script (solution code)
        - flags.txt (flag for completion)
        - public directory for participant files

        Returns:
            bool: True if all core requirements are satisfied, False otherwise.

        Example:
            >>> checklist = ChallengeChecklist(
            ...     has_cyberedu_dir=True,
            ...     has_writeup_dir=True,
            ...     has_writeup_md=True,
            ...     has_solver=True,
            ...     has_description_md=True,
            ...     has_flags_txt=True,
            ...     has_public_dir=True
            ... )
            >>> checklist.is_complete()
            True
        """
        return all(
            [
                self.has_cyberedu_dir,
                self.has_writeup_dir,
                self.has_writeup_md,
                self.has_solver,
                self.has_description_md,
                self.has_flags_txt,
                self.has_public_dir,
            ]
        )


class ValidationReport(BaseModel):
    """Complete validation results for a challenge after structural and content analysis.

    This report is generated by the Validation Agent and contains:
    - Overall validity assessment
    - Detailed checklist of file requirements
    - Specific issues found with severity levels
    - Structure quality score
    - Presence of key content (writeup, solve script)

    Attributes:
        challenge_id (str): Unique identifier for the challenge
        is_valid (bool): Whether challenge passes all validation checks
        checklist (Optional[ChallengeChecklist]): File requirements checklist
        issues (List[ValidationIssue]): List of problems found during validation
        structure_score (float): Quality score from 0.0 to 1.0 for file structure
        has_writeup (bool): Whether challenge has a complete writeup document
        has_solve_script (bool): Whether challenge includes a solver/solution script
        files_found (Dict[str, List[str]]): Discovered files grouped by category

    Example:
        >>> report = ValidationReport(
        ...     challenge_id="crypto_001",
        ...     is_valid=True,
        ...     structure_score=0.95,
        ...     has_writeup=True,
        ...     has_solve_script=True,
        ...     files_found={
        ...         "writeup": ["writeup.md", "description.md"],
        ...         "solver": ["solve.py"]
        ...     }
        ... )
    """

    challenge_id: str
    is_valid: bool
    checklist: Optional[ChallengeChecklist] = None
    issues: List[ValidationIssue] = Field(default_factory=list)
    structure_score: float = 0.0  # 0.0 to 1.0
    has_writeup: bool = False
    has_solve_script: bool = False
    files_found: Dict[str, List[str]] = Field(default_factory=dict)


class RankingScore(BaseModel):
    """Individual scoring from one evaluation perspective in the Ranking Agent.

    The Ranking Agent uses multiple personas (Pedagogical, Technical) to evaluate
    generated content from different viewpoints. This class represents one persona's
    assessment and recommendations.

    Attributes:
        score (int): Numerical rating from 1 (poor) to 10 (excellent)
        persona (str): Evaluation perspective - either "Pedagogical" or "Technical"
            - Pedagogical: Focuses on learning value, clarity, progression difficulty
            - Technical: Focuses on correctness, elegance, security implications
        justification (str): Detailed explanation of the score and reasoning
        improvements (List[str]): Specific recommendations for improvement
        dimension_scores (Optional[Dict[str, int]]): Per-dimension scores (1-10) for rubric anchoring.
            Technical dimensions: correctness, completeness, technical_accuracy, code_quality, logical_validity.
            Pedagogical dimensions: sections_structure, cognitive_load, scaffolding_reproducibility,
            relevance_curriculum, skill_level_awareness, human_language_context.

    Example:
        >>> pedagogical_score = RankingScore(
        ...     score=8,
        ...     persona="Pedagogical",
        ...     justification="Clear problem statement and good progression in difficulty",
        ...     improvements=[
        ...         "Add more comments in solution code",
        ...         "Include learning resource references"
        ...     ],
        ...     dimension_scores={
        ...         "sections_structure": 9,
        ...         "cognitive_load": 8,
        ...         "scaffolding_reproducibility": 7
        ...     }
        ... )
    """

    score: int = Field(ge=1, le=10)
    persona: str  # 'Pedagogical' or 'Technical'
    justification: str
    improvements: List[str] = Field(default_factory=list)
    dimension_scores: Optional[Dict[str, int]] = None


class RankingReport(BaseModel):
    """Final evaluation report combining all ranking perspectives for generated content.

    After the Ranking Agent evaluates content through multiple personas, this report
    synthesizes the results into an overall assessment. It includes scores from both
    pedagogical and technical reviewers, computed average score, and overall difficulty
    classification.

    Attributes:
        challenge_id (str): Unique identifier for the evaluated challenge
        overall_score (float): Synthesized score (typically average of all persona scores)
        pedagogical_review (RankingScore): Assessment from educational perspective.
            Contains dimension_scores mapping pedagogical dimensions to 1-10 scores:
            sections_structure, cognitive_load, scaffolding_reproducibility,
            relevance_curriculum, skill_level_awareness, human_language_context.
        technical_review (RankingScore): Assessment from technical perspective.
            Contains dimension_scores mapping technical dimensions to 1-10 scores:
            correctness, completeness, technical_accuracy, code_quality, logical_validity.
        technical_rank (str): Classification of challenge difficulty level
            - "Beginner": Introductory concepts, basic techniques
            - "Intermediate": Multi-step exploitation, some domain knowledge required
            - "Advanced": Complex techniques, deep understanding of exploitation
        dimension_scores (Optional[Dict[str, float]]): Aggregated dimension scores across
            both personas. Merges technical and pedagogical dimension_scores into a single
            flat dict (float values). Shared dimension names are averaged; unique names
            are taken as-is. None if neither persona provided dimension_scores.

    Example:
        >>> report = RankingReport(
        ...     challenge_id="crypto_001",
        ...     overall_score=8.5,
        ...     pedagogical_review=RankingScore(
        ...         score=8,
        ...         persona="Pedagogical",
        ...         justification="Clear progression and learning objectives",
        ...         dimension_scores={"sections_structure": 9, "cognitive_load": 8}
        ...     ),
        ...     technical_review=RankingScore(
        ...         score=9,
        ...         persona="Technical",
        ...         justification="Demonstrates correct security concepts",
        ...         dimension_scores={"correctness": 9, "completeness": 8}
        ...     ),
        ...     technical_rank="Intermediate",
        ...     dimension_scores={"sections_structure": 9.0, "cognitive_load": 8.0,
        ...                       "correctness": 9.0, "completeness": 8.0}
        ... )
    """

    challenge_id: str
    overall_score: float
    pedagogical_review: RankingScore
    technical_review: RankingScore
    technical_rank: str  # e.g., 'Beginner', 'Intermediate', 'Advanced'
    dimension_scores: Optional[Dict[str, float]] = None
    overall_scores: Optional[Dict[str, float]] = None
    """Sensitivity analysis: all three scoring views computed in every run.
    Keys: 'mean', 'min', 'weighted_<tech*100>_<ped*100>' (e.g. 'weighted_50_50').
    The active SCORING_POLICY determines the top-level overall_score.
    None when not populated (e.g. legacy data or error fallback reports)."""


class PedagogicalRubricScores(BaseModel):
    """Objective section presence and closure scores for writeup evaluation.

    Aligned with the course writeup guidelines: required sections and
    presence/closure checks so evaluation gives objective criteria alongside
    optional LLM judgment. Used by evaluation_service and can feed Ranking Agent.

    Attributes:
        section_scores (Dict[str, bool]): Per-section presence (True if detected).
            Keys match the course schema: abstract_tldr, objectives, skills,
            definitions, reproducibility_step0, thought_process, step_by_step,
            solution_script, conclusion, extra_resources.
        section_count_present (int): Number of sections present (0–10).
        closure_score (float): 0.0–1.0; conclusion present and abstract length ok.
        rubric_score_normalized (float): 0.0–1.0; combined section + closure.
    """

    section_scores: Dict[str, bool] = Field(default_factory=dict)
    section_count_present: int = 0
    closure_score: float = 0.0
    rubric_score_normalized: float = 0.0


class EvaluationReport(BaseModel):
    """Structured evaluation output for generated course.

    Returned by evaluation_service.evaluate_writeup(). Measures faithfulness,
    relevance, and quality via pedagogical rubric (objective) and optional
    LLM/Ragas metrics. Can feed reporting or the Ranking Agent.

    Attributes:
        challenge_id (Optional[str]): Set when provided to evaluate_writeup.
        category (Optional[str]): Challenge category when provided.
        rubric (PedagogicalRubricScores): Objective section/closure scores.
        faithfulness_score (Optional[float]): 0.0–1.0 if computed (e.g. Ragas).
        relevance_score (Optional[float]): 0.0–1.0 if computed.
        quality_score (Optional[float]): 0.0–1.0 or 1–10 scale if from LLM.
        llm_judgment (Optional[str]): Short qualitative summary when use_llm=True.
        summary (str): One-line summary suitable for reporting.
    """

    challenge_id: Optional[str] = None
    category: Optional[str] = None
    rubric: PedagogicalRubricScores = Field(default_factory=PedagogicalRubricScores)
    faithfulness_score: Optional[float] = None
    relevance_score: Optional[float] = None
    quality_score: Optional[float] = None
    llm_judgment: Optional[str] = None
    summary: str = ""


class CodeAnalysisResult(BaseModel):
    """Result of static analysis of a solver script.

    Provides extra technical context (complexity, patterns, issues) for the
    Ranking Agent or reporting. Interface supports full implementation later
    (e.g. multi-language, security linters).

    Attributes:
        language (str): Detected or inferred language (e.g. python).
        valid_syntax (bool): True if code parses/compiles.
        cyclomatic_complexity (Optional[int]): McCabe complexity if computed.
        lines_of_code (int): Approximate LOC.
        issues (List[str]): Short descriptions of detected issues (e.g. hardcoded secret).
        patterns (List[str]): Detected patterns (e.g. uses_requests, uses_subprocess).
        summary (str): One-line summary for reporting.
    """

    language: str = "unknown"
    valid_syntax: bool = False
    cyclomatic_complexity: Optional[int] = None
    lines_of_code: int = 0
    issues: List[str] = Field(default_factory=list)
    patterns: List[str] = Field(default_factory=list)
    summary: str = ""


class WriteupMapping(BaseModel):
    """Mapping of a challenge writeup to curriculum/taxonomy IDs.

    Each writeup is tagged with MITRE ATT&CK technique(s), CWE(s), and OWASP
    WSTG scenario(s) detected in or inferred from the writeup text. Output feeds
    ranking, reporting, and curriculum tagging. IDs align with Content Generation
    guidance from the cybersecurity glossary sources.

    Attributes:
        challenge_id (str): Unique identifier for the challenge.
        attack_technique_ids (List[str]): MITRE ATT&CK technique IDs (e.g. T1566, T1566.001).
        cwe_ids (List[str]): CWE weakness IDs (e.g. CWE-79, CWE-89).
        owasp_wstg_ids (List[str]): OWASP WSTG scenario IDs (e.g. WSTG-INPV-05).

    Example:
        >>> mapping = WriteupMapping(
        ...     challenge_id="web_001",
        ...     attack_technique_ids=["T1059.001"],
        ...     cwe_ids=["CWE-79", "CWE-89"],
        ...     owasp_wstg_ids=["WSTG-INPV-01", "WSTG-INPV-05"]
        ... )
    """

    challenge_id: str
    attack_technique_ids: List[str] = Field(default_factory=list)
    cwe_ids: List[str] = Field(default_factory=list)
    owasp_wstg_ids: List[str] = Field(default_factory=list)
