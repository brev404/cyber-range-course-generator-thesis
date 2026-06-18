# System Architecture for Automated Cyber Range Content Validation

This document details the proposed modular architecture for the Master's Dissertation project: "Automated System based on Intelligent Agents for Cyber Range Content Validation." It leverages Python, intelligent agents orchestrated by LangChain/LangGraph, and various AI/ML technologies to achieve its goals of content validation, generation, and ranking.

## 1. Folder Structure

The project adopts a clear, modular folder structure to organize code, data, and configurations.

```
.
├── docs/
│   ├── reference/           # Architecture, conventions, and reference docs
│   │   ├── ARCHITECTURE.md
│   │   ├── brief.md
│   │   └── CONVENTIONS.md
│   ├── initial_project_information/   # cerinte.pdf, first_gem.md
│   └── pedagogical_expertise/         # Reference documents and domain knowledge
├── data/
│   ├── input/               # Raw challenge archives (.zip files uploaded by user)
│   ├── processed/           # Extracted and pre-processed challenge files (unzipped contents)
│   ├── knowledge_base/      # Curated documents for RAG (e.g., existing high-quality write-ups,
│   │                        # technical documentation, glossaries, challenge templates)
│   └── outputs/             # Validation reports (.json), ranking reports (.json). Generated courses
│                            # (course.md) live under PROCESSED_DIR/.../cyberedu/write-up/course.md
├── src/
│   ├── main.py              # Main entry point for the application (e.g., CLI)
│   ├── config/              # Configuration settings for LLMs, paths, thresholds, etc.
│   │   └── settings.py      # Pydantic-based configuration management
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── coordinator_agent.py          # Orchestrates the overall workflow of other agents
│   │   ├── validation_agent.py           # Handles structural and initial content integrity checks
│   │   ├── content_generation_agent.py   # Generates courses (Cybersecurity Expert Educator persona)
│   │   ├── ranking_agent.py              # Evaluates content quality (Learning/Pedagogical & Technical Expert personas)
│   │   └── hitl_agent.py                 # Manages human intervention points (Human-in-the-Loop)
│   ├── core/
│   │   ├── __init__.py
│   │   ├── state.py                      # Defines the LangGraph state for the agent workflow
│   │   └── graph.py                      # Defines the LangGraph agent workflow
│   ├── services/                         # Integrations with external technologies
│   │   ├── __init__.py
│   │   ├── llm_service.py                # Wrapper for interacting with various LLM APIs
│   │   ├── vector_db_service.py          # Interface for the Vector Database (RAG)
│   │   ├── langsmith_service.py          # Setup for LangSmith tracing and monitoring
│   ├── models/                           # Pydantic models for strict data validation and structure
│   │   ├── __init__.py
│   │   ├── challenge_models.py           # Models for challenge metadata, files
│   │   ├── report_models.py              # Models for validation and ranking reports
│   └── utils/                            # General utility functions (e.g., file operations, logging)
│       └── __init__.py
├── tests/                                # Unit and integration tests
├── .env.example                          # Example environment variables file
├── requirements.txt                      # Project dependencies
└── README.md
```

## 2. System Architecture (Components and Services)

```mermaid
flowchart TB
    subgraph CLI["CLI & entry"]
        MAIN["main.py"]
        MAIN --> PIPELINE["Pipeline (--analyse-contest / --all, --organize, --analyze, --map-docs, --validate)"]
        MAIN --> GRAPH_ENTRY["Graph (--generate-courses / --run-graph)"]
    end

    subgraph CORE["Core (LangGraph)"]
        STATE["state.py\nAgentState"]
        GRAPH["graph.py\nStateGraph"]
        GRAPH --> STATE
    end

    subgraph AGENTS["Agents"]
        COORD["coordinator_agent"]
        VALID["validation_agent\n(Python-only)"]
        CONTENT["content_generation_agent\n(LLM)"]
        TERM["course_terminology_agent\n(Python-only)"]
        MAPPING["mapping_agent\n(Python-only)"]
        RANK["ranking_agent\n(LLM)"]
        HITL["hitl_agent"]
    end

    subgraph SERVICES["Services"]
        LLM["llm_service\n(OpenAI, Anthropic, Google)"]
        VDB["vector_db_service\n(RAG)"]
        LANG["langsmith_service\n(tracing)"]
end

    subgraph DATA["Data"]
        INPUT["data/input\n(raw archives)"]
        PROCESSED["data/processed\n(organized challenges)"]
        KB["data/knowledge_base\n(RAG source)"]
        OUT["data/outputs\nreports; cyberedu/write-up/course.md"]
    end

    subgraph VALIDATORS["Validators"]
        STRUCT["validate_challenge_structure"]
        TERMCHECK["terminology_checker"]
        CONTENTCHECK["course_content_checker"]
    end
GRAPH_ENTRY --> GRAPH
    GRAPH --> COORD
    GRAPH --> VALID
    GRAPH --> CONTENT
    GRAPH --> TERM
    GRAPH --> MAPPING
    GRAPH --> RANK
    GRAPH --> HITL

    CONTENT --> LLM
    CONTENT --> VDB
    RANK --> LLM
    COORD --> PROCESSED
    VALID --> STRUCT
    TERM --> TERMCHECK
    VDB --> KB
    LANG -.-> GRAPH

    INPUT --> PIPELINE
    PIPELINE --> PROCESSED
    CONTENT --> OUT
    RANK --> OUT
```

## 3. Pipeline Data Flow (LangGraph)

End-to-end flow of control and state between graph nodes.

```mermaid
flowchart LR
    START([START]) --> COORD

    subgraph PREP["Preparation"]
        COORD["coordinator"]
    end

    COORD --> VALID["validator\n(Python-only)"]

    subgraph GENERATION["Generation & checks"]
        VALID --> ROUTE_V{"stop_on_validation_fail\n& critical issue?"}
        ROUTE_V -->|no| CONTENT["content_generation\n(LLM + RAG)"]
        ROUTE_V -->|yes| END1([END])
        CONTENT --> TERM["course_terminology_checker\n(Python-only)"]
        TERM --> ROUTE_T{"terminology\nblock & issues?"}
        ROUTE_T -->|no / warn / annotate| MAP
        ROUTE_T -->|yes, under max rounds| REFINE
        MAP["mapping\n(Python-only)"] --> RANK["ranking\n(LLM)"]
    end

    subgraph DECISION["Ranking decision"]
        RANK --> ROUTE_R{"all scores ≥ threshold\nand no low scores?"}
        ROUTE_R -->|yes| END2([END])
        ROUTE_R -->|no, under max refinement| REFINE["refinement_step"]
        ROUTE_R -->|no, max rounds or low score| HITL["hitl"]
    end

    subgraph HITL_BLOCK["Human-in-the-loop"]
        HITL --> ROUTE_H{"approved or\nmax iterations?"}
        ROUTE_H -->|yes| END3([END])
        ROUTE_H -->|no| REFINE
    end

    REFINE --> CONTENT

    COORD:::python
    VALID:::python
    TERM:::python
    MAP:::python
    CONTENT:::llm
    RANK:::llm

    classDef python fill:#c8e6c9
    classDef llm fill:#bbdefb
```

## 4. State Flow (What Each Node Reads and Writes)

```mermaid
flowchart LR
    subgraph STATE_FLOW["AgentState fields by node"]
        S0["(initial)\nchallenges, config flags"]
        S1["coordinator\n→ organized_challenges\n→ challenge_ids"]
        S2["validator\n← organized_challenges\n→ validation_reports"]
        S3["content_generation\n← organized_challenges, KB\n→ generated_courses\n→ generated_solve_scripts"]
        S4["course_terminology_checker\n← generated_courses\n→ course_terminology_issues"]
        S5["mapping\n← generated_courses\n→ writeup_mappings"]
        S6["ranking\n← generated_courses, writeup_mappings\n→ ranking_reports"]
        S7["refinement_step\n→ content_generation_subset_ids\n→ refinement_count"]
        S8["hitl\n← ranking_reports\n→ human_feedback, hitl_approved"]
    end

    S0 --> S1 --> S2 --> S3 --> S4 --> S5 --> S6
    S6 -.->|low score| S7
    S7 --> S3
    S6 -.->|max rounds| S8
    S8 -.->|not approved| S7
```

## 5. External and Internal Dependencies

```mermaid
flowchart TB
    subgraph EXTERNAL["External"]
        API["LLM APIs\n(OpenAI, Anthropic, Google)"]
        LANGSMITH["LangSmith\n(tracing)"]
    end

    subgraph APP["Application"]
        subgraph LLM_NODES["Nodes using LLM"]
            CG["Content Generation"]
            RANK["Ranking"]
        end
        subgraph PY_NODES["Python-only nodes"]
            COORD["Coordinator"]
            VAL["Validator"]
            TERM["Course terminology"]
            MAP["Mapping"]
        end
        RAG["Vector DB (RAG)\n← knowledge_base"]
    end

    CG --> API
    RANK --> API
    CG --> RAG
    APP -.-> LANGSMITH
```

**Summary:** LLM nodes: Content Generation, Ranking. Python-only: Validation, Course terminology checker, Mapping. See `FLOW_AND_LLM.md` for token-saving options.

## 6. High-Level Data Flow

The system has two phases: **(1) Pipeline** (organize → analyze → map-docs → validate) and **(2) Graph** (validation → content generation → terminology check → mapping → ranking → HITL/refinement).

### Pipeline (CLI: `--analyse-contest` or `--all`)

2.  **Analyze:** Structure, file types, and dependencies are analyzed. Output: logs and statistics.
3.  **Map-docs:** Official PDFs are linked to challenges via text matching. Output: challenge → documentation mapping.
4.  **Validate:** Structural and Step 0 reproducibility checks. Output: validation reports.

### Graph (CLI: `--generate-courses` or `--run-graph`)

1.  **Coordinator:** Loads organized challenges from `data/processed/` into state (`organized_challenges`, `challenge_ids`). RAG is ingested from `data/knowledge_base/` before the graph runs.
2.  **Validation:** Structural checks on each challenge; optionally routes to END if `stop_on_validation_fail` and critical issues.

3.  **Content Generation (LLM + RAG):** Queries the Vector Database for relevant context. Uses the LLM Service to generate pedagogical **courses** (output: `course.md` in `cyberedu/write-up/`). See `TERMINOLOGY.md` for course vs writeup.
4.  **Course Terminology Checker (Python-only):** Validates ATT&CK/CWE/OWASP IDs in generated courses.
5.  **Mapping (Python-only):** Extracts MITRE ATT&CK, CWE, OWASP IDs for writeup mapping.
6.  **Ranking (LLM):** Pedagogical and technical reviewers evaluate the generated course. If scores are below threshold, the graph routes to refinement or HITL.
7.  **HITL / Refinement:** Human review at ranking checkpoints; refinement loops back to content generation when not approved.
8.  **Final Output:** `course.md` per challenge in `cyberedu/write-up/`; `validation_report.json`, `ranking_report.json` in `data/outputs/`.

**Knowledge base and dictionary:** All glossary elements (NIST, ATT&CK, CyBOK, CWE, OWASP WSTG) live in `data/knowledge_base/`. Agents load term sets from these files for RAG and taxonomy tagging.

## 7. Technology Choices and Rationale

*   **Python:** The core programming language, chosen for its extensive AI/ML ecosystem, readability, and compatibility with LangChain/LangGraph.
*   **Pydantic:** Used for defining data models (`src/models/`). Ensures strict data validation, type checking, and structured data handling across agents and components, critical for maintaining consistency and reducing errors in complex data flows.
*   **LangChain / LangGraph:**
    *   **LangGraph:** Provides the state machine for orchestrating complex multi-agent interactions, managing conversation history, and enabling `Human-in-the-Loop` (HITL) checkpoints. This is foundational for the iterative and collaborative nature of the agents.
    *   **LangChain:** Provides foundational components for LLM interactions, tool integrations, and prompt management within the agents.
*   **LLM APIs (OpenAI, Google, Claude):** Chosen for their advanced generative capabilities, allowing for testing and comparison of different models across experimental configurations. The `LLM Service` abstracts these for flexible integration.
*   **Vector Database (e.g., ChromaDB / FAISS / Custom):** Essential for implementing the **Retrieval Augmented Generation (RAG)** system. Stores vectorized embeddings of the project's knowledge base, enabling efficient semantic search and grounding LLM outputs in specific, relevant documents. Initial development might use an in-memory solution, with future consideration for more persistent or scalable options.
*   **Ragas / DeepEval:** Critical for evaluating the performance of the RAG system and the overall quality of LLM-generated content. These frameworks provide metrics to assess aspects like faithfulness, context relevance, and answer correctness, directly supporting the ranking system and academic validation.
*   **LangSmith:** Indispensable for observability, debugging, and monitoring of complex agent workflows. It provides detailed traces of LLM calls, tool usage, and agent decisions, which is crucial for understanding agent behavior, optimizing prompts, and identifying bottlenecks during development and experimentation.
*   **Optional Static Code Analysis Tools (e.g., Bandit for Python, ESLint for JS, Clang-Tidy for C/C++):** These tools will augment the technical validation by providing objective, rule-based analysis of code. Their outputs can be fed to the `Ranking Agent`'s "Hardcore Technical Expert" persona or used by the `Validation Agent` to enrich the `validation_report.json` with concrete security or quality findings. Their optional nature allows for flexibility in implementation scope.

This architecture provides a robust, extensible, and research-oriented framework for achieving the project's goals.
