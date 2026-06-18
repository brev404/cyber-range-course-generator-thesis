# Context Creator — Multi-Agent Course Generation from CTF Challenges

A multi-agent [LangGraph](https://github.com/langchain-ai/langgraph) pipeline that automatically generates structured educational course content from Capture-the-Flag (CTF) / Cyber Range challenge files. The pipeline analyzes challenge materials, generates course content (concepts, exercises, solutions), evaluates quality via LLM judges, and supports a human-in-the-loop refinement loop.

This repository is the reference implementation developed for the master's thesis **„Sistem automatizat bazat pe agenți inteligenți pentru validarea și generarea conținutului educațional în platforme de tip Cyber Range”** (*Automated intelligent-agent-based system for the validation and generation of educational content in Cyber Range platforms*) by **Bogdan-George Carp**, master's program *Tehnologii Multimedia în Aplicații de Biometrie și Securitatea Informației* (BIOSINF), **Faculty of Electronics, Telecommunications and Information Technology (ETTI)**, National University of Science and Technology POLITEHNICA Bucharest, 2026.

> Scientific coordinator: Ș.L. Georgian Nicolae

## Architecture

The pipeline consists of two stages:

**Pre-graph pipeline** (`src/main.py` flags): sequential Python scripts for challenge ingestion and knowledge base construction.

**LangGraph graph** (`src/run_graph.py`): an 8-node state machine:

```
coordinator → validator → content_generation → course_terminology_checker
           → mapping → ranking → refinement_step → hitl
```

LLM calls are made only in `content_generation` (course generation) and `ranking` (LLM-as-judge evaluation). All other nodes are pure Python.

See `docs/reference/ARCHITECTURE.md` for a full description of the graph, node responsibilities, and data flow.

## Installation

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/brev404/cyber-range-course-generator-thesis.git
cd cyber-range-course-generator-thesis
cp .env.example .env                          # fill in your LLM API key(s)
uv venv && uv pip install -r requirements.txt  # install dependencies
```

Configure your LLM provider in `.env`:

```
OPENAI_API_KEY=sk-...         # or ANTHROPIC_API_KEY / GOOGLE_API_KEY
LLM_DEFAULT_PROVIDER=openai   # openai | anthropic | google | openrouter | deepseek
LLM_TEMPERATURE=0             # 0 for reproducibility
```

RAG uses local sentence-transformers embeddings by default (no API key required). Set `EMBEDDING_PROVIDER=openai` to use OpenAI embeddings instead.

## Usage

### Pre-graph pipeline (challenge ingestion)

```bash
# Run all four steps in sequence
uv run python src/main.py --analyse-contest

# Or run steps individually
uv run python src/main.py --organize    # step 1: organize challenge archives
uv run python src/main.py --analyze     # step 2: analyze challenge files
uv run python src/main.py --map-docs    # step 3: map PDF documentation
uv run python src/main.py --validate    # step 4: validate structure
```

Place raw challenge archives in `data/input/`. Processed challenges are written to `data/processed/`.

### Graph (course generation)

```bash
uv run python src/main.py --generate-courses
```

Key options:

- `--provider anthropic` — override LLM provider
- `--model claude-sonnet-4-6` — override model
- `--skip-ranking` — skip LLM judge evaluation
- `--content-subset N` — limit to N challenges
- `--no-rag` — disable RAG retrieval (ablation)
- `--output-language ro` — generate in Romanian (`en` default)

Generated course content is written to `cyberedu/write-up/course.md` alongside each challenge. Author `writeup.md` files are never overwritten.

### TUI (optional)

```bash
uv run python src/tui.py
```

Launches a Textual-based terminal UI for configuring and monitoring pipeline runs.

### Tests

```bash
PYTHONPATH=. uv run pytest -q
```

## Challenge Data Availability

The CTF challenge dataset used in the thesis evaluation is not included in this repository. Challenge data is available on reasonable request to the author.

Experiment outputs (scores, rankings, run configurations) from the thesis evaluation are also not included here and are available on request.

## Documentation

- `docs/reference/ARCHITECTURE.md` — graph nodes, data flow, state machine
- `docs/reference/FLOW_AND_LLM.md` — CLI options, LLM call budget, provider setup
- `docs/reference/CHALLENGE_STRUCTURE.md` — expected challenge directory layout
- `docs/reference/CONVENTIONS.md` — code conventions, terminology
- `docs/reference/REPRODUCIBILITY.md` — reproducibility guidance for pipeline runs
- `docs/reference/TERMINOLOGY.md` — course vs. writeup distinction
- `docs/reference/FEEDBACK_LOOP.md` — ranking and refinement loop
- `docs/reference/HITL_WORKFLOW.md` — human-in-the-loop interface
- `docs/ranking_criterias/` — rubric sources and criteria for the LLM judge dimensions

## License

Released under the [MIT License](LICENSE). © 2026 Bogdan-George Carp.

## Citation

```bibtex
@mastersthesis{carp2026coursegen,
  title  = {Sistem automatizat bazat pe agenți inteligenți pentru validarea și
            generarea conținutului educațional în platforme de tip Cyber Range},
  author = {Carp, Bogdan-George},
  school = {Faculty of Electronics, Telecommunications and Information Technology (ETTI),
            National University of Science and Technology POLITEHNICA Bucharest},
  year   = {2026},
  type   = {Master's thesis (Disertație)},
  note   = {In Romanian. English title: Automated intelligent-agent-based system for the
            validation and generation of educational content in Cyber Range platforms}
}
```
