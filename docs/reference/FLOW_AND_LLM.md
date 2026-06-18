# Flow and LLM Usage

Short reference for which pipeline steps use the LLM vs Python-only logic.

## LLM used

- **Content Generation** — RAG + research-based course and solve-script generation.
- **Ranking** — Pedagogical and technical rubric evaluation (LLM-based scoring).

## Python-only (no LLM)

- **Validation** — Structural checks via `validate_challenge_structure.validate_challenge` and rule-based Step 0 (reproducibility, metadata). No LLM calls.
- **Mapping** — Rule-based ID loading from the knowledge base, regex parsing of course text, and validation against loaded ATT&CK/CWE/OWASP IDs. No LLM calls.

## Parallel LLM calls

When your API supports high RPM (e.g. Gemini 2.5 Flash Lite 4k RPM), set **LLM_MAX_CONCURRENT** in `.env` to run content generation and ranking with multiple concurrent LLM calls:

- **LLM_MAX_CONCURRENT=1** (default): Sequential; one challenge at a time.
- **LLM_MAX_CONCURRENT=4** or **8**: Up to 4 or 8 challenges processed in parallel (ThreadPoolExecutor). Tune to stay within your rate limit.

Content generation and ranking both respect this setting.

## Token usage

To minimize token usage, run validation and mapping as-is (they do not consume LLM tokens). State options (see `src/core/state.py`):

- **skip_ranking** (bool): If True, the ranking node skips all LLM calls and leaves `ranking_reports` unchanged.
- **ranking_subset_ids** (list of challenge IDs): If set, ranking runs only for those challenges; others are skipped.
- **content_generation_subset_ids** (list of challenge IDs): If set, content generation runs the LLM only for those challenges; others are skipped. Challenges that already have an entry in `generated_courses` are skipped (conditional content generation).
- **stop_on_validation_fail** (bool): If True, after validation the graph routes to END when any report has a critical (HIGH or CRITICAL) issue, so content generation and ranking are not run on invalid challenges.

## CLI for graph

Run the graph with token-saving options:

- **./venv/bin/python src/main.py --generate-courses** — primary entry (ingests RAG, then runs graph). `--run-graph` is equivalent. Use `./.venv/bin/python` if venv is named `.venv`.
- **./venv/bin/python -m src.run_graph** — dedicated entry point (same flags).

Flags: `--skip-ranking`, `--ranking-subset id1,id2`, `--content-subset id1,id2`, `--stop-on-validation-fail`.
