# Reproducibility

This document describes reproducibility requirements and practices for the Cyber Range Content Validation project. There are two aspects: **content reproducibility** (what the student has) and **pipeline reproducibility** (generation quality consistency).

## 1. Content reproducibility

**Purpose:** Ensure learners can follow along and reproduce the challenge experience.

### Requirements

- **Step 0 = metadata and documented resources:** Challenge name, category, difficulty; description or link; documented resources (binary, PCAP, source, public files, deployment) when applicable.
- **Explicit "what the student has":** Generated courses must include an explicit section stating what the student has access to: challenge description, public files, deployment (if any). Do not assume they have source, writeup, or the flag.
- **"You do NOT have access to":** Where applicable, state what the student does *not* have (e.g. server source, author writeup, flag in advance).

See the course writeup guidelines (§6) for full reproducibility requirements and the Step 0 checklist.

### Verification

```bash
./venv/bin/python -m src.utils.verify_outputs --check-courses
```

See `docs/reference/OUTPUT_VERIFICATION.md` for details.

---

## 2. Pipeline reproducibility

**Purpose:** Given the same settings, ensure consistent generation quality across courses and runs.

### Settings that affect generation quality

| Setting | Effect on quality |
|---------|-------------------|
| `LLM_TEMPERATURE` | Higher = more variation; 0 = deterministic; 0.7 default |
| `LLM_DEFAULT_PROVIDER` | Different providers/models → different outputs |
| `LLM_DEFAULT_MODEL` | Model choice affects quality and style |
| `RAG_CHUNK_SIZE` | Chunk size affects RAG context for content generation |
| `RAG_CHUNK_OVERLAP` | Overlap affects RAG context continuity |
| `EMBEDDING_PROVIDER` / `EMBEDDING_MODEL` | Embeddings affect RAG retrieval |
| `CONTENT_GENERATION_MAX_TOKENS` | Limits course length; truncation if too low |
| `CONTENT_GENERATION_SOLVE_MAX_TOKENS` | Limits solve script length |
| `RANKING_MAX_TOKENS` | Limits ranking justification length; truncation if too low |
| `RANKING_PASS_THRESHOLD` | Threshold for refinement; affects when HITL triggers |
| `MAX_REFINEMENT_ROUNDS` | How many refinement loops before HITL |
| `LLM_MAX_CONCURRENT` | Parallelism; can change ordering and timing |

### Fixed settings for comparable quality

Using the same `.env` (or config snapshot) supports comparable quality across runs. Pin `LLM_DEFAULT_MODEL`, `LLM_DEFAULT_PROVIDER`, and RAG settings for reproducibility.

### Deterministic runs

For more deterministic output:

1. Set `LLM_TEMPERATURE=0` in `.env`.
2. Use the same `LLM_DEFAULT_MODEL` and `LLM_DEFAULT_PROVIDER` across runs.
3. Keep RAG settings (`RAG_CHUNK_SIZE`, `RAG_CHUNK_OVERLAP`, `EMBEDDING_PROVIDER`, `EMBEDDING_MODEL`) fixed.

### Config snapshot (optional)

To version or snapshot config for reproducible runs:

1. **Copy `.env`:** `cp .env config_snapshot_YYYYMMDD.env` (e.g. `config_snapshot_20250202.env`).
2. **Export settings:** Optionally create a script that dumps current settings to a file for audit.
3. **Restore for a run:** `cp config_snapshot_YYYYMMDD.env .env` before running the pipeline.

Store snapshots outside version control (add `config_snapshot_*.env` to `.gitignore` if they contain secrets) or in a secure location.

### Automatic run config export

At each pipeline start, the main runner:

1. **Logs** a "Run config (reproducibility)" block to `logs/main_runner.log` with all settings that affect generation quality (secrets redacted as "set"/"unset").
2. **Saves** to `data/outputs/run_config_latest.json` (overwrites each run). Use this to compare or restore the last run's config.

Disable file save with `SAVE_RUN_CONFIG=false` in `.env`.

---

## 3. How to reproduce a run

To reproduce a specific pipeline run:

1. **Same `.env`:** Use the same environment variables (or config snapshot) as the original run.
2. **Same KB state:** Ensure `data/knowledge_base/` is unchanged. If RAG embeddings are persisted (`CHROMA_PERSIST_DIR`), use the same Chroma DB or rebuild with the same KB files.
3. **Same challenge set:** Use the same processed challenges in `data/processed/` (e.g. same archive, same organization).

For maximum determinism, also set `LLM_TEMPERATURE=0`.
