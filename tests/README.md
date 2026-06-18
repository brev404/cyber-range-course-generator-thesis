# Tests

End-to-end tests for env configuration, pipeline (organize + validate), mapping, and graph.

## Running tests

From the project root, use the project venv (per `docs/reference/CONVENTIONS.md` § III.5):

```bash
./venv/bin/python -m pytest tests/
```

Verbose output:

```bash
./venv/bin/python -m pytest tests/ -v
```

## Fixtures vs real paths

- **Fixtures**: Simple challenges live under `tests/fixtures/challenges/` (raw) and `tests/fixtures/processed/` (processed layout). E2E tests patch `RAW_CHALLENGES_SOURCE`, `PROCESSED_DIR`, and optionally `OFFICIAL_DOCS_SOURCE` to these paths so the suite does not depend on external machine-specific paths.
- **Real paths**: Set `REAL_CHALLENGES_DIR` and `REAL_DOCS_DIR` env vars to point at local challenge data; an optional test runs the full pipeline against them and is skipped when those env vars are unset (e.g. CI).

See `tests/fixtures/README.md` for fixture layout and usage.
