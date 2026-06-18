# E2E test fixtures

Minimal challenge fixtures used to verify **env configuration** and **mapping** without depending on external machine-specific paths.

## Layout

Structure matches `docs/reference/CHALLENGE_STRUCTURE.md`:

- **Raw challenges** (for pipeline organize step): `tests/fixtures/challenges/`
  - `crypto/test_crypto_01/` — simple hash challenge
  - `web/test_web_01/` — simple form challenge
  - Each has: `cyberedu/write-up/description.md` (≥80 chars), optional `writeup.md`, `solve.py`, `challenge-flags.txt`, `public/`

- **Processed challenges** (for validate and graph): `tests/fixtures/processed/raw_challenges/`
  - Same categories and challenge names with the same internal structure
  - Used when `PROCESSED_DIR` is set to `tests/fixtures/processed` so the coordinator and validator find challenges without running organize

## Usage

- **E2E with fixtures only**: Set `RAW_CHALLENGES_SOURCE` and/or `PROCESSED_DIR` (and optionally `OFFICIAL_DOCS_SOURCE`) to these paths or a temp copy. Tests do this via monkeypatch so the suite does not depend on external paths.
- **E2E with real data**: Set `REAL_CHALLENGES_DIR` and `REAL_DOCS_DIR` env vars to point at local challenge data; optional tests skip in CI when those env vars are unset.

## Running E2E

From project root with venv:

```bash
./venv/bin/python -m pytest tests/
```

See `tests/README.md` for more options and fixture vs real-path usage.
