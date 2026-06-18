"""Pre-graph sequential processing scripts for CTF challenge data.

CLI scripts for the 4 pre-processing steps (organize, analyze, map docs,
validate structure), invoked via ``uv run python src/pipeline/<script>.py``,
plus supporting KB / term-database builders. ``repair_challenge_structure``
is imported by ``src/core/graph.py``.

Inputs:  Raw challenge archives (data/raw/), contest directory trees
Outputs: Organized challenge structure (data/processed/), metadata KB
         (data/kb/), term databases for RAG
"""
