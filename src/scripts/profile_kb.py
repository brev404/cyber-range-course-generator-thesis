"""Profile the knowledge base for RAG optimization.

Measures chunk size distribution, embedding performance, retrieval quality,
and duplicate chunk detection. Outputs a report to docs/reference/archive/KB_PROFILING.md (or --output PATH).

Usage:
    python -m src.scripts.profile_kb [--output PATH] [--chunk-sizes 500,1000,1500,2000] [--overlaps 0,100,200,300]

Options:
    --output PATH    Write report to PATH (default: docs/reference/archive/KB_PROFILING.md)
    --chunk-sizes    Comma-separated chunk sizes to test (default: 500,1000,1500,2000)
    --overlaps       Comma-separated overlaps to test (default: 0,100,200,300)
    --skip-embed     Skip embedding/retrieval timing (faster, chunk stats only)
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

# Project root for imports
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.config.settings import settings  # noqa: E402
from src.services.vector_db_service import (  # noqa: E402
    VectorDBService,
    VectorDBServiceError,
    _load_documents_from_dir,
)


def _chunk_documents(chunk_size: int, chunk_overlap: int):
    """Load and chunk KB documents with given parameters."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    directory = Path(settings.KNOWLEDGE_BASE_DIR)
    documents = _load_documents_from_dir(directory)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return splitter.split_documents(documents)


def _count_duplicates(chunks) -> tuple[int, set[str]]:
    """Count exact duplicate chunks by content hash. Returns (duplicate_count, seen_hashes)."""
    seen: set[str] = set()
    duplicates = 0
    for chunk in chunks:
        h = hashlib.md5(chunk.page_content.encode("utf-8")).hexdigest()
        if h in seen:
            duplicates += 1
        else:
            seen.add(h)
    return duplicates, seen


def _chunk_stats(chunks) -> dict:
    """Compute chunk size statistics."""
    if not chunks:
        return {"total": 0, "avg_len": 0, "min_len": 0, "max_len": 0, "lengths": []}
    lengths = [len(c.page_content) for c in chunks]
    return {
        "total": len(chunks),
        "avg_len": sum(lengths) / len(lengths),
        "min_len": min(lengths),
        "max_len": max(lengths),
        "lengths": lengths,
    }


def _run_profiling(
    chunk_sizes: list[int],
    overlaps: list[int],
    skip_embed: bool,
) -> dict:
    """Run full profiling and return results dict."""
    directory = Path(settings.KNOWLEDGE_BASE_DIR)
    documents = _load_documents_from_dir(directory)
    if not documents:
        return {"error": "No documents found in knowledge base", "documents": 0}

    results: dict = {
        "documents": len(documents),
        "kb_dir": str(directory),
        "chunk_strategies": [],
        "embedding_provider": settings.EMBEDDING_PROVIDER,
        "embedding_model": (
            settings.EMBEDDING_MODEL
            if settings.EMBEDDING_PROVIDER in ("openai", "auto")
            and settings.OPENAI_API_KEY
            else settings.LOCAL_EMBEDDING_MODEL
        ),
    }

    # Default strategy (current settings)
    default_chunks = _chunk_documents(
        settings.RAG_CHUNK_SIZE, settings.RAG_CHUNK_OVERLAP
    )
    dup_count, _ = _count_duplicates(default_chunks)
    stats = _chunk_stats(default_chunks)
    results["default_strategy"] = {
        "chunk_size": settings.RAG_CHUNK_SIZE,
        "chunk_overlap": settings.RAG_CHUNK_OVERLAP,
        "total_chunks": stats["total"],
        "avg_chunk_len": round(stats["avg_len"], 1),
        "min_chunk_len": stats["min_len"],
        "max_chunk_len": stats["max_len"],
        "duplicate_chunks": dup_count,
        "duplicate_pct": (
            round(100 * dup_count / stats["total"], 2) if stats["total"] else 0
        ),
    }

    # Test different strategies
    for cs in chunk_sizes:
        for ov in overlaps:
            if ov >= cs:
                continue
            chunks = _chunk_documents(cs, ov)
            dup_count, _ = _count_duplicates(chunks)
            s = _chunk_stats(chunks)
            results["chunk_strategies"].append(
                {
                    "chunk_size": cs,
                    "chunk_overlap": ov,
                    "total_chunks": s["total"],
                    "avg_chunk_len": round(s["avg_len"], 1),
                    "duplicate_chunks": dup_count,
                    "duplicate_pct": (
                        round(100 * dup_count / s["total"], 2) if s["total"] else 0
                    ),
                }
            )

    if skip_embed:
        return results

    # Embedding and retrieval timing (use default strategy)
    try:
        svc = VectorDBService()
        t0 = time.perf_counter()
        n = svc.ingest_knowledge_base(replace=True)
        embed_time = time.perf_counter() - t0
        results["embedding"] = {
            "chunks_ingested": n,
            "time_seconds": round(embed_time, 2),
        }

        # Sample retrieval queries
        sample_queries = [
            "XSS vulnerability",
            "SQL injection",
            "buffer overflow",
            "MITRE ATT&CK technique",
            "CWE weakness",
        ]
        retrieval_times = []
        retrieval_scores = []
        for q in sample_queries:
            t0 = time.perf_counter()
            hits = svc.similarity_search(q, k=5)
            retrieval_times.append(time.perf_counter() - t0)
            if hits:
                scores = [h.get("score") for h in hits if h.get("score") is not None]
                retrieval_scores.extend(scores)

        results["retrieval"] = {
            "sample_queries": sample_queries,
            "avg_retrieval_time_ms": round(
                1000 * sum(retrieval_times) / len(retrieval_times), 2
            ),
            "avg_relevance_score": (
                round(sum(retrieval_scores) / len(retrieval_scores), 4)
                if retrieval_scores
                else None
            ),
        }
    except VectorDBServiceError as e:
        results["embedding"] = {"error": str(e)}
        results["retrieval"] = {"error": str(e)}

    return results


def _format_report(data: dict) -> str:
    """Format profiling results as Markdown."""
    lines = [
        "# Knowledge Base Profiling Report",
        "",
        "Generated by `python -m src.scripts.profile_kb`.",
        "",
        "## Summary",
        "",
        f"- **KB directory:** `{data.get('kb_dir', 'N/A')}`",
        f"- **Documents loaded:** {data.get('documents', 0)}",
        f"- **Embedding provider:** {data.get('embedding_provider', 'N/A')}",
        f"- **Embedding model:** {data.get('embedding_model', 'N/A')}",
        "",
    ]

    if "error" in data:
        lines.extend(["## Error", "", str(data["error"]), ""])
        return "\n".join(lines)

    # Default strategy
    ds = data.get("default_strategy", {})
    lines.extend(
        [
            "## Default Chunking Strategy (Current Settings)",
            "",
            f"- **Chunk size:** {ds.get('chunk_size', 'N/A')}",
            f"- **Chunk overlap:** {ds.get('chunk_overlap', 'N/A')}",
            f"- **Total chunks:** {ds.get('total_chunks', 'N/A')}",
            f"- **Avg chunk length:** {ds.get('avg_chunk_len', 'N/A')} chars",
            f"- **Min/Max chunk length:** {ds.get('min_chunk_len', 'N/A')} / {ds.get('max_chunk_len', 'N/A')}",
            f"- **Duplicate chunks:** {ds.get('duplicate_chunks', 0)} ({ds.get('duplicate_pct', 0)}%)",
            "",
        ]
    )

    # Chunk strategies comparison
    strategies = data.get("chunk_strategies", [])
    if strategies:
        lines.extend(
            [
                "## Chunk Strategy Comparison",
                "",
                "| Chunk Size | Overlap | Total Chunks | Avg Length | Duplicates | Dup % |",
                "|------------|---------|--------------|------------|------------|-------|",
            ]
        )
        for s in strategies:
            lines.append(
                f"| {s['chunk_size']} | {s['chunk_overlap']} | {s['total_chunks']} | "
                f"{s['avg_chunk_len']} | {s['duplicate_chunks']} | {s['duplicate_pct']}% |"
            )
        lines.extend(["", ""])

    # Embedding
    emb = data.get("embedding", {})
    if "error" in emb:
        lines.extend(["## Embedding", "", f"Error: {emb['error']}", ""])
    else:
        lines.extend(
            [
                "## Embedding Performance",
                "",
                f"- **Chunks ingested:** {emb.get('chunks_ingested', 'N/A')}",
                f"- **Time:** {emb.get('time_seconds', 'N/A')} seconds",
                "",
            ]
        )

    # Retrieval
    ret = data.get("retrieval", {})
    if "error" in ret:
        lines.extend(["## Retrieval", "", f"Error: {ret['error']}", ""])
    else:
        lines.extend(
            [
                "## Retrieval Performance",
                "",
                f"- **Sample queries:** {', '.join(repr(q) for q in ret.get('sample_queries', []))}",
                f"- **Avg retrieval time:** {ret.get('avg_retrieval_time_ms', 'N/A')} ms",
                f"- **Avg relevance score:** {ret.get('avg_relevance_score', 'N/A')}",
                "",
            ]
        )

    lines.extend(
        [
            "## Recommendations",
            "",
            "- **Chunking:** Current defaults (1000/200) balance chunk count and context. "
            "Smaller chunks (500) increase retrieval granularity but may fragment context; "
            "larger (1500–2000) preserve context but reduce precision.",
            "- **Deduplication:** Implement content-hash deduplication before ingestion to avoid "
            "duplicate chunks from overlapping regions or repeated content across files.",
            "- **Refresh:** Re-run ingest after adding/updating KB files. See `docs/reference/KB_REFRESH.md`.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile KB for RAG optimization")
    parser.add_argument(
        "--output",
        type=Path,
        default=_project_root / "docs" / "reference" / "archive" / "KB_PROFILING.md",
        help="Output report path",
    )
    parser.add_argument(
        "--chunk-sizes",
        type=str,
        default="500,1000,1500,2000",
        help="Comma-separated chunk sizes to test",
    )
    parser.add_argument(
        "--overlaps",
        type=str,
        default="0,100,200,300",
        help="Comma-separated overlaps to test",
    )
    parser.add_argument(
        "--skip-embed",
        action="store_true",
        help="Skip embedding/retrieval timing",
    )
    args = parser.parse_args()

    chunk_sizes = [int(x.strip()) for x in args.chunk_sizes.split(",")]
    overlaps = [int(x.strip()) for x in args.overlaps.split(",")]

    data = _run_profiling(chunk_sizes, overlaps, args.skip_embed)
    report = _format_report(data)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Report written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
