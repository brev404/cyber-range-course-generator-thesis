"""LRU embedding cache for vector_db_service.

B — RAG embedding cache: avoids re-embedding identical query text across
refinement rounds. Each process gets one shared cache instance; cleared at
process start (no disk persistence needed).

Usage:
    from src.services.embedding_cache import get_process_cache

    cache = get_process_cache(embed_fn=my_embedder.embed_query, maxsize=256)
    vector = cache.embed("XSS vulnerability")
"""

from __future__ import annotations

from threading import Lock
from typing import Any, Callable, Optional

from loguru import logger


class EmbeddingCache:
    """Thread-safe LRU cache keyed on query text.

    Args:
        embed_fn: Callable[str] -> embedding (list[float] or array).
        maxsize: Maximum number of cached entries (LRU eviction).
    """

    def __init__(
        self,
        embed_fn: Callable[[str], Any],
        maxsize: int = 256,
    ) -> None:
        self._embed_fn = embed_fn
        self._maxsize = maxsize
        self._lock = Lock()
        self._hits = 0
        self._misses = 0
        # OrderedDict-based LRU: most-recently-used at right (last)
        from collections import OrderedDict

        self._cache: OrderedDict[str, Any] = OrderedDict()

    def embed(self, query: str) -> Any:
        """Return embedding for query. Hits cache on repeated identical queries.

        Thread-safe: safe for concurrent workers in ThreadPoolExecutor.
        """
        with self._lock:
            if query in self._cache:
                self._hits += 1
                # Move to end (most recently used)
                self._cache.move_to_end(query)
                return self._cache[query]
            self._misses += 1

        # Call embed_fn *outside* the lock to avoid blocking other threads.
        result = self._embed_fn(query)

        with self._lock:
            # Check again in case another thread embedded same query meanwhile.
            if query in self._cache:
                self._cache.move_to_end(query)
                return self._cache[query]
            self._cache[query] = result
            self._cache.move_to_end(query)
            # Evict LRU entry if over maxsize
            if len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)  # remove oldest (left end)

        return result

    def stats(self) -> dict[str, int | float]:
        """Return hit/miss counters for end-of-run logging."""
        with self._lock:
            total = self._hits + self._misses
            hit_ratio = self._hits / total if total > 0 else 0.0
            return {
                "hits": self._hits,
                "misses": self._misses,
                "hit_ratio": hit_ratio,
                "cached_entries": len(self._cache),
            }

    def log_stats(self) -> None:
        """Emit cache stats via loguru at INFO level."""
        s = self.stats()
        logger.info(
            "Embedding cache stats: hits={} misses={} ratio={:.1%} entries={}",
            s["hits"],
            s["misses"],
            s["hit_ratio"],
            s["cached_entries"],
        )


# Process-level singleton cache.  Initialized lazily via get_process_cache().
_process_cache: Optional[EmbeddingCache] = None
_process_cache_lock = Lock()


def get_process_cache(
    embed_fn: Optional[Callable[[str], Any]] = None,
    maxsize: int = 256,
) -> EmbeddingCache:
    """Return (or create) the process-level embedding cache singleton.

    The first call must supply embed_fn.  Subsequent calls may omit it.
    Cache is NOT replaced on subsequent calls (singleton semantics).

    Args:
        embed_fn: The embedding function to use if creating the cache.
        maxsize: LRU capacity (only used on first call).

    Returns:
        The shared EmbeddingCache instance.
    """
    global _process_cache
    with _process_cache_lock:
        if _process_cache is None:
            if embed_fn is None:
                raise ValueError(
                    "get_process_cache: embed_fn is required on first call"
                )
            _process_cache = EmbeddingCache(embed_fn=embed_fn, maxsize=maxsize)
            logger.debug("Embedding cache created (maxsize={})", maxsize)
        return _process_cache


def reset_process_cache() -> None:
    """Clear the process-level cache singleton (used in tests / new experiment runs)."""
    global _process_cache
    with _process_cache_lock:
        if _process_cache is not None:
            _process_cache.log_stats()
        _process_cache = None
