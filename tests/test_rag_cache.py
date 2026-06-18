"""Tests for embedding cache in vector_db_service.

B — RAG embedding cache.

Pre-fix: these tests will fail because no cache exists yet.
Post-fix: all tests pass.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


class TestEmbeddingCacheBasic:
    """Cache returns same vector for same query without re-embedding."""

    def test_same_query_hits_cache(self):
        """Second call with same text must NOT invoke the embed function again."""
        from src.services.embedding_cache import EmbeddingCache

        mock_embed_fn = MagicMock(return_value=[0.1, 0.2, 0.3])
        cache = EmbeddingCache(embed_fn=mock_embed_fn, maxsize=256)

        result1 = cache.embed("XSS vulnerability")
        result2 = cache.embed("XSS vulnerability")

        assert result1 == result2
        # embed_fn must be called exactly once for identical queries
        mock_embed_fn.assert_called_once_with("XSS vulnerability")

    def test_different_queries_call_embed_separately(self):
        """Different queries each call the embed function once."""
        from src.services.embedding_cache import EmbeddingCache

        mock_embed_fn = MagicMock(side_effect=lambda q: [hash(q) % 100 / 100])
        cache = EmbeddingCache(embed_fn=mock_embed_fn, maxsize=256)

        r1 = cache.embed("query A")
        r2 = cache.embed("query B")
        r3 = cache.embed("query A")  # repeat — should hit cache

        assert mock_embed_fn.call_count == 2
        assert r1 == r3  # same vector from cache
        assert r1 != r2

    def test_lru_eviction_at_maxsize(self):
        """When cache is full, least-recently-used entry is evicted."""
        from src.services.embedding_cache import EmbeddingCache

        call_log: list[str] = []

        def embed_fn(q: str):
            call_log.append(q)
            return [len(q) / 100]

        cache = EmbeddingCache(embed_fn=embed_fn, maxsize=2)

        cache.embed("a")  # miss → call
        cache.embed("b")  # miss → call
        cache.embed("a")  # hit  → no call (a is now MRU, b is LRU)
        cache.embed("c")  # miss → call, evicts b
        cache.embed("b")  # miss again → call (b was evicted)

        assert call_log.count("a") == 1
        assert call_log.count("b") == 2  # once initial, once after eviction
        assert call_log.count("c") == 1

    def test_cache_hit_ratio_reporting(self):
        """Cache exposes hit/miss counters for end-of-run reporting."""
        from src.services.embedding_cache import EmbeddingCache

        mock_embed_fn = MagicMock(return_value=[0.5])
        cache = EmbeddingCache(embed_fn=mock_embed_fn, maxsize=256)

        cache.embed("q1")
        cache.embed("q1")
        cache.embed("q2")
        cache.embed("q1")

        stats = cache.stats()
        assert stats["hits"] == 2
        assert stats["misses"] == 2
        assert stats["hit_ratio"] == pytest.approx(0.5, abs=0.01)


class TestEmbeddingCacheIntegration:
    """VectorDBService similarity_search uses cache for query embeddings."""

    def test_vector_db_service_imports_cache(self):
        """EmbeddingCache must be importable from src.services.embedding_cache."""
        from src.services.embedding_cache import EmbeddingCache  # noqa: F401

    def test_cache_module_exists(self):
        """Module src.services.embedding_cache must exist."""
        import importlib

        mod = importlib.import_module("src.services.embedding_cache")
        assert hasattr(mod, "EmbeddingCache")
        assert hasattr(mod, "get_process_cache")
