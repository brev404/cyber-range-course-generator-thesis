"""Vector database integration for RAG over the knowledge base.

This module provides a Retrieval-Augmented Generation (RAG) service using ChromaDB:
- Ingest documents from KNOWLEDGE_BASE_DIR (.md, .txt)
- Chunk and embed with configurable embeddings:
  - **Local (free):** sentence-transformers, no API key (EMBEDDING_PROVIDER=local or auto without key)
  - **OpenAI (good for test):** text-embedding-3-small when OPENAI_API_KEY set (EMBEDDING_PROVIDER=openai or auto)
- Store in Chroma (persistent or in-memory)
- Similarity search for context-aware content generation and validation

Usage:
    from src.services.vector_db_service import VectorDBService, VectorDBServiceError

    service = VectorDBService()
    service.ingest_knowledge_base()  # Works with local (free) or OpenAI embeddings
    results = service.similarity_search("XSS vulnerability mitigation", k=5)
    retriever = service.get_retriever(k=5)  # For LangChain RAG chains
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from src.config.settings import settings


class VectorDBServiceError(Exception):
    """Raised when vector DB operations fail (missing config, ingest errors, etc.)."""

    pass


def _default_persist_dir() -> Path:
    """Return the directory for Chroma persistence. Uses DATA_DIR/chroma_db if not set."""
    if settings.CHROMA_PERSIST_DIR is not None:
        return settings.CHROMA_PERSIST_DIR
    return settings.DATA_DIR / "chroma_db"


def _load_documents_from_dir(directory: Path) -> list[Any]:
    """Load .md and .txt files from a directory into LangChain Document objects.

    Args:
        directory: Root path to scan recursively for .md and .txt files.

    Returns:
        List of Document instances with page_content and metadata (source path).
    """
    from langchain_core.documents import Document

    documents: list[Document] = []
    suffixes = (".md", ".txt")
    for path in directory.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            if not text.strip():
                continue
            documents.append(
                Document(
                    page_content=text,
                    metadata={"source": str(path), "filename": path.name},
                )
            )
        except OSError as e:
            logger.warning("Could not read {}: {}", path, e)
    return documents


def _get_embeddings() -> Any:
    """Return the embedding model: OpenAI if configured, else local (free, no API key).

    Uses EMBEDDING_PROVIDER: auto (OpenAI if key set, else local), openai, or local.
    """
    provider = settings.EMBEDDING_PROVIDER

    use_openai = provider == "openai" or (
        provider == "auto" and bool(settings.OPENAI_API_KEY)
    )

    if use_openai:
        if not settings.OPENAI_API_KEY:
            raise VectorDBServiceError(
                "EMBEDDING_PROVIDER=openai requires OPENAI_API_KEY. "
                "Set OPENAI_API_KEY in .env or use EMBEDDING_PROVIDER=local for free embeddings."
            )
        from langchain_openai import OpenAIEmbeddings

        logger.debug("RAG embeddings: OpenAI ({})", settings.EMBEDDING_MODEL)
        return OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            api_key=settings.OPENAI_API_KEY,
        )

    # Local (free): sentence-transformers, no API key
    # Prefer langchain_huggingface; fall back to langchain_community if not installed
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
        except ImportError as e:
            raise VectorDBServiceError(
                "Local embeddings require either 'langchain-huggingface' or 'langchain-community'. "
                "Install with: pip install langchain-community sentence-transformers"
            ) from e

    model = settings.LOCAL_EMBEDDING_MODEL
    logger.debug("RAG embeddings: local ({})", model)
    return HuggingFaceEmbeddings(
        model_name=model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def _relevance_score_fn(raw_score: float) -> float:
    """Normalize Chroma raw score to [0, 1] to avoid UserWarning.

    Chroma may return cosine similarity ([-1, 1]) or distance; this maps to [0, 1].
    """
    return max(0.0, min(1.0, (float(raw_score) + 1.0) / 2.0))


def _get_vector_store(
    persist_directory: Optional[Path] = None,
    collection_name: Optional[str] = None,
    embedding: Optional[Any] = None,
) -> Any:
    """Build a Chroma vector store (persistent or in-memory).

    Args:
        persist_directory: If set, Chroma persists to this path. If None, in-memory.
        collection_name: Chroma collection name.
        embedding: Embedding function (e.g. OpenAIEmbeddings).

    Returns:
        Chroma vector store instance.
    """
    from langchain_chroma import Chroma

    collection = collection_name or settings.CHROMA_COLLECTION_NAME
    emb = embedding or _get_embeddings()

    if persist_directory is not None:
        persist_directory = Path(persist_directory)
        persist_directory.mkdir(parents=True, exist_ok=True)
        return Chroma(
            collection_name=collection,
            embedding_function=emb,
            persist_directory=str(persist_directory),
            relevance_score_fn=_relevance_score_fn,
        )
    return Chroma(
        collection_name=collection,
        embedding_function=emb,
        relevance_score_fn=_relevance_score_fn,
    )


class VectorDBService:
    """RAG service over the knowledge base using ChromaDB.

    Embeddings: local (free, sentence-transformers) or OpenAI (set OPENAI_API_KEY).
    Ingest documents from KNOWLEDGE_BASE_DIR, then run similarity_search or
    get_retriever() for use in LangChain RAG chains.
    """

    def __init__(
        self,
        persist_directory: Optional[Path] = None,
        collection_name: Optional[str] = None,
    ) -> None:
        """Initialize the service. Does not load documents; call ingest_knowledge_base().

        Args:
            persist_directory: Override default Chroma persist path. None = use settings
                or DATA_DIR/chroma_db.
            collection_name: Override default collection name.
        """
        self._persist_dir = (
            persist_directory
            if persist_directory is not None
            else _default_persist_dir()
        )
        self._collection_name = collection_name or settings.CHROMA_COLLECTION_NAME
        self._vector_store: Optional[Any] = None
        self._embedding = None

    def _ensure_embedding(self) -> Any:
        """Lazily create embedding model."""
        if self._embedding is None:
            self._embedding = _get_embeddings()
        return self._embedding

    def _ensure_vector_store(self, create_empty: bool = False) -> Any:
        """Get or create the Chroma vector store. create_empty=True avoids loading existing."""
        if self._vector_store is not None:
            return self._vector_store
        emb = self._ensure_embedding()
        self._vector_store = _get_vector_store(
            persist_directory=self._persist_dir,
            collection_name=self._collection_name,
            embedding=emb,
        )
        return self._vector_store

    def ingest_knowledge_base(
        self,
        source_dir: Optional[Path] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        replace: bool = True,
    ) -> int:
        """Load documents from the knowledge base dir, chunk, embed, and add to Chroma.

        If replace=True (default), clears the collection first so each run reflects
        the current directory contents. Set replace=False to append to existing.

        Args:
            source_dir: Directory to load from. Default: settings.KNOWLEDGE_BASE_DIR.
            chunk_size: Character chunk size. Default: settings.RAG_CHUNK_SIZE.
            chunk_overlap: Overlap between chunks. Default: settings.RAG_CHUNK_OVERLAP.
            replace: If True, clear collection before adding (idempotent ingest).

        Returns:
            Number of document chunks added to the store.

        Raises:
            VectorDBServiceError: If OPENAI_API_KEY is missing or ingest fails.
        """
        directory = source_dir or settings.KNOWLEDGE_BASE_DIR
        directory = Path(directory)
        if not directory.is_dir():
            logger.warning("Knowledge base dir does not exist: {}", directory)
            return 0

        documents = _load_documents_from_dir(directory)
        if not documents:
            logger.info("No .md or .txt files found in {}", directory)
            return 0

        from langchain_chroma import Chroma
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        chunk_size = chunk_size if chunk_size is not None else settings.RAG_CHUNK_SIZE
        chunk_overlap = (
            chunk_overlap if chunk_overlap is not None else settings.RAG_CHUNK_OVERLAP
        )
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        chunks = splitter.split_documents(documents)
        if not chunks:
            return 0

        # Deduplication: skip chunks with identical content
        seen_hashes: set[str] = set()
        unique_chunks: list[Any] = []
        duplicates_skipped = 0
        for chunk in chunks:
            content_hash = hashlib.md5(chunk.page_content.encode("utf-8")).hexdigest()
            if content_hash in seen_hashes:
                duplicates_skipped += 1
                continue
            seen_hashes.add(content_hash)
            unique_chunks.append(chunk)
        chunks = unique_chunks
        if duplicates_skipped > 0:
            logger.info(
                "Deduplication: skipped {} duplicate chunk(s) ({} unique remaining)",
                duplicates_skipped,
                len(chunks),
            )

        emb = self._ensure_embedding()
        persist_dir = str(self._persist_dir)
        self._persist_dir.mkdir(parents=True, exist_ok=True)

        if replace:
            self.clear()
            # Ensure persisted collection is removed so from_documents replaces it
            try:
                tmp = Chroma(
                    collection_name=self._collection_name,
                    embedding_function=emb,
                    persist_directory=persist_dir,
                )
                if hasattr(tmp, "_client") and tmp._client is not None:
                    tmp._client.delete_collection(name=self._collection_name)
            except Exception as e:
                logger.debug(
                    "Could not delete existing collection (may not exist): {}", e
                )

        self._vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=emb,
            collection_name=self._collection_name,
            persist_directory=persist_dir,
            relevance_score_fn=_relevance_score_fn,
        )
        logger.info(
            "Ingested {} chunks from {} into Chroma (collection={})",
            len(chunks),
            directory,
            self._collection_name,
        )
        return len(chunks)

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        metadata_filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Return the top-k most similar chunks to the query.

        Args:
            query: Search query text.
            k: Number of results to return.
            metadata_filter: Optional ChromaDB where-clause filter, e.g.
                {"contest_id": {"$eq": "raw_challenges"}}. None = no filter.
            **kwargs: Passed to the vector store similarity_search_with_score if supported.

        Returns:
            List of dicts with keys: content, metadata, score (if available).
        """
        store = self._ensure_vector_store()
        if metadata_filter is not None:
            kwargs["filter"] = metadata_filter
        try:
            # Prefer with score when available
            results = store.similarity_search_with_relevance_scores(
                query, k=k, **kwargs
            )
        except Exception:
            filter_kwarg = (
                {"filter": metadata_filter} if metadata_filter is not None else {}
            )
            results = [
                (doc, 1.0)
                for doc in store.similarity_search(query, k=k, **filter_kwarg)
            ]
        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": score,
            }
            for doc, score in results
        ]

    def get_retriever(
        self,
        k: int = 5,
        **kwargs: Any,
    ) -> Any:
        """Return a LangChain retriever for use in RAG chains.

        Args:
            k: Default number of documents to retrieve per query.
            **kwargs: Passed to as_retriever().

        Returns:
            A LangChain retriever (e.g. for use with LCEL or create_retrieval_chain).
        """
        store = self._ensure_vector_store()
        return store.as_retriever(search_kwargs={"k": k, **kwargs})

    def clear(self) -> None:
        """Remove all documents from the collection. Persisted store is cleared on disk on next use."""
        if self._vector_store is not None:
            try:
                self._vector_store.delete_collection()
            except Exception as e:
                logger.warning("Could not delete Chroma collection: {}", e)
            self._vector_store = None
        logger.info("Vector store cleared (collection={})", self._collection_name)
