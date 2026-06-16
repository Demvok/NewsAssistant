"""Indexer module for ChromaDB persistence.

Manages embedding generation, vector storage in ChromaDB, and
index rebuild operations.
"""

import logging
from typing import Optional

import chromadb

from core.embeddings import embed_batch, get_embeddings_client

logger = logging.getLogger(__name__)

_chroma_client: Optional[chromadb.Client] = None
_collection = None


def initialize_chroma_db(
    db_path: str,
    embedding_base_url: str,
    embedding_model: str,
) -> None:
    """Initialize ChromaDB instance.

    Args:
        db_path: Path to ChromaDB storage directory
        embedding_base_url: Base URL for the embedding API
        embedding_model: Name of the embedding model to use

    Raises:
        Exception: If initialization fails
    """
    global _chroma_client, _collection

    logger.info(f"Initializing ChromaDB: db_path={db_path}")

    try:
        # Initialize embeddings client
        get_embeddings_client(embedding_base_url, embedding_model)

        # Initialize ChromaDB client with persistent storage (new API)
        _chroma_client = chromadb.PersistentClient(path=db_path)

        # Get or create collection
        _collection = _chroma_client.get_or_create_collection(
            name="news_corpus",
            metadata={"hnsw:space": "cosine", "dimension": 384},
        )

        logger.info(
            f"ChromaDB initialized successfully: "
            f"db_path={db_path}, collection=news_corpus"
        )
    except Exception as e:
        logger.error(f"Failed to initialize ChromaDB: {str(e)}")
        raise


def index_chunks(
    chunks: list[dict],
    embedding_base_url: str,
    embedding_model: str,
    db_path: str,
) -> int:
    """Index chunks into ChromaDB.

    Args:
        chunks: List of chunks with content and metadata
        embedding_base_url: Base URL for the embedding API
        embedding_model: Name of the embedding model to use
        db_path: Path to ChromaDB storage directory

    Returns:
        Number of chunks indexed

    Raises:
        ValueError: If chunks is empty or invalid
        Exception: If indexing fails
    """
    if not chunks:
        raise ValueError("Chunks list cannot be empty")

    # Ensure ChromaDB is initialized
    if _chroma_client is None or _collection is None:
        initialize_chroma_db(db_path, embedding_base_url, embedding_model)

    logger.info(f"Indexing {len(chunks)} chunks into ChromaDB")

    try:
        # Extract texts for embedding
        texts_raw = [chunk.get("content", "") for chunk in chunks]
        chunk_ids = [chunk.get("id") for chunk in chunks]

        # Prepare metadata (include chunk positions and preview for better UI snippets)
        # NOTE: ChromaDB only stores string metadata, so convert all values to strings
        metadatas = []
        for chunk in chunks:
            metadata = {
                "document_id": str(chunk["document_id"]),
                "chunk_order": str(chunk["chunk_order"]),
                "source": str(chunk.get("metadata", {}).get("source", "")),
                "filename": str(chunk.get("metadata", {}).get("filename", "")),
                "article_id": str(chunk.get("metadata", {}).get("article_id", "")),
                "chunk_id": str(chunk.get("id", "")),
                "start_char": str(chunk.get("start_char", 0)),
                "end_char": str(chunk.get("end_char", 0)),
                "chunk_preview": str(chunk.get("content", "")[:500] if chunk.get("content") else ""),
            }
            metadatas.append(metadata)

        # Sanitize texts and align with metadata/ids
        valid_texts = []
        valid_ids = []
        valid_metadatas = []
        for cid, text, md in zip(chunk_ids, texts_raw, metadatas):
            if text is None:
                logger.warning(f"Skipping chunk {cid}: content is None")
                continue
            if not isinstance(text, str):
                text = str(text)
            text = text.strip()
            if not text:
                logger.warning(f"Skipping chunk {cid}: empty after strip")
                continue
            valid_texts.append(text)
            valid_ids.append(cid)
            valid_metadatas.append(md)

        if not valid_texts:
            raise ValueError("No valid texts to embed")

        # Debug: log types and sample of texts before embedding
        try:
            sample_types = [type(t).__name__ for t in valid_texts[:5]]
            logger.info(f"Embedding input types: {sample_types}")
            logger.info(f"Embedding sample text (truncated): {valid_texts[0][:200]!r}")
            # Also print to stdout to ensure visibility in CLI
            print("[DEBUG] Embedding input types:", sample_types)
            print("[DEBUG] Embedding sample (truncated):", repr(valid_texts[0][:200]))
        except Exception:
            logger.info("Failed to log embedding sample")

        # Debug: log sample metadata being stored
        try:
            print(f"[DEBUG] First metadata sample: {valid_metadatas[0]}")
            print(f"[DEBUG] Metadata keys: {list(valid_metadatas[0].keys())}")
        except Exception:
            pass

        # Generate embeddings for sanitized texts
        embeddings = embed_batch(valid_texts)

        # Add to collection
        _collection.add(
            ids=valid_ids,
            embeddings=embeddings,
            documents=valid_texts,
            metadatas=valid_metadatas,
        )

        logger.info(f"Indexed {len(chunks)} chunks successfully")
        return len(chunks)

    except Exception as e:
        logger.error(f"Failed to index chunks: {str(e)}")
        raise


def rebuild_index(
    documents: list[dict],
    chunk_size: int,
    chunk_overlap: int,
    embedding_base_url: str,
    embedding_model: str,
    db_path: str,
) -> dict:
    """Rebuild the entire index from documents.

    Args:
        documents: List of documents to re-index
        chunk_size: Chunk size for splitting
        chunk_overlap: Chunk overlap for splitting
        embedding_base_url: Base URL for the embedding API
        embedding_model: Name of the embedding model to use
        db_path: Path to ChromaDB storage directory

    Returns:
        Dictionary with indexing statistics

    Raises:
        ValueError: If documents is empty
    """
    if not documents:
        raise ValueError("Documents list cannot be empty")

    logger.info(f"Rebuilding index from {len(documents)} documents")

    # Clear existing collection
    delete_collection()

    # Initialize fresh
    initialize_chroma_db(db_path, embedding_base_url, embedding_model)

    # Import here to avoid circular imports
    from ingestion.chunker import chunk_batch

    # Chunk documents
    chunks = chunk_batch(documents, chunk_size, chunk_overlap)
    logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")

    # Index chunks
    indexed_count = index_chunks(chunks, embedding_base_url, embedding_model, db_path)

    stats = {
        "num_documents": len(documents),
        "num_chunks": indexed_count,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
    }

    logger.info(f"Index rebuilt: {stats}")
    return stats


def get_collection_stats() -> dict:
    """Get statistics about the indexed collection.

    Returns:
        Collection metadata and statistics

    Raises:
        RuntimeError: If collection is not initialized
    """
    if _collection is None:
        raise RuntimeError("Collection not initialized. Call initialize_chroma_db first.")

    try:
        count = _collection.count()
        stats = {
            "collection_name": _collection.name,
            "total_chunks": count,
            "metadata": _collection.metadata if hasattr(_collection, "metadata") else {},
        }
        logger.debug(f"Collection stats: {stats}")
        return stats
    except Exception as e:
        logger.error(f"Failed to get collection stats: {str(e)}")
        raise


def delete_collection() -> None:
    """Delete and clear the current collection."""
    global _collection, _chroma_client

    if _chroma_client is None:
        logger.warning("ChromaDB client not initialized")
        return

    try:
        if _collection is not None:
            _chroma_client.delete_collection(name="news_corpus")
            _collection = None
            logger.info("Collection deleted successfully")
    except Exception as e:
        logger.warning(f"Failed to delete collection: {str(e)}")
        _collection = None


def get_chroma_client() -> Optional[chromadb.Client]:
    """Get the ChromaDB client instance.

    Returns:
        ChromaDB client or None if not initialized
    """
    return _chroma_client


def get_collection() -> Optional[chromadb.Collection]:
    """Get the ChromaDB collection instance.

    Returns:
        ChromaDB collection or None if not initialized
    """
    return _collection
