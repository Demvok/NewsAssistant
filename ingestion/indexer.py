"""Indexer module for ChromaDB persistence.

Manages embedding generation, vector storage in ChromaDB, and
index rebuild operations.
"""

import logging
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

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

        # Initialize ChromaDB client with persistent storage
        settings = ChromaSettings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=db_path,
            anonymized_telemetry=False,
        )
        _chroma_client = chromadb.Client(settings)

        # Get or create collection
        _collection = _chroma_client.get_or_create_collection(
            name="news_corpus",
            metadata={"hnsw:space": "cosine"},
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
        texts = [chunk["content"] for chunk in chunks]
        chunk_ids = [chunk["id"] for chunk in chunks]

        # Generate embeddings
        embeddings = embed_batch(texts)

        # Prepare metadata
        metadatas = []
        for chunk in chunks:
            metadata = {
                "document_id": chunk["document_id"],
                "chunk_order": chunk["chunk_order"],
                "source": chunk["metadata"].get("source", ""),
                "filename": chunk["metadata"].get("filename", ""),
            }
            metadatas.append(metadata)

        # Add to collection
        _collection.add(
            ids=chunk_ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
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
