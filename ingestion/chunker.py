"""Document chunker module.

Splits documents into manageable chunks with configurable size and overlap,
preserving chunk metadata and ordering.
"""

import logging
import hashlib
from typing import Optional
from uuid import uuid4
from config import settings

logger = logging.getLogger(__name__)


def chunk_document(
    content: str,
    document_id: str,
    chunk_size: int,
    overlap: int,
    metadata: Optional[dict] = None,
) -> list[dict]:
    """Split document content into chunks.

    Args:
        content: Document text content
        document_id: ID of the source document
        chunk_size: Target chunk size in characters
        overlap: Character overlap between consecutive chunks
        metadata: Additional metadata for the document

    Returns:
        List of chunk dictionaries with content and metadata

    Raises:
        ValueError: If chunk_size <= 0 or overlap >= chunk_size
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be < chunk_size")

    metadata = metadata or {}
    chunks = []

    if not content or len(content.strip()) == 0:
        logger.warning(f"Empty content for document {document_id}")
        return chunks

    # Calculate stride (non-overlapping distance between chunks)
    stride = chunk_size - overlap

    for i in range(0, len(content), stride):
        chunk_text = content[i : i + chunk_size]

        # Skip very small trailing chunks
        if len(chunk_text.strip()) < 10:
            continue

        chunk_order = len(chunks)
        chunk_id = str(uuid4())

        chunk = {
            "id": chunk_id,
            "content": chunk_text,
            "document_id": document_id,
            "chunk_order": chunk_order,
            "start_char": i,
            "end_char": min(i + chunk_size, len(content)),
            "metadata": {
                **metadata,
                "chunk_size": len(chunk_text),
                "is_last": (i + chunk_size) >= len(content),
            },
        }
        chunks.append(chunk)

    logger.debug(
        f"Chunked document {document_id}: {len(chunks)} chunks, "
        f"chunk_size={chunk_size}, overlap={overlap}"
    )
    return chunks


def chunk_batch(
    documents: list[dict],
    chunk_size: int = None,
    overlap: int = None,
) -> list[dict]:
    """Chunk a batch of documents.

    Args:
        documents: List of document dictionaries with 'id', 'content', and optional 'metadata'
        chunk_size: Target chunk size in characters (defaults to settings.chunk_size)
        overlap: Character overlap between consecutive chunks (defaults to settings.chunk_overlap)

    Returns:
        List of all chunks from all documents with metadata

    Raises:
        ValueError: If documents list is empty or chunk parameters invalid
    """
    if chunk_size is None:
        chunk_size = settings.chunk_size
    if overlap is None:
        overlap = settings.chunk_overlap
    
    if not documents:
        raise ValueError("documents list cannot be empty")

    all_chunks = []

    for doc in documents:
        if not isinstance(doc, dict):
            logger.warning(f"Skipping non-dict document: {type(doc)}")
            continue

        doc_id = doc.get("id")
        content = doc.get("content", "")

        if not doc_id:
            logger.warning("Skipping document without 'id' field")
            continue

        doc_metadata = doc.get("metadata", {})
        # Preserve source file info if available
        if "source" in doc:
            doc_metadata["source"] = doc["source"]
        if "filename" in doc:
            doc_metadata["filename"] = doc["filename"]

        chunks = chunk_document(
            content=content,
            document_id=doc_id,
            chunk_size=chunk_size,
            overlap=overlap,
            metadata=doc_metadata,
        )
        all_chunks.extend(chunks)

    logger.info(
        f"Chunked {len(documents)} documents into {len(all_chunks)} chunks"
    )
    return all_chunks


def validate_chunks(chunks: list[dict]) -> bool:
    """Validate chunk integrity and consistency.

    Args:
        chunks: List of chunks to validate

    Returns:
        True if all chunks are valid

    Raises:
        ValueError: If chunks are invalid
    """
    if not chunks:
        logger.warning("Empty chunks list for validation")
        return True

    required_fields = {"id", "content", "document_id", "chunk_order"}

    for i, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            raise ValueError(f"Chunk {i} is not a dictionary")

        missing_fields = required_fields - set(chunk.keys())
        if missing_fields:
            raise ValueError(
                f"Chunk {i} missing required fields: {missing_fields}"
            )

        if not chunk["id"]:
            raise ValueError(f"Chunk {i} has empty id")
        if not chunk["content"]:
            raise ValueError(f"Chunk {i} has empty content")
        if not chunk["document_id"]:
            raise ValueError(f"Chunk {i} has empty document_id")
        if chunk["chunk_order"] < 0:
            raise ValueError(f"Chunk {i} has invalid chunk_order")

    logger.info(f"Validated {len(chunks)} chunks: all valid")
    return True
