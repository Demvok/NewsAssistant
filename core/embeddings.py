"""Embeddings module for local embedding model integration.

Provides functions to generate embeddings using the local embedding
model via LM Studio or alternative providers.
"""

import logging
from typing import Optional

from langchain_community.embeddings import OpenAIEmbeddings

logger = logging.getLogger(__name__)

_embeddings_client: Optional[OpenAIEmbeddings] = None


def get_embeddings_client(base_url: str, model_name: str) -> OpenAIEmbeddings:
    """Initialize and return an embeddings client.

    Args:
        base_url: Base URL of the embeddings API endpoint
        model_name: Name of the embedding model to use

    Returns:
        Configured OpenAIEmbeddings client instance
    """
    global _embeddings_client

    if _embeddings_client is not None:
        logger.debug("Returning cached embeddings client")
        return _embeddings_client

    logger.info(
        f"Initializing embeddings client: base_url={base_url}, model={model_name}"
    )

    try:
        _embeddings_client = OpenAIEmbeddings(
            model=model_name,
            openai_api_base=base_url,
            openai_api_key="not-needed",
        )
        logger.info("Embeddings client initialized successfully")
        return _embeddings_client
    except Exception as e:
        logger.error(f"Failed to initialize embeddings client: {str(e)}")
        raise


def embed_text(text: str, client: Optional[OpenAIEmbeddings] = None) -> list[float]:
    """Generate embedding for a given text.

    Args:
        text: Input text to embed
        client: Optional pre-initialized embeddings client

    Returns:
        Embedding vector as list of floats

    Raises:
        ValueError: If text is empty or embedding fails
    """
    if not text or not text.strip():
        raise ValueError("Text cannot be empty")

    if client is None:
        client = _embeddings_client
        if client is None:
            raise RuntimeError(
                "No embeddings client available. Call get_embeddings_client first."
            )

    try:
        embedding = client.embed_query(text)
        logger.debug(f"Generated embedding: text_len={len(text)}, embedding_dim={len(embedding)}")
        return embedding
    except Exception as e:
        logger.error(f"Failed to embed text: {str(e)}")
        raise


def embed_batch(texts: list[str], client: Optional[OpenAIEmbeddings] = None) -> list[list[float]]:
    """Generate embeddings for a batch of texts.

    Args:
        texts: List of input texts
        client: Optional pre-initialized embeddings client

    Returns:
        List of embedding vectors

    Raises:
        ValueError: If texts list is empty
    """
    if not texts:
        raise ValueError("Texts list cannot be empty")

    if client is None:
        client = _embeddings_client
        if client is None:
            raise RuntimeError(
                "No embeddings client available. Call get_embeddings_client first."
            )

    try:
        embeddings = client.embed_documents(texts)
        logger.debug(
            f"Generated batch embeddings: num_texts={len(texts)}, "
            f"embedding_dim={len(embeddings[0]) if embeddings else 0}"
        )
        return embeddings
    except Exception as e:
        logger.error(f"Failed to embed batch: {str(e)}")
        raise


def reset_client() -> None:
    """Reset the global embeddings client (useful for testing)."""
    global _embeddings_client
    _embeddings_client = None
    logger.info("Embeddings client reset")
