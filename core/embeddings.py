"""Embeddings module using langchain-openai OpenAIEmbeddings only.

Per project requirements, the system uses a local embedding model (embeddinggemma)
via LM Studio's OpenAI-compatible API. Legacy HTTP fallback support removed to
ensure consistent behavior and surface clear errors when integration is missing.
"""

import logging
from typing import Optional, List

logger = logging.getLogger(__name__)

try:
    from langchain_openai import OpenAIEmbeddings  # type: ignore
except Exception as e:
    OpenAIEmbeddings = None  # type: ignore
    _import_error = e

_embeddings_client: Optional[OpenAIEmbeddings] = None
_embedding_base_url: Optional[str] = None
_embedding_model: Optional[str] = None


def get_embeddings_client(base_url: str, model_name: str) -> OpenAIEmbeddings:
    """Initialize and return an OpenAIEmbeddings client from langchain_openai.

    Raises RuntimeError with actionable instructions if langchain_openai is missing
    or initialization fails.
    """
    global _embeddings_client, _embedding_base_url, _embedding_model

    if OpenAIEmbeddings is None:
        raise RuntimeError(
            "langchain_openai.OpenAIEmbeddings is not available.\n"
            "Install it with: pip install -U langchain-openai"
        )

    if _embeddings_client is not None:
        return _embeddings_client

    _embedding_base_url = base_url
    _embedding_model = model_name

    try:
        _embeddings_client = OpenAIEmbeddings(
            model=model_name,
            openai_api_base=base_url,
            openai_api_key="not-needed",
        )
        logger.info("Embeddings client initialized via langchain_openai")
        return _embeddings_client
    except Exception as e:
        logger.error(f"Failed to initialize OpenAIEmbeddings: {e}")
        raise RuntimeError(
            "Failed to initialize OpenAIEmbeddings. Ensure LM Studio is reachable at the provided base_url "
            "and the embedding model name is correct. Original error: " + str(e)
        )


def embed_text(text: str, client: Optional[OpenAIEmbeddings] = None) -> List[float]:
    """Generate embedding for a given text using OpenAIEmbeddings.

    Requires get_embeddings_client(...) to have been called beforehand (or pass client).
    """
    if not text or not text.strip():
        raise ValueError("Text cannot be empty")

    if client is None:
        client = _embeddings_client

    if client is None:
        raise RuntimeError("Embeddings client not initialized. Call get_embeddings_client(base_url, model_name) first.")

    # Prefer embed_query method for single queries
    if hasattr(client, "embed_query"):
        return client.embed_query(text)

    # Fallback to batch method and return first vector
    if hasattr(client, "embed_documents"):
        return client.embed_documents([text])[0]

    raise RuntimeError("OpenAIEmbeddings client does not expose known embed methods")


def embed_batch(texts: List[str], client: Optional[OpenAIEmbeddings] = None) -> List[List[float]]:
    """Generate embeddings for a batch of texts using OpenAIEmbeddings.

    Requires get_embeddings_client(...) to have been called beforehand (or pass client).
    """
    if not texts:
        raise ValueError("Texts list cannot be empty")

    if client is None:
        client = _embeddings_client

    if client is None:
        raise RuntimeError("Embeddings client not initialized. Call get_embeddings_client(base_url, model_name) first.")

    if not hasattr(client, "embed_documents"):
        raise RuntimeError("OpenAIEmbeddings client does not expose embed_documents")

    return client.embed_documents(texts)


def reset_client() -> None:
    """Reset the global embeddings client (useful for testing)."""
    global _embeddings_client, _embedding_base_url, _embedding_model
    _embeddings_client = None
    _embedding_base_url = None
    _embedding_model = None
    logger.info("Embeddings client reset")
