"""Embeddings module for local embedding model integration.

Provides functions to generate embeddings using the local embedding
model via LM Studio or alternative providers. Includes an HTTP fallback
for environments where the LangChain OpenAIEmbeddings integration fails.
"""

import logging
from typing import Optional, List
import requests

# Try the updated langchain-openai package first (recommended). Fall back to
# community integration if not available. If neither is installed, the module
# will use an HTTP fallback to the OpenAI-compatible /embeddings endpoint.
try:
    from langchain_openai import OpenAIEmbeddings  # type: ignore
except Exception:
    try:
        from langchain_community.embeddings import OpenAIEmbeddings  # type: ignore
    except Exception:
        OpenAIEmbeddings = None  # type: ignore

logger = logging.getLogger(__name__)

_embeddings_client: Optional[object] = None
_embedding_base_url: Optional[str] = None
_embedding_model: Optional[str] = None


def get_embeddings_client(base_url: str, model_name: str) -> Optional[object]:
    """Initialize and return an embeddings client if available.

    If the langchain_openai/OpenAIEmbeddings integration is not installed the
    function configures the HTTP fallback settings and returns None.
    """
    global _embeddings_client, _embedding_base_url, _embedding_model

    if _embeddings_client is not None:
        logger.debug("Returning cached embeddings client")
        return _embeddings_client

    logger.info(f"Initializing embeddings client: base_url={base_url}, model={model_name}")

    _embedding_base_url = base_url
    _embedding_model = model_name

    if OpenAIEmbeddings is None:
        logger.warning("OpenAIEmbeddings integration not found; using HTTP fallback")
        _embeddings_client = None
        return None

    try:
        # The OpenAIEmbeddings constructor signature across packages accepts
        # model and openai_api_base/openai_api_key when using OpenAI-compatible
        # runtimes like LM Studio.
        _embeddings_client = OpenAIEmbeddings(
            model=model_name,
            openai_api_base=base_url,
            openai_api_key="not-needed",
        )
        logger.info("Embeddings client initialized successfully")
        return _embeddings_client
    except Exception as e:
        logger.warning(f"OpenAIEmbeddings init failed: {str(e)} - HTTP fallback will be used")
        _embeddings_client = None
        return None


def _embed_via_http(texts: List[str]) -> List[List[float]]:
    """Fallback: Call LM Studio/OpenAI-compatible embeddings endpoint directly via HTTP.

    Sends a single string when only one input is provided to avoid server
    side validation issues that sometimes occur with single-element lists.
    """
    if not _embedding_base_url or not _embedding_model:
        raise RuntimeError("Embedding base URL or model not configured for HTTP fallback")

    url = _embedding_base_url.rstrip("/") + "/embeddings"
    # Use a string for single-item batches (some endpoints validate types strictly)
    payload_input = texts[0] if len(texts) == 1 else texts
    payload = {"model": _embedding_model, "input": payload_input}
    try:
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        embeddings = [item.get("embedding") for item in data.get("data", [])]
        return embeddings
    except Exception as e:
        logger.error(f"HTTP embeddings request failed: {str(e)}")
        raise


def embed_text(text: str, client: Optional[object] = None) -> list[float]:
    """Generate embedding for a given text.

    Uses the installed OpenAIEmbeddings client when available, otherwise
    falls back to the HTTP endpoint.
    """
    if not text or not text.strip():
        raise ValueError("Text cannot be empty")

    if client is None:
        client = _embeddings_client

    try:
        if client is not None:
            # Try common method names across different integration packages
            if hasattr(client, "embed_query"):
                embedding = client.embed_query(text)
            elif hasattr(client, "embed_documents"):
                embedding = client.embed_documents([text])[0]
            elif hasattr(client, "embed"):
                # Some wrappers accept either a single string or a list
                out = client.embed(text)
                # If embed returns a list-of-lists for a single input, grab first
                if isinstance(out, list) and out and isinstance(out[0], list):
                    embedding = out[0]
                else:
                    embedding = out
            else:
                raise RuntimeError("Embeddings client does not expose a known embed method")
        else:
            embedding = _embed_via_http([text])[0]

        logger.debug(f"Generated embedding: text_len={len(text)}, embedding_dim={len(embedding)}")
        return embedding
    except Exception as e:
        logger.error(f"Failed to embed text: {str(e)}")
        raise


def embed_batch(texts: List[str], client: Optional[object] = None) -> List[List[float]]:
    """Generate embeddings for a batch of texts.

    Tries a client batch method first, then falls back to the HTTP API.
    """
    if not texts:
        raise ValueError("Texts list cannot be empty")

    if client is None:
        client = _embeddings_client

    # Try client method first, fall back to direct HTTP call
    try:
        if client is not None:
            if hasattr(client, "embed_documents"):
                embeddings = client.embed_documents(texts)
            elif hasattr(client, "embed"):
                embeddings = client.embed(texts)
                # Normalize possible single-item responses
                if isinstance(embeddings, list) and embeddings and not isinstance(embeddings[0], list) and len(texts) == 1:
                    embeddings = [embeddings]
            else:
                raise RuntimeError("Embeddings client does not expose a known batch embed method")

            logger.debug(
                f"Generated batch embeddings: num_texts={len(texts)}, embedding_dim={len(embeddings[0]) if embeddings else 0}"
            )
            return embeddings
        else:
            return _embed_via_http(texts)
    except Exception as e:
        logger.warning(f"Client batch embedding failed: {str(e)} — falling back to HTTP")
        return _embed_via_http(texts)


def reset_client() -> None:
    """Reset the global embeddings client (useful for testing)."""
    global _embeddings_client, _embedding_base_url, _embedding_model
    _embeddings_client = None
    _embedding_base_url = None
    _embedding_model = None
    logger.info("Embeddings client reset")
