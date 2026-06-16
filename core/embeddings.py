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

# Optional tokenizer support for detokenization
_tokenizer = None
_tokenizer_name: Optional[str] = None

_embeddings_client: Optional[OpenAIEmbeddings] = None
_embedding_base_url: Optional[str] = None
_embedding_model: Optional[str] = None

# Detokenization helper using optional HuggingFace tokenizer
try:
    from transformers import AutoTokenizer
except Exception:
    AutoTokenizer = None

_tokenizer_cache = {}


def _try_detokenize(token_ids: list, tokenizer_name: Optional[str] = None) -> Optional[str]:
    """Attempt to detokenize a list of token IDs using a HuggingFace tokenizer.

    Returns decoded string on success, or None if tokenizer is unavailable or decoding fails.
    """
    global _tokenizer_cache
    # Prefer explicit tokenizer name, otherwise try to use embedding model name
    name = tokenizer_name
    if not name:
        try:
            from config import settings

            name = settings.embedding_model
        except Exception:
            name = None

    if AutoTokenizer is None or not name:
        return None

    # Cache tokenizer instances
    if name not in _tokenizer_cache:
        try:
            _tokenizer_cache[name] = AutoTokenizer.from_pretrained(name, use_fast=False)
        except Exception:
            try:
                # Fallback to a generic tokenizer (gpt2) if specific one is not available
                _tokenizer_cache[name] = AutoTokenizer.from_pretrained("gpt2", use_fast=False)
            except Exception:
                _tokenizer_cache[name] = None

    tokenizer = _tokenizer_cache.get(name)
    if tokenizer is None:
        return None

    try:
        # Ensure list of ints
        ids = [int(x) for x in token_ids]
        return tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    except Exception:
        return None


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
        # Monkeypatch the underlying OpenAI client create/acreate methods to ensure
        # the 'input' field is always a string or list of strings. Some OpenAI
        # client variants or model mappings may inadvertently send token ids
        # (lists of ints) which LM Studio rejects. Coerce non-string inputs to
        # their string representations to avoid 400 errors.
        try:
            client = getattr(_embeddings_client, "client", None)
            if client is not None and hasattr(client, "create"):
                orig_create = client.create

                def _coerce_item(item):
                    # If list/tuple of ints -> detokenize or join numbers
                    if isinstance(item, (list, tuple)):
                        if all(isinstance(x, int) for x in item):
                            detok = _try_detokenize(item, _embedding_model)
                            if detok is not None:
                                return detok
                            return " ".join(str(x) for x in item)
                        # For other lists, stringify elements
                        return [str(x) for x in item]
                    if isinstance(item, bytes):
                        return item.decode("utf-8", errors="ignore")
                    if isinstance(item, str):
                        return item
                    return str(item)

                def _coerce_input_and_create(*args, **kwargs):
                    if "input" in kwargs:
                        inp = kwargs["input"]
                        if isinstance(inp, list):
                            coerced = []
                            for item in inp:
                                coerced.append(_coerce_item(item))
                            kwargs["input"] = coerced
                    return orig_create(*args, **kwargs)

                client.create = _coerce_input_and_create

            # Async variant if present
            if client is not None and hasattr(client, "acreate"):
                orig_acreate = client.acreate

                async def _coerce_input_and_acreate(*args, **kwargs):
                    if "input" in kwargs:
                        inp = kwargs["input"]
                        if isinstance(inp, list):
                            coerced = []
                            for item in inp:
                                coerced.append(_coerce_item(item))
                            kwargs["input"] = coerced
                    return await orig_acreate(*args, **kwargs)

                client.acreate = _coerce_input_and_acreate
        except Exception:
            logger.info("Could not patch underlying client.create; continuing without coercion")

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

    Uses embed_documents(...) consistently to avoid differences between
    embed_query implementations across packages.
    """
    if not text or not text.strip():
        raise ValueError("Text cannot be empty")

    if client is None:
        client = _embeddings_client

    if client is None:
        raise RuntimeError("Embeddings client not initialized. Call get_embeddings_client(base_url, model_name) first.")

    # Normalize and detect token-id lists encoded as strings
    import json

    # If original input was a list/tuple of ints, convert to space-separated
    if isinstance(text, (list, tuple)) and all(isinstance(x, int) for x in text):
        coerced = " ".join(str(x) for x in text)
    else:
        t_str = str(text)
        if t_str.strip().startswith("[") and t_str.strip().endswith("]"):
            try:
                parsed = json.loads(t_str)
                if isinstance(parsed, (list, tuple)) and all(isinstance(x, int) for x in parsed):
                    coerced = " ".join(str(x) for x in parsed)
                else:
                    coerced = t_str
            except Exception:
                coerced = t_str
        else:
            coerced = t_str

    try:
        # Use batch method for consistent payload shape
        embeddings = client.embed_documents([coerced])
        if not embeddings or not isinstance(embeddings, list):
            raise RuntimeError("Invalid embedding response from OpenAIEmbeddings")
        return embeddings[0]

    except Exception as e:
        logger.error(
            "Failed to embed text: %s",
            {
                "error": str(e),
                "input_type": type(text).__name__,
                "input_preview": repr(coerced)[:200],
            },
        )
        raise


def embed_batch(texts: List[str], client: Optional[OpenAIEmbeddings] = None) -> List[List[float]]:
    """Generate embeddings for a batch of texts using OpenAIEmbeddings.

    Uses embed_documents(...) and coerces inputs to plain strings.
    """
    if not texts:
        raise ValueError("Texts list cannot be empty")

    if client is None:
        client = _embeddings_client

    if client is None:
        raise RuntimeError("Embeddings client not initialized. Call get_embeddings_client(base_url, model_name) first.")

    # Coerce to plain strings and handle token-id lists specially
    import json

    coerced_texts = []
    coerced_info = []
    import json
    from config import settings

    for t in texts:
        # If already a string, try to detect JSON-encoded list
        if isinstance(t, str):
            t_str = t
            # Detect JSON list encoding like "[1,2,3]"
            if t_str.strip().startswith("[") and t_str.strip().endswith("]"):
                try:
                    parsed = json.loads(t_str)
                    if isinstance(parsed, (list, tuple)) and all(isinstance(x, int) for x in parsed):
                        # Attempt detokenization using tokenizer if available
                        detok = _try_detokenize(parsed, settings.embedding_model)
                        if detok is not None:
                            coerced_texts.append(detok)
                            coerced_info.append("detokenized_parsed_list")
                            continue
                        # Fallback: Convert token id list to spaced numbers
                        coerced = " ".join(str(x) for x in parsed)
                        coerced_texts.append(coerced)
                        coerced_info.append("parsed_list")
                        continue
                except Exception:
                    pass
            coerced_texts.append(t_str)
            coerced_info.append("str")
            continue

        # If it's a list/tuple of ints, try detokenization
        if isinstance(t, (list, tuple)) and all(isinstance(x, int) for x in t):
            detok = _try_detokenize(t, settings.embedding_model)
            if detok is not None:
                coerced_texts.append(detok)
                coerced_info.append("detokenized_list_of_ints")
                continue
            coerced = " ".join(str(x) for x in t)
            coerced_texts.append(coerced)
            coerced_info.append("list_of_ints")
            continue

        # Fallback: coerce to string
        coerced_texts.append(str(t))
        coerced_info.append(type(t).__name__)

    try:
        logger.debug(f"Embedding batch input types: {coerced_info[:5]}")
        embeddings = client.embed_documents(coerced_texts)
        if not embeddings or not isinstance(embeddings, list):
            raise RuntimeError("Invalid batch embedding response from OpenAIEmbeddings")
        return embeddings

    except Exception as e:
        sample_types = [type(t).__name__ for t in coerced_texts[:5]]
        logger.error(
            "Failed to embed batch: %s",
            {
                "error": str(e),
                "num_texts": len(coerced_texts),
                "sample_types": sample_types,
                "sample_preview": repr(coerced_texts[0])[:200] if coerced_texts else "",
            },
        )
        raise


def reset_client() -> None:
    """Reset the global embeddings client (useful for testing)."""
    global _embeddings_client, _embedding_base_url, _embedding_model
    _embeddings_client = None
    _embedding_base_url = None
    _embedding_model = None
    logger.info("Embeddings client reset")
