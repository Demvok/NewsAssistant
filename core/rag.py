"""RAG (Retrieval-Augmented Generation) module.

Implements the retrieval pipeline: query embedding, ChromaDB retrieval,
context injection, and answer generation with citations.
"""

import logging
from typing import Optional

from core.schemas import RetrievalResult, Chunk
from core.tools import search_articles
from ingestion.indexer import get_collection

logger = logging.getLogger(__name__)


def retrieve_context(
    query: str,
    top_k: int = 5,
    embedding_base_url: Optional[str] = None,
    embedding_model: Optional[str] = None,
) -> list[dict]:
    """Backward-compatible wrapper for article search."""
    return search_articles(
        query=query,
        top_k=top_k,
        embedding_base_url=embedding_base_url,
        embedding_model=embedding_model,
    )


def inject_context_into_prompt(query: str, context: list[dict]) -> str:
    """Assemble a prompt with retrieved context.

    Args:
        query: Original user query
        context: Retrieved document chunks

    Returns:
        Formatted prompt with context injected
    """
    if not context:
        logger.warning("No context provided for prompt injection")
        return f"Question: {query}\n\nNo relevant context available."

    # Format retrieved context
    context_text = "---\nRETRIEVED CONTEXT:\n"
    for i, result in enumerate(context, 1):
        content = result.get("content", "")
        source = result.get("source", "")
        similarity = result.get("similarity_score", 0)

        context_text += f"\n[Source {i}] (Relevance: {similarity:.2%})\n"
        if source:
            context_text += f"Source: {source}\n"
        context_text += f"Content: {content}\n"

    context_text += "---\n\n"

    # Assemble final prompt
    prompt = (
        f"{context_text}"
        f"Question: {query}\n\n"
        f"Please answer the question based on the provided context. "
        f"If the context does not contain relevant information, "
        f"you may provide general knowledge but indicate that it is not from the corpus.\n\n"
        f"Answer:"
    )

    logger.debug(f"Injected context into prompt: context_chunks={len(context)}")
    return prompt


def format_answer_with_sources(answer: str, context: list[dict]) -> dict:
    """Format the final answer with source citations.

    Args:
        answer: Generated answer text
        context: Retrieved context used for generation

    Returns:
        Dictionary with formatted answer and sources
    """
    sources = []
    for result in context:
        source_info = {
            "filename": result.get("filename", ""),
            "source": result.get("source", ""),
            "similarity": result.get("similarity_score", 0),
        }
        sources.append(source_info)

    formatted_response = {
        "answer": answer,
        "sources": sources,
        "num_sources": len(sources),
        "metadata": {
            "retrieval_enabled": True,
            "context_chunks": len(context),
        },
    }

    return formatted_response


def generate_with_rag(
    query: str,
    llm_client,
    rag_enabled: bool = True,
    top_k: int = 5,
    embedding_base_url: Optional[str] = None,
    embedding_model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 512,
) -> dict:
    """Generate an answer using RAG pipeline.

    Args:
        query: User query
        llm_client: LLM client instance for generation
        rag_enabled: Whether to use retrieval
        top_k: Number of retrieved chunks if RAG is enabled
        embedding_base_url: Optional base URL for embeddings
        embedding_model: Optional embedding model name
        temperature: Generation temperature
        max_tokens: Maximum tokens to generate

    Returns:
        Dictionary with answer, context, and metadata

    Raises:
        ValueError: If query is empty or llm_client is invalid
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    if llm_client is None:
        raise ValueError("LLM client is required")

    logger.info(f"Generating answer with RAG: enabled={rag_enabled}, top_k={top_k}")

    context = []
    final_prompt = query

    # Retrieve context if RAG is enabled
    if rag_enabled:
        try:
            context = retrieve_context(
                query,
                top_k=top_k,
                embedding_base_url=embedding_base_url,
                embedding_model=embedding_model,
            )
            final_prompt = inject_context_into_prompt(query, context)
        except Exception as e:
            logger.warning(f"RAG retrieval failed, using query only: {str(e)}")
            context = []
            final_prompt = query

    # Generate answer using LLM
    try:
        from core.llm_client import LMStudioClient

        if isinstance(llm_client, LMStudioClient):
            response = llm_client.complete(
                prompt=final_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            answer = response.text
        else:
            # LangChain LLM
            answer = llm_client.invoke(final_prompt)
            if hasattr(answer, "content"):
                answer = answer.content

        logger.info(f"Generated answer: {len(answer)} characters")

        # Format final response
        result = {
            "answer": answer,
            "query": query,
            "context": context,
            "num_sources": len(context),
            "metadata": {
                "rag_enabled": rag_enabled,
                "top_k": top_k,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "prompt_length": len(final_prompt),
                "answer_length": len(answer),
            },
        }

        return result

    except Exception as e:
        logger.error(f"Failed to generate answer: {str(e)}")
        raise
