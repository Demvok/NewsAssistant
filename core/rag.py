"""RAG (Retrieval-Augmented Generation) module.

Implements the retrieval pipeline: query embedding, ChromaDB retrieval,
context injection, and answer generation with citations.
"""

import logging
from typing import Optional

from core.schemas import RetrievalResult, Chunk
from core.embeddings import embed_text, get_embeddings_client
from ingestion.indexer import get_collection

logger = logging.getLogger(__name__)


def retrieve_context(
    query: str,
    top_k: int = 5,
    embedding_base_url: Optional[str] = None,
    embedding_model: Optional[str] = None,
) -> list[dict]:
    """Retrieve relevant documents (articles) for a query.

    This function embeds the query, searches top chunks, aggregates results by
    document_id, and returns document-level entries. Each returned dict includes
    document_id, filename, source, similarity_score, content (full text when
    available) and a snippet.
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    collection = get_collection()
    if collection is None:
        raise RuntimeError(
            "Collection not initialized. Please build the knowledge base first."
        )

    logger.debug(f"Retrieving documents for query: {query[:100]}...")

    try:
        # Ensure embeddings client available when requested
        if embedding_base_url and embedding_model:
            get_embeddings_client(embedding_base_url, embedding_model)

        # Embed the query
        query_embedding = embed_text(query)

        # Fetch more chunks than final documents to get good coverage per document
        chunk_results_k = max(top_k * 10, 50)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=chunk_results_k,
            include=["documents", "metadatas", "distances"],
        )

        # Collect chunks and aggregate by document_id
        doc_scores: dict[str, dict] = {}
        if results and results.get("documents"):
            docs = results["documents"][0]
            metadatas = results.get("metadatas", [])[0] if results.get("metadatas") else []
            distances = results.get("distances", [])[0] if results.get("distances") else []

            for doc_text, metadata, distance in zip(docs, metadatas, distances):
                document_id = (
                    metadata.get("document_id")
                    or metadata.get("document")
                    or metadata.get("filename")
                    or "unknown"
                )
                # capture article_id if present (SQL ingestion)
                article_id = metadata.get("article_id") or metadata.get("articleId")

                similarity = 1 - (distance if distance is not None else 0)
                start_char = metadata.get("start_char")
                end_char = metadata.get("end_char")
                chunk_preview = metadata.get("chunk_preview") or (doc_text[:1000] if doc_text else "")

                entry = doc_scores.get(document_id)
                chunk_entry = {
                    "text": doc_text,
                    "start_char": start_char,
                    "end_char": end_char,
                    "preview": chunk_preview,
                    "similarity": similarity,
                }

                if entry is None:
                    doc_scores[document_id] = {
                        "document_id": document_id,
                        "article_id": article_id,
                        "filename": metadata.get("filename", ""),
                        "source": metadata.get("source", ""),
                        "best_similarity": similarity,
                        "best_chunk": chunk_entry,
                        "chunks": [chunk_entry],
                    }
                else:
                    # preserve article_id if missing
                    if not entry.get("article_id") and article_id:
                        entry["article_id"] = article_id
                    entry["chunks"].append(chunk_entry)
                    if similarity > entry["best_similarity"]:
                        entry["best_similarity"] = similarity
                        entry["best_chunk"] = chunk_entry

        # Rank documents by best_similarity and return top_k
        ranked = sorted(
            doc_scores.values(), key=lambda d: d["best_similarity"], reverse=True
        )[:top_k]

        # Try to load full text for each document when source path is available
        from ingestion.loader import load_document
        final_results = []
        for d in ranked:
            full_text = None
            try:
                src = d.get("source")
                if src:
                    # load_document expects a filepath; if it fails, ignore and keep snippet
                    doc_obj = load_document(src)
                    full_text = doc_obj.get("content")
            except Exception:
                full_text = None

            result = {
                "document_id": d.get("document_id"),
                "article_id": d.get("article_id"),
                "filename": d.get("filename", ""),
                "source": d.get("source", ""),
                "similarity_score": d.get("best_similarity", 0.0),
                "snippet": d.get("best_chunk", {}).get("preview", ""),
                "content": full_text if full_text is not None else d.get("best_chunk", {}).get("text", ""),
            }
            final_results.append(result)

        top_similarity = final_results[0]["similarity_score"] if final_results else 0.0
        logger.info(
            f"Retrieved {len(final_results)} documents for query. Top similarity: {top_similarity:.3f}"
        )
        return final_results

    except Exception as e:
        logger.error(f"Failed to retrieve context: {str(e)}")
        raise


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
