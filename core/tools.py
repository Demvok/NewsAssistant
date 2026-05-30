"""Tool calling and function registry module.

Implements a registry of callable tools that the assistant can use to
interact with the corpus and perform specific analytical tasks.
"""

import logging
from typing import Callable, Any, Optional
from datetime import datetime

from ingestion.indexer import get_collection

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Registry for callable tools."""

    def __init__(self):
        """Initialize the tool registry."""
        self.tools: dict[str, dict[str, Any]] = {}

    def register(self, name: str, func: Callable, description: str):
        """Register a new tool.
        
        Args:
            name: Tool name
            func: Callable function
            description: Tool description and usage
        """
        self.tools[name] = {
            "func": func,
            "description": description,
            "name": name,
        }
        logger.debug(f"Registered tool: {name}")

    def get_tool(self, name: str) -> Optional[Callable]:
        """Retrieve a tool by name.
        
        Args:
            name: Tool name
            
        Returns:
            Callable tool function or None if not found
        """
        if name not in self.tools:
            logger.warning(f"Tool not found: {name}")
            return None
        return self.tools[name]["func"]

    def list_tools(self) -> list[dict]:
        """List all available tools with descriptions.
        
        Returns:
            List of tool metadata
        """
        return [
            {"name": name, "description": tool["description"]}
            for name, tool in self.tools.items()
        ]


def search_by_topic(topic: str) -> dict:
    """Search documents by topic using semantic similarity.
    
    Performs a semantic search on the corpus for the given topic.
    
    Args:
        topic: Topic name or description to search for
        
    Returns:
        Dictionary with search results including matching documents and metadata
        
    Raises:
        ValueError: If topic is empty
        RuntimeError: If collection is not initialized
    """
    if not topic or not topic.strip():
        raise ValueError("Topic cannot be empty")
    
    collection = get_collection()
    if collection is None:
        raise RuntimeError(
            "Collection not initialized. Please build the knowledge base first."
        )
    
    logger.info(f"Searching for topic: {topic}")
    
    try:
        from core.embeddings import embed_text
        
        # Generate embedding for the topic query
        topic_embedding = embed_text(topic)
        
        # Query collection with semantic search
        results = collection.query(
            query_embeddings=[topic_embedding],
            n_results=10,
            include=["documents", "metadatas", "distances"],
        )
        
        # Format results
        matched_documents = []
        if results and results["documents"] and len(results["documents"]) > 0:
            docs = results["documents"][0]
            metadatas = results["metadatas"][0] if results["metadatas"] else []
            distances = results["distances"][0] if results["distances"] else []
            
            for i, (doc_text, metadata, distance) in enumerate(
                zip(docs, metadatas, distances)
            ):
                similarity_score = 1 - distance
                matched_documents.append({
                    "rank": i + 1,
                    "content": doc_text,
                    "similarity": round(similarity_score, 4),
                    "source": metadata.get("source", ""),
                    "filename": metadata.get("filename", ""),
                    "document_id": metadata.get("document_id", ""),
                })
        
        logger.info(f"Found {len(matched_documents)} documents for topic: {topic}")
        
        return {
            "topic": topic,
            "num_results": len(matched_documents),
            "results": matched_documents,
            "metadata": {
                "search_type": "semantic",
                "timestamp": datetime.now().isoformat(),
            },
        }
    
    except Exception as e:
        logger.error(f"Failed to search by topic: {str(e)}")
        raise


def filter_by_date(start_date: str, end_date: str) -> dict:
    """Filter documents by date range.
    
    Retrieves documents published within the specified date range.
    Note: Requires document metadata to include 'date' or 'published_date' field.
    
    Args:
        start_date: Start date (ISO format: YYYY-MM-DD)
        end_date: End date (ISO format: YYYY-MM-DD)
        
    Returns:
        Dictionary with filtered documents and metadata
        
    Raises:
        ValueError: If dates are invalid or start_date > end_date
        RuntimeError: If collection is not initialized
    """
    try:
        start = datetime.fromisoformat(start_date).date()
        end = datetime.fromisoformat(end_date).date()
    except ValueError as e:
        raise ValueError(f"Invalid date format. Expected ISO format (YYYY-MM-DD): {str(e)}")
    
    if start > end:
        raise ValueError("start_date cannot be after end_date")
    
    collection = get_collection()
    if collection is None:
        raise RuntimeError(
            "Collection not initialized. Please build the knowledge base first."
        )
    
    logger.info(f"Filtering documents by date range: {start_date} to {end_date}")
    
    try:
        # Get all documents from collection
        results = collection.get(
            include=["documents", "metadatas"]
        )
        
        filtered_documents = []
        if results and results["documents"]:
            docs = results["documents"]
            metadatas = results["metadatas"] if results["metadatas"] else []
            
            for doc_text, metadata in zip(docs, metadatas):
                # Check for date in metadata
                doc_date_str = metadata.get("date") or metadata.get("published_date")
                if not doc_date_str:
                    continue
                
                try:
                    doc_date = datetime.fromisoformat(doc_date_str).date()
                    if start <= doc_date <= end:
                        filtered_documents.append({
                            "content": doc_text,
                            "source": metadata.get("source", ""),
                            "filename": metadata.get("filename", ""),
                            "document_id": metadata.get("document_id", ""),
                            "date": str(doc_date),
                        })
                except (ValueError, TypeError):
                    pass
        
        logger.info(
            f"Filtered {len(filtered_documents)} documents in date range: "
            f"{start_date} to {end_date}"
        )
        
        return {
            "start_date": start_date,
            "end_date": end_date,
            "num_results": len(filtered_documents),
            "results": filtered_documents,
            "metadata": {
                "filter_type": "date_range",
                "timestamp": datetime.now().isoformat(),
            },
        }
    
    except Exception as e:
        logger.error(f"Failed to filter by date: {str(e)}")
        raise


def count_keyword_mentions(keyword: str) -> dict:
    """Count keyword mentions in corpus.
    
    Searches the corpus for occurrences of a keyword and returns
    count statistics and document distribution.
    
    Args:
        keyword: Keyword to search for (case-insensitive)
        
    Returns:
        Dictionary with count, distribution, and matched chunks
        
    Raises:
        ValueError: If keyword is empty
        RuntimeError: If collection is not initialized
    """
    if not keyword or not keyword.strip():
        raise ValueError("Keyword cannot be empty")
    
    collection = get_collection()
    if collection is None:
        raise RuntimeError(
            "Collection not initialized. Please build the knowledge base first."
        )
    
    keyword_lower = keyword.lower()
    logger.info(f"Counting mentions of keyword: {keyword}")
    
    try:
        # Get all documents from collection
        results = collection.get(
            include=["documents", "metadatas"]
        )
        
        total_mentions = 0
        document_distribution = {}
        matched_chunks = []
        
        if results and results["documents"]:
            docs = results["documents"]
            metadatas = results["metadatas"] if results["metadatas"] else []
            
            for i, (doc_text, metadata) in enumerate(zip(docs, metadatas)):
                # Count occurrences
                mentions = doc_text.lower().count(keyword_lower)
                
                if mentions > 0:
                    total_mentions += mentions
                    doc_id = metadata.get("document_id", "")
                    filename = metadata.get("filename", "")
                    
                    if doc_id not in document_distribution:
                        document_distribution[doc_id] = {
                            "filename": filename,
                            "mention_count": 0,
                            "chunk_count": 0,
                        }
                    
                    document_distribution[doc_id]["mention_count"] += mentions
                    document_distribution[doc_id]["chunk_count"] += 1
                    
                    matched_chunks.append({
                        "content_preview": doc_text[:200] + "..." if len(doc_text) > 200 else doc_text,
                        "mention_count": mentions,
                        "source": metadata.get("source", ""),
                        "filename": filename,
                        "document_id": doc_id,
                    })
        
        # Sort by mention count
        matched_chunks = sorted(
            matched_chunks,
            key=lambda x: x["mention_count"],
            reverse=True
        )[:20]  # Top 20
        
        logger.info(
            f"Found {total_mentions} mentions of keyword '{keyword}' "
            f"across {len(document_distribution)} documents"
        )
        
        return {
            "keyword": keyword,
            "total_mentions": total_mentions,
            "num_documents": len(document_distribution),
            "document_distribution": document_distribution,
            "top_chunks": matched_chunks,
            "metadata": {
                "search_type": "keyword",
                "timestamp": datetime.now().isoformat(),
            },
        }
    
    except Exception as e:
        logger.error(f"Failed to count keyword mentions: {str(e)}")
        raise


def corpus_statistics() -> dict:
    """Get comprehensive corpus statistics.
    
    Returns statistics about the indexed corpus including
    total documents, chunks, size metrics, and topic distribution.
    
    Returns:
        Dictionary with detailed corpus statistics
        
    Raises:
        RuntimeError: If collection is not initialized
    """
    collection = get_collection()
    if collection is None:
        raise RuntimeError(
            "Collection not initialized. Please build the knowledge base first."
        )
    
    logger.info("Generating corpus statistics")
    
    try:
        # Get collection count
        total_chunks = collection.count()
        
        # Get all documents to analyze
        results = collection.get(
            include=["documents", "metadatas"]
        )
        
        # Analyze statistics
        unique_documents = set()
        total_content_length = 0
        chunk_lengths = []
        sources = {}
        
        if results and results["documents"]:
            docs = results["documents"]
            metadatas = results["metadatas"] if results["metadatas"] else []
            
            for doc_text, metadata in zip(docs, metadatas):
                doc_id = metadata.get("document_id", "")
                unique_documents.add(doc_id)
                
                total_content_length += len(doc_text)
                chunk_lengths.append(len(doc_text))
                
                source = metadata.get("source", "unknown")
                if source not in sources:
                    sources[source] = 0
                sources[source] += 1
        
        # Calculate averages
        num_documents = len(unique_documents)
        avg_chunk_length = (
            sum(chunk_lengths) / len(chunk_lengths) if chunk_lengths else 0
        )
        min_chunk_length = min(chunk_lengths) if chunk_lengths else 0
        max_chunk_length = max(chunk_lengths) if chunk_lengths else 0
        
        logger.info(
            f"Corpus statistics: documents={num_documents}, "
            f"chunks={total_chunks}, "
            f"total_content={total_content_length} chars"
        )
        
        return {
            "num_documents": num_documents,
            "num_chunks": total_chunks,
            "total_content_length": total_content_length,
            "avg_chunk_length": round(avg_chunk_length, 2),
            "min_chunk_length": min_chunk_length,
            "max_chunk_length": max_chunk_length,
            "source_distribution": sources,
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "collection_name": collection.name,
            },
        }
    
    except Exception as e:
        logger.error(f"Failed to generate corpus statistics: {str(e)}")
        raise


def create_default_tools() -> ToolRegistry:
    """Create and populate the default tool registry.
    
    Default tools include:
    - search_by_topic: Search documents by topic using semantic similarity
    - filter_by_date: Filter documents by date range
    - count_keyword_mentions: Count keyword mentions in corpus
    - corpus_statistics: Return corpus metadata and statistics
    
    Returns:
        Populated ToolRegistry instance
    """
    registry = ToolRegistry()
    
    registry.register(
        name="search_by_topic",
        func=search_by_topic,
        description=(
            "Search documents by topic using semantic similarity. "
            "Returns top 10 most relevant documents for the given topic."
        ),
    )
    
    registry.register(
        name="filter_by_date",
        func=filter_by_date,
        description=(
            "Filter documents by date range. "
            "Requires ISO format dates (YYYY-MM-DD)."
        ),
    )
    
    registry.register(
        name="count_keyword_mentions",
        func=count_keyword_mentions,
        description=(
            "Count keyword mentions in corpus. "
            "Returns total mentions, document distribution, and top chunks."
        ),
    )
    
    registry.register(
        name="corpus_statistics",
        func=corpus_statistics,
        description=(
            "Get comprehensive corpus statistics including document count, "
            "chunk count, size metrics, and source distribution."
        ),
    )
    
    logger.info("Default tool registry created with 4 tools")
    return registry
