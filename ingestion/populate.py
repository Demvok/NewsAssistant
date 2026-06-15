"""Automated knowledge base population from SQL database."""

import logging
from typing import Dict, Optional
from datetime import datetime

from ingestion.sql_loader import SQLArticleLoader, convert_sql_articles_to_documents
from ingestion.chunker import chunk_batch
from ingestion.indexer import (
    initialize_chroma_db,
    index_chunks,
    delete_collection,
)

logger = logging.getLogger(__name__)


def populate_knowledge_base_from_sql(
    database_url: str,
    embedding_base_url: str,
    embedding_model: str,
    chroma_db_path: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    clear_existing: bool = False,
) -> Dict:
    """Populate ChromaDB knowledge base from SQL database articles.

    This function:
    1. Connects to the SQL database
    2. Loads all articles (or filtered subset)
    3. Chunks them with specified overlap
    4. Generates embeddings
    5. Indexes into ChromaDB

    Args:
        database_url: SQL database connection string
        embedding_base_url: Base URL for embedding model
        embedding_model: Name of embedding model
        chroma_db_path: Path to ChromaDB persistence directory
        chunk_size: Size of each chunk in characters
        chunk_overlap: Overlap between consecutive chunks
        clear_existing: If True, delete existing index before repopulating

    Returns:
        Dictionary with statistics:
        - num_articles: Total articles loaded
        - num_documents: Total documents created
        - num_chunks: Total chunks created
        - num_indexed: Chunks successfully indexed
        - topics_count: Number of unique topics
        - errors: List of error messages if any

    Raises:
        ValueError: If database_url is invalid
        Exception: If database connection or indexing fails
    """
    stats = {
        "num_articles": 0,
        "num_documents": 0,
        "num_chunks": 0,
        "num_indexed": 0,
        "topics_count": 0,
        "errors": [],
    }

    try:
        logger.info("Starting knowledge base population from SQL...")

        # Initialize SQL loader
        logger.info("Connecting to SQL database...")
        loader = SQLArticleLoader(database_url)

        # Get topics
        topics_map = loader.get_topics()
        stats["topics_count"] = len(topics_map)
        logger.info(f"Found {len(topics_map)} topics")

        # Load all articles
        logger.info("Loading articles from database...")
        sql_articles = loader.load_all_articles()
        stats["num_articles"] = len(sql_articles)

        if not sql_articles:
            logger.warning("No articles found in database")
            return stats

        # Convert to document format
        logger.info("Converting articles to document format...")
        documents = convert_sql_articles_to_documents(sql_articles, topics_map)
        stats["num_documents"] = len(documents)

        # Clear existing index if requested
        if clear_existing:
            logger.info("Clearing existing index...")
            delete_collection()

        # Initialize ChromaDB
        logger.info("Initializing ChromaDB...")
        initialize_chroma_db(chroma_db_path, embedding_base_url, embedding_model)

        # Chunk documents
        logger.info(f"Chunking {len(documents)} documents...")
        chunks = chunk_batch(documents, chunk_size, chunk_overlap)
        stats["num_chunks"] = len(chunks)
        logger.info(f"Created {len(chunks)} chunks")

        # Index chunks
        logger.info(f"Indexing {len(chunks)} chunks into ChromaDB...")
        num_indexed = index_chunks(chunks, embedding_base_url, embedding_model, chroma_db_path)
        stats["num_indexed"] = num_indexed

        logger.info(
            f"✅ Knowledge base population complete:\n"
            f"  - Articles: {stats['num_articles']}\n"
            f"  - Documents: {stats['num_documents']}\n"
            f"  - Chunks: {stats['num_chunks']}\n"
            f"  - Indexed: {stats['num_indexed']}\n"
            f"  - Topics: {stats['topics_count']}"
        )

        loader.close()
        return stats

    except Exception as e:
        error_msg = f"Failed to populate knowledge base: {str(e)}"
        logger.error(error_msg)
        stats["errors"].append(error_msg)
        raise


def populate_knowledge_base_incremental(
    database_url: str,
    embedding_base_url: str,
    embedding_model: str,
    chroma_db_path: str,
    since_date: Optional[datetime] = None,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> Dict:
    """Incrementally add new articles to existing knowledge base.

    Loads only articles created/modified since a specific date and adds them
    to the existing ChromaDB index without clearing it.

    Args:
        database_url: SQL database connection string
        embedding_base_url: Base URL for embedding model
        embedding_model: Name of embedding model
        chroma_db_path: Path to ChromaDB persistence directory
        since_date: Only load articles after this datetime (if None, loads all)
        chunk_size: Size of each chunk in characters
        chunk_overlap: Overlap between consecutive chunks

    Returns:
        Dictionary with statistics (same as populate_knowledge_base_from_sql)

    Raises:
        ValueError: If database_url is invalid
        Exception: If database connection or indexing fails
    """
    stats = {
        "num_articles": 0,
        "num_documents": 0,
        "num_chunks": 0,
        "num_indexed": 0,
        "topics_count": 0,
        "errors": [],
    }

    try:
        logger.info("Starting incremental knowledge base update...")

        # Initialize SQL loader
        loader = SQLArticleLoader(database_url)

        # Get topics
        topics_map = loader.get_topics()
        stats["topics_count"] = len(topics_map)

        # Load articles (all if since_date is None)
        if since_date:
            logger.info(f"Loading articles since {since_date}...")
            sql_articles = loader.load_articles_since(since_date)
        else:
            logger.info("Loading all articles...")
            sql_articles = loader.load_all_articles()

        stats["num_articles"] = len(sql_articles)

        if not sql_articles:
            logger.info("No new articles to add")
            loader.close()
            return stats

        # Convert to document format
        documents = convert_sql_articles_to_documents(sql_articles, topics_map)
        stats["num_documents"] = len(documents)

        # Initialize ChromaDB (doesn't clear existing)
        initialize_chroma_db(chroma_db_path, embedding_base_url, embedding_model)

        # Chunk documents
        chunks = chunk_batch(documents, chunk_size, chunk_overlap)
        stats["num_chunks"] = len(chunks)

        # Index chunks
        num_indexed = index_chunks(chunks, embedding_base_url, embedding_model, chroma_db_path)
        stats["num_indexed"] = num_indexed

        logger.info(
            f"✅ Incremental update complete:\n"
            f"  - New articles: {stats['num_articles']}\n"
            f"  - Chunks indexed: {stats['num_indexed']}"
        )

        loader.close()
        return stats

    except Exception as e:
        error_msg = f"Failed to incrementally update knowledge base: {str(e)}"
        logger.error(error_msg)
        stats["errors"].append(error_msg)
        raise
