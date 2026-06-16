#!/usr/bin/env python
"""Rebuild ChromaDB knowledge base with updated metadata."""
import os
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from config import settings
    from ingestion.indexer import delete_collection
    from ingestion.populate import populate_knowledge_base_from_sql
    
    logger.info("Loading configuration...")
    
    logger.info("Clearing existing ChromaDB collection...")
    delete_collection()
    
    logger.info("Rebuilding knowledge base from SQL...")
    stats = populate_knowledge_base_from_sql(
        database_url=settings.database_url,
        embedding_base_url=settings.lm_studio_base_url,
        embedding_model=settings.embedding_model,
        chroma_db_path=settings.project_root / settings.chroma_db_dir,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        clear_existing=False,  # Already cleared above
    )
    
    logger.info(f"Rebuild complete: {stats}")
    
except Exception as e:
    logger.error(f"Rebuild failed: {e}", exc_info=True)
    sys.exit(1)
