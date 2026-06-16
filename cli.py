#!/usr/bin/env python
"""CLI tool for managing knowledge base population from SQL database.

Usage:
    python cli.py populate-kb --clear
    python cli.py populate-kb --incremental
    python cli.py kb-status
"""

import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

import click
import logging
import sys
from datetime import datetime
from pathlib import Path

from config import settings
from core.utils import setup_logging
from ingestion.populate import (
    populate_knowledge_base_from_sql,
    populate_knowledge_base_incremental,
)
from ingestion.indexer import get_collection_stats, get_chroma_client

# Configure logging
logger = setup_logging(level="INFO")


@click.group()
def cli():
    """AI News Intelligence Assistant CLI."""
    pass


@cli.command()
@click.option(
    "--clear",
    is_flag=True,
    help="Clear existing ChromaDB index before populating",
)
@click.option(
    "--incremental",
    is_flag=True,
    help="Only add new articles (don't clear existing)",
)
def populate_kb(clear, incremental):
    """Populate knowledge base from SQL database.

    By default, this will populate the full knowledge base. Use --incremental
    to only add new articles, or --clear to start fresh.
    """
    try:
        if not settings.database_url:
            click.echo("[ERROR] DATABASE_URL not configured in .env", err=True)
            sys.exit(1)

        click.echo("Knowledge Base Population Tool")
        click.echo("=" * 50)

        if incremental:
            click.echo("\n[INFO] Mode: Incremental (adding only new articles)")
            stats = populate_knowledge_base_incremental(
                database_url=settings.database_url,
                embedding_base_url=settings.lm_studio_base_url,
                embedding_model=settings.embedding_model,
                chroma_db_path=settings.project_root / settings.chroma_db_dir,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
            )
        else:
            click.echo(f"\n[INFO] Mode: Full Population (clear={clear})")
            stats = populate_knowledge_base_from_sql(
                database_url=settings.database_url,
                embedding_base_url=settings.lm_studio_base_url,
                embedding_model=settings.embedding_model,
                chroma_db_path=settings.project_root / settings.chroma_db_dir,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
                clear_existing=clear,
            )

        # Display results
        click.echo("\nPopulation Results:")
        click.echo("-" * 50)
        click.echo(f"  Articles loaded: {stats['num_articles']}")
        click.echo(f"  Documents created: {stats['num_documents']}")
        click.echo(f"  Chunks created: {stats['num_chunks']}")
        click.echo(f"  Chunks indexed: {stats['num_indexed']}")
        click.echo(f"  Topics found: {stats['topics_count']}")

        if stats["errors"]:
            click.echo("\nErrors:")
            for error in stats["errors"]:
                click.echo(f"  - {error}")

        if stats["num_indexed"] > 0:
            click.echo("\n[SUCCESS] Knowledge base populated successfully!")
        else:
            click.echo("\n[WARNING] No chunks were indexed", err=True)

    except Exception as e:
        click.echo(f"[ERROR] {str(e)}", err=True)
        logger.error(f"Populate failed: {str(e)}", exc_info=True)
        sys.exit(1)


@cli.command()
def kb_status():
    """Show knowledge base status."""
    try:
        from ingestion.indexer import initialize_chroma_db, get_collection

        # Initialize ChromaDB
        initialize_chroma_db(
            db_path=settings.project_root / settings.chroma_db_dir,
            embedding_base_url=settings.lm_studio_base_url,
            embedding_model=settings.embedding_model,
        )

        collection = get_collection()

        if collection is None:
            click.echo("[ERROR] Knowledge base not initialized")
            sys.exit(1)

        try:
            stats = get_collection_stats()
        except Exception:
            click.echo("[WARNING] ChromaDB initialized but empty")
            sys.exit(0)

        click.echo("Knowledge Base Status")
        click.echo("=" * 50)
        click.echo(f"Collection name: {stats.get('collection_name', 'N/A')}")
        click.echo(f"Total chunks: {stats.get('total_chunks', 0)}")
        click.echo(f"ChromaDB path: {settings.project_root / settings.chroma_db_dir}")
        click.echo(f"Distance metric: {settings.distance_metric}")
        click.echo(f"Top-K retrieval: {settings.top_k_retrieval}")

        if stats.get("total_chunks", 0) > 0:
            click.echo("\n[SUCCESS] Knowledge base is ready for search and retrieval")
        else:
            click.echo("\n[WARNING] Knowledge base is empty. Run 'populate-kb' to add articles.")

    except Exception as e:
        click.echo(f"[ERROR] {str(e)}", err=True)
        logger.error(f"Status check failed: {str(e)}", exc_info=True)
        sys.exit(1)


@cli.command()
@click.option(
    "--topic-id",
    type=int,
    help="Optional topic ID to load specific category",
)
def load_articles(topic_id):
    """Load and display articles from SQL database."""
    try:
        from ingestion.sql_loader import SQLArticleLoader

        if not settings.database_url:
            click.echo("[ERROR] DATABASE_URL not configured in .env", err=True)
            sys.exit(1)

        loader = SQLArticleLoader(settings.database_url)

        if topic_id:
            click.echo(f"Loading articles for topic {topic_id}...")
            articles = loader.load_articles_by_topic(topic_id)
        else:
            click.echo("Loading all articles...")
            articles = loader.load_all_articles()

        click.echo(f"\nFound {len(articles)} articles")
        click.echo("=" * 70)

        for i, article in enumerate(articles[:10], 1):  # Show first 10
            click.echo(
                f"\n{i}. ID: {article['id']}\n"
                f"   Title: {article['title'][:60]}...\n"
                f"   Length: {len(article['content'])} chars\n"
                f"   Date: {article['date']}\n"
                f"   URL: {article['url'][:60]}..."
            )

        if len(articles) > 10:
            click.echo(f"\n... and {len(articles) - 10} more articles")

        loader.close()

    except Exception as e:
        click.echo(f"[ERROR] {str(e)}", err=True)
        logger.error(f"Load articles failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
