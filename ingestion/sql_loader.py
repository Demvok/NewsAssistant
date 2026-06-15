"""SQL database loader for ingesting news articles into the knowledge base.

Connects to MySQL/PostgreSQL to load articles from the news_analysis_database_v3.dimArticle table
and prepares them for chunking and embedding into ChromaDB.
"""

import logging
from typing import Optional, List, Dict
from datetime import datetime

from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool

logger = logging.getLogger(__name__)


class SQLArticleLoader:
    """Loader for fetching articles from SQL database."""

    def __init__(self, database_url: str):
        """Initialize SQL connection.

        Args:
            database_url: SQLAlchemy connection string
                Example: mysql+pymysql://user:pass@host:3306/db

        Raises:
            ValueError: If database_url is invalid
        """
        if not database_url:
            raise ValueError("database_url cannot be empty")

        self.database_url = database_url
        self.engine = None
        self._connect()

    def _connect(self) -> None:
        """Create database connection pool."""
        try:
            self.engine = create_engine(
                self.database_url,
                poolclass=QueuePool,
                pool_size=5,
                max_overflow=10,
                pool_pre_ping=True,
                echo=False,
            )
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("Database connection established successfully")
        except Exception as e:
            logger.error(f"Failed to connect to database: {str(e)}")
            raise

    def load_all_articles(self) -> List[Dict]:
        """Load all articles from dimArticle table.

        Returns:
            List of article dictionaries with keys:
            - id, title, content, date, url, topic_id

        Raises:
            Exception: If query fails
        """
        if not self.engine:
            raise RuntimeError("Database not connected")

        try:
            query = text("""
                SELECT 
                    article_id,
                    title,
                    content,
                    article_date,
                    url,
                    fk_topic_id as topic_id,
                    created_at
                FROM dimArticle
                WHERE content IS NOT NULL AND content != ''
                ORDER BY article_id DESC
            """)

            with self.engine.connect() as conn:
                result = conn.execute(query)
                rows = result.fetchall()

            articles = []
            for row in rows:
                articles.append({
                    "id": row.article_id,
                    "title": row.title or "Untitled",
                    "content": row.content,
                    "date": row.article_date,
                    "url": row.url,
                    "topic_id": row.topic_id,
                    "created_at": row.created_at,
                })

            logger.info(f"Loaded {len(articles)} articles from database")
            return articles

        except Exception as e:
            logger.error(f"Failed to load articles: {str(e)}")
            raise

    def load_articles_by_topic(self, topic_id: int) -> List[Dict]:
        """Load articles filtered by topic ID.

        Args:
            topic_id: Topic ID to filter by

        Returns:
            List of article dictionaries for the given topic

        Raises:
            Exception: If query fails
        """
        if not self.engine:
            raise RuntimeError("Database not connected")

        try:
            query = text("""
                SELECT 
                    article_id,
                    title,
                    content,
                    article_date,
                    url,
                    fk_topic_id as topic_id,
                    created_at
                FROM dimArticle
                WHERE fk_topic_id = :topic_id 
                  AND content IS NOT NULL 
                  AND content != ''
                ORDER BY article_id DESC
            """)

            with self.engine.connect() as conn:
                result = conn.execute(query, {"topic_id": topic_id})
                rows = result.fetchall()

            articles = []
            for row in rows:
                articles.append({
                    "id": row.article_id,
                    "title": row.title or "Untitled",
                    "content": row.content,
                    "date": row.article_date,
                    "url": row.url,
                    "topic_id": row.topic_id,
                    "created_at": row.created_at,
                })

            logger.info(f"Loaded {len(articles)} articles for topic {topic_id}")
            return articles

        except Exception as e:
            logger.error(f"Failed to load articles by topic: {str(e)}")
            raise

    def load_articles_since(self, since_date: datetime) -> List[Dict]:
        """Load articles created/modified after a specific date.

        Args:
            since_date: Only load articles after this datetime

        Returns:
            List of recent article dictionaries

        Raises:
            Exception: If query fails
        """
        if not self.engine:
            raise RuntimeError("Database not connected")

        try:
            query = text("""
                SELECT 
                    article_id,
                    title,
                    content,
                    article_date,
                    url,
                    fk_topic_id as topic_id,
                    created_at
                FROM dimArticle
                WHERE created_at >= :since_date 
                  AND content IS NOT NULL 
                  AND content != ''
                ORDER BY article_id DESC
            """)

            with self.engine.connect() as conn:
                result = conn.execute(query, {"since_date": since_date})
                rows = result.fetchall()

            articles = []
            for row in rows:
                articles.append({
                    "id": row.article_id,
                    "title": row.title or "Untitled",
                    "content": row.content,
                    "date": row.article_date,
                    "url": row.url,
                    "topic_id": row.topic_id,
                    "created_at": row.created_at,
                })

            logger.info(f"Loaded {len(articles)} articles since {since_date}")
            return articles

        except Exception as e:
            logger.error(f"Failed to load articles since date: {str(e)}")
            raise

    def get_topics(self) -> Dict[int, str]:
        """Get mapping of topic IDs to topic names.

        Returns:
            Dictionary mapping topic_id -> topic_name

        Raises:
            Exception: If query fails
        """
        if not self.engine:
            raise RuntimeError("Database not connected")

        try:
            query = text("""
                SELECT topic_id, topic_name
                FROM dimTopic
                ORDER BY topic_id
            """)

            with self.engine.connect() as conn:
                result = conn.execute(query)
                rows = result.fetchall()

            topics = {row.topic_id: row.topic_name for row in rows}
            logger.info(f"Loaded {len(topics)} topics")
            return topics

        except Exception as e:
            logger.error(f"Failed to load topics: {str(e)}")
            return {}

    def get_article_count(self) -> int:
        """Get total count of articles in database.

        Returns:
            Number of articles with content

        Raises:
            Exception: If query fails
        """
        if not self.engine:
            raise RuntimeError("Database not connected")

        try:
            query = text("""
                SELECT COUNT(*) as count
                FROM dimArticle
                WHERE content IS NOT NULL AND content != ''
            """)

            with self.engine.connect() as conn:
                result = conn.execute(query)
                count = result.scalar()

            logger.info(f"Database has {count} articles")
            return count

        except Exception as e:
            logger.error(f"Failed to get article count: {str(e)}")
            raise

    def close(self) -> None:
        """Close database connection pool."""
        if self.engine:
            self.engine.dispose()
            logger.info("Database connection closed")


def convert_sql_articles_to_documents(
    sql_articles: List[Dict],
    topics_map: Optional[Dict[int, str]] = None,
) -> List[Dict]:
    """Convert SQL articles to document format for ingestion.

    Args:
        sql_articles: Articles from SQL database
        topics_map: Optional mapping of topic_id -> topic_name

    Returns:
        List of documents ready for chunking with keys:
        - id, title, content, metadata (source, filename, url, date, topic)
    """
    documents = []

    for article in sql_articles:
        topic_name = "Unknown"
        if topics_map and article.get("topic_id") in topics_map:
            topic_name = topics_map[article["topic_id"]]

        doc = {
            "id": f"article_{article['id']}",
            "title": article["title"],
            "content": article["content"],
            "metadata": {
                "source": "MySQL news_analysis_database_v3.dimArticle",
                "filename": f"article_{article['id']}.txt",
                "url": article.get("url", ""),
                "article_id": str(article["id"]),
                "date": str(article.get("date", "")),
                "topic": topic_name,
                "created_at": str(article.get("created_at", "")),
            },
        }
        documents.append(doc)

    logger.info(f"Converted {len(documents)} SQL articles to document format")
    return documents
