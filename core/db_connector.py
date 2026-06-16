"""Database connector module for article storage and retrieval.

Manages connection to the database storing articles, events, opinions, and other
article-related metadata. Uses SQLAlchemy ORM for all database operations.

Configuration via config.py:
- settings.database_url: Connection string from .env (DATABASE_URL)
"""

import time
from datetime import datetime
from contextlib import contextmanager
import threading
import logging

from sqlalchemy import (
    create_engine,
    text,
    Column,
    Integer,
    String,
    ForeignKey,
    Float,
    TIMESTAMP,
    Boolean,
    func,
)
from sqlalchemy.orm import (
    declarative_base,
    sessionmaker,
    relationship,
    scoped_session,
)
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from sqlalchemy.orm.exc import DetachedInstanceError

logger = logging.getLogger(__name__)

# ============================================================================
# Database Configuration
# ============================================================================


def _resolve_database_url() -> str:
    """Resolve the database URL from config.

    Uses settings.database_url which loads from .env (DATABASE_URL).

    Raises:
        ValueError: If DATABASE_URL is not configured in .env.
    """
    from config import settings

    if not settings.database_url:
        raise ValueError(
            "DATABASE_URL is not configured in .env file. "
            "Set DATABASE_URL in .env (e.g., mysql+pymysql://user:pass@host:3306/db)"
        )

    return settings.database_url


DATABASE_URL = _resolve_database_url()

# ============================================================================
# Database Engine & Session Management
# ============================================================================

# Create engine with improved connection pooling settings
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,  # Check connection validity before using
    pool_recycle=1800,  # Recycle connections after 30 minutes
    pool_size=15,  # Default pool size
    max_overflow=25,  # Allow extra connections when pool is full
    pool_timeout=30,  # Timeout when waiting for connection from pool
    echo=False,  # Set to True for SQL debugging
)

SessionFactory = sessionmaker(bind=engine, expire_on_commit=False)
Session = scoped_session(SessionFactory)

# Thread-local storage to track connection state
_local = threading.local()
_local.is_session_valid = True


def refresh_connection() -> None:
    """Refresh database connection when issues occur.

    Disposes existing connections and creates a new engine with the same settings.
    """
    global engine, SessionFactory, Session
    logger.info("Refreshing database connection")

    # Close any existing connections in the pool
    engine.dispose()

    # Create a new engine with the same settings
    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,
        pool_recycle=1800,
        pool_size=15,
        max_overflow=25,
        pool_timeout=30,
    )

    # Create new session factory and scoped session
    SessionFactory = sessionmaker(bind=engine, expire_on_commit=False)
    Session = scoped_session(SessionFactory)

    # Reset session validity flag
    _local.is_session_valid = True
    logger.info("Database connection refreshed")


@contextmanager
def _get_session():
    """Context manager for database session management with automatic retry.

    Handles connection failures with retry logic and automatic rollback on errors.

    Yields:
        SQLAlchemy Session: Database session for queries.

    Raises:
        OperationalError: If connection fails after max retries.
        DetachedInstanceError: If ORM object is detached from session.
        SQLAlchemyError: For other database errors.
    """
    session = None
    retry_count = 0
    max_retries = 3

    while retry_count < max_retries:
        try:
            # Create a new session
            session = Session()

            # Test the connection with a simple query before proceeding
            session.execute(text("SELECT 1"))

            # If we get here, the connection is good
            break
        except (OperationalError, SQLAlchemyError) as e:
            retry_count += 1
            logger.warning(
                f"Connection error on session creation (attempt {retry_count}/{max_retries}): {e}"
            )

            if session:
                session.close()

            # Refresh the connection on failure
            refresh_connection()

            if retry_count >= max_retries:
                logger.error(
                    f"Failed to establish database connection after {max_retries} attempts"
                )
                raise

            # Wait before retrying
            time.sleep(1)

    try:
        yield session
        session.commit()
    except OperationalError as e:
        session.rollback()
        _local.is_session_valid = False
        refresh_connection()
        logger.error(f"Connection lost, please retry: {e}")
        raise
    except DetachedInstanceError as e:
        session.rollback()
        logger.warning(f"DetachedInstanceError occurred: {e}")
        raise
    except Exception as e:
        session.rollback()
        logger.error(f"Database error: {str(e)}")
        raise
    finally:
        if session:
            # Make sure to close the session in all cases
            session.close()


# ============================================================================
# ORM Models
# ============================================================================

Base = declarative_base()


class DimArticle(Base):
    """Dimension table for articles."""

    __tablename__ = "dimArticle"

    article_id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(150), nullable=True)
    fk_topic_id = Column(
        Integer,
        ForeignKey("dimTopic.topic_id", ondelete="CASCADE"),
        nullable=False,
        default=1,
    )
    url = Column(String(150), nullable=False)
    article_date = Column(TIMESTAMP, nullable=True)
    created_at = Column(
        TIMESTAMP, server_default=func.current_timestamp(), nullable=True
    )
    modified_at = Column(
        TIMESTAMP,
        server_default=func.current_timestamp(),
        onupdate=func.current_timestamp(),
        nullable=True,
    )
    content = Column(String, nullable=True)

    # Relationships
    events = relationship(
        "DimEvent", back_populates="article", cascade="all, delete-orphan"
    )
    opinions = relationship(
        "DimOpinion", back_populates="article", cascade="all, delete-orphan"
    )
    chunks = relationship(
        "DimArticleChunks", back_populates="article", cascade="all, delete-orphan"
    )
    topic = relationship("DimTopic", back_populates="articles")


class DimEvent(Base):
    """Dimension table for events extracted from articles."""

    __tablename__ = "dimEvents"

    event_id = Column(Integer, primary_key=True, autoincrement=True)
    event_title = Column(String(50), nullable=True)
    fk_origin_article_id = Column(
        Integer, ForeignKey("dimArticle.article_id", ondelete="CASCADE"), nullable=False
    )
    description = Column(String(200), nullable=True)
    created_at = Column(
        TIMESTAMP, server_default=func.current_timestamp(), nullable=True
    )
    modified_at = Column(
        TIMESTAMP,
        server_default=func.current_timestamp(),
        onupdate=func.current_timestamp(),
        nullable=True,
    )
    relevance_score = Column(Float, nullable=True)
    influence_score = Column(Float, nullable=True)
    novelty_score = Column(Float, nullable=True)
    event_hotness = Column(Float, nullable=True)
    is_selected = Column(Boolean, nullable=True)

    # Relationships
    article = relationship("DimArticle", back_populates="events")


class DimOpinion(Base):
    """Dimension table for opinions extracted from articles."""

    __tablename__ = "dimOpinion"

    opinion_id = Column(Integer, primary_key=True, autoincrement=True)
    fk_origin_article_id = Column(
        Integer, ForeignKey("dimArticle.article_id", ondelete="CASCADE"), nullable=False
    )
    fk_person_id = Column(
        Integer, ForeignKey("dimPerson.person_id", ondelete="CASCADE"), nullable=False
    )
    created_at = Column(
        TIMESTAMP, server_default=func.current_timestamp(), nullable=True
    )
    modified_at = Column(
        TIMESTAMP,
        server_default=func.current_timestamp(),
        onupdate=func.current_timestamp(),
        nullable=True,
    )
    citation = Column(String(300), nullable=True)
    sentiment_score = Column(Float, nullable=True)
    inconsistency_flag = Column(Boolean, nullable=True)
    inconsistency_with_id = Column(String(200), nullable=True)
    inconsistency_comment = Column(String(200), nullable=True)
    controversy_score = Column(Float, nullable=True)
    relevancy_score = Column(Float, nullable=True)
    contribution_score = Column(Float, nullable=True)
    opinion_hotness = Column(Float, nullable=True)
    is_selected = Column(Boolean, nullable=True)

    # Relationships
    article = relationship("DimArticle", back_populates="opinions")
    person = relationship("DimPerson", back_populates="opinions")


class DimPerson(Base):
    """Dimension table for people mentioned in articles."""

    __tablename__ = "dimPerson"

    person_id = Column(Integer, primary_key=True, autoincrement=True)
    person_name = Column(String(100), nullable=False)
    image_url = Column(String(100), nullable=True)

    # Relationships
    opinions = relationship("DimOpinion", back_populates="person")
    attitudes = relationship("FctAttitude", back_populates="person")


class DimArticleChunks(Base):
    """Dimension table for article chunks."""

    __tablename__ = "dimArticleChunks"

    chunk_id = Column(Integer, primary_key=True, autoincrement=True)
    fk_article_id = Column(
        Integer, ForeignKey("dimArticle.article_id", ondelete="CASCADE"), nullable=False
    )
    start_index = Column(Integer, nullable=True)
    end_index = Column(Integer, nullable=True)
    is_processed = Column(Boolean, nullable=True)

    # Relationships
    article = relationship("DimArticle", back_populates="chunks")


class DimTopic(Base):
    """Dimension table for topics."""

    __tablename__ = "dimTopic"

    topic_id = Column(Integer, primary_key=True, autoincrement=True)
    topic_name = Column(String(50), nullable=False)
    query = Column(String(100), nullable=False)
    source = Column(String(100), nullable=True)

    # Relationships
    attitudes = relationship("FctAttitude", back_populates="topic")
    articles = relationship("DimArticle", back_populates="topic")


class FctAttitude(Base):
    """Fact table for person attitudes toward topics."""

    __tablename__ = "fctAttitude"

    fk_topic_id = Column(
        Integer, ForeignKey("dimTopic.topic_id"), primary_key=True, nullable=False
    )
    fk_person_id = Column(
        Integer, ForeignKey("dimPerson.person_id"), primary_key=True, nullable=False
    )
    created_at = Column(
        TIMESTAMP, server_default=func.current_timestamp(), nullable=True
    )
    modified_at = Column(
        TIMESTAMP,
        server_default=func.current_timestamp(),
        onupdate=func.current_timestamp(),
        nullable=True,
    )
    sentiment_deviation = Column(Float, nullable=True)
    stance = Column(String(30), nullable=True)
    person_summary = Column(String(300), nullable=True)

    # Relationships
    topic = relationship("DimTopic", back_populates="attitudes")
    person = relationship("DimPerson", back_populates="attitudes")


# ============================================================================
# Query Execution Functions
# ============================================================================


def execute_query(query: str):
    """Execute a raw SQL query and return results.

    Args:
        query: SQL query string.

    Returns:
        Query results if the query returns rows, None otherwise.
    """
    with engine.connect() as conn:
        result = conn.execute(text(query))
        if result.returns_rows:
            data = result.fetchall()
            return data
        return None


def get_article_by_id(article_id: int) -> dict | None:
    """Fetch an article by its article_id using ORM session.

    Returns a dictionary with keys: article_id, title, content, url, article_date
    or None if not found.
    """
    try:
        with _get_session() as session:
            row = session.query(DimArticle).filter(DimArticle.article_id == int(article_id)).one_or_none()
            if row is None:
                return None
            return {
                "article_id": int(row.article_id),
                "title": row.title,
                "content": row.content,
                "url": row.url,
                "article_date": row.article_date.isoformat() if row.article_date else None,
                "created_at": row.created_at.isoformat() if row.created_at else None,
            }
    except Exception as e:
        logger.error(f"Failed to fetch article by id {article_id}: {e}")
        return None


def get_article_by_id(article_id: int) -> dict | None:
    """Fetch an article by its article_id using ORM session.

    Returns a dictionary with keys: article_id, title, content, url, article_date
    or None if not found.
    """
    try:
        with _get_session() as session:
            row = session.query(DimArticle).filter(DimArticle.article_id == int(article_id)).one_or_none()
            if row is None:
                return None
            return {
                "article_id": int(row.article_id),
                "title": row.title,
                "content": row.content,
                "url": row.url,
                "article_date": row.article_date.isoformat() if row.article_date else None,
                "created_at": row.created_at.isoformat() if row.created_at else None,
            }
    except Exception as e:
        logger.error(f"Failed to fetch article by id {article_id}: {e}")
        return None
