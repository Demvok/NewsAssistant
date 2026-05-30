"""Core module for AI News Intelligence Assistant.

Provides central functionality for LLM access, embeddings, RAG, agents, 
tools, evaluation, and utility functions.
"""

from core.tools import (
    ToolRegistry,
    create_default_tools,
    search_by_topic,
    filter_by_date,
    count_keyword_mentions,
    corpus_statistics,
)

__all__ = [
    "ToolRegistry",
    "create_default_tools",
    "search_by_topic",
    "filter_by_date",
    "count_keyword_mentions",
    "corpus_statistics",
]
