"""Data schemas and types for the application.

Defines dataclasses and type hints for consistent data representation
across the application.
"""

from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime


@dataclass
class Document:
    """Represents a document in the corpus."""

    id: str
    title: str
    content: str
    source: str
    metadata: dict = field(default_factory=dict)
    created_at: Optional[datetime] = None


@dataclass
class Chunk:
    """Represents a text chunk from a document."""

    id: str
    content: str
    document_id: str
    chunk_order: int
    metadata: dict = field(default_factory=dict)
    embedding: Optional[list[float]] = None


@dataclass
class RetrievalResult:
    """Represents a retrieval result."""

    chunk: Chunk
    similarity_score: float
    metadata: dict = field(default_factory=dict)


@dataclass
class ChatMessage:
    """Represents a chat message."""

    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)


@dataclass
class TestCase:
    """Represents a benchmark test case."""

    id: str
    category: str
    prompt: str
    expected_answer: str
    evaluation_criteria: str
    metadata: dict = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Represents evaluation results for a test case."""

    test_case_id: str
    response: str
    score: float
    judge_reasoning: str
    metadata: dict = field(default_factory=dict)


@dataclass
class ExperimentRun:
    """Represents a single experiment run with specific parameters."""

    prompt: str
    temperature: float
    top_p: float
    top_k: int
    response: str
    response_length: int
    metadata: dict = field(default_factory=dict)
