"""Configuration module for the AI News Intelligence Assistant.

Centralizes all configuration settings including model endpoints,
embedding models, database paths, and application parameters.

Configuration is loaded from environment variables via .env file using Pydantic Settings.
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

# Load .env file
load_dotenv()


class LMStudioSettings(BaseModel):
    """LM Studio model server configuration."""

    base_url: str = Field(
        default="http://localhost:1234/v1",
        description="LM Studio base URL for API calls",
    )
    chat_model: str = Field(
        default="gemma-4-e4b",
        description="Chat/generation model name",
    )
    embedding_model: str = Field(
        default="embeddinggemma-300M-GGUF",
        description="Embedding model name",
    )
    request_timeout: int = Field(
        default=60,
        gt=0,
        description="Request timeout in seconds",
    )


class GenerationSettings(BaseModel):
    """LLM generation/decoding parameters."""

    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature",
    )
    top_p: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling parameter",
    )
    top_k: int = Field(
        default=40,
        ge=0,
        description="Top-k sampling parameter",
    )
    max_tokens: int = Field(
        default=512,
        gt=0,
        description="Maximum tokens to generate",
    )


class ChunkingSettings(BaseModel):
    """Document chunking configuration."""

    chunk_size: int = Field(
        default=512,
        gt=0,
        description="Size of each chunk in tokens/characters",
    )
    chunk_overlap: int = Field(
        default=50,
        ge=0,
        description="Overlap between consecutive chunks",
    )

    @field_validator("chunk_overlap")
    @classmethod
    def validate_overlap(cls, v: int, info) -> int:
        """Ensure overlap is less than chunk size."""
        chunk_size = info.data.get("chunk_size", 512)
        if v >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return v


class ChromaDBSettings(BaseModel):
    """ChromaDB vector database configuration."""

    persist_directory: Optional[str] = Field(
        default=None,
        description="ChromaDB persistence directory (auto-set to data/chroma_db if None)",
    )
    distance_metric: str = Field(
        default="cosine",
        pattern="^(cosine|euclidean|manhattan)$",
        description="Distance metric for similarity search",
    )
    top_k_retrieval: int = Field(
        default=5,
        gt=0,
        description="Number of chunks to retrieve per query",
    )


class RAGSettings(BaseModel):
    """Retrieval-Augmented Generation settings."""

    enabled_by_default: bool = Field(
        default=True,
        description="Enable RAG by default",
    )
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    chromadb: ChromaDBSettings = Field(default_factory=ChromaDBSettings)


class AppSettings(BaseModel):
    """Application-level settings."""

    name: str = "AI News Intelligence Assistant"
    version: str = "0.1.0"
    multi_agent_enabled: bool = Field(
        default=False,
        description="Enable multi-agent reasoning by default",
    )
    log_level: str = Field(
        default="INFO",
        pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$",
    )
    streamlit_theme: str = Field(
        default="dark",
    )


class Settings(BaseModel):
    """Main configuration class combining all settings."""

    # Metadata
    project_root: Path = Field(default_factory=lambda: Path(__file__).parent)

    # Data directories
    data_dir: Optional[Path] = None
    raw_data_dir: Optional[Path] = None
    processed_data_dir: Optional[Path] = None
    chroma_db_dir: Optional[Path] = None
    benchmark_dir: Optional[Path] = None
    logs_dir: Optional[Path] = None

    # Settings groups
    lm_studio: LMStudioSettings = Field(default_factory=LMStudioSettings)
    generation: GenerationSettings = Field(default_factory=GenerationSettings)
    rag: RAGSettings = Field(default_factory=RAGSettings)
    app: AppSettings = Field(default_factory=AppSettings)

    def __init__(self, **data):
        super().__init__(**data)
        # Load settings from environment variables
        self._load_from_env()

        # Initialize directories with defaults
        if self.data_dir is None:
            self.data_dir = self.project_root / "data"
        if self.raw_data_dir is None:
            self.raw_data_dir = self.data_dir / "raw"
        if self.processed_data_dir is None:
            self.processed_data_dir = self.data_dir / "processed"
        if self.chroma_db_dir is None:
            self.chroma_db_dir = self.data_dir / "chroma_db"
        if self.benchmark_dir is None:
            self.benchmark_dir = self.data_dir / "benchmark"
        if self.logs_dir is None:
            self.logs_dir = self.project_root / "logs"

        # Set ChromaDB persist directory if not already set
        if self.rag.chromadb.persist_directory is None:
            self.rag.chromadb.persist_directory = str(self.chroma_db_dir)

        # Create directories if they don't exist
        self._ensure_directories()

    def _load_from_env(self) -> None:
        """Load settings from environment variables."""
        # LM Studio settings
        self.lm_studio = LMStudioSettings(
            base_url=os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1"),
            chat_model=os.getenv("CHAT_MODEL", "gemma-4-e4b"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "embeddinggemma-300M-GGUF"),
            request_timeout=int(os.getenv("REQUEST_TIMEOUT", "60")),
        )

        # Generation settings
        self.generation = GenerationSettings(
            temperature=float(os.getenv("DEFAULT_TEMPERATURE", "0.7")),
            top_p=float(os.getenv("DEFAULT_TOP_P", "0.95")),
            top_k=int(os.getenv("DEFAULT_TOP_K", "40")),
            max_tokens=int(os.getenv("MAX_TOKENS", "512")),
        )

        # Chunking settings
        chunking = ChunkingSettings(
            chunk_size=int(os.getenv("CHUNK_SIZE", "512")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "50")),
        )

        # ChromaDB settings
        chromadb = ChromaDBSettings(
            persist_directory=os.getenv("CHROMA_DB_DIR"),
            distance_metric=os.getenv("CHROMA_DISTANCE_METRIC", "cosine"),
            top_k_retrieval=int(os.getenv("TOP_K_RETRIEVAL", "5")),
        )

        # RAG settings
        self.rag = RAGSettings(
            enabled_by_default=os.getenv("RAG_ENABLED_DEFAULT", "true").lower() == "true",
            chunking=chunking,
            chromadb=chromadb,
        )

        # App settings
        self.app = AppSettings(
            multi_agent_enabled=os.getenv("MULTI_AGENT_ENABLED_DEFAULT", "false").lower() == "true",
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            streamlit_theme=os.getenv("STREAMLIT_THEME", "dark"),
        )

    def _ensure_directories(self) -> None:
        """Create all required directories if they don't exist."""
        for directory in [
            self.data_dir,
            self.raw_data_dir,
            self.processed_data_dir,
            self.chroma_db_dir,
            self.benchmark_dir,
            self.logs_dir,
        ]:
            if directory:
                directory.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict:
        """Export settings as dictionary for logging/debugging."""
        return {
            "app_name": self.app.name,
            "app_version": self.app.version,
            "lm_studio": {
                "base_url": self.lm_studio.base_url,
                "chat_model": self.lm_studio.chat_model,
                "embedding_model": self.lm_studio.embedding_model,
                "request_timeout": self.lm_studio.request_timeout,
            },
            "generation": {
                "temperature": self.generation.temperature,
                "top_p": self.generation.top_p,
                "top_k": self.generation.top_k,
                "max_tokens": self.generation.max_tokens,
            },
            "rag": {
                "enabled_by_default": self.rag.enabled_by_default,
                "chunking": {
                    "chunk_size": self.rag.chunking.chunk_size,
                    "chunk_overlap": self.rag.chunking.chunk_overlap,
                },
                "chromadb": {
                    "persist_directory": self.rag.chromadb.persist_directory,
                    "distance_metric": self.rag.chromadb.distance_metric,
                    "top_k_retrieval": self.rag.chromadb.top_k_retrieval,
                },
            },
            "app": {
                "multi_agent_enabled": self.app.multi_agent_enabled,
                "log_level": self.app.log_level,
                "streamlit_theme": self.app.streamlit_theme,
            },
            "directories": {
                "project_root": str(self.project_root),
                "data_dir": str(self.data_dir),
                "raw_data_dir": str(self.raw_data_dir),
                "processed_data_dir": str(self.processed_data_dir),
                "chroma_db_dir": str(self.chroma_db_dir),
                "benchmark_dir": str(self.benchmark_dir),
                "logs_dir": str(self.logs_dir),
            },
        }


# Load configuration at module level
settings = Settings()

# Export commonly used settings for backward compatibility
PROJECT_ROOT = settings.project_root
DATA_DIR = settings.data_dir
RAW_DATA_DIR = settings.raw_data_dir
PROCESSED_DATA_DIR = settings.processed_data_dir
CHROMA_DB_DIR = settings.chroma_db_dir
BENCHMARK_DIR = settings.benchmark_dir
LOGS_DIR = settings.logs_dir

LM_STUDIO_BASE_URL = settings.lm_studio.base_url
CHAT_MODEL = settings.lm_studio.chat_model
EMBEDDING_MODEL = settings.lm_studio.embedding_model
REQUEST_TIMEOUT = settings.lm_studio.request_timeout

DEFAULT_TEMPERATURE = settings.generation.temperature
DEFAULT_TOP_P = settings.generation.top_p
DEFAULT_TOP_K = settings.generation.top_k
MAX_TOKENS = settings.generation.max_tokens

RAG_ENABLED_DEFAULT = settings.rag.enabled_by_default
TOP_K_RETRIEVAL = settings.rag.chromadb.top_k_retrieval
CHUNK_SIZE = settings.rag.chunking.chunk_size
CHUNK_OVERLAP = settings.rag.chunking.chunk_overlap

MULTI_AGENT_ENABLED_DEFAULT = settings.app.multi_agent_enabled
LOG_LEVEL = settings.app.log_level
STREAMLIT_THEME = settings.app.streamlit_theme

APP_NAME = settings.app.name
APP_VERSION = settings.app.version
