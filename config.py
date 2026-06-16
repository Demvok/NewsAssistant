"""Configuration management - load from config.yaml and .env.

All configuration values come from:
1. config.yaml - non-sensitive settings (checked into git)
2. .env - sensitive data only (not in git)

No hardcoded defaults or legacy fallbacks. All values externalized.
"""

import logging
import os
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, field_validator

load_dotenv()
logger = logging.getLogger(__name__)


class Settings(BaseModel):
    """Complete application settings from config.yaml and .env."""

    # Application metadata
    app_name: str
    app_version: str
    log_level: str
    streamlit_theme: str

    # LM Studio (from .env, with fallback to config.yaml)
    lm_studio_base_url: str
    chat_model: str
    embedding_model: str
    request_timeout: int

    # Database (from .env)
    database_url: Optional[str] = None

    # Generation parameters
    temperature: float
    top_p: float
    top_k: int
    max_tokens: int

    # Document chunking
    chunk_size: int
    chunk_overlap: int

    # RAG settings
    rag_enabled_default: bool
    distance_metric: str
    top_k_retrieval: int

    # Features
    multi_agent_enabled_default: bool

    # Directories (relative to project root)
    data_dir: str
    raw_data_dir: str
    processed_data_dir: str
    chroma_db_dir: str
    benchmark_dir: str
    logs_dir: str

    # Project root (for resolving relative paths)
    project_root: Path = Path(__file__).parent

    @field_validator("chunk_overlap")
    @classmethod
    def validate_overlap(cls, v: int, info) -> int:
        """Chunk overlap must be less than chunk size."""
        chunk_size = info.data.get("chunk_size", 512)
        if v >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return v

    def get_absolute_path(self, relative_path: str) -> Path:
        """Convert relative directory path to absolute."""
        return self.project_root / relative_path


def load_settings() -> Settings:
    """Load configuration from config.yaml and environment.
    
    Raises:
        FileNotFoundError: If config.yaml is missing
        ValueError: If required configuration is missing
    """
    project_root = Path(__file__).parent
    config_path = project_root / "config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"config.yaml required at {config_path}")

    # Load YAML
    with open(config_path) as f:
        config = yaml.safe_load(f)
        if not config:
            raise ValueError("config.yaml is empty")

    # Build settings from config.yaml + environment overrides
    app_cfg = config.get("application", {})
    lm_cfg = config.get("lm_studio", {})
    gen_cfg = config.get("generation", {})
    rag_cfg = config.get("rag", {})
    dir_cfg = config.get("directories", {})
    models_cfg = config.get("models", {})

    settings_data = {
        # Application
        "app_name": app_cfg.get("name"),
        "app_version": app_cfg.get("version"),
        "log_level": app_cfg.get("log_level"),
        "streamlit_theme": app_cfg.get("streamlit_theme"),
        "multi_agent_enabled_default": app_cfg.get("multi_agent_enabled_default"),

        # LM Studio - from .env with fallback to config.yaml
        "lm_studio_base_url": os.getenv("LM_STUDIO_BASE_URL") or lm_cfg.get("base_url"),
        "chat_model": os.getenv("CHAT_MODEL") or models_cfg.get("chat_model"),
        "embedding_model": os.getenv("EMBEDDING_MODEL") or models_cfg.get("embedding_model"),
        "request_timeout": lm_cfg.get("request_timeout"),

        # Database - from .env
        "database_url": os.getenv("DATABASE_URL"),

        # Generation
        "temperature": gen_cfg.get("temperature"),
        "top_p": gen_cfg.get("top_p"),
        "top_k": gen_cfg.get("top_k"),
        "max_tokens": gen_cfg.get("max_tokens"),

        # Chunking
        "chunk_size": rag_cfg.get("chunking", {}).get("chunk_size"),
        "chunk_overlap": rag_cfg.get("chunking", {}).get("chunk_overlap"),

        # RAG
        "rag_enabled_default": rag_cfg.get("enabled_by_default"),
        "distance_metric": rag_cfg.get("chromadb", {}).get("distance_metric"),
        "top_k_retrieval": rag_cfg.get("chromadb", {}).get("top_k_retrieval"),

        # Directories
        "data_dir": dir_cfg.get("data"),
        "raw_data_dir": dir_cfg.get("raw_data"),
        "processed_data_dir": dir_cfg.get("processed_data"),
        "chroma_db_dir": dir_cfg.get("chroma_db"),
        "benchmark_dir": dir_cfg.get("benchmark"),
        "logs_dir": dir_cfg.get("logs"),
    }

    try:
        settings = Settings(**settings_data)
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        raise

    # Create directories
    for dir_name in [
        settings.data_dir,
        settings.raw_data_dir,
        settings.processed_data_dir,
        settings.chroma_db_dir,
        settings.benchmark_dir,
        settings.logs_dir,
    ]:
        (settings.project_root / dir_name).mkdir(parents=True, exist_ok=True)

    return settings


# Global settings instance
settings = load_settings()
