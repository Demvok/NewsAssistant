"""Utility functions for the application.

Provides common helpers for logging, text processing, file I/O,
and other cross-cutting concerns.
"""

import logging
import json
import re
from pathlib import Path
from typing import Any


def setup_logging(log_dir: str = "logs", level: str = "INFO") -> logging.Logger:
    """Configure logging for the application.
    
    Args:
        log_dir: Directory for log files
        level: Logging level
        
    Returns:
        Configured logger instance
    """
    # Create log directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # Create logger
    logger = logging.getLogger("NewsAssistant")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level.upper()))
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler(log_path / "app.log")
    file_handler.setLevel(getattr(logging, level.upper()))
    file_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    return logger


def normalize_text(text: str) -> str:
    """Normalize text for processing.
    
    Args:
        text: Raw text input
        
    Returns:
        Cleaned and normalized text
    """
    if not text:
        return ""
    
    # Remove extra whitespace
    text = " ".join(text.split())
    
    # Remove special characters but keep punctuation
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    
    return text.strip()


def format_output(data: Any, format_type: str = "text") -> str:
    """Format output for display.
    
    Args:
        data: Data to format
        format_type: Output format ('text', 'json', 'markdown')
        
    Returns:
        Formatted string
    """
    if format_type == "json":
        return json.dumps(data, indent=2)
    elif format_type == "markdown":
        if isinstance(data, dict):
            lines = ["| Key | Value |", "|-----|-------|"]
            for key, value in data.items():
                lines.append(f"| {key} | {value} |")
            return "\n".join(lines)
        return str(data)
    else:  # text
        return str(data)


def save_json(data: dict, filepath: str) -> None:
    """Save data to JSON file.
    
    Args:
        data: Data to save
        filepath: Output file path
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(filepath: str) -> dict:
    """Load data from JSON file.
    
    Args:
        filepath: Input file path
        
    Returns:
        Loaded data
    """
    path = Path(filepath)
    
    if not path.exists():
        return {}
    
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
