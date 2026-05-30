"""Document loader module.

Loads documents from various file formats (PDF, TXT, etc.) and
extracts text content while preserving metadata.
"""

import logging
from pathlib import Path
from typing import Optional
from datetime import datetime

from pypdf import PdfReader

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = {".pdf", ".txt"}


def extract_text_from_pdf(filepath: str) -> str:
    """Extract text from PDF file.

    Args:
        filepath: Path to PDF file

    Returns:
        Extracted text content

    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If PDF cannot be read
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {filepath}")

    try:
        reader = PdfReader(filepath)
        text_content = []
        for page in reader.pages:
            text_content.append(page.extract_text())
        return "\n".join(text_content)
    except Exception as e:
        logger.error(f"Failed to extract text from PDF {filepath}: {str(e)}")
        raise ValueError(f"Cannot read PDF file {filepath}: {str(e)}")


def extract_text_from_txt(filepath: str) -> str:
    """Extract text from text file.

    Args:
        filepath: Path to TXT file

    Returns:
        Text content

    Raises:
        FileNotFoundError: If file does not exist
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Text file not found: {filepath}")

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.error(f"Failed to read text file {filepath}: {str(e)}")
        raise


def load_document(filepath: str) -> dict:
    """Load a document from file.

    Args:
        filepath: Path to document file

    Returns:
        Dictionary with content, metadata, and source info

    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If file format is not supported
    """
    path = Path(filepath)

    if path.suffix.lower() not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported file format: {path.suffix}. "
            f"Supported formats: {', '.join(SUPPORTED_FORMATS)}"
        )

    if path.suffix.lower() == ".pdf":
        content = extract_text_from_pdf(filepath)
    elif path.suffix.lower() == ".txt":
        content = extract_text_from_txt(filepath)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    document = {
        "id": path.stem,
        "title": path.stem,
        "content": content,
        "source": str(path.absolute()),
        "filename": path.name,
        "file_size": path.stat().st_size,
        "created_at": datetime.fromtimestamp(path.stat().st_ctime),
        "modified_at": datetime.fromtimestamp(path.stat().st_mtime),
        "metadata": {
            "format": path.suffix.lower(),
            "num_chars": len(content),
        },
    }

    logger.info(
        f"Loaded document: id={document['id']}, size={len(content)} chars, "
        f"source={path.name}"
    )
    return document


def load_from_directory(directory: str, file_pattern: str = "*") -> list[dict]:
    """Load documents from a directory.

    Args:
        directory: Directory path containing documents
        file_pattern: File pattern to match (e.g., "*.txt", "*.pdf")

    Returns:
        List of loaded documents

    Raises:
        FileNotFoundError: If directory does not exist
    """
    dir_path = Path(directory)

    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    if not dir_path.is_dir():
        raise ValueError(f"Path is not a directory: {directory}")

    documents = []
    errors = []

    for file_path in dir_path.glob(file_pattern):
        if file_path.suffix.lower() not in SUPPORTED_FORMATS:
            continue

        try:
            doc = load_document(str(file_path))
            documents.append(doc)
        except Exception as e:
            error_msg = f"Failed to load {file_path.name}: {str(e)}"
            logger.warning(error_msg)
            errors.append(error_msg)

    logger.info(
        f"Loaded {len(documents)} documents from {directory}. "
        f"Errors: {len(errors)}"
    )

    if errors:
        logger.warning(f"Loading errors: {errors}")

    return documents
