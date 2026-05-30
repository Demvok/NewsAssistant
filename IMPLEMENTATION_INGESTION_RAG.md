## Implementation: Document Ingestion and RAG Modules

This document describes the implementation of the document ingestion, storage, and retrieval-augmented generation (RAG) modules for the AI News Intelligence Assistant.

### Implemented Modules

#### 1. **ingestion/loader.py** - Document Loading and Text Extraction
Loads documents from various file formats and extracts text content while preserving metadata.

**Key Functions:**
- `load_document(filepath)` - Load a single document and extract metadata
- `load_from_directory(directory, file_pattern)` - Batch load documents from a directory
- `extract_text_from_pdf(filepath)` - Extract text from PDF files using PyPDF
- `extract_text_from_txt(filepath)` - Extract text from plain text files

**Features:**
- Supports PDF and TXT file formats (easily extensible)
- Preserves document metadata (source, filename, creation date, file size)
- Robust error handling with detailed logging
- Batch loading with error tolerance

**Example Usage:**
```python
from ingestion.loader import load_document, load_from_directory

# Load single document
doc = load_document("path/to/article.pdf")

# Load all documents from directory
docs = load_from_directory("data/raw/", "*.pdf")
```

---

#### 2. **ingestion/chunker.py** - Text Chunking with Overlap
Splits documents into manageable chunks with configurable size and overlap, preserving chunk metadata and ordering.

**Key Functions:**
- `chunk_document(content, document_id, chunk_size, overlap, metadata)` - Split a single document
- `chunk_batch(documents, chunk_size, overlap)` - Chunk multiple documents
- `validate_chunks(chunks)` - Validate chunk integrity and consistency

**Features:**
- Configurable chunk size and overlap
- Preserves source metadata in each chunk
- Generates unique chunk IDs
- Tracks chunk order and position within source document
- Validates all chunks before use

**Chunk Structure:**
```python
{
    "id": "unique-uuid",
    "content": "chunk text...",
    "document_id": "source-doc-id",
    "chunk_order": 0,
    "start_char": 0,
    "end_char": 512,
    "metadata": {
        "source": "article.pdf",
        "chunk_size": 512,
        "is_last": False
    }
}
```

**Example Usage:**
```python
from ingestion.chunker import chunk_batch, validate_chunks

chunks = chunk_batch(
    documents=docs,
    chunk_size=512,
    overlap=50
)
validate_chunks(chunks)
```

---

#### 3. **ingestion/indexer.py** - ChromaDB Vector Storage
Manages embedding generation, vector storage in ChromaDB, and index rebuild operations.

**Key Functions:**
- `initialize_chroma_db(db_path, embedding_base_url, embedding_model)` - Initialize ChromaDB
- `index_chunks(chunks, ...)` - Add chunks to ChromaDB with embeddings
- `rebuild_index(documents, ...)` - Rebuild entire index from scratch
- `get_collection_stats()` - Get collection statistics
- `delete_collection()` - Clear the collection

**Features:**
- Uses embeddinggemma-300M-GGUF for local embeddings
- ChromaDB with persistent disk storage
- Cosine distance metric for similarity search
- Batch embedding generation for efficiency
- Preserves chunk metadata in ChromaDB

**Example Usage:**
```python
from ingestion.indexer import initialize_chroma_db, index_chunks, rebuild_index
from config import settings

# Initialize
initialize_chroma_db(
    db_path=str(settings.chroma_db_dir),
    embedding_base_url=settings.lm_studio.base_url,
    embedding_model=settings.lm_studio.embedding_model
)

# Index chunks
num_indexed = index_chunks(chunks, ...)

# Or rebuild entire index
stats = rebuild_index(documents, ...)
```

---

#### 4. **core/embeddings.py** - Embedding Generation (Enhanced)
Provides functions to generate embeddings using the local embedding model via LM Studio.

**Key Functions:**
- `get_embeddings_client(base_url, model_name)` - Initialize embeddings client
- `embed_text(text, client)` - Generate embedding for a single text
- `embed_batch(texts, client)` - Generate embeddings for multiple texts
- `reset_client()` - Reset cached client instance

**Features:**
- Uses LangChain's OpenAIEmbeddings with LM Studio
- Client caching for efficiency
- Global state management for consistent embeddings

---

#### 5. **core/rag.py** - Retrieval-Augmented Generation
Implements the complete RAG pipeline: query embedding, ChromaDB retrieval, context injection, and answer generation.

**Key Functions:**
- `retrieve_context(query, top_k, ...)` - Retrieve relevant chunks for a query
- `inject_context_into_prompt(query, context)` - Assemble prompt with context
- `format_answer_with_sources(answer, context)` - Format answer with citations
- `generate_with_rag(query, llm_client, ...)` - End-to-end RAG generation

**Features:**
- Semantic retrieval using embeddings
- Similarity score calculation (1 - cosine distance)
- Context formatting with source attribution
- Fallback to non-RAG if retrieval fails
- Comprehensive error handling
- Supports both direct LMStudioClient and LangChain LLM interfaces

**RAG Response Structure:**
```python
{
    "answer": "generated answer text...",
    "query": "original query",
    "context": [
        {
            "content": "retrieved chunk...",
            "similarity_score": 0.87,
            "source": "article.pdf",
            "filename": "article.pdf"
        },
        ...
    ],
    "num_sources": 5,
    "metadata": {
        "rag_enabled": True,
        "top_k": 5,
        "temperature": 0.7,
        "max_tokens": 512,
        "prompt_length": 2048,
        "answer_length": 256
    }
}
```

**Example Usage:**
```python
from core.rag import generate_with_rag
from core.llm_client import get_llm_client
from config import settings

# Initialize LLM client
llm_client = get_llm_client(
    base_url=settings.lm_studio.base_url,
    model_name=settings.lm_studio.chat_model
)

# Generate with RAG
result = generate_with_rag(
    query="What is the latest on green energy?",
    llm_client=llm_client,
    rag_enabled=True,
    top_k=5,
    embedding_base_url=settings.lm_studio.base_url,
    embedding_model=settings.lm_studio.embedding_model
)

print(result["answer"])
print(f"Sources: {result['num_sources']}")
```

---

### Configuration Integration

All modules respect the centralized configuration in `config.py`:

```python
# Chunking
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# ChromaDB
CHROMA_DB_DIR = "data/chroma_db"
TOP_K_RETRIEVAL = 5

# Embeddings & LLM
EMBEDDING_MODEL = "embeddinggemma-300M-GGUF"
CHAT_MODEL = "gemma-4-e4b"
LM_STUDIO_BASE_URL = "http://localhost:1234/v1"
```

---

### Data Flow Diagram

```
User Documents (PDF/TXT)
         ↓
    [LOADER] → Extract text + metadata
         ↓
    [CHUNKER] → Split with overlap, preserve metadata
         ↓
    [EMBEDDINGS] → Generate embeddings (embeddinggemma-300M-GGUF)
         ↓
    [INDEXER] → Store in ChromaDB with embeddings
         ↓
    [RETRIEVAL] → Query embedding + similarity search
         ↓
    [RAG] → Inject context + LLM generation
         ↓
    Answer with sources
```

---

### Error Handling

All modules include comprehensive error handling:

- **Loader**: FileNotFoundError, ValueError for unsupported formats
- **Chunker**: ValueError for invalid parameters, empty content handling
- **Indexer**: RuntimeError if ChromaDB not initialized, Exception logging
- **RAG**: ValueError for empty queries, RuntimeError if collection not available
- **Embeddings**: ValueError for empty text, RuntimeError if client not initialized

All errors are logged with context for debugging.

---

### Testing

Test files validate the implementation:

- `test_ingestion.py` - Basic chunking and configuration validation
- `test_loader_chunker.py` - Document loading and batch chunking

Run tests:
```bash
python test_ingestion.py
python test_loader_chunker.py
```

---

### Next Steps

The implemented modules are ready for integration with:

1. **Streamlit UI** (pages/2_Knowledge_Base.py) - Upload, ingest, and manage documents
2. **Chat System** (core/llm_client.py) - Already supports RAG generation
3. **Multi-agent System** (core/agents.py) - Can use RAG context for reasoning
4. **Benchmark & Experiments** - RAG can be toggled for evaluation

---

### Dependencies

Required packages (in requirements.txt):
- langchain
- langchain-community
- langchain-openai
- chromadb
- pypdf
- openai (for OpenAI-compatible API)

All dependencies are included in the project requirements.
