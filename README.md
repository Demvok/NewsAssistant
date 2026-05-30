# AI News Intelligence Assistant

A full-stack generative AI application for analyzing CNN news articles using local LLMs, retrieval-augmented generation (RAG), multi-agent reasoning, and benchmark evaluation.

## Overview

This is a comprehensive course project demonstrating modern AI engineering practices:

- **Local Model Integration**: Uses LM Studio with local models (gemma-4-e4b for chat, embeddinggemma-300M-GGUF for embeddings)
- **Retrieval-Augmented Generation**: Grounds responses in a curated CNN news corpus
- **Knowledge Base Management**: Document ingestion, chunking, embedding, and semantic search via ChromaDB
- **Multi-Agent Reasoning**: Orchestrated pipeline with analyzer, critic, and synthesizer agents
- **Tool Calling**: Registry of analytical tools for corpus exploration
- **Benchmark Evaluation**: LLM-as-Judge evaluation on structured test datasets
- **Parameter Experiments**: Controlled experiments comparing generation parameters

## Core Topics

The application analyzes a corpus focused on three key themes:

- 🌱 **Green Energy** - Renewable energy, climate initiatives, sustainability
- 💼 **U.S. Trade War** - Trade policies, tariffs, economic impacts
- 🔕 **Social Media Censorship** - Content moderation, freedom of speech

## Project Structure

```
NewsAssistant/
├── app.py                 # Main Streamlit entry point
├── config.py              # Configuration and environment settings
├── requirements.txt       # Python dependencies
├── .env.example           # Environment variables template
├── README.md              # This file
│
├── core/                  # Core application logic
│   ├── llm_client.py      # LM Studio integration
│   ├── embeddings.py      # Embedding model client
│   ├── rag.py             # RAG pipeline
│   ├── agents.py          # Multi-agent orchestration
│   ├── tools.py           # Tool registry and implementations
│   ├── evaluator.py       # Benchmark evaluation
│   ├── schemas.py         # Data models and types
│   └── utils.py           # Utility functions
│
├── ingestion/             # Document ingestion pipeline
│   ├── loader.py          # Document loading from files
│   ├── chunker.py         # Text chunking with overlap
│   └── indexer.py         # ChromaDB indexing
│
├── pages/                 # Streamlit application pages
│   ├── 1_Chat.py          # Main chat interface
│   ├── 2_Knowledge_Base.py # Document management
│   ├── 3_Experiments.py    # Parameter experiments
│   └── 4_Benchmark.py      # Evaluation dashboard
│
├── data/                  # Data storage
│   ├── raw/               # Raw documents
│   ├── processed/         # Processed documents
│   ├── chroma_db/         # ChromaDB vector store
│   └── benchmark/         # Benchmark test cases
│
└── logs/                  # Application logs
```

## Requirements

- **Python**: 3.13.7
- **LM Studio**: Running locally with models loaded
  - Chat Model: `gemma-4-e4b`
  - Embedding Model: `embeddinggemma-300M-GGUF`

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd NewsAssistant
```

### 2. Create virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your settings
# Ensure LM_STUDIO_BASE_URL points to your local LM Studio instance
```

### 5. Prepare data directories

```bash
mkdir -p data/raw data/processed data/chroma_db data/benchmark
mkdir -p logs
```

### 6. Start LM Studio

Ensure LM Studio is running and both models are loaded:
- `gemma-4-e4b` on port 1234 (or configured endpoint)
- `embeddinggemma-300M-GGUF` for embeddings

## Usage

### Running the Application

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

### Pages

#### 1. 🗣️ Chat
Main conversation interface with the AI assistant.

Features:
- Multi-turn conversation
- Toggle RAG (Retrieval-Augmented Generation)
- Toggle multi-agent reasoning
- Adjust generation parameters (temperature, top-p, top-k)
- View retrieval context when RAG is enabled
- Display agent reasoning traces

#### 2. 📚 Knowledge Base
Manage and search the document corpus.

Features:
- Upload documents (PDF, TXT, etc.)
- Automatic chunking and embedding
- Semantic search over the corpus
- View retrieval results with similarity scores
- Corpus statistics and analytics
- Rebuild index operations

#### 3. 🧪 Experiments
Run controlled experiments with different parameters.

Features:
- Compare generation quality across temperatures
- Repeated sampling for consistency analysis
- Output length tracking
- Diversity metrics
- Visual comparisons and tables
- Export results for reporting

#### 4. 📊 Benchmark
Evaluate assistant performance on test datasets.

Features:
- Load structured test cases
- Batch evaluation on all cases
- LLM-as-Judge scoring
- Result aggregation by category
- Performance metrics and summaries
- Export results

## Architecture

### Layered Design

The application follows a clean layered architecture:

```
┌─────────────────────────────────────┐
│  UI Layer (Streamlit Pages)         │
├─────────────────────────────────────┤
│  Application Core                   │
│  (RAG, Agents, Tools, Evaluation)   │
├─────────────────────────────────────┤
│  Infrastructure                     │
│  (LLM Client, Embeddings, ChromaDB) │
├─────────────────────────────────────┤
│  Data Layer                         │
│  (Documents, Vectors, Configs)      │
└─────────────────────────────────────┘
```

### Key Components

**LLM Client** (`core/llm_client.py`)
- Manages LM Studio API interactions
- Handles timeouts and retries
- Centralizes model configuration

**Embeddings** (`core/embeddings.py`)
- Generates text embeddings
- Batch processing support
- Local model integration

**RAG Pipeline** (`core/rag.py`)
- Query embedding
- ChromaDB retrieval
- Context injection
- Citation generation

**Multi-Agent** (`core/agents.py`)
- Agent role definitions
- Orchestration logic
- State management
- Trace generation

**Tools** (`core/tools.py`)
- Tool registry pattern
- Corpus filtering by topic/date
- Keyword counting
- Statistics aggregation

**Evaluation** (`core/evaluator.py`)
- LLM-as-Judge scoring
- Batch test execution
- Result aggregation
- Category-based analysis

## Configuration

All settings are managed through environment variables in `.env`:

```bash
# LM Studio API
LM_STUDIO_BASE_URL=http://localhost:1234/v1
CHAT_MODEL=gemma-4-e4b
EMBEDDING_MODEL=embeddinggemma-300M-GGUF

# Generation
DEFAULT_TEMPERATURE=0.7
DEFAULT_TOP_P=0.95
DEFAULT_TOP_K=40
MAX_TOKENS=512

# RAG
RAG_ENABLED_DEFAULT=true
TOP_K_RETRIEVAL=5
CHUNK_SIZE=512
CHUNK_OVERLAP=50

# Multi-agent
MULTI_AGENT_ENABLED_DEFAULT=false

# Logging
LOG_LEVEL=INFO
```

## Data Format

### Test Cases (Benchmark)

JSON format for benchmark test cases:

```json
{
  "id": "case-001",
  "category": "green_energy",
  "prompt": "What are the latest developments in solar energy?",
  "expected_answer": "Solar energy advancements...",
  "evaluation_criteria": "Factual accuracy, comprehensiveness, relevance"
}
```

### Documents

Supported formats:
- PDF (.pdf)
- Plain text (.txt)
- Word documents (.docx)

Metadata preservation:
- Source filename
- Document title
- Publication date (if available)
- Topic/category tags

## Development

### Project Structure Philosophy

- **Modularity**: Clear separation of concerns across layers
- **Testability**: Small, focused functions with clear contracts
- **Maintainability**: Descriptive names, docstrings, type hints
- **Extensibility**: Tool registry, agent roles, and configurable pipelines

### Adding New Tools

```python
# In core/tools.py
def tool_new_analysis(param: str) -> dict:
    """Analyze corpus with new logic."""
    pass

# Register in create_default_tools()
registry.register("new_analysis", tool_new_analysis, "Analysis description")
```

### Adding New Pages

1. Create `pages/X_PageName.py`
2. Implement `render_page_name()` function
3. Streamlit auto-discovers the page

### Logging

The application uses Python's standard logging. Important events are logged to both console and files in `logs/`.

```python
import logging
logger = logging.getLogger(__name__)
logger.info("Event message")
logger.error("Error message")
```

## Performance Considerations

- **Embeddings**: Pre-computed and cached in ChromaDB
- **Retrieval**: Top-K limiting for efficiency
- **Chunking**: Balance between context size and search precision
- **Multi-agent**: Sequential by design (can be parallelized)

## Troubleshooting

### LM Studio Connection Issues

1. Verify LM Studio is running: `curl http://localhost:1234/v1/models`
2. Check `LM_STUDIO_BASE_URL` in `.env`
3. Ensure models are loaded

### Empty Search Results

1. Check if documents are loaded in Knowledge Base
2. Rebuild the ChromaDB index
3. Verify chunk size and overlap settings

### Low Generation Quality

1. Adjust temperature (lower = more focused, higher = more creative)
2. Increase `top_k_retrieval` for more context
3. Improve document quality or coverage

## Future Enhancements

- [ ] Web UI deployment
- [ ] Async processing for large batches
- [ ] GraphQL/REST API
- [ ] Multi-model support (Claude, Llama, etc.)
- [ ] Advanced visualization dashboards
- [ ] Export to multiple formats (PDF, DOCX, HTML)
- [ ] Fine-tuning pipelines
- [ ] Caching and memoization

## References

- [Streamlit Documentation](https://docs.streamlit.io/)
- [LangChain](https://python.langchain.com/)
- [ChromaDB](https://www.trychroma.com/)
- [LM Studio](https://lmstudio.ai/)

## License

This project is for educational purposes.

## Author

Created as a comprehensive course project demonstrating modern AI engineering practices.

---

**Last Updated**: May 2026
