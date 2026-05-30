# Tools Module Documentation

## Overview

The `core/tools.py` module implements a **tool registry** and provides four core analytical tools for the News Intelligence Assistant. These tools enable the assistant to perform targeted queries and analysis on the news corpus.

## Architecture

### ToolRegistry Class

A registry pattern implementation for managing callable tools.

```python
registry = ToolRegistry()
registry.register("tool_name", tool_function, "Description")
tool = registry.get_tool("tool_name")
all_tools = registry.list_tools()
```

**Methods:**
- `register(name, func, description)` — Register a tool function
- `get_tool(name)` — Retrieve a tool by name
- `list_tools()` — List all available tools with metadata

---

## Tools

### 1. search_by_topic

**Purpose:** Semantic search for documents by topic

**Signature:**
```python
def search_by_topic(topic: str) -> dict
```

**Parameters:**
- `topic` (str): Topic description or keyword (e.g., "green energy", "trade war", "censorship")

**Returns:**
```python
{
    "topic": str,
    "num_results": int,
    "results": [
        {
            "rank": int,
            "content": str,
            "similarity": float,  # 0-1
            "source": str,
            "filename": str,
            "document_id": str
        },
        ...
    ],
    "metadata": {
        "search_type": "semantic",
        "timestamp": str  # ISO format
    }
}
```

**Behavior:**
- Generates embedding for the topic query
- Performs semantic similarity search against indexed chunks
- Returns top 10 results sorted by relevance
- Includes document metadata and similarity scores

**Exceptions:**
- `ValueError` — If topic is empty or whitespace-only
- `RuntimeError` — If knowledge base not initialized

**Example:**
```python
result = search_by_topic("climate change and renewable energy")
print(f"Found {result['num_results']} relevant documents")
for doc in result['results']:
    print(f"  {doc['rank']}. {doc['source']} (relevance: {doc['similarity']:.2%})")
```

---

### 2. filter_by_date

**Purpose:** Filter documents by publication date range

**Signature:**
```python
def filter_by_date(start_date: str, end_date: str) -> dict
```

**Parameters:**
- `start_date` (str): Start date (ISO format: YYYY-MM-DD)
- `end_date` (str): End date (ISO format: YYYY-MM-DD)

**Returns:**
```python
{
    "start_date": str,
    "end_date": str,
    "num_results": int,
    "results": [
        {
            "content": str,
            "source": str,
            "filename": str,
            "document_id": str,
            "date": str
        },
        ...
    ],
    "metadata": {
        "filter_type": "date_range",
        "timestamp": str  # ISO format
    }
}
```

**Behavior:**
- Retrieves all indexed chunks
- Filters by document metadata `date` or `published_date` fields
- Returns documents within the specified range (inclusive)
- Requires document metadata to include date information

**Exceptions:**
- `ValueError` — If dates not in ISO format or start_date > end_date
- `RuntimeError` — If knowledge base not initialized

**Example:**
```python
result = filter_by_date("2024-01-01", "2024-03-31")
print(f"Found {result['num_results']} documents in Q1 2024")
```

---

### 3. count_keyword_mentions

**Purpose:** Count keyword occurrences in the corpus

**Signature:**
```python
def count_keyword_mentions(keyword: str) -> dict
```

**Parameters:**
- `keyword` (str): Keyword to search for (case-insensitive)

**Returns:**
```python
{
    "keyword": str,
    "total_mentions": int,
    "num_documents": int,
    "document_distribution": {
        "doc_id": {
            "filename": str,
            "mention_count": int,
            "chunk_count": int
        },
        ...
    },
    "top_chunks": [
        {
            "content_preview": str,
            "mention_count": int,
            "source": str,
            "filename": str,
            "document_id": str
        },
        ...
    ],
    "metadata": {
        "search_type": "keyword",
        "timestamp": str  # ISO format
    }
}
```

**Behavior:**
- Performs case-insensitive keyword search across entire corpus
- Counts total mentions across all chunks
- Tracks distribution by document
- Returns top 20 chunks by mention count
- Useful for frequency analysis and trending topics

**Exceptions:**
- `ValueError` — If keyword is empty or whitespace-only
- `RuntimeError` — If knowledge base not initialized

**Example:**
```python
result = count_keyword_mentions("tariff")
print(f"Keyword 'tariff' mentioned {result['total_mentions']} times")
print(f"Found in {result['num_documents']} documents")
for doc_id, dist in result['document_distribution'].items():
    print(f"  {dist['filename']}: {dist['mention_count']} mentions")
```

---

### 4. corpus_statistics

**Purpose:** Get comprehensive statistics about the indexed corpus

**Signature:**
```python
def corpus_statistics() -> dict
```

**Parameters:** None

**Returns:**
```python
{
    "num_documents": int,
    "num_chunks": int,
    "total_content_length": int,  # characters
    "avg_chunk_length": float,
    "min_chunk_length": int,
    "max_chunk_length": int,
    "source_distribution": {
        "source_name": int,  # count
        ...
    },
    "metadata": {
        "timestamp": str,  # ISO format
        "collection_name": str
    }
}
```

**Behavior:**
- Retrieves metadata from ChromaDB collection
- Analyzes chunk size distribution
- Tracks document and source statistics
- Computes aggregate metrics

**Exceptions:**
- `RuntimeError` — If knowledge base not initialized

**Example:**
```python
stats = corpus_statistics()
print(f"Corpus Summary:")
print(f"  Documents: {stats['num_documents']}")
print(f"  Chunks: {stats['num_chunks']}")
print(f"  Avg chunk size: {stats['avg_chunk_length']:.0f} characters")
print(f"  Sources: {', '.join(stats['source_distribution'].keys())}")
```

---

## Creating and Using the Tool Registry

### Basic Setup

```python
from core.tools import create_default_tools

# Create registry with all default tools
registry = create_default_tools()

# List available tools
for tool_info in registry.list_tools():
    print(f"{tool_info['name']}: {tool_info['description']}")

# Use a tool
search_tool = registry.get_tool("search_by_topic")
results = search_tool("climate change")
```

### Integration with Agents

```python
from core.tools import create_default_tools

registry = create_default_tools()

# Retrieve tool by name
tool = registry.get_tool("count_keyword_mentions")

# Execute tool
try:
    result = tool("sustainable energy")
    print(result)
except Exception as e:
    print(f"Tool error: {e}")
```

---

## Error Handling

All tools implement consistent error handling:

1. **Validation errors** — `ValueError` for invalid inputs
   - Empty or whitespace-only strings
   - Invalid date formats
   - Invalid date ranges

2. **Runtime errors** — `RuntimeError` for uninitialized state
   - Collection not initialized
   - Knowledge base not built

3. **Generic exceptions** — Logged and re-raised for unexpected errors

**Example:**
```python
from core.tools import search_by_topic

try:
    result = search_by_topic("test")
except ValueError as e:
    print(f"Invalid input: {e}")
except RuntimeError as e:
    print(f"System not initialized: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

---

## Logging

All tools include logging at appropriate levels:

- `DEBUG` — Detailed operation information
- `INFO` — Operation summaries (tool invocation, results count)
- `WARNING` — Non-critical issues (empty results, missing metadata)
- `ERROR` — Exceptions and failures

Check logs in `logs/` directory for detailed execution traces.

---

## Integration Points

### ChromaDB Integration
- Tools query the `news_corpus` collection
- Requires metadata: `document_id`, `source`, `filename`, optional `date`

### Embeddings Integration
- `search_by_topic` uses `embed_text()` for query embedding
- Requires embeddings client initialized via `ingestion.indexer`

### Metadata Requirements

For full functionality, ensure documents include:
- `document_id` — Unique document identifier
- `source` — Source URL or publication
- `filename` — Original filename
- `date` (optional) — Publication date in ISO format for date filtering
- Any custom metadata for topic categorization

---

## Performance Considerations

1. **search_by_topic** — O(n) similarity search, returns top 10
2. **filter_by_date** — O(n) scan + filter, lightweight
3. **count_keyword_mentions** — O(n) scan, case-insensitive matching
4. **corpus_statistics** — O(n) collection scan, computed once

For large corpora (>100k chunks), consider:
- Caching statistics results
- Using topic-specific filters before semantic search
- Batch operations for multiple queries

---

## Future Enhancements

Potential tools to add:
- `search_by_source` — Filter by document source
- `get_trending_topics` — Identify trending keywords over time
- `sentiment_analysis` — Analyze tone and sentiment
- `topic_clustering` — Cluster documents by topic
- `generate_summary` — Summarize corpus by topic
