# Quick Reference: Tools Module

## Module Location
```
NewsAssistant/
├── core/
│   ├── __init__.py          (exports tools)
│   └── tools.py             (implementation - 430+ lines)
├── test_tools.py            (unit tests)
└── examples_tools_usage.py  (integration examples)
```

## Import Styles

### Style 1: Registry Import
```python
from core import create_default_tools

registry = create_default_tools()
tool = registry.get_tool("search_by_topic")
result = tool("query")
```

### Style 2: Direct Import
```python
from core import search_by_topic, filter_by_date, count_keyword_mentions, corpus_statistics

result = search_by_topic("topic")
```

### Style 3: Module Import
```python
from core.tools import ToolRegistry, create_default_tools

registry = create_default_tools()
```

## Tool Signatures

### 1. search_by_topic
```python
def search_by_topic(topic: str) -> dict
# Returns: {"topic", "num_results", "results", "metadata"}
```

### 2. filter_by_date
```python
def filter_by_date(start_date: str, end_date: str) -> dict
# Format: "YYYY-MM-DD"
# Returns: {"start_date", "end_date", "num_results", "results", "metadata"}
```

### 3. count_keyword_mentions
```python
def count_keyword_mentions(keyword: str) -> dict
# Returns: {"keyword", "total_mentions", "num_documents", "document_distribution", "top_chunks", "metadata"}
```

### 4. corpus_statistics
```python
def corpus_statistics() -> dict
# Returns: {"num_documents", "num_chunks", "total_content_length", "avg_chunk_length", "min_chunk_length", "max_chunk_length", "source_distribution", "metadata"}
```

## Error Handling

```python
from core.tools import search_by_topic

try:
    result = search_by_topic("climate change")
except ValueError as e:
    # Input validation failed
    # Possible causes:
    # - Empty or whitespace-only topic
    # - Invalid date format
    # - Invalid date range
    print(f"Validation error: {e}")
except RuntimeError as e:
    # Runtime error
    # Possible causes:
    # - Knowledge base not initialized
    # - Collection not built
    print(f"Runtime error: {e}")
except Exception as e:
    # Other exceptions
    print(f"Unexpected error: {e}")
```

## Registry API

```python
registry = ToolRegistry()

# Register a tool
registry.register(
    name="my_tool",
    func=my_function,
    description="My tool description"
)

# Get a tool
tool = registry.get_tool("my_tool")
if tool:
    result = tool(arg1, arg2)

# List all tools
tools = registry.list_tools()
for tool_info in tools:
    print(f"{tool_info['name']}: {tool_info['description']}")
```

## Common Patterns

### Pattern 1: Search by Topic
```python
from core import search_by_topic

results = search_by_topic("renewable energy")
for doc in results['results']:
    print(f"{doc['rank']}. {doc['source']} ({doc['similarity']:.0%})")
```

### Pattern 2: Date Range Filter
```python
from core import filter_by_date

docs = filter_by_date("2024-01-01", "2024-03-31")
print(f"Found {docs['num_results']} documents in Q1")
```

### Pattern 3: Keyword Analysis
```python
from core import count_keyword_mentions

stats = count_keyword_mentions("tariff")
print(f"'{stats['keyword']}' mentioned {stats['total_mentions']} times")
print(f"Appears in {stats['num_documents']} documents")

for doc_id, dist in stats['document_distribution'].items():
    print(f"  {dist['filename']}: {dist['mention_count']} mentions")
```

### Pattern 4: Corpus Overview
```python
from core import corpus_statistics

stats = corpus_statistics()
print(f"Corpus: {stats['num_documents']} docs, {stats['num_chunks']} chunks")
print(f"Size: {stats['total_content_length']} chars")
print(f"Avg chunk: {stats['avg_chunk_length']:.0f} chars")
print(f"Sources: {', '.join(stats['source_distribution'].keys())}")
```

## Agent Integration

```python
from core import create_default_tools

def agent_setup():
    registry = create_default_tools()
    return registry

def agent_execute(registry, tool_name, tool_args):
    tool = registry.get_tool(tool_name)
    if tool is None:
        return {"error": f"Tool not found: {tool_name}"}
    
    try:
        result = tool(**tool_args)
        return {"success": True, "data": result}
    except (ValueError, RuntimeError) as e:
        return {"error": str(e)}
```

## Chat Integration

```python
from core import create_default_tools

registry = create_default_tools()

def process_chat_message(user_message):
    # Determine which tool to use
    if "search" in user_message or "find" in user_message:
        tool = registry.get_tool("search_by_topic")
        # Extract topic and execute
    elif "when" in user_message:
        tool = registry.get_tool("filter_by_date")
        # Extract dates and execute
    elif "count" in user_message or "how many" in user_message:
        tool = registry.get_tool("count_keyword_mentions")
        # Extract keyword and execute
    else:
        tool = registry.get_tool("corpus_statistics")
        # Return corpus overview
```

## Troubleshooting

### Problem: "Collection not initialized"
**Solution:** Build knowledge base first
```python
from ingestion.indexer import initialize_chroma_db
initialize_chroma_db(db_path, embedding_base_url, embedding_model)
```

### Problem: "Topic cannot be empty"
**Solution:** Validate input before calling tool
```python
topic = user_input.strip()
if not topic:
    print("Please provide a topic")
else:
    result = search_by_topic(topic)
```

### Problem: "Invalid date format"
**Solution:** Use ISO format (YYYY-MM-DD)
```python
from datetime import date
start = date(2024, 1, 1).isoformat()  # "2024-01-01"
end = date(2024, 12, 31).isoformat()   # "2024-12-31"
result = filter_by_date(start, end)
```

### Problem: Tool not found
**Solution:** Check tool name spelling
```python
registry = create_default_tools()
for tool in registry.list_tools():
    print(tool['name'])  # Print available tools
```

## Performance Tips

1. **Caching:** Cache corpus_statistics() result
2. **Filtering:** Use filter_by_date() before search_by_topic()
3. **Batch:** Use registry for multiple operations
4. **Async:** Consider async wrappers for long operations

## Testing

Run tests:
```bash
python test_tools.py
python examples_tools_usage.py
```

## Documentation Files

- **TOOLS_DOCUMENTATION.md** — Complete API reference
- **TOOLS_IMPLEMENTATION_SUMMARY.md** — Implementation details
- **TOOLS_COMPLETE.md** — Final completion report
- **examples_tools_usage.py** — Usage patterns and examples
- **test_tools.py** — Unit tests

## Support

For issues or questions:
1. Check error message
2. Review usage examples
3. Check test cases
4. Review documentation
