# Knowledge Base Population Guide

## Overview

Your News Intelligence Assistant now supports **semi-automatic knowledge base population from your MySQL database**. The system can load 260+ CNN news articles from `news_analysis_database_v3.dimArticle` and automatically chunk, embed, and index them into ChromaDB for semantic search.

## Current Status

✅ **ChromaDB**: Empty (0 chunks) and ready for population
✅ **SQL Connection**: Configured in `.env` → `mysql+pymysql://admin:gvce924b@192.168.8.98:3306/news_analysis_database_v3`
✅ **Database**: Contains 260 news articles in `dimArticle` table
✅ **CLI Tools**: Created for population and management

## Quick Start

### 1. Start LM Studio
Before populating, ensure LM Studio is running with both models:
- Chat Model: `gemma-4-e4b` (for generation)
- Embedding Model: `embeddinggemma-300M-GGUF` (for embeddings)

LM Studio should be accessible at: http://localhost:1234/v1

### 2. Check Database Connection
```bash
python cli.py load-articles
```
This displays first 10 articles from your database.

### 3. Populate Knowledge Base
```bash
python cli.py populate-kb --clear
```
This will:
- Load all 260 articles from MySQL
- Split them into chunks (512 chars, 50 char overlap)
- Generate embeddings for each chunk
- Index into ChromaDB
- Create topic metadata

Expected time: ~3-5 minutes (depends on LM Studio embedding speed)

### 4. Verify Population
```bash
python cli.py kb-status
```
Shows ChromaDB statistics.

## Advanced Usage

### Populate with Topic Filtering
```bash
# Load articles only for a specific topic
python cli.py load-articles --topic-id 1
```

### Incremental Updates
Add only new articles without clearing existing index:
```bash
python cli.py populate-kb --incremental
```

### Check Article Count
```bash
python cli.py load-articles
```

## Architecture

### New Modules

1. **`ingestion/sql_loader.py`** - SQLArticleLoader class
   - Connects to MySQL database
   - Loads articles with metadata (title, content, date, url, topic)
   - Supports filtering by topic or date
   - Handles connection pooling

2. **`ingestion/populate.py`** - Population functions
   - `populate_knowledge_base_from_sql()` - Full population with optional clearing
   - `populate_knowledge_base_incremental()` - Incremental updates
   - Orchestrates: SQL load → Chunking → Embedding → Indexing

3. **`cli.py`** - Command-line interface
   - `populate-kb` - Populate/repopulate knowledge base
   - `kb-status` - Show ChromaDB status
   - `load-articles` - Preview articles from database

### Data Flow
```
MySQL dimArticle table
        ↓
SQLArticleLoader.load_all_articles()
        ↓
convert_sql_articles_to_documents()
        ↓
chunk_batch() [ingestion/chunker.py]
        ↓
embed_batch() [core/embeddings.py]
        ↓
ChromaDB index_chunks()
        ↓
ChromaDB (news_corpus collection)
```

## Metadata Preserved

Each indexed chunk retains source metadata:
- `source`: "MySQL news_analysis_database_v3.dimArticle"
- `filename`: "article_<id>.txt"
- `url`: Original CNN article URL
- `article_id`: Database article ID
- `date`: Publication date
- `topic`: Topic name (from dimTopic mapping)
- `created_at`: When article was added to database

## Troubleshooting

### "Failed to connect to database"
- Ensure MySQL server is running
- Check DATABASE_URL in `.env` is correct
- Verify username/password and network access

### "No module named 'pymysql'"
- Run: `pip install pymysql sqlalchemy click`

### "Failed to embed batch"
- Ensure LM Studio is running on http://localhost:1234/v1
- Verify embedding model `embeddinggemma-300M-GGUF` is loaded in LM Studio
- Check `LM_STUDIO_BASE_URL` and `EMBEDDING_MODEL` in `.env`

### "Knowledge base is empty after populate-kb"
- Check LM Studio embedding model is working
- Review logs: `logs/NewsAssistant.log`
- Try: `python cli.py kb-status` to see what's happening

## Configuration

Settings in `.env`:
```env
# Database
DATABASE_URL=mysql+pymysql://admin:gvce924b@192.168.8.98:3306/news_analysis_database_v3

# Chunking
CHUNK_SIZE=512            # Characters per chunk
CHUNK_OVERLAP=50          # Overlap between chunks

# ChromaDB
CHROMA_DISTANCE_METRIC=cosine   # Similarity metric
TOP_K_RETRIEVAL=5               # Default results per search

# LM Studio
LM_STUDIO_BASE_URL=http://localhost:1234/v1
EMBEDDING_MODEL=text-embedding-embeddinggemma-300m
```

## Performance Notes

- **SQL Loading**: ~1-2 seconds for 260 articles
- **Chunking**: ~0.5 seconds for ~4,000 chunks
- **Embedding**: ~2-4 minutes (network dependent on LM Studio)
- **Total Population**: ~3-5 minutes first time
- **Incremental Updates**: Depends on number of new articles

## Next Steps

1. ✅ Ensure LM Studio is running
2. ✅ Run `python cli.py load-articles` to verify database connection
3. ✅ Run `python cli.py populate-kb --clear` to populate knowledge base
4. ✅ Use Streamlit Chat page with RAG enabled to query the corpus
5. ✅ Run `python cli.py kb-status` to verify

## Example: After Population

Once populated, the Knowledge Base page will show:
- **Total Chunks**: ~4,000 chunks
- **Collection**: news_corpus
- **Distance Metric**: cosine
- **Chunk Statistics**: avg size, min/max sizes, total characters
- **Documents by Topic**: breakdown by category

The Chat page can then:
- Retrieve relevant chunks for queries
- Show source articles and URLs
- Ground responses in the actual CNN article content
