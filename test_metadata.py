#!/usr/bin/env python
"""Inspect ChromaDB metadata to verify chunk_preview and article_id are stored."""
import os
import chromadb

db_path = os.path.join("data", "chroma_db")
client = chromadb.PersistentClient(path=db_path)

# List existing collections
try:
    collections = client.list_collections()
    collection_names = [c.name for c in collections]
    print(f"Existing collections: {collection_names}")
    
    if not collection_names:
        print("\nNo collections found. Need to rebuild index first.")
        # Try to rebuild
        from ingestion.indexer import populate_knowledge_base_from_sql
        print("Rebuilding knowledge base from SQL...")
        populate_knowledge_base_from_sql()
        print("Knowledge base rebuilt!")
        collection_names = [c.name for c in client.list_collections()]
    
    # Get the first collection (usually "news_corpus")
    collection_name = collection_names[0] if collection_names else "news_corpus"
    print(f"\nUsing collection: {collection_name}")
    collection = client.get_collection(name=collection_name)
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
print(f"Total chunks in ChromaDB: {collection.count()}")

# Fetch a few chunks to inspect metadata
results = collection.get(limit=3, include=["documents", "metadatas"])
for i, (doc, meta) in enumerate(zip(results["documents"], results["metadatas"])):
    print(f"\n--- Chunk {i} ---")
    print(f"Text preview (first 100 chars): {doc[:100] if doc else 'NONE'}")
    print(f"Metadata keys: {list(meta.keys())}")
    print(f"chunk_preview exists: {'chunk_preview' in meta}")
    if 'chunk_preview' in meta:
        print(f"chunk_preview (first 100 chars): {meta['chunk_preview'][:100]}")
    else:
        print("WARNING: chunk_preview MISSING from metadata!")
    
    print(f"article_id: {meta.get('article_id', 'MISSING')}")
    print(f"filename: {meta.get('filename', 'MISSING')}")
    print(f"source: {meta.get('source', 'MISSING')}")
