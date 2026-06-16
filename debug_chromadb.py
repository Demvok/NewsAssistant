#!/usr/bin/env python
"""Check full metadata including all fields from ChromaDB."""
import os
import chromadb
import json

db_path = os.path.join("data", "chroma_db")
client = chromadb.PersistentClient(path=db_path)

collections = client.list_collections()
collection_name = collections[0].name if collections else "news_corpus"
collection = client.get_collection(name=collection_name)

print(f"Total chunks: {collection.count()}")

# Get raw results without filtering
results = collection.get(limit=1, include=["documents", "metadatas", "embeddings"])

print("\nRaw chunk data (first chunk):")
print(f"Document ID: {results['ids'][0]}")
print(f"Metadata: {json.dumps(results['metadatas'][0], indent=2)}")
print(f"Document text (first 100 chars): {results['documents'][0][:100] if results['documents'] else 'NONE'}")

# Try querying with include parameter
print("\n--- Query with all include options ---")
results2 = collection.query(
    query_texts=["trade war"],
    n_results=3,
    include=["documents", "metadatas", "distances"]
)

print(f"Query returned {len(results2['ids'][0])} results")
if results2['metadatas'][0]:
    print(f"First result metadata keys: {list(results2['metadatas'][0][0].keys())}")
    print(f"First result metadata: {json.dumps(results2['metadatas'][0][0], indent=2)}")
