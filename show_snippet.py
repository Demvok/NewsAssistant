#!/usr/bin/env python
"""Search for 'green energy' and show first result's snippet."""
import os
import chromadb
from config import Settings
from core.embeddings import get_embeddings_client, embed_text

# Initialize config
settings = Settings()

# Initialize embeddings
get_embeddings_client(
    settings.lm_studio.base_url,
    settings.lm_studio.embedding_model
)

# Initialize ChromaDB
db_path = settings.rag.chromadb.persist_directory
client = chromadb.PersistentClient(path=db_path)
collection = client.get_collection(name="news_corpus")

# Search
query = "green energy"
print(f"Searching for: '{query}'")
print("=" * 80)

# Embed and search
query_emb = embed_text(query)
results = collection.query(
    query_embeddings=[query_emb],
    n_results=5,
    include=["documents", "metadatas", "distances"]
)

if results and results["documents"] and results["documents"][0]:
    # Get first result
    doc_text = results["documents"][0][0]
    metadata = results["metadatas"][0][0]
    distance = results["distances"][0][0]
    similarity = 1 - distance
    
    print(f"\nFirst result:")
    print(f"Document ID: {metadata.get('document_id', 'N/A')}")
    print(f"Article ID: {metadata.get('article_id', 'N/A')}")
    print(f"Filename: {metadata.get('filename', 'N/A')}")
    print(f"Similarity: {similarity:.3f}")
    
    # Show chunk_preview from metadata
    snippet = metadata.get('chunk_preview', '')
    print(f"\n--- chunk_preview in metadata (first 300 chars) ---")
    print(snippet[:300] if snippet else "EMPTY")
    
    # Show the actual chunk document from ChromaDB
    print(f"\n--- Actual chunk document in ChromaDB (first 300 chars) ---")
    print(doc_text[:300] if doc_text else "EMPTY")
    
    # Show chunk details
    print(f"\n--- Chunk metadata ---")
    print(f"start_char: {metadata.get('start_char', 'N/A')}")
    print(f"end_char: {metadata.get('end_char', 'N/A')}")
    print(f"chunk_order: {metadata.get('chunk_order', 'N/A')}")
    
else:
    print("No results found!")
