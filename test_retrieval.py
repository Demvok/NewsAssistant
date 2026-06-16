#!/usr/bin/env python
"""Test actual retrieval functionality - check if articles work and search works."""
import os
import sys
from config import settings

# Initialize embeddings client first
from core.embeddings import get_embeddings_client, embed_text
get_embeddings_client(
    settings.lm_studio_base_url,
    settings.embedding_model
)

# Direct ChromaDB access
import chromadb

db_path = os.path.join("data", "chroma_db")
client = chromadb.PersistentClient(path=db_path)

try:
    collections = client.list_collections()
    collection_name = collections[0].name if collections else None
    
    if not collection_name:
        print("ERROR: No ChromaDB collection found!")
        sys.exit(1)
    
    collection = client.get_collection(name=collection_name)
    print(f"[OK] Collection exists: {collection_name}")
    print(f"[OK] Total chunks: {collection.count()}")
    
    # Test 1: Get a raw chunk and see what metadata and content we have
    print("\n=== TEST 1: Raw chunk from ChromaDB ===")
    results = collection.get(limit=1, include=["documents", "metadatas"])
    if results["documents"]:
        chunk_text = results["documents"][0]
        chunk_meta = results["metadatas"][0]
        print(f"Chunk text (first 150 chars): {chunk_text[:150]}")
        print(f"Metadata keys: {list(chunk_meta.keys())}")
        for key in ['document_id', 'article_id', 'filename', 'chunk_preview']:
            print(f"  {key}: {chunk_meta.get(key, 'MISSING')}")
    
    # Test 2: Try a real search query
    print("\n=== TEST 2: Semantic search query ===")
    search_query = "trade war"
    print(f"Search query: '{search_query}'")
    
    query_emb = embed_text(search_query)
    print(f"[OK] Query embedded (dim={len(query_emb)})")
    
    search_results = collection.query(
        query_embeddings=[query_emb],
        n_results=3,
        include=["documents", "metadatas", "distances"]
    )
    
    if search_results["documents"] and search_results["documents"][0]:
        for i, (doc, meta, dist) in enumerate(zip(
            search_results["documents"][0],
            search_results["metadatas"][0],
            search_results["distances"][0]
        )):
            similarity = 1 - dist
            print(f"\n--- Result {i+1} (similarity {similarity:.3f}) ---")
            print(f"Article ID from metadata: {meta.get('article_id', 'MISSING')}")
            print(f"Filename: {meta.get('filename', 'MISSING')}")
            print(f"Chunk text (first 200 chars): {doc[:200] if doc else 'EMPTY'}")
            chunk_preview = meta.get('chunk_preview', 'MISSING')
            if chunk_preview and chunk_preview != 'MISSING':
                print(f"Chunk preview (first 150 chars): {chunk_preview[:150]}")
            else:
                print(f"Chunk preview field: {chunk_preview}")
    
    # Test 3: Try to get full article from SQL database
    print("\n=== TEST 3: Load article from SQL DB ===")
    from core.db_connector import get_article_by_id
    
    if search_results["metadatas"] and search_results["metadatas"][0]:
        first_result_meta = search_results["metadatas"][0][0]
        article_id_str = first_result_meta.get('article_id', '')
        
        if article_id_str and article_id_str != 'MISSING':
            try:
                article_id = int(article_id_str)
                article = get_article_by_id(article_id)
                if article:
                    print(f"[OK] Article {article_id} loaded from SQL")
                    print(f"Title: {article.get('title', 'NO TITLE')}")
                    print(f"Content (first 300 chars): {article.get('content', 'EMPTY')[:300]}")
                else:
                    print(f"[ERROR] Article {article_id} NOT found in SQL DB")
            except Exception as e:
                print(f"[ERROR] Error loading article: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"[ERROR] No valid article_id in metadata")
            print(f"Full first result metadata: {first_result_meta}")
    
    print("\n[OK] All tests completed successfully")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
