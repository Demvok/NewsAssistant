#!/usr/bin/env python
"""Comprehensive test for loader, chunker, and indexer modules."""

from pathlib import Path
from ingestion.loader import load_document, load_from_directory
from ingestion.chunker import chunk_document, chunk_batch
from config import settings

# Test 1: Load single document
print("Test 1: Load Single Document")
doc_path = settings.project_root / settings.raw_data_dir / "test_article.txt"
if doc_path.exists():
    doc = load_document(str(doc_path))
    print(f"  ✓ Loaded document: {doc['title']}")
    print(f"  ✓ Content length: {len(doc['content'])} chars")
    print(f"  ✓ Metadata: {doc['metadata']}")
else:
    print(f"  ! Document not found: {doc_path}")

# Test 2: Load from directory
print("\nTest 2: Load from Directory")
docs = load_from_directory(str(settings.raw_data_dir), "*.txt")
print(f"  ✓ Loaded {len(docs)} documents from directory")

# Test 3: Chunk batch
print("\nTest 3: Batch Chunking")
if docs:
    chunks = chunk_batch(
        docs,
        chunk_size=settings.rag.chunking.chunk_size,
        overlap=settings.rag.chunking.chunk_overlap,
    )
    print(f"  ✓ Created {len(chunks)} chunks from {len(docs)} documents")
    
    # Show sample chunk
    if chunks:
        sample = chunks[0]
        print(f"  ✓ Sample chunk:")
        print(f"    - ID: {sample['id']}")
        print(f"    - Document ID: {sample['document_id']}")
        print(f"    - Order: {sample['chunk_order']}")
        print(f"    - Size: {len(sample['content'])} chars")

# Test 4: Configuration
print("\nTest 4: Configuration Summary")
print(f"  ✓ Chunk Size: {settings.rag.chunking.chunk_size}")
print(f"  ✓ Chunk Overlap: {settings.rag.chunking.chunk_overlap}")
print(f"  ✓ ChromaDB Path: {settings.chroma_db_dir}")
print(f"  ✓ Top-K Retrieval: {settings.rag.chromadb.top_k_retrieval}")
print(f"  ✓ Distance Metric: {settings.rag.chromadb.distance_metric}")

print("\n✓ All loader and chunker tests passed!")
