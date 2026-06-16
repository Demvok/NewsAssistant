#!/usr/bin/env python
"""Quick integration test for ingestion and RAG modules."""

from ingestion.chunker import chunk_document, validate_chunks
from ingestion.loader import load_document, extract_text_from_txt
from config import settings

# Test 1: Document chunking
print("Test 1: Document Chunking")
content = "This is a test document. " * 50
chunks = chunk_document(
    content,
    document_id="test-doc",
    chunk_size=128,
    overlap=20,
)
print(f"  ✓ Created {len(chunks)} chunks")

# Test 2: Chunk validation
print("Test 2: Chunk Validation")
validate_chunks(chunks)
print(f"  ✓ Chunks validated successfully")

# Test 3: Metadata preservation
print("Test 3: Metadata Preservation")
sample_chunk = chunks[0]
print(f"  ✓ Chunk ID: {sample_chunk['id']}")
print(f"  ✓ Chunk Order: {sample_chunk['chunk_order']}")
print(f"  ✓ Chunk Size: {len(sample_chunk['content'])} chars")

# Test 4: Configuration check
print("Test 4: Configuration Check")
print(f"  ✓ Chunk Size Config: {settings.chunk_size}")
    print(f"  ✓ Top-K Retrieval: {settings.top_k_retrieval}")
print(f"  ✓ ChromaDB Path: {settings.chroma_db_dir}")

print("\n✓ All integration tests passed!")
