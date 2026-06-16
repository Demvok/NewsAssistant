import os
import chromadb
import json
from typing import List, Dict

# Define the ChromaDB path based on debug_chromadb.py
db_path = os.path.join("data", "chroma_db")
print(f"Attempting to connect to ChromaDB at: {db_path}")

try:
    client = chromadb.PersistentClient(path=db_path)
except Exception as e:
    print(f"Error connecting to ChromaDB. Ensure the database is initialized and data exists. Error: {e}")
    exit(1)

collections = client.list_collections()

if not collections:
    print("No collections found in ChromaDB.")
    exit(0)

# Assuming the first collection is the target corpus, as per debug_chromadb.py logic
collection_name = collections[0].name
try:
    collection = client.get_collection(name=collection_name)
except Exception as e:
    print(f"Error accessing collection {collection_name}. Error: {e}")
    exit(1)

total_chunks = collection.count()
if total_chunks == 0:
    print("The target collection is empty.")
    exit(0)

# Fetch all documents to calculate the average chunk size
try:
    results = collection.get(include=["documents"])
except Exception as e:
    print(f"Error retrieving documents from ChromaDB: {e}")
    exit(1)

all_chunks: List[str] = results['documents']

if not all_chunks:
    print("Retrieved no document content.")
    exit(0)

total_length = sum(len(chunk) for chunk in all_chunks)
average_length = total_length / len(all_chunks)

print(f"--- Analysis Complete ---")
print(f"Total chunks retrieved: {len(all_chunks)}")
print(f"Total cumulative character length: {total_length}")
print(f"Average chunk size (characters): {average_length:.2f}")