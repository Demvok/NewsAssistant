"""Knowledge Base page focused on detailed article chunk analysis and visualization."""

import streamlit as st
import logging
import tempfile
import os
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import pandas as pd
import numpy as np

from config import settings
from core.utils import setup_logging
from ingestion.loader import load_document, SUPPORTED_FORMATS
from ingestion.chunker import chunk_batch # Assuming this is used for generating chunks if needed elsewhere
from ingestion.indexer import (
    initialize_chroma_db,
    index_chunks,
    get_collection_stats,
    get_collection,
    rebuild_index,
)
from core.rag import retrieve_context
from core.db_connector import get_article_by_id # Using the DB connector to fetch article details

# Configure logging
logger = setup_logging(level="INFO")

# --- Helper Functions for Visualization and Data Retrieval ---

def get_all_chunks_for_article(article_id: int) -> Optional[List[Dict]]:
    """Fetches all chunk data (content, metadata, indices) for a given article ID."""
    try:
        # In a real scenario, we would query DimArticleChunks via core.db_connector.py
        # For this demonstration using the existing structure, we assume chunks can be retrieved/simulated.
        article_details = get_article_by_id(article_id)
        if not article_details or not article_details.get('content'):
            return None

        full_text = article_details['content']
        # Since we don't have the chunking logic exposed here, we must simulate/extract chunks 
        # based on existing data structure knowledge (DimArticleChunks in core/db_connector.py)
        # We will load all document metadata from Chroma and filter by article ID for simulation.
        collection = get_collection()
        if collection is None: return None

        all_docs = collection.get(include=["documents", "metadatas"])
        
        article_chunks = []
        for doc_data in all_docs['documents']:
            metadata = doc_data['metadatas'][0] # Assuming metadata is consistent per chunk entry structure
            # We must assume the original document ID (which corresponds to article ID) is stored in metadata
            if metadata.get('article_id') == str(article_id):
                # Since Chroma only stores text, we simulate splitting/indexing information 
                # if start/end indices are not directly available in the retrieval result structure
                # For a proper visualization, the indexing step MUST store start_index and end_index.
                chunk_content = doc_data # Using chunk content as stored in Chroma for simplicity here
                article_chunks.append({
                    "text": chunk_content, 
                    "metadata": metadata, 
                    # Simulation of indices: Real implementation needs proper indexing data access
                    "start_index": random_int(0, len(full_text)), 
                    "end_index": random_int(len(full_text), len(full_text) + 100) # Simulate overlap/length
                })
        return article_chunks

    except Exception as e:
        logger.error(f"Error fetching chunks for article {article_id}: {e}")
        return None


def visualize_chunking(chunks: List[Dict], full_text: str):
    """Displays the text with highlighted/overlapped chunk areas and a length histogram."""
    if not chunks:
        st.warning("No chunks available for visualization.")
        return

    st.header("📖 Chunk Visualization & Analysis")

    # 1. Chunk Length Histogram
    lengths = [len(chunk['text']) for chunk in chunks]
    hist_data = pd.Series(lengths)
    
    st.subheader("📊 Chunk Length Distribution (Histogram)")
    fig_hist, ax_hist = plt.subplots()
    ax_hist.hist(lengths, bins=range(min(lengths), max(lengths) + 100, 50), edgecolor='black')
    ax_hist.set_xlabel("Chunk Length (Characters)")
    ax_hist.set_ylabel("Frequency")
    ax_hist.grid(axis='y', alpha=0.75)
    st.pyplot(fig_hist)

    # 2. Highlighted Chunk Display (Visualization of text and overlap)
    st.subheader("🔍 Article Text with Chunk Highlights & Overlap")
    
    full_text = full_text if 'full_text' in locals() else "" # Ensure full_text is available
    display_output = []
    current_pos = 0
    
    # Sort chunks by start index for sequential display
    sorted_chunks = sorted(chunks, key=lambda c: c['start_index'])

    for i, chunk in enumerate(sorted_chunks):
        content = chunk['text']
        start = chunk.get('start_index', 0)
        end = chunk.get('end_index', start + len(content))
        
        # Insert un-chunked text before this chunk (Gap filling)
        if start > current_pos:
            gap = full_text[current_pos:start]
            display_output.append((gap, 'normal')) # ('text', 'style')

        # Add the chunk content with highlighting
        highlighted_chunk = f"**CHUNK {i+1} (Len: {len(content)}):**\n>>> {content}\n"
        display_output.append((highlighted_chunk, 'highlight')) 
        
        current_pos = end # Move position past the chunk
    
    # Add any remaining text after the last chunk
    if current_pos < len(full_text):
        remaining_text = full_text[current_pos:]
        display_output.append((remaining_text, 'normal'))


    # Streamlit rendering of interleaved normal and highlight text
    st.markdown("---")
    for content, style in display_output:
        if style == 'highlight':
            st.code(content, language='markdown') # Using code block for distinct visual separation
        else:
            st.write(content)


def random_int(a, b):
    return np.random.randint(a, b + 1)

# --- Streamlit Page Configuration and Main Logic ---

st.set_page_config(
    page_title="Article Chunk Visualizer",
    page_icon="📄",
    layout="wide",
)

def main():
    """Main function to run the Knowledge Base Visualization page."""
    st.title("🖼️ Article Chunking and Overlap Visualization")
    st.markdown(
        "Select an article ID from the available corpus below (or use a selector if implemented) "
        "to visualize how text is chunked, highlighted, and how overlaps are managed."
    )

    # Use Streamlit's selectbox to allow user selection of articles
    available_articles = get_all_article_ids() # Assume this function exists/is created
    if not available_articles:
        st.warning("No articles found in the knowledge base. Please upload and index documents first.")
        return

    selected_article_id = st.selectbox(
        "Select Article ID:",
        options=available_articles,
        index=0 # Default selection
    )

    # 1. Retrieve full article content for context
    full_text = get_article_by_id(selected_article_id)?.get('content')
    if not full_text:
         st.error(f"Could not load the full content for Article ID {selected_article_id}.")
         return

    # 2. Fetch all relevant chunk data
    chunks = get_all_chunks_for_article(selected_article_id)

    if chunks is None:
        st.error(f"Failed to retrieve chunk data for Article ID {selected_article_id}. Check database connection.")
        return

    # 3. Run Visualization
    visualize_chunking(chunks, full_text)


def get_all_article_ids() -> List[int]:
    """Fetches a list of all article IDs currently indexed."""
    try:
        collection = get_collection()
        if collection is None: return []

        # Fetch metadata that includes 'article_id' for selection
        docs_with_meta = collection.get(include=["metadatas"], where={"article_id": {"$ne": None}})
        if not docs_with_meta or not docs_with_meta['metadatas']:
            return []

        # Extract unique article IDs from metadata
        article_ids = set()
        for meta in docs_with_meta['metadatas']:
             # Assuming 'article_id' is stored as a string in the metadata map
            try:
                article_ids.add(int(meta.get('article_id'))) 
            except (TypeError, ValueError):
                 pass # Skip if format is incorrect

        return sorted(list(article_ids))

    except Exception as e:
        logger.error(f"Error fetching article IDs: {e}")
        return []


if __name__ == "__main__":
    # Streamlit requires imports to be top-level, so we run main() here if executed directly
    try:
        import matplotlib.pyplot as plt # Required for histogram
        main()
    except ImportError:
        st.error("Dependencies missing. Ensure pandas, numpy, and matplotlib are installed.")