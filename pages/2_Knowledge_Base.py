"""Knowledge Base page for document management and search.

Handles document upload, ingestion, embedding generation,
ChromaDB persistence, corpus search, and analytics visualization.
"""

import streamlit as st
import logging
import tempfile
import os
from pathlib import Path
from typing import Optional, List, Dict

from config import settings
from core.utils import setup_logging
from ingestion.loader import load_document, SUPPORTED_FORMATS
from ingestion.chunker import chunk_batch
from ingestion.indexer import (
    initialize_chroma_db,
    index_chunks,
    get_collection_stats,
    get_collection,
    rebuild_index,
)
from core.rag import retrieve_context

# Configure logging
logger = setup_logging(level="INFO")

# Configure page
st.set_page_config(
    page_title="Knowledge Base - AI News Intelligence Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Knowledge Base")
st.markdown("Manage documents, search corpus, and view statistics")


# Initialize session state
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "index_built" not in st.session_state:
    st.session_state.index_built = False
if "search_results" not in st.session_state:
    st.session_state.search_results = []

# Initialize ChromaDB connection
try:
    initialize_chroma_db(
        db_path=settings.project_root / settings.chroma_db_dir,
        embedding_base_url=settings.lm_studio_base_url,
        embedding_model=settings.embedding_model,
    )
except Exception as e:
    logger.warning(f"ChromaDB initialization warning: {str(e)}")


def get_collection_documents() -> Dict[str, int]:
    """Get count of documents in the collection by type."""
    collection = get_collection()
    if collection is None:
        return {}

    try:
        all_data = collection.get(include=["metadatas"])
        if not all_data or not all_data.get("metadatas"):
            return {}

        doc_count = {}
        for metadata in all_data["metadatas"]:
            filename = metadata.get("filename", "unknown")
            if filename not in doc_count:
                doc_count[filename] = 0
            doc_count[filename] += 1

        return doc_count
    except Exception as e:
        logger.error(f"Failed to get collection documents: {str(e)}")
        return {}


def get_chunk_statistics() -> Optional[Dict]:
    """Get statistics about chunks in the collection."""
    collection = get_collection()
    if collection is None:
        return None

    try:
        all_data = collection.get(include=["metadatas", "documents"])
        if not all_data or not all_data.get("documents"):
            return None

        chunk_sizes = [len(doc) for doc in all_data.get("documents", [])]
        num_chunks = len(chunk_sizes)

        if not chunk_sizes:
            return None

        stats = {
            "total_chunks": num_chunks,
            "avg_chunk_size": sum(chunk_sizes) / len(chunk_sizes),
            "min_chunk_size": min(chunk_sizes),
            "max_chunk_size": max(chunk_sizes),
            "total_characters": sum(chunk_sizes),
        }
        return stats
    except Exception as e:
        logger.error(f"Failed to get chunk statistics: {str(e)}")
        return None


def load_and_index_files(uploaded_files) -> Dict:
    """Load uploaded files and index them into ChromaDB."""
    results = {
        "loaded": 0,
        "indexed": 0,
        "errors": [],
    }

    if not uploaded_files:
        return results

    with st.spinner("Processing documents..."):
        documents = []
        temp_files = []

        try:
            # Load documents
            for uploaded_file in uploaded_files:
                try:
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=Path(uploaded_file.name).suffix,
                    ) as tmp_file:
                        tmp_file.write(uploaded_file.getbuffer())
                        temp_files.append(tmp_file.name)

                    # Load document
                    doc = load_document(tmp_file.name)
                    documents.append(doc)
                    results["loaded"] += 1
                    logger.info(f"Loaded document: {uploaded_file.name}")

                except Exception as e:
                    error_msg = f"Failed to load {uploaded_file.name}: {str(e)}"
                    results["errors"].append(error_msg)
                    logger.error(error_msg)

            if not documents:
                results["errors"].append("No documents were successfully loaded")
                return results

            # Index documents
            try:
                stats = rebuild_index(
                    documents=documents,
                    chunk_size=settings.chunk_size,
                    chunk_overlap=settings.chunk_overlap,
                    embedding_base_url=settings.lm_studio_base_url,
                    embedding_model=settings.embedding_model,
                    db_path=settings.project_root / settings.chroma_db_dir,
                )
                results["indexed"] = stats.get("num_chunks", 0)
                st.session_state.index_built = True
                logger.info(f"Indexed documents: {stats}")

            except Exception as e:
                error_msg = f"Indexing failed: {str(e)}"
                results["errors"].append(error_msg)
                logger.error(error_msg)

        finally:
            # Clean up temp files
            for temp_file in temp_files:
                try:
                    os.unlink(temp_file)
                except Exception as e:
                    logger.warning(f"Failed to clean temp file {temp_file}: {str(e)}")

    return results


# Main layout
tab_stats, tab_search, tab_upload = st.tabs(["📊 Statistics", "🔍 Search", "📤 Upload"])

with tab_upload:
    st.subheader("Upload Documents")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(
            f"**Supported formats:** {', '.join(SUPPORTED_FORMATS)}\n\n"
            "Upload PDF or TXT documents to add them to the knowledge base."
        )

    with col2:
        if st.button("🗑️ Clear Upload Queue", key="clear_queue"):
            st.session_state.uploaded_files = []
            st.success("Upload queue cleared")

    # File uploader
    uploaded_files = st.file_uploader(
        label="Choose documents",
        type=list(ext.lstrip(".") for ext in SUPPORTED_FORMATS),
        accept_multiple_files=True,
        help="Upload one or more PDF or TXT files",
    )

    if uploaded_files:
        st.write(f"**Files selected:** {len(uploaded_files)}")
        for uploaded_file in uploaded_files:
            st.caption(f"📄 {uploaded_file.name} ({uploaded_file.size} bytes)")

        if st.button("✅ Process and Index Documents", key="process_docs"):
            results = load_and_index_files(uploaded_files)

            # Display results
            if results["loaded"] > 0:
                st.success(
                    f"✅ Loaded {results['loaded']} document(s), "
                    f"created {results['indexed']} chunks"
                )

            if results["errors"]:
                with st.expander("❌ Errors"):
                    for error in results["errors"]:
                        st.error(error)

            if results["indexed"] > 0:
                st.session_state.uploaded_files = []
                st.balloons()

    st.markdown("---")

    # Rebuild index section
    st.subheader("Rebuild Index")
    st.markdown(
        "Rebuild the entire index from documents in the raw data directory. "
        "This will re-embed and re-index all documents."
    )

    if st.button("🔄 Rebuild Index from Raw Data", key="rebuild_index"):
        with st.spinner("Rebuilding index..."):
            try:
                raw_docs_path = settings.project_root / settings.raw_data_dir
                if not raw_docs_path.exists() or not list(raw_docs_path.glob("*")):
                    st.warning(
                        f"No documents found in {raw_docs_path}. "
                        "Upload documents first."
                    )
                else:
                    from ingestion.loader import load_from_directory

                    documents = load_from_directory(str(raw_docs_path), "*")

                    if documents:
                        stats = rebuild_index(
                            documents=documents,
                            chunk_size=settings.chunk_size,
                            chunk_overlap=settings.chunk_overlap,
                            embedding_base_url=settings.lm_studio_base_url,
                            embedding_model=settings.embedding_model,
                            db_path=settings.project_root / settings.chroma_db_dir,
                        )
                        st.success(
                            f"✅ Index rebuilt: {stats['num_chunks']} chunks "
                            f"from {stats['num_documents']} documents"
                        )
                        st.session_state.index_built = True
                    else:
                        st.error("No documents could be loaded from raw data directory")

            except Exception as e:
                st.error(f"❌ Rebuild failed: {str(e)}")
                logger.error(f"Index rebuild failed: {str(e)}")


with tab_search:
    st.subheader("Search Corpus")

    search_query = st.text_input(
        "Enter search query",
        placeholder="e.g., renewable energy policy",
        help="Search the indexed documents using semantic similarity",
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        top_k = st.slider(
            "Number of results",
            min_value=1,
            max_value=10,
            value=settings.top_k_retrieval,
            help="How many top results to retrieve",
        )

    with col2:
        st.metric("Top-K", top_k)

    if st.button("🔍 Search", key="search_btn"):
        if not search_query.strip():
            st.warning("Please enter a search query")
        else:
            try:
                results = retrieve_context(
                    query=search_query,
                    top_k=top_k,
                    embedding_base_url=settings.lm_studio_base_url,
                    embedding_model=settings.embedding_model,
                )

                st.session_state.search_results = results

                # Debug helper: allow raw chunk-level inspection
                debug = st.checkbox("Show debug: raw chunk-level results", key="dbg_raw_chunks")

                if results:
                    st.success(f"Found {len(results)} matching documents")

                    if debug:
                        try:
                            from core.embeddings import embed_text
                            from ingestion.indexer import get_collection

                            collection = get_collection()
                            q_emb = embed_text(search_query)
                            # fetch many chunk-level results for inspection
                            raw = collection.query(
                                query_embeddings=[q_emb],
                                n_results=max(50, 10 * settings.top_k_retrieval),
                                include=["documents", "metadatas", "distances"],
                            )
                            st.markdown("### Raw chunk-level results")
                            st.write("Documents (first 5 chunks):")
                            docs = raw.get("documents", [])[0] if raw.get("documents") else []
                            metadatas = raw.get("metadatas", [])[0] if raw.get("metadatas") else []
                            distances = raw.get("distances", [])[0] if raw.get("distances") else []

                            rows = []
                            for i, (d, m, dist) in enumerate(zip(docs, metadatas, distances)):
                                rows.append({
                                    "index": i,
                                    "preview_len": len(d) if d else 0,
                                    "metadata": m,
                                    "distance": dist,
                                })
                            st.write(rows[:50])
                        except Exception as e:
                            st.warning(f"Failed to fetch raw chunk-level results: {e}")

                    for i, result in enumerate(results, 1):
                        doc_key = f"view_full_{result.get('document_id', i)}"
                        with st.expander(
                            f"📄 Result {i} | {result.get('filename','(no title)')} | "
                            f"Similarity: {result.get('similarity_score',0.0):.3f}",
                            expanded=(i == 1),
                        ):
                            col1, col2 = st.columns([3, 1])

                            with col1:
                                st.write(f"**Title:** {result.get('filename', result.get('document_id', ''))}")

                                # Load full article immediately to show first N lines
                                full_text = None
                                try:
                                    article_id = result.get('article_id')
                                    if article_id:
                                        from core.db_connector import get_article_by_id
                                        art = get_article_by_id(int(article_id))
                                        if art and art.get('content'):
                                            full_text = art.get('content')
                                    
                                    # Fallback to file-based loader if SQL entry not available
                                    if full_text is None:
                                        from ingestion.loader import load_document
                                        src = result.get('source')
                                        if src:
                                            doc_obj = load_document(src)
                                            full_text = doc_obj.get('content')
                                except Exception as e:
                                    st.warning(f"Could not load article: {str(e)}")
                                    full_text = None
                                
                                # Display first N lines of article
                                if full_text:
                                    lines = full_text.split('\n')
                                    first_lines = '\n'.join([line for line in lines if line.strip()][:10])
                                    st.write("**First 10 lines:**")
                                    st.text(first_lines)
                                else:
                                    st.warning("Could not load article content")

                            with col2:
                                st.metric("Similarity", f"{result.get('similarity_score',0.0):.3f}")
                                st.caption(f"Document ID: {result.get('document_id','')}")
                                st.caption(f"Article ID: {result.get('article_id','N/A')}")


                else:
                    st.info("No matching documents found. Try a different query.")

            except RuntimeError as e:
                st.error(f"❌ Search error: {str(e)}")
                logger.error(f"Search failed: {str(e)}")
            except Exception as e:
                st.error(f"❌ Unexpected error: {str(e)}")
                logger.error(f"Unexpected error during search: {str(e)}")

    # Display recent search results
    if st.session_state.search_results:
        st.markdown("---")
        st.markdown("### Recent Search Results")

        cols = st.columns(len(st.session_state.search_results[:3]))
        for col, result in zip(cols, st.session_state.search_results[:3]):
            with col:
                st.metric(
                    label=f"{result['filename'][:15]}...",
                    value=f"{result['similarity_score']:.2f}",
                    label_visibility="collapsed",
                )


with tab_stats:
    st.subheader("Corpus Statistics")

    try:
        collection = get_collection()

        if collection is None:
            st.info("Knowledge base is empty. Upload documents to get started.")
        else:
            stats = get_collection_stats()

            if stats:
                # Collection overview
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Total Chunks", stats["total_chunks"])

                with col2:
                    st.metric("Collection", stats["collection_name"])

                with col3:
                    st.metric(
                        "Distance Metric",
                        settings.distance_metric,
                    )

                st.markdown("---")

                # Chunk statistics
                chunk_stats = get_chunk_statistics()
                if chunk_stats:
                    st.subheader("Chunk Statistics")

                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric(
                            "Avg Chunk Size",
                            f"{chunk_stats['avg_chunk_size']:.0f}",
                            "characters",
                        )

                    with col2:
                        st.metric(
                            "Min Chunk Size",
                            f"{chunk_stats['min_chunk_size']}",
                            "characters",
                        )

                    with col3:
                        st.metric(
                            "Max Chunk Size",
                            f"{chunk_stats['max_chunk_size']}",
                            "characters",
                        )

                    with col4:
                        st.metric(
                            "Total Size",
                            f"{chunk_stats['total_characters'] / 1024:.1f}",
                            "KB",
                        )

                st.markdown("---")

                # Document breakdown
                doc_counts = get_collection_documents()
                if doc_counts:
                    st.subheader("Documents in Index")

                    cols = st.columns(min(len(doc_counts), 3))
                    for col, (filename, count) in zip(
                        cols, sorted(doc_counts.items())[:3]
                    ):
                        with col:
                            st.metric(filename[:20], count, "chunks")

                    if len(doc_counts) > 3:
                        with st.expander(f"Show all {len(doc_counts)} documents"):
                            for filename, count in sorted(doc_counts.items()):
                                st.write(f"- **{filename}**: {count} chunks")

            else:
                st.info("No statistics available.")

    except RuntimeError as e:
        st.warning(f"⚠️ {str(e)}")
    except Exception as e:
        st.error(f"❌ Error loading statistics: {str(e)}")
        logger.error(f"Failed to load statistics: {str(e)}")

    st.markdown("---")

    # Configuration
    st.subheader("Chunking Configuration")
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Chunk Size", settings.chunk_size, "characters")

    with col2:
        st.metric("Chunk Overlap", settings.chunk_overlap, "characters")

    st.markdown("---")

    # Data directory info
    st.subheader("Data Directories")
    col1, col2 = st.columns(2)

    with col1:
        st.caption("Raw Data Directory")
        st.code(str(settings.raw_data_dir))

    with col2:
        st.caption("ChromaDB Directory")
        st.code(str(settings.chroma_db_dir))


if __name__ == "__main__":
    pass
