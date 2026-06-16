"""Main Streamlit application entry point.

AI News Intelligence Assistant - A full-stack generative AI application
for analyzing CNN news articles using RAG, multi-agent reasoning, and
evaluation benchmarks.

This is the main application runner. Execute with:
    streamlit run app.py
"""

import streamlit as st
import logging
from config import settings
from core.utils import setup_logging

# Configure logging
logger = setup_logging(level=settings.log_level)

# Configure Streamlit page
st.set_page_config(
    page_title=settings.app_name,
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Application title
st.title(f"📰 {settings.app_name}")
st.markdown(f"*Version {settings.app_version}*")

# Initialize session state
if "initialized" not in st.session_state:
    st.session_state.initialized = True
    logger.info("Session initialized")

# Sidebar configuration
with st.sidebar:
    st.markdown(
        "**Pages:**\n"
        "- 🗣️ **Chat** - Main conversation interface\n"
        "- 📚 **Knowledge Base** - Document management\n"
        "- 🧪 **Experiments** - Parameter analysis\n"
        "- 📊 **Benchmark** - Evaluation on test sets"
    )

# Main content area
st.markdown("## 🏠 Home Dashboard")

st.markdown("---")

st.markdown("""
### About This Application

This is a **full-stack generative AI course project** demonstrating:

- **Local LLM Integration**: Uses LM Studio with local models (gemma-4-e4b)
- **Retrieval-Augmented Generation (RAG)**: Grounds responses in CNN news articles
- **Knowledge Base**: Document ingestion, chunking, and semantic search via ChromaDB
- **Multi-Agent Reasoning**: Orchestrated reasoning pipeline with analyzer, critic, and synthesizer
- **Tool Calling**: Function registry for corpus analysis and filtering
- **Benchmark Evaluation**: LLM-as-Judge evaluation on test datasets
- **Parameter Experiments**: Controlled experiments comparing generation parameters

### Core Topics

The application analyzes a corpus focused on three key themes:
- 🌱 **Green Energy** - Renewable energy and sustainability initiatives
- 💼 **U.S. Trade War** - Trade policies, tariffs, and economic impacts
- 🔕 **Social Media Censorship** - Content moderation and freedom of speech

### Getting Started

1. **Navigate to 🗣️ Chat** to start a conversation
2. **Use 📚 Knowledge Base** to upload or search documents
3. **Try 🧪 Experiments** to compare generation parameters
4. **Check 📊 Benchmark** to evaluate system performance

---

**Powered by**: Python • Streamlit • LM Studio • ChromaDB • LangChain
""")

logger.info("App rendered successfully")
