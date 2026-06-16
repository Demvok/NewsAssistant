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
    st.session_state.rag_enabled = True
    st.session_state.multi_agent_enabled = False
    st.session_state.chat_history = []
    logger.info("Session initialized")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.subheader("Features")
    st.session_state.rag_enabled = st.checkbox(
        "Enable RAG (Retrieval-Augmented Generation)",
        value=st.session_state.rag_enabled,
        help="Use document corpus for grounding responses"
    )
    
    st.session_state.multi_agent_enabled = st.checkbox(
        "Enable Multi-Agent Mode",
        value=st.session_state.multi_agent_enabled,
        help="Use multiple agents for reasoning (analyzer → critic → synthesizer)"
    )
    
    st.subheader("Generation Parameters")
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=0.7,
        step=0.1,
        help="Controls randomness: 0 = deterministic, 2 = maximum randomness"
    )
    
    top_p = st.slider(
        "Top-P (Nucleus Sampling)",
        min_value=0.0,
        max_value=1.0,
        value=0.95,
        step=0.05,
        help="Controls diversity via nucleus sampling"
    )
    
    top_k = st.slider(
        "Top-K",
        min_value=1,
        max_value=100,
        value=40,
        step=1,
        help="Limits the number of highest-probability tokens"
    )
    
    st.markdown("---")
    st.markdown(
        "**Pages:**\n"
        "- 🗣️ **Chat** - Main conversation interface\n"
        "- 📚 **Knowledge Base** - Document management\n"
        "- 🧪 **Experiments** - Parameter analysis\n"
        "- 📊 **Benchmark** - Evaluation on test sets"
    )

# Main content area
st.markdown("## Welcome to the AI News Intelligence Assistant")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="RAG Status",
        value="Enabled" if st.session_state.rag_enabled else "Disabled",
        delta="Using corpus" if st.session_state.rag_enabled else "Inference only"
    )

with col2:
    st.metric(
        label="Multi-Agent",
        value="Enabled" if st.session_state.multi_agent_enabled else "Disabled",
        delta="Multi-step reasoning" if st.session_state.multi_agent_enabled else "Single-pass"
    )

with col3:
    st.metric(
        label="Temperature",
        value=f"{temperature:.1f}",
        delta="Controlled" if temperature < 0.7 else "Creative" if temperature > 0.7 else "Balanced"
    )

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
