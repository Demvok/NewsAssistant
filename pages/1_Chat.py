"""Chat page for the AI News Intelligence Assistant.

Main user interface for multi-turn conversation with support for:
- RAG (Retrieval-Augmented Generation)
- Multi-agent reasoning
- Tool calling
- Generation parameter configuration
- Chat history and session state management
"""

import streamlit as st
import logging
from datetime import datetime
from typing import Optional

from config import (
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_TOP_K,
    MAX_TOKENS,
    TOP_K_RETRIEVAL,
)
from core.llm_client import LMStudioClient
from core.schemas import ChatMessage
from core.rag import retrieve_context
from core.utils import setup_logging

# Configure logging
logger = setup_logging(level="INFO")

# Configure page
st.set_page_config(
    page_title="Chat - AI News Intelligence Assistant",
    page_icon="🗣️",
    layout="wide",
)

st.title("🗣️ Chat")
st.markdown("Analyze news articles with multi-turn conversation")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    
if "rag_enabled" not in st.session_state:
    st.session_state.rag_enabled = True
    
if "multi_agent_enabled" not in st.session_state:
    st.session_state.multi_agent_enabled = False
    
if "tool_calling_enabled" not in st.session_state:
    st.session_state.tool_calling_enabled = False
    
if "temperature" not in st.session_state:
    st.session_state.temperature = DEFAULT_TEMPERATURE
    
if "top_p" not in st.session_state:
    st.session_state.top_p = DEFAULT_TOP_P
    
if "top_k" not in st.session_state:
    st.session_state.top_k = DEFAULT_TOP_K


# Sidebar controls
with st.sidebar:
    st.header("⚙️ Chat Configuration")
    
    st.subheader("Features")
    st.session_state.rag_enabled = st.checkbox(
        "🔍 RAG (Retrieval-Augmented Generation)",
        value=st.session_state.rag_enabled,
        help="Ground responses in the document corpus",
    )
    
    st.session_state.multi_agent_enabled = st.checkbox(
        "🤖 Multi-Agent Mode",
        value=st.session_state.multi_agent_enabled,
        help="Use analyzer → critic → synthesizer pipeline",
    )
    
    st.session_state.tool_calling_enabled = st.checkbox(
        "🔧 Tool Calling",
        value=st.session_state.tool_calling_enabled,
        help="Allow function calling for corpus analysis",
    )
    
    st.markdown("---")
    st.subheader("Generation Parameters")
    
    st.session_state.temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=st.session_state.temperature,
        step=0.1,
        help="Higher = more creative, Lower = more focused",
    )
    
    st.session_state.top_p = st.slider(
        "Top-P (Nucleus Sampling)",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.top_p,
        step=0.05,
        help="Controls diversity of responses",
    )
    
    st.session_state.top_k = st.slider(
        "Top-K",
        min_value=1,
        max_value=100,
        value=st.session_state.top_k,
        step=1,
        help="Limits vocabulary to top k most probable tokens",
    )
    
    st.markdown("---")
    st.subheader("Retrieval Settings")
    
    top_k_retrieval = st.slider(
        "Top-K Chunks",
        min_value=1,
        max_value=20,
        value=TOP_K_RETRIEVAL,
        step=1,
        help="Number of document chunks to retrieve",
    )
    
    if st.button("🗑️ Clear Chat History", key="clear_history"):
        st.session_state.chat_history = []
        st.success("Chat history cleared")
        st.rerun()


# Main chat area
def render_chat_history():
    """Render all chat messages from history."""
    for message in st.session_state.chat_history:
        role = message.get("role", "user")
        content = message.get("content", "")
        
        if role == "user":
            with st.chat_message("user", avatar="👤"):
                st.markdown(content)
        else:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(content.get("answer", "") if isinstance(content, dict) else content)
                
                # Display retrieval context if available
                if isinstance(content, dict) and "retrieved_chunks" in content and content["retrieved_chunks"]:
                    with st.expander("📚 Retrieved Context", expanded=False):
                        for i, chunk in enumerate(content["retrieved_chunks"], 1):
                            col1, col2 = st.columns([1, 4])
                            with col1:
                                st.metric(
                                    label=f"Chunk {i}",
                                    value=f"{chunk.get('similarity_score', 0):.3f}",
                                    label_visibility="collapsed"
                                )
                            with col2:
                                st.write(
                                    f"**Source**: {chunk.get('filename', 'Unknown')}\n\n"
                                    f"{chunk.get('content', '')[:500]}..."
                                )
                
                # Display agent trace if available
                if isinstance(content, dict) and "agent_trace" in content and content["agent_trace"]:
                    with st.expander("🔗 Agent Reasoning Trace", expanded=False):
                        for step in content["agent_trace"]:
                            st.write(f"**{step.get('agent', 'Unknown')}**: {step.get('reasoning', '')}")
                
                # Display tool calls if available
                if isinstance(content, dict) and "tool_calls" in content and content["tool_calls"]:
                    with st.expander("🔧 Tool Calls", expanded=False):
                        for tool_call in content["tool_calls"]:
                            st.code(
                                f"Tool: {tool_call.get('name', 'unknown')}\n"
                                f"Args: {tool_call.get('args', {})}\n"
                                f"Result: {tool_call.get('result', '')}",
                                language="python"
                            )


def format_rag_context(retrieved_chunks: list) -> str:
    """Format retrieved chunks into a context string for the prompt.
    
    Args:
        retrieved_chunks: List of retrieved chunk dictionaries
        
    Returns:
        Formatted context string
    """
    if not retrieved_chunks:
        return ""
    
    context_parts = ["Retrieved Context:"]
    for i, chunk in enumerate(retrieved_chunks, 1):
        source = chunk.get("filename", "Unknown")
        content = chunk.get("content", "")
        score = chunk.get("similarity_score", 0)
        context_parts.append(f"\n[Chunk {i}] (Source: {source}, Similarity: {score:.3f})")
        context_parts.append(content)
    
    return "\n".join(context_parts)


def build_prompt(
    user_message: str,
    rag_enabled: bool,
    retrieved_chunks: Optional[list] = None,
) -> str:
    """Build the complete prompt for the LLM.
    
    Args:
        user_message: The user's query
        rag_enabled: Whether RAG is enabled
        retrieved_chunks: Retrieved context chunks (if RAG enabled)
        
    Returns:
        Formatted prompt string
    """
    system_prompt = (
        "You are an AI News Intelligence Assistant specialized in analyzing CNN news articles. "
        "You focus on three key themes: green energy, the U.S. trade war, and social media censorship. "
        "Provide analytical, grounded responses based on the provided context when available. "
        "When answering, cite specific sources from the retrieved documents."
    )
    
    prompt = f"System: {system_prompt}\n\n"
    
    if rag_enabled and retrieved_chunks:
        prompt += f"{format_rag_context(retrieved_chunks)}\n\n"
    
    # Add recent chat history for context
    history_context = ""
    recent_history = st.session_state.chat_history[-4:-1] if len(st.session_state.chat_history) > 1 else []
    if recent_history:
        history_context = "Recent conversation:\n"
        for msg in recent_history:
            role = "User" if msg.get("role") == "user" else "Assistant"
            content = msg.get("content", "")
            if isinstance(content, dict):
                content = content.get("answer", "")
            history_context += f"{role}: {content}\n"
        prompt += f"{history_context}\n"
    
    prompt += f"User: {user_message}\nAssistant:"
    return prompt


def generate_response(
    user_message: str,
    llm_client: LMStudioClient,
    rag_enabled: bool,
    multi_agent_enabled: bool,
    tool_calling_enabled: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    top_k_retrieval: int,
) -> dict:
    """Generate an AI response to the user message.
    
    Args:
        user_message: The user's query
        llm_client: LM Studio client for generation
        rag_enabled: Whether to use RAG
        multi_agent_enabled: Whether to use multi-agent reasoning
        tool_calling_enabled: Whether to enable tool calling
        temperature: Generation temperature
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        top_k_retrieval: Number of chunks to retrieve
        
    Returns:
        Response dictionary with answer and metadata
    """
    response_data = {
        "answer": "",
        "retrieved_chunks": [],
        "agent_trace": [],
        "tool_calls": [],
        "error": None,
    }
    
    try:
        # Step 1: Retrieve context if RAG is enabled
        if rag_enabled:
            try:
                retrieved_chunks = retrieve_context(
                    query=user_message,
                    top_k=top_k_retrieval,
                )
                response_data["retrieved_chunks"] = retrieved_chunks
                logger.info(f"Retrieved {len(retrieved_chunks)} chunks")
            except Exception as e:
                logger.warning(f"RAG retrieval failed: {str(e)}")
                response_data["retrieved_chunks"] = []
        
        # Step 2: Build prompt with context
        prompt = build_prompt(
            user_message=user_message,
            rag_enabled=rag_enabled,
            retrieved_chunks=response_data["retrieved_chunks"],
        )
        
        # Step 3: Generate response (placeholder for multi-agent and tool calling)
        # Note: These are stubbed for now as the agents and tools modules are not fully implemented
        if multi_agent_enabled:
            # Placeholder for multi-agent pipeline
            response_data["agent_trace"] = [
                {"agent": "Analyzer", "reasoning": "Analyzing the query and retrieved context..."},
                {"agent": "Critic", "reasoning": "Evaluating the analysis for accuracy..."},
                {"agent": "Synthesizer", "reasoning": "Synthesizing the final answer..."},
            ]
            logger.debug("Multi-agent mode enabled (placeholder)")
        
        if tool_calling_enabled:
            # Placeholder for tool calling
            response_data["tool_calls"] = []
            logger.debug("Tool calling mode enabled (placeholder)")
        
        # Generate the actual response
        try:
            llm_response = llm_client.complete(
                prompt=prompt,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_tokens=MAX_TOKENS,
            )
            response_data["answer"] = llm_response.text.strip()
            logger.info(
                f"Generated response: {len(response_data['answer'])} chars, "
                f"latency={llm_response.latency_ms:.0f}ms"
            )
        except Exception as e:
            error_msg = f"Generation failed: {str(e)}"
            response_data["error"] = error_msg
            response_data["answer"] = f"❌ {error_msg}"
            logger.error(error_msg)
    
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        response_data["error"] = error_msg
        response_data["answer"] = f"❌ {error_msg}"
        logger.error(error_msg)
    
    return response_data


# Main chat interface
st.subheader("Conversation")

# Display chat history
render_chat_history()

# Chat input
col1, col2 = st.columns([1, 0.15])

with col1:
    user_input = st.chat_input("Ask a question about news articles...")

with col2:
    send_button = st.button("Send", use_container_width=True, key="send_chat")

# Process user input
if (user_input or send_button) and user_input:
    # Add user message to history
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_input,
        "timestamp": datetime.now().isoformat(),
    })
    
    # Display user message
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)
    
    # Generate response with progress indicator
    with st.spinner("🤔 Thinking..."):
        try:
            # Initialize LLM client
            from config import settings
            
            llm_client = LMStudioClient(
                base_url=settings.lm_studio.base_url,
                model_name=settings.lm_studio.chat_model,
                timeout=settings.lm_studio.request_timeout,
            )
            
            # Generate response
            response_data = generate_response(
                user_message=user_input,
                llm_client=llm_client,
                rag_enabled=st.session_state.rag_enabled,
                multi_agent_enabled=st.session_state.multi_agent_enabled,
                tool_calling_enabled=st.session_state.tool_calling_enabled,
                temperature=st.session_state.temperature,
                top_p=st.session_state.top_p,
                top_k=st.session_state.top_k,
                top_k_retrieval=top_k_retrieval,
            )
            
            # Add assistant response to history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response_data,
                "timestamp": datetime.now().isoformat(),
            })
            
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            logger.error(error_msg)
            st.error(error_msg)
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": {
                    "answer": f"❌ {error_msg}",
                    "retrieved_chunks": [],
                    "agent_trace": [],
                    "tool_calls": [],
                    "error": error_msg,
                },
                "timestamp": datetime.now().isoformat(),
            })
    
    # Re-render chat to show new messages
    st.rerun()


# Footer with stats
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Messages",
        value=len(st.session_state.chat_history),
        delta="in this session",
    )

with col2:
    status = "Enabled" if st.session_state.rag_enabled else "Disabled"
    st.metric(label="RAG", value=status)

with col3:
    status = "Enabled" if st.session_state.multi_agent_enabled else "Disabled"
    st.metric(label="Multi-Agent", value=status)

with col4:
    status = "Enabled" if st.session_state.tool_calling_enabled else "Disabled"
    st.metric(label="Tool Calling", value=status)

logger.info("Chat page rendered successfully")
