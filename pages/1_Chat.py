"""Chat page for the AI News Intelligence Assistant.

Main user interface for multi-turn conversation with support for:
- RAG (Retrieval-Augmented Generation)
- Multi-agent reasoning
- Tool calling
- Generation parameter configuration
- Chat history and session state management
"""

import json
import logging
import re
from datetime import datetime
from typing import Any, Optional

import streamlit as st

from config import settings
from core.agents import run_multi_agent_pipeline
from core.llm_client import LMStudioClient
from core.react import run_react_loop
from core.tools import (
    create_default_tools,
    count_keyword_mentions,
    # filter_articles_by_date,
    get_article,
    search_articles,
)
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
    st.session_state.tool_calling_enabled = True
    
if "temperature" not in st.session_state:
    st.session_state.temperature = settings.temperature
    
if "top_p" not in st.session_state:
    st.session_state.top_p = settings.top_p
    
if "top_k" not in st.session_state:
    st.session_state.top_k = settings.top_k


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
        value=settings.top_k_retrieval,
        step=1,
        help="Number of document chunks to retrieve",
    )
    
    if st.button("🗑️ Clear Chat History", key="clear_history"):
        st.session_state.chat_history = []
        st.success("Chat history cleared")
        st.rerun()

    if st.button("🔄 Reset Model Params", key="reset_params"):
        st.session_state.temperature = settings.temperature
        st.session_state.top_p = settings.top_p
        st.session_state.top_k = settings.top_k
        st.success("Model parameters reset to default.")
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
                            st.markdown(f"**Chunk {i}**")
                            st.code(
                                json.dumps(_summarize_retrieved_chunk(chunk), ensure_ascii=False, indent=2),
                                language="json",
                            )
                
                # Display agent trace if available
                if isinstance(content, dict) and "agent_trace" in content and content["agent_trace"]:
                    _render_agent_trace(content["agent_trace"])
                
                # Display tool calls if available
                if isinstance(content, dict) and "tool_calls" in content and content["tool_calls"]:
                    with st.expander("🔧 Tool Calls", expanded=False):
                        for tool_call in content["tool_calls"]:
                            agent_name = tool_call.get("agent_name")
                            name = tool_call.get('name', 'unknown')
                            args = tool_call.get('args', {})
                            result = tool_call.get('result', '')

                            if agent_name:
                                st.markdown(f"**Agent**: {agent_name}")
                            st.markdown(f"**Tool Called**: `{name}`")
                            st.markdown("**Parameters:**")
                            # Display arguments clearly, using code block for structure if complex
                            if args:
                                st.code(str(args), language="json")
                            else:
                                st.info("No specific parameters provided.")

                            st.markdown("**Result**:")
                            st.code(result, language="text")


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
    retrieved_chunks: list[dict],
    tool_context_text: str,
) -> str:
    """Build the standard generation prompt for non-ReAct responses."""
    if rag_enabled and retrieved_chunks:
        context_text = format_rag_context(retrieved_chunks)
    elif tool_context_text:
        context_text = tool_context_text
    else:
        context_text = ""

    if context_text:
        return (
            "You are a helpful news intelligence assistant. Use the provided context when relevant.\n\n"
            f"Context:\n{context_text}\n\n"
            f"User question: {user_message}\n\n"
            "Answer clearly and concisely using only the available information."
        )

    return (
        "You are a helpful news intelligence assistant.\n\n"
        f"User question: {user_message}\n\n"
        "Answer clearly and concisely."
    )


def _get_available_tools() -> dict[str, dict]:
    """Get tool descriptions for the ReAct loop to choose from."""
    return {
        "search_articles": {
            "description": "Search for relevant articles using semantic similarity. Use when you need to find articles related to a topic or query.",
            "params": ["query (str)", "top_k (int, optional, default=5)"],
        },
        "get_article": {
            "description": "Fetch a specific article by its ID and get its full content and metadata.",
            "params": ["article_id (int)"],
        },
        # "filter_articles_by_date": {
        #     "description": "Find articles published within a specific date range.",
        #     "params": ["start_date (str, ISO format YYYY-MM-DD)", "end_date (str, ISO format YYYY-MM-DD)"],
        # },
        "count_keyword_mentions": {
            "description": "Count how many times a keyword appears in the article corpus.",
            "params": ["keyword (str)"],
        },
    }


def _render_react_trace_from_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert the ReAct module steps into a compact trace format for the UI."""
    trace = []
    for step in steps:
        trace.append(
            {
                "thought": step.get("thought", ""),
                "action": step.get("action", ""),
                "tool": step.get("tool_name"),
                "input": step.get("tool_input", {}),
                "observation": step.get("observation", ""),
            }
        )
    return trace


def _summarize_retrieved_chunk(chunk: dict[str, Any]) -> dict[str, Any]:
    """Project a retrieved chunk down to the fields shown in chat."""
    return {
        "article_id": chunk.get("article_id"),
        "similarity_score": chunk.get("similarity_score", 0.0),
        "snippet": chunk.get("snippet") or chunk.get("chunk_preview") or chunk.get("content", "")[:200],
    }


def _collect_tool_calls(agent_trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten tool calls from grouped agent traces for the UI."""
    tool_calls: list[dict[str, Any]] = []
    for item in agent_trace or []:
        for tool_call in item.get("tool_calls", []) or []:
            tool_calls.append(
                {
                    "agent_name": item.get("agent_name", "Agent"),
                    "role": item.get("role", ""),
                    **tool_call,
                }
            )
    return tool_calls


def _render_agent_cycle(cycle: dict[str, Any], index: int) -> None:
    """Render one agent ReAct cycle with its nested reasoning steps."""
    agent_name = cycle.get("agent_name", f"Agent {index}")
    role = cycle.get("role", "")
    mode = cycle.get("mode", "react")
    answer = cycle.get("output_text", "")
    reasoning = cycle.get("reasoning", "")
    error = cycle.get("error", "")
    steps = cycle.get("steps", []) or []
    tool_calls = cycle.get("tool_calls", []) or []

    header = f"{index}. {agent_name}"
    if role:
        header += f" · {role}"
    if mode:
        header += f" · {mode}"

    with st.expander(header, expanded=index == 1):
        st.markdown(answer or "No answer returned.")
        if reasoning:
            st.caption(reasoning)
        if error:
            st.warning(error)

        if steps:
            st.markdown("**ReAct Steps**")
            for step_index, step in enumerate(steps, 1):
                st.markdown(f"**Step {step_index}**")
                st.markdown(f"**Thought:** {step.get('thought', '')}")
                st.markdown(f"**Action:** {step.get('action', '')}")
                if step.get("tool_name"):
                    st.markdown(f"**Tool:** {step.get('tool_name')}")
                if step.get("tool_input"):
                    st.code(json.dumps(step.get("tool_input", {}), ensure_ascii=False, indent=2), language="json")
                if step.get("observation"):
                    st.code(step.get("observation", ""), language="text")

        if tool_calls:
            with st.expander("Tool Calls", expanded=False):
                for tool_call in tool_calls:
                    st.markdown(f"**{tool_call.get('name', 'unknown')}**")
                    if tool_call.get("args"):
                        st.code(json.dumps(tool_call.get("args", {}), ensure_ascii=False, indent=2), language="json")
                    if tool_call.get("result") is not None:
                        st.code(json.dumps(tool_call.get("result"), ensure_ascii=False, indent=2), language="json")


def _render_agent_trace(agent_trace: list[dict[str, Any]]) -> None:
    """Render either grouped agent cycles or the legacy flat trace list."""
    if not agent_trace:
        return

    first_item = agent_trace[0]
    if isinstance(first_item, dict) and "steps" in first_item:
        with st.expander("🧠 Agent Reasoning Trace", expanded=False):
            for index, cycle in enumerate(agent_trace, 1):
                _render_agent_cycle(cycle, index)
        return

    with st.expander("🔗 Agent Reasoning Trace", expanded=False):
        for step in agent_trace:
            st.write(f"**Thought**: {step.get('thought', '')}\n\n**Action**: {step.get('action', '')} (**Tool**: {step.get('tool', 'N/A')})")



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
    """Generate an AI response to the user message."""
    response_data = {
        "answer": "",
        "retrieved_chunks": [],
        "agent_trace": [],
        "tool_calls": [],
        "error": None,
    }

    try:
        retrieved_chunks = []
        if rag_enabled:
            try:
                retrieved_chunks = search_articles(
                    query=user_message,
                    top_k=top_k_retrieval,
                )
                response_data["retrieved_chunks"] = [_summarize_retrieved_chunk(chunk) for chunk in retrieved_chunks]
                logger.info(f"Retrieved {len(retrieved_chunks)} chunks")
            except Exception as e:
                logger.warning(f"RAG retrieval failed: {str(e)}")
                retrieved_chunks = []

        if multi_agent_enabled:
            logger.info("Executing multi-agent pipeline")
            if retrieved_chunks:
                logger.info(f"Retrieved {len(retrieved_chunks)} chunks for agents")

            agent_output = run_multi_agent_pipeline(
                user_message,
                context=retrieved_chunks,
                llm_client=llm_client,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_iterations=5,
            )
            response_data["agent_trace"] = agent_output.get("agent_trace", agent_output.get("traces", []))
            response_data["tool_calls"] = _collect_tool_calls(response_data["agent_trace"])
            response_data["answer"] = agent_output.get("final_answer", "")
            if agent_output.get("error_message"):
                response_data["error"] = agent_output.get("error_message")
            logger.info(f"Multi-agent response generated: {len(response_data['answer'])} chars")
            return response_data

        if tool_calling_enabled:
            logger.info("Executing ReAct reasoning loop")
            react_output = run_react_loop(
                query=user_message,
                llm_client=llm_client,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_iterations=5,
                retrieved_context=retrieved_chunks,
            )
            response_data["answer"] = react_output.get("answer", "")
            response_data["agent_trace"] = [
                {
                    "agent_name": "ReAct",
                    "role": "Tool-Using Analyst",
                    "input_text": user_message,
                    "output_text": react_output.get("answer", ""),
                    "reasoning": "Single-agent ReAct reasoning cycle",
                    "timestamp": datetime.now().isoformat(),
                    "mode": react_output.get("mode", "react"),
                    "steps": react_output.get("steps", []),
                    "tool_calls": react_output.get("tool_calls", []),
                    "error": react_output.get("error") or "",
                }
            ]
            response_data["tool_calls"] = react_output.get("tool_calls", [])
            response_data["error"] = react_output.get("error")
            logger.info(f"ReAct completed with {len(response_data['tool_calls'])} tool calls")
            return response_data

        # Handle standard RAG mode
        tool_context_text = ""
        if retrieved_chunks:
            tool_context_text = json.dumps(retrieved_chunks, ensure_ascii=False, default=str, indent=2)

        prompt = build_prompt(
            user_message=user_message,
            rag_enabled=rag_enabled,
            retrieved_chunks=retrieved_chunks,
            tool_context_text=tool_context_text,
        )

        try:
            llm_response = llm_client.complete(
                prompt=prompt,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_tokens=settings.max_tokens,
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
                base_url=settings.lm_studio_base_url,
                model_name=settings.chat_model,
                timeout=settings.request_timeout,
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
