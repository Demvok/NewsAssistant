"""
Quick Reference: Multi-Agent System API
========================================

This module provides multi-agent reasoning for the News Intelligence Assistant
using LangGraph orchestration with three specialized agents:
- Journalist (initial analysis)
- Fact Checker (verification)
- Editor (final synthesis)
"""

# ============================================================================
# QUICK START
# ============================================================================

from core.agents import run_multi_agent_pipeline

# Basic usage
query = "What are the key trends in renewable energy?"
result = run_multi_agent_pipeline(query)

print(result['final_answer'])


# ============================================================================
# WITH CONTEXT (RAG)
# ============================================================================

from core.rag import retrieve_context

# Retrieve relevant documents
context = retrieve_context(query, top_k=5)

# Run multi-agent pipeline with context
result = run_multi_agent_pipeline(query, context)

print("Final Answer:", result['final_answer'])


# ============================================================================
# ACCESSING TRACES
# ============================================================================

# Get all agent execution traces
for trace in result['traces']:
    print(f"\n{trace['agent_name']} ({trace['role']})")
    print(f"  Input: {trace['input_text'][:100]}...")
    print(f"  Reasoning: {trace['reasoning']}")
    print(f"  Timestamp: {trace['timestamp']}")


# ============================================================================
# VIEWING INTERMEDIATE OUTPUTS
# ============================================================================

# See all stages of analysis
print("=== Journalist (Initial Analysis) ===")
print(result['journalist_response'])
print("\n=== Fact Checker (Verification) ===")
print(result['fact_checker_critique'])
print("\n=== Editor (Final Synthesis) ===")
print(result['final_answer'])


# ============================================================================
# ERROR HANDLING
# ============================================================================

try:
    result = run_multi_agent_pipeline("")  # Empty query
except ValueError as e:
    print(f"Input error: {e}")

# Check for errors in result
if result['error_message']:
    print(f"Pipeline error: {result['error_message']}")
else:
    print("Pipeline executed successfully")


# ============================================================================
# OUTPUT STRUCTURE
# ============================================================================

"""
run_multi_agent_pipeline() returns:
{
    "final_answer": str,              # Editor's synthesized answer
    "journalist_response": str,       # Initial analysis
    "fact_checker_critique": str,    # Critical evaluation
    "traces": list[AgentTrace],      # All agent execution traces
    "error_message": str              # "" if successful
}

AgentTrace contains:
{
    "agent_name": str,       # "Journalist", "Fact Checker", "Editor"
    "role": str,             # Full role description
    "input_text": str,       # Input to agent
    "output_text": str,      # Agent's response
    "reasoning": str,        # Why agent produced this output
    "timestamp": str         # ISO format timestamp
}
"""


# ============================================================================
# INTEGRATION WITH STREAMLIT
# ============================================================================

import streamlit as st
from core.agents import run_multi_agent_pipeline

# In your Streamlit page:
if st.session_state.multi_agent_mode:
    if st.session_state.rag_enabled:
        # Option 1: Use with context
        query = st.text_input("Your question:")
        if query:
            context = retrieve_context(query)
            result = run_multi_agent_pipeline(query, context)
    else:
        # Option 2: Use without context
        query = st.text_input("Your question:")
        if query:
            result = run_multi_agent_pipeline(query)
    
    # Display results
    st.write("### Final Answer")
    st.write(result['final_answer'])
    
    # Optional: Show agent traces
    with st.expander("Show Agent Traces"):
        for trace in result['traces']:
            st.write(f"**{trace['agent_name']}** ({trace['role']})")
            st.write(f"_{trace['reasoning']}_")
            st.write(trace['output_text'])
            st.write(f"⏱ {trace['timestamp']}")


# ============================================================================
# CONFIGURATION
# ============================================================================

"""
The multi-agent system uses configuration from config.py:

- LM_STUDIO_BASE_URL: Default "http://localhost:1234/v1"
- CHAT_MODEL: Default "gemma-4-e4b"
- DEFAULT_TEMPERATURE: Used by journalist & fact checker
- DEFAULT_TOP_P, DEFAULT_TOP_K: Sampling parameters
- MAX_TOKENS: Generation limit per agent

The Editor uses a lower temperature (DEFAULT_TEMPERATURE - 0.2) for 
more deterministic synthesis.

All configuration can be overridden via .env file:

    LM_STUDIO_BASE_URL=http://localhost:1234/v1
    CHAT_MODEL=gemma-4-e4b
    DEFAULT_TEMPERATURE=0.7
    DEFAULT_TOP_P=0.95
    DEFAULT_TOP_K=40
    MAX_TOKENS=512
"""


# ============================================================================
# PERFORMANCE TIPS
# ============================================================================

"""
1. Context Quality:
   - Retrieve high-quality context before calling pipeline
   - Journalists use context for grounding, fact checkers verify it
   - Poor context leads to lower quality synthesis

2. Temperature Settings:
   - Higher temperature (>0.7): More creative analysis
   - Lower temperature (<0.5): More deterministic output
   - Default 0.7 balances both

3. Token Limits:
   - Larger MAX_TOKENS = longer responses but more latency
   - Agent needs room for analysis and reasoning
   - Editor needs room for synthesis

4. Caching:
   - Graph is built once and cached in module
   - Subsequent calls reuse compiled graph
   - First call has ~10-20ms overhead

5. Parallel Use:
   - Pipeline is stateless (no shared state between calls)
   - Safe to call from multiple Streamlit sessions
   - Thread-safe graph execution
"""


# ============================================================================
# TROUBLESHOOTING
# ============================================================================

"""
Issue: "ModuleNotFoundError: No module named 'langgraph'"
Solution: pip install langgraph

Issue: Pipeline times out
Solution: 
  - Check LM Studio is running: http://localhost:1234
  - Increase request_timeout in config
  - Check model is loaded in LM Studio

Issue: Poor quality synthesis
Solution:
  - Improve retrieved context quality
  - Increase temperature for more creative analysis
  - Ensure documents are in ChromaDB

Issue: Traces are empty
Solution:
  - Traces are populated during execution
  - Check result['error_message'] for errors
  - Ensure LM Studio returns valid responses
"""


# ============================================================================
# ADVANCED: EXTENDING AGENTS
# ============================================================================

"""
To add a custom agent:

1. Define node function:
   def my_agent_node(state: AgentState) -> AgentState:
       # Process state
       output = llm_client.chat(...)
       
       # Create trace
       trace = AgentTrace(
           agent_name="My Agent",
           role="My Role",
           input_text=...,
           output_text=output.text,
           reasoning="...",
           timestamp=datetime.now().isoformat(),
       )
       state["traces"].append(trace)
       state["my_output"] = output.text
       return state

2. Add to graph (in _build_graph):
   graph.add_node("my_agent", my_agent_node)
   graph.add_edge("previous_node", "my_agent")
   graph.add_edge("my_agent", "next_node")

3. Update AgentState to include my_output field

4. Return my_output in run_multi_agent_pipeline return dict
"""
