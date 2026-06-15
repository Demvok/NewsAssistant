# Multi-Agent System Implementation with LangGraph

## Overview

Implemented a comprehensive multi-agent reasoning system for the AI News Intelligence Assistant using **LangGraph** for orchestration and state management. The system consists of three specialized agents working in a sequential pipeline for news analysis.

## Architecture

### Three-Agent Pipeline

1. **Journalist Agent** (News Analyst)
   - Role: Initial analysis and answer formation
   - Input: Query + Retrieved document context
   - Output: Comprehensive analysis with supporting evidence
   - Purpose: Forms initial response grounded in corpus

2. **Fact Checker Agent** (Verification Specialist)
   - Role: Verification and critique
   - Input: Query + Journalist's response + Original context
   - Output: Critical evaluation, identifies unsupported claims
   - Purpose: Ensures accuracy and evidence quality

3. **Editor Agent** (Synthesis & Refinement)
   - Role: Final synthesis and balance
   - Input: Journalist's response + Fact Checker's critique + All prior context
   - Output: Final polished, balanced answer
   - Purpose: Produces publication-ready synthesis

### Graph Flow

```
START → journalist → fact_checker → editor → END
```

- Linear pipeline execution
- Each agent processes the full state and updates it
- State immutability maintained throughout
- All intermediate outputs preserved for audit trail

## Data Structures

### AgentState (TypedDict)

Represents the complete state flowing through the graph:

```python
class AgentState(TypedDict):
    query: str                      # User query
    context: list[dict]             # Retrieved context
    journalist_response: str        # Initial analysis
    journalist_reasoning: str       # (Reserved for future)
    fact_checker_critique: str      # Verification critique
    fact_checker_reasoning: str     # (Reserved for future)
    editor_synthesis: str           # Final answer
    editor_reasoning: str           # (Reserved for future)
    traces: list[AgentTrace]       # All agent execution traces
    error_message: str              # Error details if any
```

### AgentTrace (TypedDict)

Represents a single agent execution step with audit information:

```python
class AgentTrace(TypedDict):
    agent_name: str        # "Journalist", "Fact Checker", "Editor"
    role: str              # Full role description
    input_text: str        # Summarized input to agent
    output_text: str       # Agent's output/response
    reasoning: str         # Why the agent produced this output
    timestamp: str         # ISO format timestamp
```

## Key Features

### 1. Trace Information

Each agent execution creates an `AgentTrace` containing:
- Agent identification (name and role)
- Input and output text (full responses)
- Reasoning explanation
- Precise timestamp for performance monitoring
- Full audit trail for transparency and debugging

Access traces via:
```python
result = run_multi_agent_pipeline(query, context)
for trace in result['traces']:
    print(f"{trace['agent_name']} ({trace['role']})")
    print(f"  Reasoning: {trace['reasoning']}")
    print(f"  Timestamp: {trace['timestamp']}")
```

### 2. Integration with Config

- Uses `settings` from `config.py` for all model parameters
- Temperature, top_p, top_k, max_tokens all configurable
- LM Studio endpoint and model names from config
- Centralized configuration management

### 3. Error Handling

- Graceful error handling at each agent level
- Pipeline continues even if individual agents error
- Error messages captured in state
- Detailed logging at INFO and DEBUG levels

### 4. LangGraph Integration

- Uses `StateGraph` for state management
- `START` and `END` special nodes for pipeline boundaries
- `.add_node()` for agent nodes
- `.add_edge()` for control flow
- `.compile()` for optimized execution
- `.invoke()` for execution with initial state

## API

### `run_multi_agent_pipeline(query: str, context: list[dict] = None) -> dict`

Main entry point for multi-agent reasoning.

**Args:**
- `query`: User query or prompt (required, non-empty)
- `context`: List of retrieved document contexts (optional)

**Returns:**
```python
{
    "final_answer": str,              # Editor's synthesized response
    "journalist_response": str,       # Initial analysis
    "fact_checker_critique": str,    # Verification critique
    "traces": list[AgentTrace],      # All agent execution traces
    "error_message": str              # "" if successful, error details otherwise
}
```

**Raises:**
- `ValueError`: If query is empty or None

### `get_agent_trace() -> list[dict]`

Convenience function for accessing traces. Returns empty list in stateless context.
For actual traces, access via pipeline output's `traces` key.

## Usage Examples

### Basic Usage

```python
from core.agents import run_multi_agent_pipeline

query = "What are the main topics in the news corpus?"
context = [
    {"content": "Green energy...", "source": "article_1.txt"},
    {"content": "Trade war...", "source": "article_2.txt"},
]

result = run_multi_agent_pipeline(query, context)

print("Final Answer:")
print(result['final_answer'])
```

### With Trace Inspection

```python
result = run_multi_agent_pipeline(query, context)

# Inspect each agent's work
for trace in result['traces']:
    print(f"\n=== {trace['agent_name']} ({trace['role']}) ===")
    print(f"Timestamp: {trace['timestamp']}")
    print(f"Reasoning: {trace['reasoning']}")
    print(f"Output (first 200 chars): {trace['output_text'][:200]}...")
```

### Viewing Complete Pipeline Output

```python
result = run_multi_agent_pipeline(query, context)

print("\n=== Journalist Analysis ===")
print(result['journalist_response'])

print("\n=== Fact Checker Critique ===")
print(result['fact_checker_critique'])

print("\n=== Final Synthesis ===")
print(result['final_answer'])

if result['error_message']:
    print(f"\nErrors: {result['error_message']}")
```

## Integration with Chat Page

In `pages/1_Chat.py`, the multi-agent system would be used like:

```python
if multi_agent_mode and rag_enabled:
    # Retrieve context first
    context = retrieve_context(user_query)
    
    # Run multi-agent pipeline
    result = run_multi_agent_pipeline(user_query, context)
    
    # Display results
    st.write("### Final Answer")
    st.write(result['final_answer'])
    
    if st.checkbox("Show reasoning traces"):
        for trace in result['traces']:
            st.write(f"**{trace['agent_name']} ({trace['role']})**")
            st.write(f"_Reasoning:_ {trace['reasoning']}")
            st.write(f"_Output:_ {trace['output_text']}")
```

## Performance Considerations

- **Sequential Execution**: Agents run one after another (not parallel)
  - Ensures each agent sees all prior outputs
  - Simplifies state management
  - Suitable for analytical tasks
  
- **Temperature Settings**:
  - Journalist: Default temperature (creative analysis)
  - Fact Checker: Default temperature (balanced evaluation)
  - Editor: Reduced temperature (0.2 lower) for more deterministic synthesis

- **Latency**:
  - Each agent makes one LLM call
  - Total latency ≈ 3× single-agent latency
  - Acceptable for background analysis

## Files Modified/Created

- **`core/agents.py`** - Main implementation (new)
- **`requirements.txt`** - Added `langgraph` dependency

## Testing

Three test files verify the implementation:

1. **`test_agents_basic.py`** - Basic imports and structure
   - Tests that all functions are callable
   - Verifies TypedDict structures
   - Tests error handling

2. **`test_agents_integration.py`** - Integration and API
   - Demonstrates output structure
   - Shows trace schema
   - Provides usage examples

Run tests:
```bash
python test_agents_basic.py
python test_agents_integration.py
```

## Future Enhancements

- **Parallel Execution**: Use LangGraph's branching for parallel agent evaluation
- **State Persistence**: Save/load agent states for offline analysis
- **Custom Routing**: Add conditional branching based on query type
- **Tool Integration**: Agents can call search, filter, or compute tools
- **Token Accounting**: Track token usage across agent chain
- **Feedback Loops**: Allow editor to send feedback back to journalist if needed

## Dependencies

- **langgraph** - Graph-based agent orchestration
- **langchain** - LLM chain abstractions
- **openai** - LM Studio API compatibility
- **pydantic** - Configuration management

All already in `requirements.txt`.
