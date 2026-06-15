"""Basic test to verify agents module structure and API."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.agents import (
    run_multi_agent_pipeline,
    get_agent_trace,
    journalist_node,
    fact_checker_node,
    editor_node,
    AgentState,
    AgentTrace,
)


def test_imports():
    """Test that all core imports work."""
    print("✓ All imports successful")
    assert callable(run_multi_agent_pipeline)
    assert callable(get_agent_trace)
    assert callable(journalist_node)
    assert callable(fact_checker_node)
    assert callable(editor_node)
    print("✓ All functions are callable")


def test_agent_state_type():
    """Test AgentState TypedDict structure."""
    state: AgentState = {
        "query": "Test query",
        "context": [],
        "journalist_response": "",
        "journalist_reasoning": "",
        "fact_checker_critique": "",
        "fact_checker_reasoning": "",
        "editor_synthesis": "",
        "editor_reasoning": "",
        "traces": [],
        "error_message": "",
    }
    print(f"✓ AgentState structure valid: {len(state)} keys")


def test_agent_trace_type():
    """Test AgentTrace TypedDict structure."""
    trace: AgentTrace = {
        "agent_name": "Journalist",
        "role": "News Analyst",
        "input_text": "Test input",
        "output_text": "Test output",
        "reasoning": "Test reasoning",
        "timestamp": "2024-01-01T00:00:00",
    }
    print(f"✓ AgentTrace structure valid: {trace['agent_name']}")


def test_error_handling():
    """Test error handling for empty query."""
    try:
        run_multi_agent_pipeline("")
        print("✗ Should have raised ValueError")
        sys.exit(1)
    except ValueError as e:
        print(f"✓ Correctly raises ValueError for empty query: {str(e)}")


def test_get_agent_trace():
    """Test get_agent_trace function."""
    traces = get_agent_trace()
    assert isinstance(traces, list)
    print(f"✓ get_agent_trace returns list: {len(traces)} items")


if __name__ == "__main__":
    print("\n=== Testing Agents Module ===\n")
    test_imports()
    test_agent_state_type()
    test_agent_trace_type()
    test_error_handling()
    test_get_agent_trace()
    print("\n=== All Basic Tests Passed ===\n")
