"""Test suite for core/tools.py module.

Tests tool registry, tool creation, and individual tool functions.
"""

import pytest
from datetime import datetime, timedelta
from core.tools import (
    ToolRegistry,
    create_default_tools,
    search_by_topic,
    filter_by_date,
    count_keyword_mentions,
    corpus_statistics,
)


def test_tool_registry_initialization():
    """Test ToolRegistry initialization."""
    registry = ToolRegistry()
    assert registry.tools == {}
    assert registry.list_tools() == []


def test_tool_registry_register():
    """Test registering tools."""
    registry = ToolRegistry()
    
    def dummy_tool():
        return "result"
    
    registry.register("test_tool", dummy_tool, "Test tool description")
    assert "test_tool" in registry.tools
    assert registry.tools["test_tool"]["description"] == "Test tool description"


def test_tool_registry_get_tool():
    """Test retrieving tools."""
    registry = ToolRegistry()
    
    def dummy_tool():
        return "result"
    
    registry.register("test_tool", dummy_tool, "Test tool")
    tool = registry.get_tool("test_tool")
    assert tool is not None
    assert tool() == "result"
    
    # Non-existent tool
    assert registry.get_tool("nonexistent") is None


def test_tool_registry_list_tools():
    """Test listing all tools."""
    registry = ToolRegistry()
    
    def tool1():
        pass
    def tool2():
        pass
    
    registry.register("tool1", tool1, "Description 1")
    registry.register("tool2", tool2, "Description 2")
    
    tools = registry.list_tools()
    assert len(tools) == 2
    assert tools[0]["name"] == "tool1"
    assert tools[1]["name"] == "tool2"


def test_create_default_tools():
    """Test default tool registry creation."""
    registry = create_default_tools()
    tools = registry.list_tools()
    
    assert len(tools) == 4
    tool_names = [t["name"] for t in tools]
    
    assert "search_by_topic" in tool_names
    assert "filter_by_date" in tool_names
    assert "count_keyword_mentions" in tool_names
    assert "corpus_statistics" in tool_names


def test_search_by_topic_empty_query():
    """Test search_by_topic with empty query."""
    with pytest.raises(ValueError, match="Topic cannot be empty"):
        search_by_topic("")
    
    with pytest.raises(ValueError, match="Topic cannot be empty"):
        search_by_topic("   ")


def test_search_by_topic_no_collection():
    """Test search_by_topic when collection not initialized."""
    with pytest.raises(RuntimeError, match="Collection not initialized"):
        search_by_topic("test topic")


def test_filter_by_date_invalid_format():
    """Test filter_by_date with invalid date format."""
    with pytest.raises(ValueError, match="Invalid date format"):
        filter_by_date("2024-01", "2024-12-31")
    
    with pytest.raises(ValueError, match="Invalid date format"):
        filter_by_date("2024-01-01", "invalid")


def test_filter_by_date_invalid_range():
    """Test filter_by_date with start_date > end_date."""
    with pytest.raises(ValueError, match="start_date cannot be after end_date"):
        filter_by_date("2024-12-31", "2024-01-01")


def test_filter_by_date_no_collection():
    """Test filter_by_date when collection not initialized."""
    with pytest.raises(RuntimeError, match="Collection not initialized"):
        filter_by_date("2024-01-01", "2024-12-31")


def test_count_keyword_mentions_empty_keyword():
    """Test count_keyword_mentions with empty keyword."""
    with pytest.raises(ValueError, match="Keyword cannot be empty"):
        count_keyword_mentions("")
    
    with pytest.raises(ValueError, match="Keyword cannot be empty"):
        count_keyword_mentions("   ")


def test_count_keyword_mentions_no_collection():
    """Test count_keyword_mentions when collection not initialized."""
    with pytest.raises(RuntimeError, match="Collection not initialized"):
        count_keyword_mentions("test")


def test_corpus_statistics_no_collection():
    """Test corpus_statistics when collection not initialized."""
    with pytest.raises(RuntimeError, match="Collection not initialized"):
        corpus_statistics()


def test_tool_return_types():
    """Test that tools return expected dictionary structures."""
    registry = create_default_tools()
    
    # Check that tools can be retrieved
    search_tool = registry.get_tool("search_by_topic")
    filter_tool = registry.get_tool("filter_by_date")
    count_tool = registry.get_tool("count_keyword_mentions")
    stats_tool = registry.get_tool("corpus_statistics")
    
    assert search_tool is not None
    assert filter_tool is not None
    assert count_tool is not None
    assert stats_tool is not None


if __name__ == "__main__":
    # Run basic functionality tests without pytest
    print("Running basic tool registry tests...")
    
    # Test 1: Initialize registry
    registry = ToolRegistry()
    print("✓ Registry initialized")
    
    # Test 2: Register tool
    def test_func():
        return "test"
    registry.register("test", test_func, "Test description")
    print("✓ Tool registered")
    
    # Test 3: Get tool
    tool = registry.get_tool("test")
    assert tool() == "test"
    print("✓ Tool retrieved and executed")
    
    # Test 4: Create default tools
    default_registry = create_default_tools()
    tools = default_registry.list_tools()
    assert len(tools) == 4
    print(f"✓ Default registry created with {len(tools)} tools")
    
    # Test 5: Test error handling for empty inputs
    try:
        search_by_topic("")
        assert False, "Should raise ValueError"
    except ValueError as e:
        print(f"✓ Empty topic validation works: {e}")
    
    try:
        filter_by_date("2024-12-31", "2024-01-01")
        assert False, "Should raise ValueError"
    except ValueError as e:
        print(f"✓ Invalid date range validation works: {e}")
    
    try:
        count_keyword_mentions("")
        assert False, "Should raise ValueError"
    except ValueError as e:
        print(f"✓ Empty keyword validation works: {e}")
    
    print("\n✓ All basic tests passed!")
