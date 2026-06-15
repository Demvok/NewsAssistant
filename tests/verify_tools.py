#!/usr/bin/env python
"""Verify tools module implementation."""

import sys

print("=" * 60)
print("Verifying core/tools.py implementation")
print("=" * 60)

try:
    # Test 1: Import from core module
    from core import create_default_tools
    print("\n✓ Test 1: Successfully imported from core module")
    
    # Test 2: Create registry
    registry = create_default_tools()
    print("✓ Test 2: Successfully created default tool registry")
    
    # Test 3: Check tool count
    tools = registry.list_tools()
    assert len(tools) == 4, f"Expected 4 tools, got {len(tools)}"
    print(f"✓ Test 3: Registry contains {len(tools)} tools")
    
    # Test 4: Verify tool names
    tool_names = [t["name"] for t in tools]
    expected_names = ["search_by_topic", "filter_by_date", "count_keyword_mentions", "corpus_statistics"]
    for name in expected_names:
        assert name in tool_names, f"Missing tool: {name}"
    print(f"✓ Test 4: All expected tools present: {', '.join(tool_names)}")
    
    # Test 5: Verify tool descriptions
    for tool in tools:
        assert "description" in tool, f"Tool {tool['name']} missing description"
        assert len(tool["description"]) > 0, f"Tool {tool['name']} has empty description"
    print("✓ Test 5: All tools have descriptions")
    
    # Test 6: Verify tool retrieval
    search_tool = registry.get_tool("search_by_topic")
    assert search_tool is not None, "Failed to get search_by_topic tool"
    print("✓ Test 6: Successfully retrieved tool from registry")
    
    # Test 7: Verify error handling for invalid inputs
    from core.tools import search_by_topic, filter_by_date, count_keyword_mentions
    
    try:
        search_by_topic("")
        print("✗ Test 7a: Should raise ValueError for empty topic")
        sys.exit(1)
    except ValueError:
        print("✓ Test 7a: Correctly validates empty topic")
    
    try:
        filter_by_date("2024-12-31", "2024-01-01")
        print("✗ Test 7b: Should raise ValueError for invalid date range")
        sys.exit(1)
    except ValueError:
        print("✓ Test 7b: Correctly validates date range")
    
    try:
        count_keyword_mentions("")
        print("✗ Test 7c: Should raise ValueError for empty keyword")
        sys.exit(1)
    except ValueError:
        print("✓ Test 7c: Correctly validates empty keyword")
    
    print("\n" + "=" * 60)
    print("✓ All verification tests passed!")
    print("=" * 60)
    
except Exception as e:
    print(f"\n✗ Verification failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
