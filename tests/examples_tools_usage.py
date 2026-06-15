"""Integration example: Using tools in the News Intelligence Assistant.

This example shows how to use the tools module in various contexts:
- Direct tool calls
- Using registry for dynamic tool selection
- Error handling patterns
- Integration with chat and agents
"""

from core import create_default_tools
from core.tools import search_by_topic, count_keyword_mentions, corpus_statistics


def example_1_direct_tool_usage():
    """Example 1: Direct tool function usage."""
    print("\n" + "=" * 60)
    print("Example 1: Direct Tool Usage")
    print("=" * 60)
    
    try:
        # Direct import and use
        from core.tools import search_by_topic, corpus_statistics
        
        # Get corpus stats
        print("\nGetting corpus statistics...")
        stats = corpus_statistics()
        print(f"  Documents: {stats['num_documents']}")
        print(f"  Chunks: {stats['num_chunks']}")
        print(f"  Avg chunk size: {stats['avg_chunk_length']:.0f} chars")
        
    except RuntimeError as e:
        print(f"  Knowledge base not initialized: {e}")


def example_2_registry_usage():
    """Example 2: Using the tool registry."""
    print("\n" + "=" * 60)
    print("Example 2: Tool Registry Usage")
    print("=" * 60)
    
    # Create default registry
    registry = create_default_tools()
    
    # List all available tools
    print("\nAvailable tools:")
    for tool_info in registry.list_tools():
        print(f"  - {tool_info['name']}")
        print(f"    {tool_info['description']}")
    
    # Use a tool from registry
    try:
        search_tool = registry.get_tool("search_by_topic")
        if search_tool:
            print("\nSearching by topic...")
            results = search_tool("green energy")
            print(f"  Found {results['num_results']} documents")
    except RuntimeError as e:
        print(f"  Knowledge base not initialized: {e}")


def example_3_error_handling():
    """Example 3: Error handling patterns."""
    print("\n" + "=" * 60)
    print("Example 3: Error Handling")
    print("=" * 60)
    
    from core.tools import search_by_topic, filter_by_date, count_keyword_mentions
    
    # Pattern 1: Input validation
    print("\nPattern 1: Input validation")
    try:
        search_by_topic("")
    except ValueError as e:
        print(f"  ✓ Caught validation error: {e}")
    
    # Pattern 2: Invalid date range
    print("\nPattern 2: Date validation")
    try:
        filter_by_date("2024-12-31", "2024-01-01")
    except ValueError as e:
        print(f"  ✓ Caught date error: {e}")
    
    # Pattern 3: Collection not initialized
    print("\nPattern 3: Runtime error handling")
    try:
        count_keyword_mentions("test")
    except RuntimeError as e:
        print(f"  ✓ Caught runtime error: Collection not initialized")


def example_4_chat_integration():
    """Example 4: Integration with chat interface.
    
    Shows how tools would be used in a chat context.
    """
    print("\n" + "=" * 60)
    print("Example 4: Chat Integration Pattern")
    print("=" * 60)
    
    registry = create_default_tools()
    
    # Simulated user messages that trigger tools
    user_queries = [
        {
            "user_message": "Tell me about renewable energy",
            "tool_name": "search_by_topic",
            "tool_args": {"topic": "renewable energy"},
        },
        {
            "user_message": "What was published in January?",
            "tool_name": "filter_by_date",
            "tool_args": {"start_date": "2024-01-01", "end_date": "2024-01-31"},
        },
        {
            "user_message": "How often is 'tariff' mentioned?",
            "tool_name": "count_keyword_mentions",
            "tool_args": {"keyword": "tariff"},
        },
        {
            "user_message": "Give me corpus statistics",
            "tool_name": "corpus_statistics",
            "tool_args": {},
        },
    ]
    
    print("\nSimulated chat interactions:")
    for i, query in enumerate(user_queries, 1):
        print(f"\n  {i}. User: {query['user_message']}")
        tool_name = query["tool_name"]
        
        tool = registry.get_tool(tool_name)
        if tool:
            try:
                print(f"     Tool: {tool_name}")
                print(f"     Args: {query['tool_args']}")
                # In real implementation: result = tool(**query['tool_args'])
                # Would process result and include in response
            except Exception as e:
                print(f"     Error: {e}")


def example_5_agent_integration():
    """Example 5: Integration with multi-agent system.
    
    Shows how tools would be exposed to agents.
    """
    print("\n" + "=" * 60)
    print("Example 5: Multi-Agent Integration Pattern")
    print("=" * 60)
    
    registry = create_default_tools()
    
    # Agent would have access to tool registry
    print("\nAgent tool interface:")
    print("  Available functions for agents:")
    
    for tool_info in registry.list_tools():
        print(f"\n  Function: {tool_info['name']}")
        print(f"  Description: {tool_info['description']}")
    
    print("\n  Agent execution pattern:")
    print("    1. Agent decides which tool to use")
    print("    2. Retrieves tool from registry")
    print("    3. Prepares arguments")
    print("    4. Executes tool")
    print("    5. Processes results")
    print("    6. Includes results in response")


def example_6_workflow():
    """Example 6: Complete workflow example."""
    print("\n" + "=" * 60)
    print("Example 6: Complete Workflow")
    print("=" * 60)
    
    registry = create_default_tools()
    
    workflow_steps = [
        ("Setup", "Create tool registry with 4 tools"),
        ("Query", "Semantic search by topic"),
        ("Filter", "Date range filtering"),
        ("Analyze", "Count keyword mentions"),
        ("Report", "Generate corpus statistics"),
        ("Integration", "Return results to user/agent"),
    ]
    
    print("\nWorkflow steps:")
    for step_num, (step_name, description) in enumerate(workflow_steps, 1):
        print(f"  {step_num}. {step_name}: {description}")
    
    print("\nExample execution sequence:")
    print("  Step 1: Registry created ✓")
    print("  Step 2: Tool 'search_by_topic' available")
    print("  Step 3: Tool 'filter_by_date' available")
    print("  Step 4: Tool 'count_keyword_mentions' available")
    print("  Step 5: Tool 'corpus_statistics' available")
    print("  Step 6: Results ready for UI/agent")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("NEWS INTELLIGENCE ASSISTANT - TOOLS INTEGRATION EXAMPLES")
    print("=" * 60)
    
    # Run examples
    example_1_direct_tool_usage()
    example_2_registry_usage()
    example_3_error_handling()
    example_4_chat_integration()
    example_5_agent_integration()
    example_6_workflow()
    
    print("\n" + "=" * 60)
    print("✓ All examples completed")
    print("=" * 60 + "\n")
