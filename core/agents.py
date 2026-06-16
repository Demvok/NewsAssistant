"""Multi-agent reasoning module with LangGraph.

Implements three specialized agents for news intelligence analysis:
- Journalist: Initial analysis and answer formation
- Fact Checker: Verification and critique of the answer
- Editor: Synthesis of feedback into final balanced response

Uses LangGraph for state management, control flow, and trace tracking.
"""

import logging
from typing import TypedDict, Any
from datetime import datetime

from langgraph.graph import StateGraph, START, END
from config import settings
from core.llm_client import get_llm_client

logger = logging.getLogger(__name__)


class AgentTrace(TypedDict):
    """Represents a single agent execution step."""

    agent_name: str
    role: str
    input_text: str
    output_text: str
    reasoning: str
    timestamp: str


class AgentState(TypedDict):
    """State passed through the multi-agent graph."""

    query: str
    context: list[dict]
    journalist_response: str
    journalist_reasoning: str
    fact_checker_critique: str
    fact_checker_reasoning: str
    editor_synthesis: str
    editor_reasoning: str
    traces: list[AgentTrace]
    error_message: str


def _create_system_prompt(role: str, guidelines: str) -> str:
    """Create a system prompt for an agent.
    
    Args:
        role: Agent role description
        guidelines: Specific guidelines for the agent
        
    Returns:
        Formatted system prompt
    """
    return f"""You are a {role} specialized in news intelligence analysis.

Your responsibilities:
{guidelines}

Respond concisely and analytically. Be direct and evidence-based."""


def journalist_node(state: AgentState) -> AgentState:
    """Journalist agent: Initial analysis and answer formation.
    
    Reads the query and context to form an initial comprehensive answer
    with supporting evidence from the corpus.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with journalist response and trace
    """
    try:
        query = state["query"]
        context = state.get("context", [])
        
        # Build context string from retrieval results
        context_str = ""
        if context:
            context_str = "\n\nRetrieved context:\n"
            for i, item in enumerate(context[:5], 1):
                content = item.get("content", "")[:300]
                source = item.get("source", "Unknown")
                context_str += f"{i}. [{source}]: {content}\n"
        
        guidelines = """1. Analyze the query comprehensively
                        2. Use retrieved context as the primary source of information
                        3. Form a clear, well-supported initial answer
                        4. Identify key claims and evidence
                        5. Note any gaps or uncertainties in the corpus"""
        
        system_prompt = _create_system_prompt("Journalist", guidelines)
        
        user_prompt = f"""Query: {query}
                        {context_str}

                        Provide a comprehensive initial answer to this query based on the retrieved context. 
                        Structure your response with:
                        1. Main answer/findings
                        2. Supporting evidence from the corpus
                        3. Key points and implications"""
        
        llm_client = get_llm_client(
            base_url=settings.lm_studio_base_url,
            model_name=settings.chat_model,
            timeout=settings.request_timeout,
        )
        
        response = llm_client.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=settings.temperature,
            top_p=settings.top_p,
            max_tokens=settings.max_tokens,
        )
        
        journalist_response = response.text
        
        # Add trace
        trace = AgentTrace(
            agent_name="Journalist",
            role="News Analyst",
            input_text=query,
            output_text=journalist_response,
            reasoning="Initial comprehensive analysis based on corpus context",
            timestamp=datetime.now().isoformat(),
        )
        
        state["journalist_response"] = journalist_response
        state["traces"].append(trace)
        
        logger.info(f"Journalist agent completed: {len(journalist_response)} chars")
        
    except Exception as e:
        logger.error(f"Journalist agent error: {str(e)}")
        state["error_message"] = f"Journalist agent failed: {str(e)}"
        state["journalist_response"] = "Error generating journalist analysis"
    
    return state


def fact_checker_node(state: AgentState) -> AgentState:
    """Fact Checker agent: Verification and critique.
    
    Reviews the journalist's answer for logical consistency, evidence quality,
    potential biases, and verifiable claims against the context.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with fact checker critique and trace
    """
    try:
        query = state["query"]
        journalist_response = state.get("journalist_response", "")
        context = state.get("context", [])
        
        context_str = ""
        if context:
            context_str = "\n\nAvailable corpus context:\n"
            for i, item in enumerate(context[:5], 1):
                content = item.get("content", "")[:200]
                source = item.get("source", "Unknown")
                context_str += f"{i}. [{source}]: {content}\n"
        
        guidelines = """1. Verify claims against provided context
2. Identify unsupported assertions
3. Check for logical consistency
4. Identify potential biases or gaps
5. Highlight evidence quality issues"""
        
        system_prompt = _create_system_prompt("Fact Checker", guidelines)
        
        user_prompt = f"""Original query: {query}

Journalist's answer:
{journalist_response}
{context_str}

Critique this analysis by:
1. Verifying each major claim against the corpus
2. Identifying any unsupported statements
3. Assessing logical consistency
4. Noting gaps or areas lacking evidence
5. Providing specific corrections or clarifications needed

Be critical but fair. Focus on evidence quality and accuracy."""
        
        llm_client = get_llm_client(
            base_url=settings.lm_studio_base_url,
            model_name=settings.chat_model,
            timeout=settings.request_timeout,
        )
        
        response = llm_client.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=settings.temperature,
            top_p=settings.top_p,
            max_tokens=settings.max_tokens,
        )
        
        critique = response.text
        
        # Add trace
        trace = AgentTrace(
            agent_name="Fact Checker",
            role="Verification Specialist",
            input_text=journalist_response[:200],
            output_text=critique,
            reasoning="Critical verification against corpus evidence",
            timestamp=datetime.now().isoformat(),
        )
        
        state["fact_checker_critique"] = critique
        state["traces"].append(trace)
        
        logger.info(f"Fact Checker agent completed: {len(critique)} chars")
        
    except Exception as e:
        logger.error(f"Fact Checker agent error: {str(e)}")
        state["error_message"] = f"Fact Checker agent failed: {str(e)}"
        state["fact_checker_critique"] = "Error generating critique"
    
    return state


def editor_node(state: AgentState) -> AgentState:
    """Editor agent: Synthesis into final balanced response.
    
    Synthesizes the journalist's analysis and fact checker's critique
    into a final, balanced, well-supported answer.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with editor synthesis and trace
    """
    try:
        query = state["query"]
        journalist_response = state.get("journalist_response", "")
        critique = state.get("fact_checker_critique", "")
        
        guidelines = """1. Synthesize journalist's analysis with fact checker's critique
2. Maintain evidence-based accuracy
3. Present balanced perspective
4. Resolve contradictions logically
5. Produce clear, well-structured final answer
6. Indicate confidence levels and areas of uncertainty"""
        
        system_prompt = _create_system_prompt("Editor", guidelines)
        
        user_prompt = f"""Original query: {query}

Initial analysis from journalist:
{journalist_response}

Critique from fact checker:
{critique}

Create a final synthesized answer that:
1. Incorporates the journalist's key findings
2. Addresses all critique points from the fact checker
3. Maintains logical consistency and evidence-based reasoning
4. Is well-organized and clear
5. Explicitly notes areas of certainty vs. uncertainty

Produce a balanced, final answer suitable for publishing."""
        
        llm_client = get_llm_client(
            base_url=settings.lm_studio_base_url,
            model_name=settings.chat_model,
            timeout=settings.request_timeout,
        )
        
        response = llm_client.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=max(0.5, settings.temperature - 0.2),
            top_p=settings.top_p,
            max_tokens=settings.max_tokens,
        )
        
        final_answer = response.text
        
        # Add trace
        trace = AgentTrace(
            agent_name="Editor",
            role="Synthesis & Refinement",
            input_text=f"Journalist + Critique ({len(journalist_response) + len(critique)} chars)",
            output_text=final_answer,
            reasoning="Synthesis of analysis and critique into final answer",
            timestamp=datetime.now().isoformat(),
        )
        
        state["editor_synthesis"] = final_answer
        state["traces"].append(trace)
        
        logger.info(f"Editor agent completed: {len(final_answer)} chars")
        
    except Exception as e:
        logger.error(f"Editor agent error: {str(e)}")
        state["error_message"] = f"Editor agent failed: {str(e)}"
        state["editor_synthesis"] = "Error generating final synthesis"
    
    return state


def _build_graph():
    """Build the LangGraph state graph for multi-agent reasoning.
    
    Returns:
        Compiled graph ready for execution
    """
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("journalist", journalist_node)
    graph.add_node("fact_checker", fact_checker_node)
    graph.add_node("editor", editor_node)
    
    # Define edges (linear pipeline)
    graph.add_edge(START, "journalist")
    graph.add_edge("journalist", "fact_checker")
    graph.add_edge("fact_checker", "editor")
    graph.add_edge("editor", END)
    
    return graph.compile()


# Global graph instance
_graph: Any = None


def _get_graph():
    """Get or initialize the compiled graph.
    
    Returns:
        Compiled LangGraph graph
    """
    global _graph
    if _graph is None:
        _graph = _build_graph()
    return _graph


def run_multi_agent_pipeline(
    query: str,
    context: list[dict] = None,
) -> dict:
    """Run multi-agent reasoning pipeline.
    
    Orchestrates journalist → fact checker → editor workflow for
    comprehensive news analysis with verification and synthesis.
    
    Args:
        query: User query or prompt for analysis
        context: Optional list of retrieved document contexts
        
    Returns:
        Dictionary containing:
            - final_answer: Synthesized response from editor
            - journalist_response: Initial analysis
            - fact_checker_critique: Verification critique
            - traces: List of all agent execution steps
            - error_message: Error details if any step failed
            
    Raises:
        ValueError: If query is empty
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")
    
    logger.info(f"Starting multi-agent pipeline for query: {query[:100]}...")
    
    # Initialize state
    initial_state: AgentState = {
        "query": query,
        "context": context or [],
        "journalist_response": "",
        "journalist_reasoning": "",
        "fact_checker_critique": "",
        "fact_checker_reasoning": "",
        "editor_synthesis": "",
        "editor_reasoning": "",
        "traces": [],
        "error_message": "",
    }
    
    try:
        graph = _get_graph()
        result = graph.invoke(initial_state)
        
        logger.info(f"Multi-agent pipeline completed with {len(result['traces'])} traces")
        
        return {
            "final_answer": result["editor_synthesis"],
            "journalist_response": result["journalist_response"],
            "fact_checker_critique": result["fact_checker_critique"],
            "traces": result["traces"],
            "error_message": result["error_message"],
        }
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        return {
            "final_answer": "Error: Multi-agent pipeline failed",
            "journalist_response": "",
            "fact_checker_critique": "",
            "traces": initial_state["traces"],
            "error_message": str(e),
        }


def get_agent_trace() -> list[dict]:
    """Retrieve traces of agent reasoning steps.
    
    This function returns trace information from the most recent execution.
    In a stateless context, traces are included in the pipeline output.
    
    Note: For direct access to traces, use run_multi_agent_pipeline() and
    access the 'traces' key in the returned dictionary.
    
    Returns:
        List of agent interaction records (empty in stateless context)
    """
    return []
