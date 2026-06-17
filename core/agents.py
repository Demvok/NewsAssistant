"""Multi-agent reasoning module with LangGraph and nested ReAct cycles.

Each node in the graph now runs its own ReAct loop so the journalist,
fact checker, and editor can all inspect context, call tools, and then
hand their results to the next stage.
"""

import logging
from typing import Any, NotRequired, TypedDict
from datetime import datetime

from langgraph.graph import StateGraph, START, END
from config import settings
from core.llm_client import get_llm_client
from core.react import run_react_loop

logger = logging.getLogger(__name__)


class AgentTrace(TypedDict):
    """Represents a single agent execution step."""

    agent_name: str
    role: str
    input_text: str
    output_text: str
    reasoning: str
    timestamp: str
    mode: NotRequired[str]
    steps: NotRequired[list[dict[str, Any]]]
    tool_calls: NotRequired[list[dict[str, Any]]]
    error: NotRequired[str]


class AgentState(TypedDict):
    """State passed through the multi-agent graph."""

    query: str
    context: list[dict]
    llm_client: Any
    temperature: float
    top_p: float
    top_k: int
    max_iterations: int
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


def _build_context_summary(context: list[dict], limit: int = 5, snippet_length: int = 700) -> str:
    """Format retrieved context for agent prompts."""
    if not context:
        return ""

    lines = []
    for index, item in enumerate(context[:limit], 1):
        content = item.get("content", "")[:snippet_length]
        source = item.get("source") or item.get("filename") or "Unknown"
        similarity = item.get("similarity_score", item.get("similarity", 0))
        lines.append(f"{index}. [{source}] similarity={similarity:.3f}")
        lines.append(content)

    return "\n".join(lines)


def _run_agent_cycle(
    *,
    agent_name: str,
    role: str,
    request_text: str,
    context: list[dict],
    llm_client: Any,
    temperature: float,
    top_p: float,
    top_k: int,
    max_iterations: int,
    agent_instructions: str,
    synthesis_instructions: str,
) -> AgentTrace:
    """Execute one ReAct-backed agent cycle and package the result."""
    react_output = run_react_loop(
        query=request_text,
        llm_client=llm_client,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_iterations=max_iterations,
        retrieved_context=context,
        agent_name=agent_name,
        agent_role=role,
        agent_instructions=agent_instructions,
        synthesis_instructions=synthesis_instructions,
    )

    output_text = react_output.get("answer", "")
    trace: AgentTrace = {
        "agent_name": agent_name,
        "role": role,
        "input_text": request_text,
        "output_text": output_text,
        "reasoning": agent_instructions.strip() or f"ReAct cycle for {agent_name}",
        "timestamp": datetime.now().isoformat(),
        "mode": react_output.get("mode", "react"),
        "steps": react_output.get("steps", []),
        "tool_calls": react_output.get("tool_calls", []),
        "error": react_output.get("error") or "",
    }
    return trace


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
        llm_client = state["llm_client"]
        temperature = state["temperature"]
        top_p = state["top_p"]
        top_k = state["top_k"]
        max_iterations = state.get("max_iterations", 4)

        context_summary = _build_context_summary(context)
        request_text = f"""Original query:
{query}

Corpus context:
{context_summary or '(no retrieved context)'}

Investigate the query as a news journalist. Use the corpus first, call tools when useful, and produce an initial answer that is as complete as the normal ReAct response.
Prioritize factual grounding, identify the strongest evidence, and note any gaps that need verification."""

        agent_instructions = (
            "Journalist cycle: develop the first full answer, inspect retrieved context, use tools to expand evidence, "
            "and keep the response grounded in the corpus."
        )

        trace = _run_agent_cycle(
            agent_name="Journalist",
            role="News Analyst",
            request_text=request_text,
            context=context,
            llm_client=llm_client,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_iterations=max_iterations,
            agent_instructions=agent_instructions,
            synthesis_instructions="Write the journalist's initial answer directly for downstream verification.",
        )

        journalist_response = trace["output_text"]
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
        llm_client = state["llm_client"]
        temperature = state["temperature"]
        top_p = state["top_p"]
        top_k = state["top_k"]
        max_iterations = state.get("max_iterations", 4)

        context_summary = _build_context_summary(context, snippet_length=500)
        request_text = f"""Original query:
{query}

Journalist draft:
{journalist_response}

Corpus context:
{context_summary or '(no retrieved context)'}

Act as a fact checker. Inspect the journalist draft claim by claim, use tools to verify or refute important points, and produce a strict but fair critique with concrete corrections."""

        agent_instructions = (
            "Fact-checker cycle: verify the journalist answer against retrieved context, call tools for supporting evidence, "
            "and identify unsupported statements or missing nuance."
        )

        trace = _run_agent_cycle(
            agent_name="Fact Checker",
            role="Verification Specialist",
            request_text=request_text,
            context=context,
            llm_client=llm_client,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_iterations=max_iterations,
            agent_instructions=agent_instructions,
            synthesis_instructions="Write a critique that downstream synthesis can fold into the final answer.",
        )

        critique = trace["output_text"]
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
        context = state.get("context", [])
        llm_client = state["llm_client"]
        temperature = state["temperature"]
        top_p = state["top_p"]
        top_k = state["top_k"]
        max_iterations = state.get("max_iterations", 4)

        context_summary = _build_context_summary(context, snippet_length=450)
        request_text = f"""Original query:
{query}

Journalist analysis:
{journalist_response}

Fact-check critique:
{critique}

Corpus context:
{context_summary or '(no retrieved context)'}

Act as the editor. Synthesize the analysis and critique into the exact answer the user needs. Resolve conflicts, preserve uncertainty, and keep the final answer direct and publication-ready."""

        agent_instructions = (
            "Editor cycle: synthesize the journalist draft and the fact-checker's critique into a final answer. "
            "Use tools only if needed to close a remaining factual gap."
        )

        trace = _run_agent_cycle(
            agent_name="Editor",
            role="Synthesis & Refinement",
            request_text=request_text,
            context=context,
            llm_client=llm_client,
            temperature=max(0.5, temperature - 0.2),
            top_p=top_p,
            top_k=top_k,
            max_iterations=max_iterations,
            agent_instructions=agent_instructions,
            synthesis_instructions="Produce the final user-facing answer from the prior agent outputs.",
        )

        final_answer = trace["output_text"]
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
    context: list[dict] | None = None,
    llm_client: Any = None,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    max_iterations: int = 4,
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

    client = llm_client or get_llm_client(
        base_url=settings.lm_studio_base_url,
        model_name=settings.chat_model,
        timeout=settings.request_timeout,
    )
    
    # Initialize state
    initial_state: AgentState = {
        "query": query,
        "context": context or [],
        "llm_client": client,
        "temperature": temperature if temperature is not None else settings.temperature,
        "top_p": top_p if top_p is not None else settings.top_p,
        "top_k": top_k if top_k is not None else settings.top_k,
        "max_iterations": max_iterations,
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
            "editor_synthesis": result["editor_synthesis"],
            "traces": result["traces"],
            "agent_trace": result["traces"],
            "error_message": result["error_message"],
        }
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        return {
            "final_answer": "Error: Multi-agent pipeline failed",
            "journalist_response": "",
            "fact_checker_critique": "",
            "traces": initial_state["traces"],
            "agent_trace": initial_state["traces"],
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
