"""ReAct reasoning loop for tool-using chat flows.

This module keeps the reasoning loop separate from the Streamlit UI and uses a
plain-text Thought/Action/Observation protocol instead of brittle JSON-only
planning.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from core.llm_client import LMStudioClient
from core.tools import ToolRegistry, create_default_tools

logger = logging.getLogger(__name__)


@dataclass
class ReactStep:
    """One reasoning step in the ReAct loop."""

    thought: str
    action: str
    tool_name: Optional[str] = None
    tool_input: dict[str, Any] = field(default_factory=dict)
    observation: str = ""


@dataclass
class ReactResult:
    """Structured output returned by the ReAct runner."""

    answer: str = ""
    steps: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    finished: bool = False
    mode: str = "react"


def _direct_tool_request(query: str) -> Optional[dict[str, Any]]:
    """Infer a deterministic tool call from a direct user request."""
    lower_query = query.lower()

    article_match = re.search(r"\b(?:article|статт(?:я|і|ю|ею)|id)\s*(?:#|:|номер\s*)?\s*(\d+)\b", lower_query)
    if article_match:
        return {"name": "get_article", "args": {"article_id": int(article_match.group(1))}}

    # date_matches = re.findall(r"\b\d{4}-\d{2}-\d{2}\b", query)
    # if len(date_matches) >= 2 and any(word in lower_query for word in ["between", "from", "to", "since", "until", "між", "від", "до"]):
    #     return {
    #         "name": "filter_articles_by_date",
    #         "args": {"start_date": date_matches[0], "end_date": date_matches[1]},
    #     }

    keyword_match = re.search(
        r"(?:count|how many|mentions?)\s+(?:of\s+)?(?:the\s+)?['\"]?([\w\- ]+?)['\"]?(?:\s+in|\s+across|\?|$)",
        query,
        re.IGNORECASE,
    )
    if keyword_match and any(word in lower_query for word in ["count", "how many", "mention"]):
        keyword = keyword_match.group(1).strip()
        if keyword:
            return {"name": "count_keyword_mentions", "args": {"keyword": keyword}}

    return None


def _build_tool_catalog(tool_registry: ToolRegistry) -> str:
    return "\n".join(f"- {tool['name']}: {tool['description']}" for tool in tool_registry.list_tools())


def _normalize_tool_name(value: str) -> str:
    text = value.strip()
    text = re.sub(r"^[`*\s]+|[`*\s]+$", "", text)
    match = re.search(r"\b([A-Za-z_][A-Za-z0-9_]*)\b", text)
    return match.group(1) if match else text


def _build_prompt(
    query: str,
    steps: list[dict[str, Any]],
    tool_registry: ToolRegistry,
    retrieved_context: Optional[list[dict[str, Any]]] = None,
) -> str:
    history_lines: list[str] = []
    for index, step in enumerate(steps, 1):
        history_lines.append(f"Step {index} Thought: {step.get('thought', '')}")
        history_lines.append(f"Step {index} Action: {step.get('action', '')}")
        if step.get("tool_name"):
            history_lines.append(f"Step {index} Tool: {step.get('tool_name')}")
        if step.get("tool_input"):
            history_lines.append(
                f"Step {index} Input: {json.dumps(step.get('tool_input', {}), ensure_ascii=False, default=str)}"
            )
        if step.get("observation"):
            history_lines.append(f"Step {index} Observation: {step.get('observation', '')[:700]}")

    history_text = "\n".join(history_lines) if history_lines else "(no previous steps)"

    context_text = ""
    if retrieved_context:
        context_lines = ["Retrieved context already available:"]
        for index, item in enumerate(retrieved_context[:5], 1):
            source = item.get("filename", "Unknown")
            similarity = item.get("similarity_score", 0)
            content = item.get("content", "")[:700]
            context_lines.append(f"{index}. Source: {source} | Similarity: {similarity:.3f}")
            context_lines.append(content)
        context_text = "\n".join(context_lines)

    return f"""You are a ReAct assistant for a news corpus.

            Available tools:
            {_build_tool_catalog(tool_registry)}

            User request:
            {query}

            {context_text}

            Previous steps: 
            {history_text}

            Return exactly one step using this format:
            Thought: brief reasoning
            Action: one of available tools or "Finish"
            Action Input: JSON object for the chosen tool

            If you are think you are DONE, answer the request with all context and return:
            Final: your final answer to the user (Do not include any tool calls, references to steps, or internal reasoning, only the final answer. Do not talk about next steps, provide answer now.)


            Do not add markdown. Do not add explanations outside the format above."""


def _parse_response(text: str) -> dict[str, Any]:
    thought_match = re.search(r"(?mi)^\*?Thought\*?:\s*(.+)$", text)
    action_match = re.search(r"(?mi)^\*?Action\*?:\s*(.+)$", text)
    final_match = re.search(r"(?mis)^\*?Final\*?:\s*(.+)$", text)
    input_match = re.search(r"(?mis)^\*?Action Input\*?:\s*(.+?)(?:\n\*?[A-Z][A-Za-z ]+\*?:|\Z)", text)

    action_input: dict[str, Any] = {}
    if input_match:
        raw_input = input_match.group(1).strip()
        try:
            action_input = json.loads(raw_input)
        except json.JSONDecodeError:
            number_match = re.search(r"\b(\d+)\b", raw_input)
            if number_match:
                action_input = {"article_id": int(number_match.group(1))}

    action_text = action_match.group(1).strip() if action_match else ""
    if action_text:
        action_text = action_text.split("(", 1)[0].strip()
        action_text = _normalize_tool_name(action_text)

    final_text = final_match.group(1).strip() if final_match else ""
    if final_text:
        final_text = re.sub(r"^\*+|\*+$", "", final_text).strip()

    return {
        "thought": thought_match.group(1).strip() if thought_match else "",
        "action": action_text,
        "final": final_text,
        "action_input": action_input,
    }


def _format_observation(result: Any) -> str:
    if isinstance(result, (dict, list)):
        return json.dumps(result, ensure_ascii=False, default=str, indent=2)
    return str(result)


def _build_synthesis_prompt(
    query: str,
    steps: list[dict[str, Any]],
    retrieved_context: Optional[list[dict[str, Any]]] = None,
    draft_answer: str = "",
) -> str:
    step_lines: list[str] = []
    for index, step in enumerate(steps, 1):
        step_lines.append(f"Step {index} Thought: {step.get('thought', '')}")
        step_lines.append(f"Step {index} Action: {step.get('action', '')}")
        if step.get("tool_name"):
            step_lines.append(f"Step {index} Tool: {step.get('tool_name')}")
        if step.get("tool_input"):
            step_lines.append(
                f"Step {index} Input: {json.dumps(step.get('tool_input', {}), ensure_ascii=False, default=str)}"
            )
        if step.get("observation"):
            step_lines.append(f"Step {index} Observation: {step.get('observation', '')[:1200]}")

    context_lines: list[str] = []
    if retrieved_context:
        context_lines.append("Retrieved context:")
        for index, item in enumerate(retrieved_context[:5], 1):
            source = item.get("filename", "Unknown")
            similarity = item.get("similarity_score", 0)
            content = item.get("content", "")[:900]
            context_lines.append(f"{index}. Source: {source} | Similarity: {similarity:.3f}")
            context_lines.append(content)

    step_text = "\n".join(step_lines) if step_lines else "(no tool steps were taken)"
    context_text = "\n".join(context_lines) if context_lines else "(no retrieved context)"

    return f"""You are writing the final answer to the user.

Original question:
{query}

Collected context:
{context_text}

Tool and reasoning trace:
{step_text}

Draft answer from the planner:
{draft_answer or '(none)'}

Write a natural language final answer that directly answers the original question.
Use the gathered evidence and results from the trace.
Summarize what was checked, what was found, and the conclusion.
Do not mention Thought/Action/Observation labels, internal steps, or that you are summarizing a trace.
Do not use markdown.
Keep the tone clear, direct, and human."""


def _synthesize_final_answer(
    query: str,
    llm_client: LMStudioClient,
    steps: list[dict[str, Any]],
    retrieved_context: Optional[list[dict[str, Any]]] = None,
    draft_answer: str = "",
    temperature: float = 0.2,
    top_p: float = 0.9,
) -> str:
    prompt = _build_synthesis_prompt(
        query=query,
        steps=steps,
        retrieved_context=retrieved_context,
        draft_answer=draft_answer,
    )
    response = llm_client.complete(
        prompt=prompt,
        temperature=temperature,
        top_p=top_p,
        top_k=40,
        max_tokens=512,
    )
    return (response.text or "").strip()


def _run_tool(tool_registry: ToolRegistry, name: str, args: dict[str, Any]) -> Any:
    tool = tool_registry.get_tool(name)
    if tool is None:
        raise RuntimeError(f"Unknown tool: {name}")
    return tool(**args)


def _coerce_action_input(action: str, action_input: dict[str, Any], thought: str, query: str) -> dict[str, Any]:
    if action_input:
        return action_input

    normalized_action = action.strip().lower()
    if normalized_action == "search_articles":
        fallback_query = thought.strip() or query.strip()
        return {"query": fallback_query, "top_k": 5}

    if normalized_action == "get_article":
        source_text = thought or query
        match = re.search(r"\b(\d+)\b", source_text)
        if match:
            return {"article_id": int(match.group(1))}

    if normalized_action == "count_keyword_mentions":
        source_text = thought or query
        keyword = source_text.strip()
        if keyword:
            return {"keyword": keyword}

    return action_input


def run_react_loop(
    query: str,
    llm_client: LMStudioClient,
    temperature: float,
    top_p: float,
    top_k: int,
    max_iterations: int = 4,
    tool_registry: Optional[ToolRegistry] = None,
    retrieved_context: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    """Run a ReAct loop and return structured output for the chat page."""
    registry = tool_registry or create_default_tools()
    result = ReactResult()

    direct_request = _direct_tool_request(query)
    if direct_request is not None:
        try:
            tool_result = _run_tool(registry, direct_request["name"], direct_request["args"])
            result.tool_calls.append(
                {
                    "name": direct_request["name"],
                    "args": direct_request["args"],
                    "result": tool_result,
                }
            )
            step = {
                "thought": "Direct request detected; skipping LLM planning.",
                "action": direct_request["name"],
                "tool_name": direct_request["name"],
                "tool_input": direct_request["args"],
                "observation": _format_observation(tool_result),
            }
            result.steps.append(step)
            if direct_request["name"] == "get_article" and isinstance(tool_result, dict):
                result.answer = tool_result.get("content") or tool_result.get("title") or _format_observation(tool_result)
            else:
                result.answer = _format_observation(tool_result)
            result.finished = True
            result.mode = "direct"
            return asdict(result)
        except Exception as exc:
            logger.error("Direct tool execution failed: %s", exc)
            result.error = str(exc)
            result.answer = f"Error during direct tool execution: {exc}"
            return asdict(result)

    steps: list[dict[str, Any]] = []
    for _iteration in range(1, max_iterations + 1):
        prompt = _build_prompt(query, steps, registry, retrieved_context=retrieved_context)
        try:
            response = llm_client.complete(
                prompt=prompt,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_tokens=None,
                stop=["Observation:", "Final:"],
            )
            text = (response.text or "").strip()
        except Exception as exc:
            logger.error("ReAct planning failed: %s", exc)
            result.error = str(exc)
            result.answer = f"Error during reasoning: {exc}"
            return asdict(result)

        if not text:
            result.error = "Empty planner response"
            result.answer = "I could not produce a valid plan for this request."
            return asdict(result)

        parsed = _parse_response(text)
        thought = parsed["thought"]
        action = parsed["action"]
        action_input = _coerce_action_input(action, parsed["action_input"], thought, query)
        final = parsed["final"]

        if action.lower() == "finish":
            steps.append({"thought": thought, "action": "Finish", "observation": final})
            result.steps = steps
            try:
                synthesized_answer = _synthesize_final_answer(
                    query=query,
                    llm_client=llm_client,
                    steps=steps,
                    retrieved_context=retrieved_context,
                    draft_answer=final,
                )
                result.answer = synthesized_answer or final or thought or "Unable to answer."
            except Exception as exc:
                logger.error("ReAct final synthesis failed: %s", exc)
                result.answer = final or thought or "Unable to answer."
            result.finished = True
            return asdict(result)

        if not action:
            result.error = "Model did not choose an action"
            result.answer = "I couldn't determine the next tool action."
            result.steps = steps
            return asdict(result)

        tool = registry.get_tool(action)
        if tool is None:
            result.error = f"Unknown tool requested: {action}"
            result.answer = f"Unknown tool requested: {action}"
            result.steps = steps
            return asdict(result)

        try:
            tool_result = tool(**action_input)
        except Exception as exc:
            logger.error("ReAct tool execution failed: %s", exc)
            steps.append(
                {
                    "thought": thought,
                    "action": action,
                    "tool_name": action,
                    "tool_input": action_input,
                    "observation": str(exc),
                }
            )
            result.error = str(exc)
            result.answer = f"Tool execution failed: {exc}"
            result.steps = steps
            return asdict(result)

        observation = _format_observation(tool_result)
        steps.append(
            {
                "thought": thought,
                "action": action,
                "tool_name": action,
                "tool_input": action_input,
                "observation": observation,
            }
        )
        result.tool_calls.append({"name": action, "args": action_input, "result": tool_result})

    result.error = "Max iterations reached"
    result.answer = "I reached the maximum number of reasoning steps before finishing."
    result.steps = steps
    return asdict(result)
