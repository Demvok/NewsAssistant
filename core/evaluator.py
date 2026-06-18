"""Benchmark evaluation and aggregation.

This module runs benchmark cases through the same assistant paths used in the
UI: standard generation, RAG, ReAct tool usage, and multi-agent reasoning.
The judge produces a multi-axis score so the UI can explain the result.
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
from datetime import datetime
from statistics import median
from typing import Any, Optional

from config import settings
from core.agents import run_multi_agent_pipeline
from core.llm_client import LMStudioClient
from core.rag import generate_with_rag
from core.react import run_react_loop
from core.tools import search_articles

logger = logging.getLogger(__name__)

DIMENSION_NAMES = [
    "role_alignment",
    "mode_adherence",
    "relevance",
    "grounding",
    "tool_usage",
    "completeness",
    "clarity",
]

DIMENSION_WEIGHTS = {
    "role_alignment": 0.12,
    "mode_adherence": 0.12,
    "relevance": 0.18,
    "grounding": 0.20,
    "tool_usage": 0.14,
    "completeness": 0.16,
    "clarity": 0.08,
}


def _normalize_mode(value: Optional[str]) -> str:
    mode = (value or "generation").strip().lower().replace("-", "_")
    aliases = {
        "basic": "generation",
        "standard": "generation",
        "rag": "rag",
        "retrieval": "rag",
        "react": "react",
        "tool": "react",
        "tool_usage": "react",
        "agent": "multi_agent",
        "multiagent": "multi_agent",
        "multi_agent": "multi_agent",
        "generation": "generation",
    }
    return aliases.get(mode, mode if mode in {"generation", "rag", "react", "multi_agent"} else "generation")


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)


def _ensure_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _safe_number(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def _summarize_context(context: list[dict[str, Any]], limit: int = 5, snippet_length: int = 650) -> str:
    if not context:
        return ""

    lines: list[str] = []
    for index, item in enumerate(context[:limit], 1):
        source = item.get("source") or item.get("filename") or item.get("title") or "Unknown"
        similarity = item.get("similarity_score", item.get("similarity", 0))
        content = _coerce_text(item.get("content", ""))[:snippet_length]
        lines.append(f"{index}. Source: {source} | Similarity: {similarity:.3f}")
        lines.append(content)
    return "\n".join(lines)


def _extract_response_payload(result: Any) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], Optional[str], str]:
    if isinstance(result, dict):
        answer = (
            result.get("answer")
            or result.get("final_answer")
            or result.get("editor_synthesis")
            or result.get("response")
            or ""
        )
        retrieved_chunks = _ensure_list(result.get("retrieved_chunks") or result.get("context") or [])
        tool_calls = _ensure_list(result.get("tool_calls") or [])
        agent_trace = _ensure_list(result.get("agent_trace") or result.get("traces") or [])
        error = result.get("error") or result.get("error_message")
        mode = _normalize_mode(result.get("assistant_mode") or result.get("mode"))
        return _coerce_text(answer), retrieved_chunks, tool_calls, agent_trace, error, mode
    return _coerce_text(result), [], [], [], None, "generation"


def _run_generation_case(
    prompt: str,
    client: LMStudioClient,
    temperature: float,
    top_p: float,
    top_k: int,
    use_rag: bool,
) -> dict[str, Any]:
    result = generate_with_rag(
        query=prompt,
        llm_client=client,
        rag_enabled=use_rag,
        top_k=top_k,
        temperature=temperature,
        max_tokens=settings.max_tokens or 512,
    )
    answer, retrieved_chunks, tool_calls, agent_trace, error, mode = _extract_response_payload(result)
    return {
        "answer": answer,
        "retrieved_chunks": retrieved_chunks,
        "tool_calls": tool_calls,
        "agent_trace": agent_trace,
        "mode": mode,
        "error": error,
        "latency_ms": result.get("metadata", {}).get("latency_ms") if isinstance(result, dict) else None,
        "metadata": result.get("metadata", {}) if isinstance(result, dict) else {},
    }


def _run_react_case(
    prompt: str,
    client: LMStudioClient,
    temperature: float,
    top_p: float,
    top_k: int,
    use_rag: bool,
) -> dict[str, Any]:
    context = []
    if use_rag:
        try:
            context = search_articles(prompt, top_k=top_k)
        except Exception as exc:
            logger.warning("RAG prefetch failed for ReAct benchmark: %s", exc)

    result = run_react_loop(
        query=prompt,
        llm_client=client,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_iterations=5,
        retrieved_context=context,
    )
    answer, retrieved_chunks, tool_calls, agent_trace, error, _mode = _extract_response_payload(result)
    if not retrieved_chunks:
        retrieved_chunks = context
    if not agent_trace:
        agent_trace = [
            {
                "agent_name": "ReAct",
                "role": "Tool-Using Analyst",
                "input_text": prompt,
                "output_text": answer,
                "reasoning": "Benchmark ReAct execution",
                "timestamp": datetime.now().isoformat(),
                "mode": result.get("mode", "react") if isinstance(result, dict) else "react",
                "steps": result.get("steps", []) if isinstance(result, dict) else [],
                "tool_calls": tool_calls,
                "error": error or "",
            }
        ]
    return {
        "answer": answer,
        "retrieved_chunks": retrieved_chunks,
        "tool_calls": tool_calls,
        "agent_trace": agent_trace,
        "mode": "react",
        "error": error,
        "latency_ms": result.get("latency_ms") if isinstance(result, dict) else None,
        "metadata": {},
    }


def _run_multi_agent_case(
    prompt: str,
    client: LMStudioClient,
    temperature: float,
    top_p: float,
    top_k: int,
    use_rag: bool,
) -> dict[str, Any]:
    context = []
    if use_rag:
        try:
            context = search_articles(prompt, top_k=top_k)
        except Exception as exc:
            logger.warning("RAG prefetch failed for multi-agent benchmark: %s", exc)

    result = run_multi_agent_pipeline(
        query=prompt,
        context=context,
        llm_client=client,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_iterations=5,
    )
    answer, retrieved_chunks, tool_calls, agent_trace, error, _mode = _extract_response_payload(result)
    if not retrieved_chunks:
        retrieved_chunks = context
    return {
        "answer": answer,
        "retrieved_chunks": retrieved_chunks,
        "tool_calls": tool_calls,
        "agent_trace": agent_trace,
        "mode": "multi_agent",
        "error": error,
        "latency_ms": None,
        "metadata": {},
    }


def run_benchmark_case(
    test_case: dict,
    client: LMStudioClient,
    use_rag: bool = True,
    temperature: Optional[float] = None,
    top_p: float | None = None,
    top_k: int | None = None,
) -> dict:
    """Run one benchmark case through the configured assistant mode."""
    test_id = test_case.get("id", "unknown")
    prompt = test_case.get("prompt", "")
    assistant_mode = _normalize_mode(test_case.get("assistant_mode") or test_case.get("mode"))
    if temperature is None:
        temperature = settings.temperature
    if top_p is None:
        top_p = settings.top_p
    if top_k is None:
        top_k = settings.top_k_retrieval

    start_time = time.perf_counter()
    try:
        if assistant_mode == "generation":
            run_data = _run_generation_case(prompt, client, temperature, top_p, top_k, use_rag=False)
        elif assistant_mode == "rag":
            run_data = _run_generation_case(prompt, client, temperature, top_p, top_k, use_rag=use_rag)
        elif assistant_mode == "react":
            run_data = _run_react_case(prompt, client, temperature, top_p, top_k, use_rag=use_rag)
        elif assistant_mode == "multi_agent":
            run_data = _run_multi_agent_case(prompt, client, temperature, top_p, top_k, use_rag=use_rag)
        else:
            logger.warning("Unknown benchmark mode '%s'; falling back to generation", assistant_mode)
            run_data = _run_generation_case(prompt, client, temperature, top_p, top_k, use_rag=False)

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        run_data.update(
            {
                "test_id": test_id,
                "topic": test_case.get("topic", "unknown"),
                "category": test_case.get("category", "unknown"),
                "assistant_mode": assistant_mode,
                "prompt": prompt,
                "expected_answer": test_case.get("expected_answer", ""),
                "evaluation_criteria": test_case.get("evaluation_criteria", ""),
                "notes": test_case.get("notes", ""),
                "retrieved_chunks_count": len(run_data.get("retrieved_chunks", [])),
                "tool_call_count": len(run_data.get("tool_calls", [])),
                "latency_ms": run_data.get("latency_ms") or elapsed_ms,
                "execution_ms": elapsed_ms,
                "status": "completed" if not run_data.get("error") else "completed_with_warnings",
            }
        )
        return run_data
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.error("Error running benchmark case %s: %s", test_id, exc, exc_info=True)
        return {
            "test_id": test_id,
            "topic": test_case.get("topic", "unknown"),
            "category": test_case.get("category", "unknown"),
            "assistant_mode": assistant_mode,
            "prompt": prompt,
            "expected_answer": test_case.get("expected_answer", ""),
            "evaluation_criteria": test_case.get("evaluation_criteria", ""),
            "notes": test_case.get("notes", ""),
            "answer": "",
            "retrieved_chunks": [],
            "tool_calls": [],
            "agent_trace": [],
            "retrieved_chunks_count": 0,
            "tool_call_count": 0,
            "latency_ms": elapsed_ms,
            "execution_ms": elapsed_ms,
            "status": "failed",
            "error": str(exc),
        }


def _clean_json_block(text: str) -> str:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group(0) if match else text


def _parse_dimension_scores(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw_scores = payload.get("scores") or payload.get("dimensions") or {}
    parsed: dict[str, dict[str, Any]] = {}
    for name in DIMENSION_NAMES:
        entry = raw_scores.get(name, {})
        if isinstance(entry, dict):
            score = _safe_number(entry.get("score"), 0.0)
            explanation = _coerce_text(entry.get("explanation") or entry.get("reason") or entry.get("why") or "")
        else:
            score = _safe_number(entry, 0.0)
            explanation = ""
        parsed[name] = {
            "score": max(0.0, min(10.0, score)),
            "explanation": explanation,
        }
    return parsed


def _calculate_weighted_score(dimensions: dict[str, dict[str, Any]]) -> float:
    total_weight = sum(DIMENSION_WEIGHTS.values())
    if total_weight <= 0:
        return 0.0
    weighted_sum = 0.0
    for name, weight in DIMENSION_WEIGHTS.items():
        weighted_sum += _safe_number(dimensions.get(name, {}).get("score"), 0.0) * weight
    return weighted_sum / total_weight


def _build_judge_prompt(
    response_text: str,
    test_case: dict,
    assistant_mode: str,
    retrieved_chunks: list[dict[str, Any]],
    tool_calls: list[dict[str, Any]],
    agent_trace: list[dict[str, Any]],
) -> str:
    context_text = _summarize_context(retrieved_chunks, limit=4, snippet_length=450) or "(no retrieved context)"
    tool_text = json.dumps(tool_calls[:6], ensure_ascii=False, default=str, indent=2) if tool_calls else "[]"
    trace_text = json.dumps(agent_trace[:3], ensure_ascii=False, default=str, indent=2) if agent_trace else "[]"

    return f"""
    You are evaluating an AI News Intelligence Assistant benchmark response.

    Selected mode: {assistant_mode}
    Topic: {test_case.get('topic', 'unknown')}
    Category: {test_case.get('category', 'unknown')}

    User prompt:
    {test_case.get('prompt', '')}

    Reference answer:
    {test_case.get('expected_answer', '')}

    Evaluation criteria:
    {test_case.get('evaluation_criteria', '')}

    Assistant response:
    {response_text}

    Retrieved context:
    {context_text}

    Tool calls:
    {tool_text}

    Agent trace:
    {trace_text}

    Score each dimension from 0 to 10 and explain each score briefly:
    - role_alignment: how well the response matches the requested assistant role and tone.
    - mode_adherence: how well the response fits the selected execution mode.
    - relevance: how directly it answers the prompt.
    - grounding: how well it uses available corpus evidence and avoids unsupported claims.
    - tool_usage: how appropriate and effective tool usage was for this mode.
    - completeness: how fully it addresses the prompt.
    - clarity: how readable, organized, and direct the answer is.

    Return JSON only in this schema:
    {{
        "scores": {{
            "role_alignment": {{"score": 0, "explanation": "..."}},
            "mode_adherence": {{"score": 0, "explanation": "..."}},
            "relevance": {{"score": 0, "explanation": "..."}},
            "grounding": {{"score": 0, "explanation": "..."}},
            "tool_usage": {{"score": 0, "explanation": "..."}},
            "completeness": {{"score": 0, "explanation": "..."}},
            "clarity": {{"score": 0, "explanation": "..."}}
        }},
        "overall_score": 0,
        "overall_reason": "...",
        "strengths": ["..."],
        "gaps": ["..."],
        "confidence": 0
    }}

    Guidance for tool_usage:
    - generation: reward direct answers and do not penalize the absence of tools.
    - rag: reward clear use of retrieved context and correct synthesis.
    - react: reward actual tool calls, sensible tool choice, and using tool output in the answer.
    - multi_agent: reward a clear analysis -> critique -> synthesis structure and visible trace usefulness.
    """


def _parse_judge_payload(text: str) -> dict[str, Any]:
    try:
        return json.loads(_clean_json_block(text))
    except Exception:
        score_match = re.search(r"(\d+(?:\.\d+)?)", text)
        score = _safe_number(score_match.group(1) if score_match else 0.0, 0.0)
        return {
            "scores": {},
            "overall_score": score,
            "overall_reason": text[:500],
            "strengths": [],
            "gaps": [],
            "confidence": 0,
        }


def evaluate_response(
    response: Any,
    test_case: dict,
    client: LMStudioClient,
    assistant_mode: Optional[str] = None,
    retrieved_chunks: Optional[list[dict[str, Any]]] = None,
    tool_calls: Optional[list[dict[str, Any]]] = None,
    agent_trace: Optional[list[dict[str, Any]]] = None,
) -> dict:
    """Evaluate a benchmark response with a structured multi-axis judge."""
    retrieved_chunks = retrieved_chunks or []
    tool_calls = tool_calls or []
    agent_trace = agent_trace or []

    if isinstance(response, dict):
        response_text = _coerce_text(response.get("answer") or response.get("final_answer") or response.get("response") or "")
        if not retrieved_chunks:
            retrieved_chunks = _ensure_list(response.get("retrieved_chunks") or response.get("context") or [])
        if not tool_calls:
            tool_calls = _ensure_list(response.get("tool_calls") or [])
        if not agent_trace:
            agent_trace = _ensure_list(response.get("agent_trace") or response.get("traces") or [])
        if not assistant_mode:
            assistant_mode = _normalize_mode(response.get("assistant_mode") or response.get("mode"))
    else:
        response_text = _coerce_text(response)

    assistant_mode = _normalize_mode(assistant_mode or test_case.get("assistant_mode") or test_case.get("mode"))
    judge_prompt = _build_judge_prompt(
        response_text=response_text,
        test_case=test_case,
        assistant_mode=assistant_mode,
        retrieved_chunks=retrieved_chunks,
        tool_calls=tool_calls,
        agent_trace=agent_trace,
    )

    try:
        judge_response = client.complete(
            prompt=judge_prompt,
            temperature=0.2,
            top_p=0.9,
            top_k=40,
            max_tokens=512,
        )
        judge_text = judge_response.text or ""
        payload = _parse_judge_payload(judge_text)
        dimensions = _parse_dimension_scores(payload)
        weighted_score = _calculate_weighted_score(dimensions)
        overall_score = _safe_number(payload.get("overall_score"), weighted_score)
        if not math.isfinite(overall_score) or overall_score <= 0:
            overall_score = weighted_score
        overall_score = max(0.0, min(10.0, overall_score))

        confidence = _safe_number(payload.get("confidence"), 0.0)
        if confidence > 1:
            confidence = confidence / 10.0

        return {
            "score": overall_score,
            "score_breakdown": dimensions,
            "reasoning": _coerce_text(payload.get("overall_reason") or judge_text[:700]),
            "strengths": _ensure_list(payload.get("strengths") or []),
            "gaps": _ensure_list(payload.get("gaps") or []),
            "confidence": max(0.0, min(1.0, confidence)),
            "judge_raw": judge_text,
            "status": "completed",
            "error": None,
        }
    except Exception as exc:
        logger.error("Error evaluating response: %s", exc, exc_info=True)
        return {
            "score": 0.0,
            "score_breakdown": {name: {"score": 0.0, "explanation": "Judge unavailable"} for name in DIMENSION_NAMES},
            "reasoning": f"Evaluation failed: {exc}",
            "strengths": [],
            "gaps": [],
            "confidence": 0.0,
            "judge_raw": "",
            "status": "failed",
            "error": str(exc),
        }


def run_full_benchmark(
    test_cases: list[dict],
    client: LMStudioClient,
    use_rag: bool = True,
    temperature: Optional[float] = None,
    top_p: float | None = None,
    top_k: int | None = None,
    temperature_grid: Optional[list[float]] = None,
    repetitions: int = 1,
    selected_modes: Optional[list[str]] = None,
) -> dict:
    """Run the full benchmark suite and return detailed per-run results."""
    selected_modes = [_normalize_mode(mode) for mode in (selected_modes or [])]
    if temperature_grid is None or not temperature_grid:
        temperature_grid = [settings.temperature if temperature is None else temperature]

    top_p = settings.top_p if top_p is None else top_p
    top_k = settings.top_k_retrieval if top_k is None else top_k
    repetitions = max(1, int(repetitions))

    results: list[dict[str, Any]] = []
    total_runs = 0
    for test_case in test_cases:
        case_mode = _normalize_mode(test_case.get("assistant_mode") or test_case.get("mode"))
        if selected_modes and case_mode not in selected_modes:
            continue

        for temp_index, temp_value in enumerate(temperature_grid, 1):
            for repeat_index in range(1, repetitions + 1):
                total_runs += 1
                logger.info(
                    "Running benchmark case %s mode=%s temp=%.2f repeat=%s",
                    test_case.get("id"),
                    case_mode,
                    temp_value,
                    repeat_index,
                )

                run_result = run_benchmark_case(
                    test_case=test_case,
                    client=client,
                    use_rag=use_rag,
                    temperature=temp_value,
                    top_p=top_p,
                    top_k=top_k,
                )

                eval_result = evaluate_response(
                    run_result,
                    test_case,
                    client,
                    assistant_mode=case_mode,
                    retrieved_chunks=run_result.get("retrieved_chunks", []),
                    tool_calls=run_result.get("tool_calls", []),
                    agent_trace=run_result.get("agent_trace", []),
                )
                run_result.update(eval_result)
                run_result.update(
                    {
                        "temperature": temp_value,
                        "temperature_index": temp_index,
                        "repeat_index": repeat_index,
                        "suite_mode": case_mode,
                    }
                )
                results.append(run_result)

    return {
        "total_tests": len(test_cases),
        "total_runs": total_runs,
        "completed": sum(1 for r in results if r.get("status") == "completed"),
        "failed": sum(1 for r in results if r.get("status") == "failed"),
        "results": results,
        "timestamp": datetime.now().isoformat(),
        "temperature_grid": temperature_grid,
        "repetitions": repetitions,
        "selected_modes": selected_modes,
    }


def _dimension_average(results: list[dict[str, Any]], dimension: str) -> float:
    scores = []
    for result in results:
        breakdown = result.get("score_breakdown", {}) or {}
        if dimension in breakdown:
            scores.append(_safe_number(breakdown[dimension].get("score"), 0.0))
    return sum(scores) / len(scores) if scores else 0.0


def _group_results(results: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        group_value = str(result.get(key, "unknown") or "unknown")
        grouped.setdefault(group_value, []).append(result)

    summary: dict[str, dict[str, Any]] = {}
    for group_value, items in grouped.items():
        scores = [_safe_number(item.get("score"), 0.0) for item in items if item.get("status") == "completed"]
        total_tool_calls = sum(_safe_int(item.get("tool_call_count"), 0) for item in items)
        total_context = sum(_safe_int(item.get("retrieved_chunks_count"), 0) for item in items)
        summary[group_value] = {
            "count": len(items),
            "completed": sum(1 for item in items if item.get("status") == "completed"),
            "failed": sum(1 for item in items if item.get("status") == "failed"),
            "avg_score": sum(scores) / len(scores) if scores else 0.0,
            "median_score": median(scores) if scores else 0.0,
            "min_score": min(scores) if scores else 0.0,
            "max_score": max(scores) if scores else 0.0,
            "completion_rate": (sum(1 for item in items if item.get("status") == "completed") / len(items)) if items else 0.0,
            "avg_latency_ms": sum(_safe_number(item.get("latency_ms"), 0.0) for item in items) / len(items) if items else 0.0,
            "avg_tool_calls": total_tool_calls / len(items) if items else 0.0,
            "avg_retrieved_chunks": total_context / len(items) if items else 0.0,
            "dimensions": {name: _dimension_average(items, name) for name in DIMENSION_NAMES},
        }
    return summary


def aggregate_results(results: list[dict]) -> dict:
    """Aggregate benchmark results by topic, category, mode, and temperature."""
    if not results:
        empty_dimensions = {name: 0.0 for name in DIMENSION_NAMES}
        return {
            "overall": {
                "count": 0,
                "completed": 0,
                "failed": 0,
                "avg_score": 0.0,
                "median_score": 0.0,
                "min_score": 0.0,
                "max_score": 0.0,
                "completion_rate": 0.0,
                "avg_latency_ms": 0.0,
                "avg_tool_calls": 0.0,
                "avg_retrieved_chunks": 0.0,
                "dimensions": empty_dimensions,
            },
            "by_topic": {},
            "by_category": {},
            "by_mode": {},
            "by_temperature": {},
        }

    completed = [result for result in results if result.get("status") == "completed"]
    scores = [_safe_number(result.get("score"), 0.0) for result in completed]
    overall = {
        "count": len(results),
        "completed": len(completed),
        "failed": sum(1 for result in results if result.get("status") == "failed"),
        "avg_score": sum(scores) / len(scores) if scores else 0.0,
        "median_score": median(scores) if scores else 0.0,
        "min_score": min(scores) if scores else 0.0,
        "max_score": max(scores) if scores else 0.0,
        "completion_rate": len(completed) / len(results) if results else 0.0,
        "avg_latency_ms": sum(_safe_number(result.get("latency_ms"), 0.0) for result in results) / len(results) if results else 0.0,
        "avg_tool_calls": sum(_safe_int(result.get("tool_call_count"), 0) for result in results) / len(results) if results else 0.0,
        "avg_retrieved_chunks": sum(_safe_int(result.get("retrieved_chunks_count"), 0) for result in results) / len(results) if results else 0.0,
        "dimensions": {name: _dimension_average(results, name) for name in DIMENSION_NAMES},
    }

    return {
        "overall": overall,
        "by_topic": _group_results(results, "topic"),
        "by_category": _group_results(results, "category"),
        "by_mode": _group_results(results, "assistant_mode"),
        "by_temperature": _group_results(results, "temperature"),
    }


def results_to_dataframe(results: dict):
    """Convert benchmark results to a dataframe for reporting."""
    import pandas as pd

    rows: list[dict[str, Any]] = []
    for result in results.get("results", []):
        breakdown = result.get("score_breakdown", {}) or {}
        rows.append(
            {
                "Test ID": result.get("test_id", ""),
                "Topic": result.get("topic", ""),
                "Category": result.get("category", ""),
                "Mode": result.get("assistant_mode", ""),
                "Temperature": result.get("temperature", settings.temperature),
                "Score": _safe_number(result.get("score"), 0.0),
                "Status": result.get("status", ""),
                "Latency (ms)": round(_safe_number(result.get("latency_ms"), 0.0), 2),
                "Execution (ms)": round(_safe_number(result.get("execution_ms"), 0.0), 2),
                "Retrieved Chunks": _safe_int(result.get("retrieved_chunks_count"), 0),
                "Tool Calls": _safe_int(result.get("tool_call_count"), 0),
                "Role": _safe_number(breakdown.get("role_alignment", {}).get("score"), 0.0),
                "Mode Adh.": _safe_number(breakdown.get("mode_adherence", {}).get("score"), 0.0),
                "Relevance": _safe_number(breakdown.get("relevance", {}).get("score"), 0.0),
                "Grounding": _safe_number(breakdown.get("grounding", {}).get("score"), 0.0),
                "Tool Usage": _safe_number(breakdown.get("tool_usage", {}).get("score"), 0.0),
                "Completeness": _safe_number(breakdown.get("completeness", {}).get("score"), 0.0),
                "Clarity": _safe_number(breakdown.get("clarity", {}).get("score"), 0.0),
                "Reasoning": result.get("reasoning", ""),
            }
        )

    return pd.DataFrame(rows)
