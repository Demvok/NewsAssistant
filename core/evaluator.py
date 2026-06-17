"""Evaluator module for benchmark and quality assessment.

Implements LLM-as-Judge evaluation, test case execution, and result
aggregation for benchmark evaluation.
"""

import logging
import json
import re
from typing import Optional
from datetime import datetime

from core.llm_client import LMStudioClient
from core.tools import search_articles
from config import settings

logger = logging.getLogger(__name__)


def run_benchmark_case(
    test_case: dict,
    client: LMStudioClient,
    use_rag: bool = True,
    temperature: float = None,
) -> dict:
    """Run a single benchmark test case.
    
    Args:
        test_case: Test case with id, category, prompt, expected_answer, evaluation_criteria
        client: LMStudioClient instance for generation
        use_rag: Whether to use RAG for context retrieval
        temperature: Generation temperature
        
    Returns:
        Results including output, judge score, reasoning, and metadata
    """
    test_id = test_case.get("id", "unknown")
    prompt = test_case.get("prompt", "")
    
    # Use default temperature if not provided
    if temperature is None:
        temperature = settings.temperature
    
    try:
        context_text = ""
        retrieved_chunks = []
        
        if use_rag:
            try:
                retrieved_chunks = search_articles(prompt, top_k=5)
                context_text = "\n\n".join(
                    [chunk.get("content", "") for chunk in retrieved_chunks[:5]]
                )
            except Exception as e:
                logger.warning(f"RAG retrieval failed for test {test_id}: {e}")
                context_text = ""
        
        # Build the prompt with context if available
        if context_text:
            full_prompt = (
                f"Based on the following context:\n\n{context_text}\n\n"
                f"Answer this question:\n{prompt}"
            )
        else:
            full_prompt = prompt
        
        # Generate response
        response = client.complete(
            prompt=full_prompt,
            temperature=temperature,
            max_tokens=512,
        )
        
        return {
            "test_id": test_id,
            "category": test_case.get("category", "unknown"),
            "prompt": prompt,
            "response": response.text,
            "retrieved_chunks": len(retrieved_chunks),
            "latency_ms": response.latency_ms,
            "status": "completed",
            "error": None,
        }
    except Exception as e:
        logger.error(f"Error running benchmark case {test_id}: {e}")
        return {
            "test_id": test_id,
            "category": test_case.get("category", "unknown"),
            "prompt": prompt,
            "response": "",
            "retrieved_chunks": 0,
            "latency_ms": 0,
            "status": "failed",
            "error": str(e),
        }


def evaluate_response(
    response: str,
    test_case: dict,
    client: LMStudioClient,
) -> dict:
    """Evaluate a response using LLM-as-Judge.
    
    Args:
        response: Generated response to evaluate
        test_case: Original test case with evaluation criteria
        client: LMStudioClient for judge evaluation
        
    Returns:
        Score (0-10) and evaluation reasoning
    """
    expected_answer = test_case.get("expected_answer", "")
    evaluation_criteria = test_case.get("evaluation_criteria", "")
    prompt = test_case.get("prompt", "")
    
    judge_prompt = (
        f"You are an expert evaluator. Evaluate the following response on a scale of 0-10.\n\n"
        f"Question: {prompt}\n\n"
        f"Expected/Reference Answer: {expected_answer}\n\n"
        f"Evaluation Criteria: {evaluation_criteria}\n\n"
        f"Actual Response: {response}\n\n"
        f"Provide a JSON response with the following format:\n"
        f'{{"score": <0-10>, "reasoning": "<brief explanation>"}}'
    )
    
    try:
        judge_response = client.complete(
            prompt=judge_prompt,
            temperature=0.3,  # Low temperature for consistency
            max_tokens=256,
        )
        
        # Extract JSON from response
        json_match = re.search(r'\{.*\}', judge_response.text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            score = float(result.get("score", 0))
            reasoning = str(result.get("reasoning", ""))
        else:
            # Fallback: try to extract a number
            numbers = re.findall(r'\d+', judge_response.text)
            score = float(numbers[0]) if numbers else 5
            reasoning = judge_response.text[:200]
        
        score = min(10, max(0, score))  # Clamp to 0-10
        
        return {
            "score": score,
            "reasoning": reasoning,
            "status": "completed",
            "error": None,
        }
    except Exception as e:
        logger.error(f"Error evaluating response: {e}")
        return {
            "score": 0,
            "reasoning": f"Evaluation failed: {str(e)}",
            "status": "failed",
            "error": str(e),
        }


def run_full_benchmark(
    test_cases: list[dict],
    client: LMStudioClient,
    use_rag: bool = True,
    temperature: float = None,
) -> dict:
    """Run benchmark on all test cases.
    
    Args:
        test_cases: List of test cases
        client: LMStudioClient instance
        use_rag: Whether to use RAG
        temperature: Generation temperature
        
    Returns:
        Summary results with per-category scores and overall metrics
    """
    # Use default temperature if not provided
    if temperature is None:
        temperature = settings.temperature
    
    results = []
    total = len(test_cases)
    
    for idx, test_case in enumerate(test_cases):
        logger.info(f"Running test case {idx+1}/{total}: {test_case.get('id')}")
        
        # Run test case
        run_result = run_benchmark_case(test_case, client, use_rag, temperature)
        
        if run_result["status"] == "completed":
            # Evaluate response
            eval_result = evaluate_response(
                run_result["response"],
                test_case,
                client,
            )
            run_result.update(eval_result)
        else:
            run_result["score"] = 0
            run_result["reasoning"] = "Test execution failed"
        
        results.append(run_result)
    
    return {
        "total_tests": total,
        "completed": sum(1 for r in results if r["status"] == "completed"),
        "failed": sum(1 for r in results if r["status"] == "failed"),
        "results": results,
        "timestamp": datetime.now().isoformat(),
    }


def aggregate_results(results: list[dict]) -> dict:
    """Aggregate benchmark results by category.
    
    Args:
        results: List of individual test results
        
    Returns:
        Aggregated statistics by category and overall
    """
    if not results:
        return {
            "overall": {"count": 0, "avg_score": 0, "min_score": 0, "max_score": 0},
            "by_category": {},
        }
    
    # Overall statistics
    scores = [r.get("score", 0) for r in results if r.get("status") == "completed"]
    
    overall = {
        "count": len(results),
        "avg_score": sum(scores) / len(scores) if scores else 0,
        "min_score": min(scores) if scores else 0,
        "max_score": max(scores) if scores else 0,
        "median_score": sorted(scores)[len(scores)//2] if scores else 0,
    }
    
    # By category
    by_category = {}
    for result in results:
        category = result.get("category", "unknown")
        if category not in by_category:
            by_category[category] = []
        by_category[category].append(result.get("score", 0))
    
    by_category_stats = {}
    for category, cat_scores in by_category.items():
        by_category_stats[category] = {
            "count": len(cat_scores),
            "avg_score": sum(cat_scores) / len(cat_scores) if cat_scores else 0,
            "min_score": min(cat_scores) if cat_scores else 0,
            "max_score": max(cat_scores) if cat_scores else 0,
        }
    
    return {
        "overall": overall,
        "by_category": by_category_stats,
    }
