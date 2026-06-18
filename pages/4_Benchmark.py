"""Benchmark page for the AI News Intelligence Assistant.

This page runs benchmark cases through the same assistant paths used in the
chat interface:
- standard generation
- RAG retrieval
- ReAct tool use
- multi-agent reasoning

It also exposes topic/category summaries, per-dimension judge scores, and
exportable results for reporting.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import settings
from core.evaluator import aggregate_results, results_to_dataframe, run_full_benchmark
from core.llm_client import LMStudioClient
from core.utils import setup_logging

logger = setup_logging(level="INFO")

st.set_page_config(
    page_title="Benchmark - AI News Intelligence Assistant",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Benchmark")
st.markdown("Run assistant-mode benchmarks, inspect judge breakdowns, and export experiment-ready results.")

BENCHMARK_DIR = settings.project_root / settings.benchmark_dir
BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

DIMENSION_LABELS = {
    "role_alignment": "Role",
    "mode_adherence": "Mode",
    "relevance": "Relevance",
    "grounding": "Grounding",
    "tool_usage": "Tool Usage",
    "completeness": "Completeness",
    "clarity": "Clarity",
}

MODE_LABELS = {
    "generation": "Generation",
    "rag": "RAG",
    "react": "ReAct",
    "multi_agent": "Multi-Agent",
}

TOPIC_LABELS = {
    "green_energy": "Green Energy",
    "trade_war": "Trade War",
    "censorship": "Censorship",
}


if "benchmark_dataset" not in st.session_state:
    st.session_state.benchmark_dataset = []
if "benchmark_results" not in st.session_state:
    st.session_state.benchmark_results = None
if "benchmark_running" not in st.session_state:
    st.session_state.benchmark_running = False
if "benchmark_run_history" not in st.session_state:
    st.session_state.benchmark_run_history = []


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def _dataset_path(name: str) -> Path:
    return BENCHMARK_DIR / name


def load_benchmark_dataset(filepath: Path) -> list[dict[str, Any]]:
    try:
        with filepath.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            if isinstance(data.get("test_cases"), list):
                return data["test_cases"]
            if isinstance(data.get("cases"), list):
                return data["cases"]
        return []
    except Exception as exc:
        logger.error("Failed to load benchmark dataset %s: %s", filepath, exc)
        st.error(f"Failed to load dataset: {exc}")
        return []


def save_benchmark_dataset(dataset: list[dict[str, Any]], filename: str) -> Path:
    filepath = _dataset_path(filename)
    payload = {
        "test_cases": dataset,
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "count": len(dataset),
            "topics": sorted({item.get("topic", "unknown") for item in dataset}),
            "categories": sorted({item.get("category", "unknown") for item in dataset}),
            "modes": sorted({item.get("assistant_mode", "generation") for item in dataset}),
        },
    }
    with filepath.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return filepath


def save_benchmark_results(results: dict[str, Any], filename: str) -> Path:
    filepath = _dataset_path(filename)
    with filepath.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False, default=str)
    return filepath


def get_available_datasets() -> list[Path]:
    datasets = sorted(BENCHMARK_DIR.glob("*.json"))
    return [path for path in datasets if not path.name.startswith("result_")]


def create_sample_dataset() -> list[dict[str, Any]]:
    return [
        {
            "id": "green_energy_gen_1",
            "topic": "green_energy",
            "category": "overview",
            "assistant_mode": "generation",
            "prompt": "Summarize the main public debate around green energy in the news corpus.",
            "expected_answer": "A concise overview of renewable energy transition, policy tradeoffs, costs, and adoption barriers.",
            "evaluation_criteria": "Answers the question directly, stays general but structured, and avoids unsupported specifics.",
            "notes": "Baseline generation on the green energy topic.",
        },
        {
            "id": "green_energy_rag_1",
            "topic": "green_energy",
            "category": "evidence_lookup",
            "assistant_mode": "rag",
            "prompt": "What concrete examples of renewable energy adoption appear in the corpus?",
            "expected_answer": "Mentions examples such as solar, wind, batteries, grid upgrades, or policy-backed clean energy efforts.",
            "evaluation_criteria": "Uses retrieved context, cites grounded examples, and does not invent unsupported projects.",
            "notes": "Checks retrieval quality and synthesis.",
        },
        {
            "id": "green_energy_react_1",
            "topic": "green_energy",
            "category": "tool_usage",
            "assistant_mode": "react",
            "prompt": "Count how many times the assistant should look for green energy mentions and explain the result at a high level.",
            "expected_answer": "A response that uses tools appropriately, explains the count or search outcome, and stays on topic.",
            "evaluation_criteria": "Actual tool use should be visible, the tool choice should match the request, and the answer should use the tool result.",
            "notes": "Tool usage benchmark for corpus analysis.",
        },
        {
            "id": "green_energy_agent_1",
            "topic": "green_energy",
            "category": "synthesis",
            "assistant_mode": "multi_agent",
            "prompt": "Give a balanced analysis of green energy progress and remaining concerns.",
            "expected_answer": "An analysis that includes progress, limitations, and a balanced conclusion.",
            "evaluation_criteria": "The final answer should reflect analysis, critique, and synthesis with visible trace usefulness.",
            "notes": "Multi-agent synthesis on the green energy topic.",
        },
        {
            "id": "trade_war_gen_1",
            "topic": "trade_war",
            "category": "overview",
            "assistant_mode": "generation",
            "prompt": "Summarize the public narrative around the U.S. trade war in the corpus.",
            "expected_answer": "A high-level summary of tariffs, supply-chain pressure, market uncertainty, and policy tension.",
            "evaluation_criteria": "Direct answer, domain-appropriate tone, and no unsupported detail.",
            "notes": "Baseline generation on trade policy.",
        },
        {
            "id": "trade_war_rag_1",
            "topic": "trade_war",
            "category": "evidence_lookup",
            "assistant_mode": "rag",
            "prompt": "Which industries or business areas are most affected by trade tensions?",
            "expected_answer": "References to manufacturing, logistics, consumer goods, agriculture, or similar affected sectors.",
            "evaluation_criteria": "Should ground the answer in retrieved passages and mention concrete affected sectors if available.",
            "notes": "Retrieval and summarization benchmark.",
        },
        {
            "id": "trade_war_react_1",
            "topic": "trade_war",
            "category": "tool_usage",
            "assistant_mode": "react",
            "prompt": "Search for a relevant article and explain what it says about tariffs or business impact.",
            "expected_answer": "A tool-driven answer that identifies the article and summarizes the tariff or business effect.",
            "evaluation_criteria": "Tool calls should be meaningful and the final answer should clearly use the observed tool output.",
            "notes": "Tool use on the trade war topic.",
        },
        {
            "id": "trade_war_agent_1",
            "topic": "trade_war",
            "category": "synthesis",
            "assistant_mode": "multi_agent",
            "prompt": "Provide a balanced assessment of trade war winners, losers, and uncertainties.",
            "expected_answer": "A synthesis with multiple perspectives, uncertainty, and a balanced final judgment.",
            "evaluation_criteria": "Should show analysis, critique, and synthesis without exaggeration.",
            "notes": "Multi-agent synthesis on trade policy.",
        },
        {
            "id": "censorship_gen_1",
            "topic": "censorship",
            "category": "overview",
            "assistant_mode": "generation",
            "prompt": "Summarize the main concerns about social network censorship raised in news coverage.",
            "expected_answer": "A concise overview of moderation, speech, bias, and platform responsibility concerns.",
            "evaluation_criteria": "Directly answers the prompt and remains broadly applicable.",
            "notes": "Baseline generation on censorship.",
        },
        {
            "id": "censorship_rag_1",
            "topic": "censorship",
            "category": "evidence_lookup",
            "assistant_mode": "rag",
            "prompt": "What kinds of moderation or misinformation responses are described in the corpus?",
            "expected_answer": "Mentions fact-checking, moderation policies, verification systems, or takedown approaches.",
            "evaluation_criteria": "Retrieves and synthesizes corpus evidence rather than giving a generic answer.",
            "notes": "RAG benchmark on content moderation.",
        },
        {
            "id": "censorship_react_1",
            "topic": "censorship",
            "category": "tool_usage",
            "assistant_mode": "react",
            "prompt": "Use a tool to inspect corpus references related to censorship and explain the key takeaway.",
            "expected_answer": "A tool-based explanation with a clear takeaway about censorship or moderation.",
            "evaluation_criteria": "Tool usage must be observable and relevant, not decorative.",
            "notes": "ReAct benchmark on censorship.",
        },
        {
            "id": "censorship_agent_1",
            "topic": "censorship",
            "category": "synthesis",
            "assistant_mode": "multi_agent",
            "prompt": "Give a balanced analysis of free speech concerns and moderation concerns on social platforms.",
            "expected_answer": "A nuanced answer that balances competing concerns and avoids one-sided framing.",
            "evaluation_criteria": "The final answer should reflect critique and synthesis, not just a first-pass summary.",
            "notes": "Multi-agent synthesis on censorship.",
        },
    ]


def _topic_counts(dataset: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for topic, items in _group_dataset(dataset, "topic").items():
        rows.append({"Topic": topic, "Count": len(items)})
    return pd.DataFrame(rows)


def _category_counts(dataset: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for category, items in _group_dataset(dataset, "category").items():
        rows.append({"Category": category, "Count": len(items)})
    return pd.DataFrame(rows)


def _group_dataset(dataset: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in dataset:
        grouped.setdefault(str(item.get(key, "unknown") or "unknown"), []).append(item)
    return grouped


# ---------------------------------------------------------------------------
# Result helpers
# ---------------------------------------------------------------------------


def _score_breakdown_frame(results: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for result in results.get("results", []):
        breakdown = result.get("score_breakdown", {}) or {}
        row = {
            "Test ID": result.get("test_id", ""),
            "Topic": result.get("topic", ""),
            "Category": result.get("category", ""),
            "Mode": result.get("assistant_mode", ""),
            "Temperature": result.get("temperature", settings.temperature),
            "Score": result.get("score", 0.0),
            "Status": result.get("status", ""),
            "Retrieved Chunks": result.get("retrieved_chunks_count", 0),
            "Tool Calls": result.get("tool_call_count", 0),
            "Latency (ms)": round(float(result.get("latency_ms", 0.0) or 0.0), 2),
            "Execution (ms)": round(float(result.get("execution_ms", 0.0) or 0.0), 2),
            "Reasoning": result.get("reasoning", ""),
        }
        for dim_key, dim_label in DIMENSION_LABELS.items():
            row[dim_label] = round(float(breakdown.get(dim_key, {}).get("score", 0.0) or 0.0), 2)
        rows.append(row)
    return pd.DataFrame(rows)


def _format_metric_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _result_label(result: dict[str, Any], index: int) -> str:
    return (
        f"#{index + 1} | {result.get('test_id', '')} | "
        f"{MODE_LABELS.get(result.get('assistant_mode', ''), result.get('assistant_mode', ''))} | "
        f"T={result.get('temperature', settings.temperature)} | Score={result.get('score', 0.0):.2f}"
    )


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------


def _bar_chart_from_summary(summary: dict[str, dict[str, Any]], title: str, x_label: str) -> None:
    if not summary:
        st.info(f"No {x_label.lower()} data to visualize yet.")
        return

    rows = []
    for label, stats in summary.items():
        rows.append(
            {
                x_label: label,
                "Avg Score": stats.get("avg_score", 0.0),
                "Count": stats.get("count", 0),
                "Completion Rate": stats.get("completion_rate", 0.0),
            }
        )

    df = pd.DataFrame(rows)
    fig = px.bar(
        df,
        x=x_label,
        y="Avg Score",
        color=x_label,
        text="Avg Score",
        hover_data=["Count", "Completion Rate"],
        title=title,
    )
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_layout(height=420, showlegend=False, yaxis=dict(range=[0, 10]))
    st.plotly_chart(fig, use_container_width=True)


def _dimension_chart(overall_dimensions: dict[str, float]) -> None:
    if not overall_dimensions:
        st.info("No dimension scores available.")
        return

    labels = [DIMENSION_LABELS.get(name, name.title()) for name in overall_dimensions.keys()]
    values = [overall_dimensions[name] for name in overall_dimensions.keys()]
    fig = go.Figure(
        data=[
            go.Scatterpolar(r=values + [values[0]], theta=labels + [labels[0]], fill="toself", name="Overall")
        ]
    )
    fig.update_layout(height=420, polar=dict(radialaxis=dict(visible=True, range=[0, 10])), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------


tab_dataset, tab_run, tab_results, tab_export = st.tabs(["Dataset", "Run", "Results", "Export"])

with tab_dataset:
    st.header("Dataset Management")
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Load Dataset")
        datasets = get_available_datasets()
        if datasets:
            selected_file = st.selectbox("Select dataset", datasets, format_func=lambda path: path.name)
            if st.button("Load Selected Dataset", use_container_width=True):
                st.session_state.benchmark_dataset = load_benchmark_dataset(selected_file)
                st.success(f"Loaded {len(st.session_state.benchmark_dataset)} test cases")
                st.rerun()
        else:
            st.info("No benchmark datasets found yet.")

    with col_right:
        st.subheader("Create Sample Dataset")
        if st.button("Create & Load Sample Dataset", use_container_width=True):
            sample_dataset = create_sample_dataset()
            save_benchmark_dataset(sample_dataset, "sample_benchmark.json")
            st.session_state.benchmark_dataset = sample_dataset
            st.success(f"Created sample dataset with {len(sample_dataset)} test cases")
            st.rerun()

    st.subheader("Current Dataset")
    if st.session_state.benchmark_dataset:
        dataset = st.session_state.benchmark_dataset
        dataset_df = pd.DataFrame(
            [
                {
                    "ID": item.get("id", ""),
                    "Topic": TOPIC_LABELS.get(item.get("topic", ""), item.get("topic", "")),
                    "Category": item.get("category", ""),
                    "Mode": MODE_LABELS.get(item.get("assistant_mode", ""), item.get("assistant_mode", "")),
                    "Prompt": (item.get("prompt", "")[:90] + "...") if len(item.get("prompt", "")) > 90 else item.get("prompt", ""),
                }
                for item in dataset
            ]
        )
        st.dataframe(dataset_df, use_container_width=True, hide_index=True)

        stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
        with stats_col1:
            st.metric("Test Cases", len(dataset))
        with stats_col2:
            st.metric("Topics", len({item.get("topic", "unknown") for item in dataset}))
        with stats_col3:
            st.metric("Categories", len({item.get("category", "unknown") for item in dataset}))
        with stats_col4:
            st.metric("Modes", len({item.get("assistant_mode", "generation") for item in dataset}))

        topic_df = _topic_counts(dataset)
        category_df = _category_counts(dataset)
        if not topic_df.empty:
            st.markdown("**Topic Distribution**")
            st.bar_chart(topic_df.set_index("Topic"))
        if not category_df.empty:
            st.markdown("**Category Distribution**")
            st.bar_chart(category_df.set_index("Category"))
    else:
        st.warning("No dataset loaded. Load an existing dataset or create the sample dataset.")

    st.subheader("Upload Custom Dataset")
    uploaded_file = st.file_uploader("Upload benchmark JSON file", type=["json"], key="benchmark_upload")
    if uploaded_file is not None:
        try:
            payload = json.load(uploaded_file)
            if isinstance(payload, dict) and isinstance(payload.get("test_cases"), list):
                dataset = payload["test_cases"]
            elif isinstance(payload, list):
                dataset = payload
            else:
                raise ValueError("Expected a list or a JSON object with a test_cases field")

            st.session_state.benchmark_dataset = dataset
            filename = f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            save_benchmark_dataset(dataset, filename)
            st.success(f"Uploaded {len(dataset)} test cases")
            st.rerun()
        except Exception as exc:
            st.error(f"Invalid benchmark dataset: {exc}")

with tab_run:
    st.header("Benchmark Run")

    if not st.session_state.benchmark_dataset:
        st.warning("Load a dataset first.")
    else:
        dataset = st.session_state.benchmark_dataset
        left, right = st.columns(2)

        with left:
            selected_modes = st.multiselect(
                "Assistant modes to benchmark",
                options=["generation", "rag", "react", "multi_agent"],
                default=["generation", "rag", "react", "multi_agent"],
                format_func=lambda mode: MODE_LABELS.get(mode, mode),
            )
            use_rag = st.checkbox("Enable RAG retrieval", value=True)
            top_p = st.slider("Top-P", min_value=0.0, max_value=1.0, value=settings.top_p, step=0.05)
            top_k = st.slider("Top-K retrieval", min_value=1, max_value=20, value=settings.top_k_retrieval, step=1)
            temperature_grid_text = st.text_input("Temperature grid", value="0.2, 0.5, 0.8")
            repetitions = st.slider("Repetitions per run", min_value=1, max_value=5, value=2, step=1)

        with right:
            st.write("**Dataset Summary**")
            st.metric("Test Cases", len(dataset))
            st.metric("Topics", len({item.get('topic', 'unknown') for item in dataset}))
            st.metric("Categories", len({item.get('category', 'unknown') for item in dataset}))
            st.metric("Modes in Dataset", len({item.get('assistant_mode', 'generation') for item in dataset}))
            st.caption("The benchmark will evaluate every selected dataset case and then judge each run with multi-axis scores.")

        st.divider()

        if st.button("🚀 Start Benchmark", type="primary", disabled=st.session_state.benchmark_running):
            try:
                temperature_grid = [float(item.strip()) for item in temperature_grid_text.split(",") if item.strip()]
            except ValueError:
                st.error("Temperature grid must be a comma-separated list of numbers.")
            else:
                if not temperature_grid:
                    st.error("Please provide at least one temperature value.")
                else:
                    st.session_state.benchmark_running = True
                    with st.spinner("Running benchmark through the assistant stack..."):
                        client = LMStudioClient(
                            base_url=settings.lm_studio_base_url,
                            model_name=settings.chat_model,
                            timeout=settings.request_timeout,
                        )
                        benchmark_results = run_full_benchmark(
                            test_cases=dataset,
                            client=client,
                            use_rag=use_rag,
                            temperature=temperature_grid[0],
                            top_p=top_p,
                            top_k=top_k,
                            temperature_grid=temperature_grid,
                            repetitions=repetitions,
                            selected_modes=selected_modes,
                        )
                        st.session_state.benchmark_results = benchmark_results
                        st.session_state.benchmark_run_history.append(benchmark_results)
                        st.session_state.benchmark_running = False
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        save_benchmark_results(benchmark_results, f"result_{timestamp}.json")
                        st.success("Benchmark completed")
                        st.rerun()

        st.subheader("Active Dataset Modes")
        mode_counts = pd.DataFrame(
            [
                {"Mode": MODE_LABELS.get(mode, mode), "Count": count}
                for mode, count in sorted(
                    {
                        item.get("assistant_mode", "generation"): sum(
                            1 for test_case in dataset if test_case.get("assistant_mode", "generation") == item.get("assistant_mode", "generation")
                        )
                        for item in dataset
                    }.items()
                )
            ]
        )
        if not mode_counts.empty:
            st.dataframe(mode_counts, use_container_width=True, hide_index=True)

with tab_results:
    st.header("Results")

    if st.session_state.benchmark_results is None:
        st.info("Run a benchmark to see results.")
    else:
        results = st.session_state.benchmark_results
        aggregated = aggregate_results(results.get("results", []))
        overall = aggregated.get("overall", {})

        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        with metric_col1:
            st.metric("Total Runs", results.get("total_runs", 0))
        with metric_col2:
            st.metric("Completed", results.get("completed", 0))
        with metric_col3:
            st.metric("Failed", results.get("failed", 0))
        with metric_col4:
            st.metric("Avg Score", f"{overall.get('avg_score', 0.0):.2f} / 10")

        st.divider()

        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            _bar_chart_from_summary(aggregated.get("by_topic", {}), "Average Score by Topic", "Topic")
        with chart_col2:
            _bar_chart_from_summary(aggregated.get("by_category", {}), "Average Score by Category", "Category")

        chart_col3, chart_col4 = st.columns(2)
        with chart_col3:
            _bar_chart_from_summary(aggregated.get("by_mode", {}), "Average Score by Mode", "Mode")
        with chart_col4:
            _bar_chart_from_summary(aggregated.get("by_temperature", {}), "Average Score by Temperature", "Temperature")

        st.divider()
        breakdown_col1, breakdown_col2 = st.columns([1, 1])
        with breakdown_col1:
            st.subheader("Judge Dimension Breakdown")
            _dimension_chart(overall.get("dimensions", {}))
        with breakdown_col2:
            st.subheader("Overall Summary")
            summary_rows = [
                ("Average Score", overall.get("avg_score", 0.0)),
                ("Median Score", overall.get("median_score", 0.0)),
                ("Minimum Score", overall.get("min_score", 0.0)),
                ("Maximum Score", overall.get("max_score", 0.0)),
                ("Completion Rate", overall.get("completion_rate", 0.0)),
                ("Average Latency (ms)", overall.get("avg_latency_ms", 0.0)),
                ("Average Tool Calls", overall.get("avg_tool_calls", 0.0)),
                ("Average Retrieved Chunks", overall.get("avg_retrieved_chunks", 0.0)),
            ]
            st.dataframe(pd.DataFrame(summary_rows, columns=["Metric", "Value"]), use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("Topic Details")
        if aggregated.get("by_topic"):
            topic_rows = []
            for topic, stats in aggregated["by_topic"].items():
                topic_rows.append(
                    {
                        "Topic": TOPIC_LABELS.get(topic, topic),
                        "Count": stats.get("count", 0),
                        "Avg Score": round(stats.get("avg_score", 0.0), 2),
                        "Completion Rate": round(stats.get("completion_rate", 0.0), 2),
                        "Avg Tool Calls": round(stats.get("avg_tool_calls", 0.0), 2),
                        "Avg Retrieved Chunks": round(stats.get("avg_retrieved_chunks", 0.0), 2),
                    }
                )
            st.dataframe(pd.DataFrame(topic_rows), use_container_width=True, hide_index=True)

        st.subheader("Category Details")
        if aggregated.get("by_category"):
            category_rows = []
            for category, stats in aggregated["by_category"].items():
                category_rows.append(
                    {
                        "Category": category,
                        "Count": stats.get("count", 0),
                        "Avg Score": round(stats.get("avg_score", 0.0), 2),
                        "Median": round(stats.get("median_score", 0.0), 2),
                        "Completion Rate": round(stats.get("completion_rate", 0.0), 2),
                    }
                )
            st.dataframe(pd.DataFrame(category_rows), use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("Detailed Runs")
        detailed_df = results_to_dataframe(results)
        st.dataframe(detailed_df, use_container_width=True, hide_index=True)

        st.subheader("Inspect One Run")
        if results.get("results"):
            run_labels = [_result_label(result, index) for index, result in enumerate(results["results"])]
            selected_label = st.selectbox("Select a run", run_labels)
            selected_index = run_labels.index(selected_label)
            selected_run = results["results"][selected_index]

            left, right = st.columns(2)
            with left:
                st.markdown("**Prompt**")
                st.text_area("Prompt", value=selected_run.get("prompt", ""), height=120, disabled=True)
                st.markdown("**Response**")
                st.text_area("Response", value=selected_run.get("answer", ""), height=220, disabled=True)
            with right:
                st.metric("Score", f"{selected_run.get('score', 0.0):.2f} / 10")
                st.write(f"**Mode:** {MODE_LABELS.get(selected_run.get('assistant_mode', ''), selected_run.get('assistant_mode', ''))}")
                st.write(f"**Topic:** {TOPIC_LABELS.get(selected_run.get('topic', ''), selected_run.get('topic', ''))}")
                st.write(f"**Category:** {selected_run.get('category', '')}")
                st.write(f"**Temperature:** {selected_run.get('temperature', settings.temperature)}")
                st.write(f"**Retrieved Chunks:** {selected_run.get('retrieved_chunks_count', 0)}")
                st.write(f"**Tool Calls:** {selected_run.get('tool_call_count', 0)}")
                if selected_run.get("error"):
                    st.error(selected_run.get("error"))

            dim_breakdown = selected_run.get("score_breakdown", {}) or {}
            breakdown_rows = []
            for dim_key, label in DIMENSION_LABELS.items():
                item = dim_breakdown.get(dim_key, {}) or {}
                breakdown_rows.append(
                    {
                        "Dimension": label,
                        "Score": round(float(item.get("score", 0.0) or 0.0), 2),
                        "Explanation": item.get("explanation", ""),
                    }
                )
            st.dataframe(pd.DataFrame(breakdown_rows), use_container_width=True, hide_index=True)

            with st.expander("Judge Reasoning", expanded=False):
                st.write(selected_run.get("reasoning", ""))
                if selected_run.get("strengths"):
                    st.markdown("**Strengths**")
                    for strength in selected_run.get("strengths", []):
                        st.write(f"- {strength}")
                if selected_run.get("gaps"):
                    st.markdown("**Gaps**")
                    for gap in selected_run.get("gaps", []):
                        st.write(f"- {gap}")
                if selected_run.get("judge_raw"):
                    st.code(selected_run.get("judge_raw", ""), language="json")

            if selected_run.get("agent_trace"):
                with st.expander("Agent Trace", expanded=False):
                    st.code(json.dumps(selected_run.get("agent_trace", []), ensure_ascii=False, indent=2, default=str), language="json")

            if selected_run.get("tool_calls"):
                with st.expander("Tool Calls", expanded=False):
                    st.code(json.dumps(selected_run.get("tool_calls", []), ensure_ascii=False, indent=2, default=str), language="json")

with tab_export:
    st.header("Export")

    if st.session_state.benchmark_results is None:
        st.info("Run a benchmark first.")
    else:
        results = st.session_state.benchmark_results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_payload = json.dumps(results, indent=2, ensure_ascii=False, default=str)
        csv_payload = results_to_dataframe(results).to_csv(index=False)

        col_json, col_csv = st.columns(2)
        with col_json:
            st.subheader("JSON")
            st.download_button(
                "Download JSON",
                data=json_payload,
                file_name=f"benchmark_results_{timestamp}.json",
                mime="application/json",
                use_container_width=True,
            )
            if st.button("Save JSON locally", use_container_width=True):
                filepath = save_benchmark_results(results, f"result_{timestamp}.json")
                st.success(f"Saved to {filepath.name}")

        with col_csv:
            st.subheader("CSV")
            st.download_button(
                "Download CSV",
                data=csv_payload,
                file_name=f"benchmark_results_{timestamp}.csv",
                mime="text/csv",
                use_container_width=True,
            )

        st.subheader("Markdown Report")
        aggregated = aggregate_results(results.get("results", []))
        overall = aggregated.get("overall", {})
        report_lines = [
            "# Benchmark Report",
            f"Generated: {datetime.now().isoformat()}",
            "",
            "## Summary",
            f"- Total runs: {results.get('total_runs', 0)}",
            f"- Completed: {results.get('completed', 0)}",
            f"- Failed: {results.get('failed', 0)}",
            f"- Average score: {overall.get('avg_score', 0.0):.2f}",
            f"- Median score: {overall.get('median_score', 0.0):.2f}",
            "",
            "## Topic Summary",
        ]
        for topic, stats in aggregated.get("by_topic", {}).items():
            report_lines.append(
                f"- {TOPIC_LABELS.get(topic, topic)}: avg={stats.get('avg_score', 0.0):.2f}, count={stats.get('count', 0)}, completion={stats.get('completion_rate', 0.0):.2f}"
            )
        report_lines.extend(["", "## Category Summary"])
        for category, stats in aggregated.get("by_category", {}).items():
            report_lines.append(
                f"- {category}: avg={stats.get('avg_score', 0.0):.2f}, count={stats.get('count', 0)}, completion={stats.get('completion_rate', 0.0):.2f}"
            )

        report_text = "\n".join(report_lines)
        st.download_button(
            "Download Markdown Report",
            data=report_text,
            file_name=f"benchmark_report_{timestamp}.md",
            mime="text/markdown",
            use_container_width=True,
        )
