"""Benchmark page for test-driven evaluation.

Supports:
- Loading and managing benchmark test cases
- Running the assistant on test cases with optional RAG
- LLM-as-Judge evaluation for automatic scoring
- Results visualization and aggregation
- Report export to CSV and JSON
"""

import streamlit as st
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from config import settings
from core.llm_client import LMStudioClient
from core.evaluator import (
    run_benchmark_case,
    evaluate_response,
    run_full_benchmark,
    aggregate_results,
)
from core.utils import setup_logging

# Configure logging
logger = setup_logging(level="INFO")

# Configure page
st.set_page_config(
    page_title="Benchmark - AI News Intelligence Assistant",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Benchmark")
st.markdown("Evaluate assistant performance on a curated test dataset")

# Initialize session state
if "benchmark_dataset" not in st.session_state:
    st.session_state.benchmark_dataset = []

if "benchmark_results" not in st.session_state:
    st.session_state.benchmark_results = None

if "benchmark_running" not in st.session_state:
    st.session_state.benchmark_running = False

if "current_test_index" not in st.session_state:
    st.session_state.current_test_index = 0


# ============================================================================
# Helper Functions
# ============================================================================


def load_benchmark_dataset(filepath: Path) -> List[Dict]:
    """Load benchmark dataset from JSON file.
    
    Args:
        filepath: Path to benchmark JSON file
        
    Returns:
        List of test case dictionaries
    """
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
        
        # Handle both list format and dict with 'test_cases' key
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and "test_cases" in data:
            return data["test_cases"]
        else:
            return []
    except Exception as e:
        logger.error(f"Error loading benchmark dataset: {e}")
        st.error(f"Failed to load dataset: {str(e)}")
        return []


def save_benchmark_dataset(dataset: List[Dict], filename: str) -> Path:
    """Save benchmark dataset to JSON file.
    
    Args:
        dataset: List of test cases
        filename: Name of the file (without path)
        
    Returns:
        Path to the saved file
    """
    (settings.project_root / settings.benchmark_dir).mkdir(parents=True, exist_ok=True)
    filepath = settings.project_root / settings.benchmark_dir / filename
    
    with open(filepath, "w") as f:
        json.dump({
            "test_cases": dataset,
            "created_at": datetime.now().isoformat(),
            "count": len(dataset),
        }, f, indent=2)
    
    logger.info(f"Saved benchmark dataset to {filepath}")
    return filepath


def save_benchmark_results(results: Dict, filename: str) -> Path:
    """Save benchmark results to JSON file.
    
    Args:
        results: Benchmark results dictionary
        filename: Name of the file (without path)
        
    Returns:
        Path to the saved file
    """
    (settings.project_root / settings.benchmark_dir).mkdir(parents=True, exist_ok=True)
    filepath = settings.project_root / settings.benchmark_dir / filename
    
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved benchmark results to {filepath}")
    return filepath


def get_available_datasets() -> List[Path]:
    """Get list of available benchmark datasets.
    
    Returns:
        List of JSON files in benchmark directory
    """
    (settings.project_root / settings.benchmark_dir).mkdir(parents=True, exist_ok=True)
    files = list((settings.project_root / settings.benchmark_dir).glob("*.json"))
    return [f for f in files if not f.name.startswith("result_")]


def create_sample_dataset() -> List[Dict]:
    """Create a sample benchmark dataset for news analysis.
    
    Returns:
        List of test cases
    """
    return [
        {
            "id": "green_energy_1",
            "category": "green_energy",
            "prompt": "What are the latest developments in green energy adoption according to CNN?",
            "expected_answer": "Should include information about renewable energy, solar/wind power, or clean energy initiatives",
            "evaluation_criteria": "Response should be grounded in corpus facts, mention specific technologies or initiatives, and provide current information",
            "notes": "Focus on CNN coverage of green energy trends",
        },
        {
            "id": "green_energy_2",
            "category": "green_energy",
            "prompt": "How is the US government addressing climate change through policy?",
            "expected_answer": "Should discuss government initiatives, policies, or regulations related to environmental protection",
            "evaluation_criteria": "Accuracy of policy names and details, relevance to climate change, grounding in corpus",
            "notes": "Evaluate knowledge of government policies",
        },
        {
            "id": "trade_war_1",
            "category": "trade_war",
            "prompt": "What are the main impacts of the US trade war on businesses?",
            "expected_answer": "Should include tariffs, supply chain disruptions, and economic effects",
            "evaluation_criteria": "Specificity of impacts, accuracy of tariff information, relevance to documented events",
            "notes": "Focus on economic and business implications",
        },
        {
            "id": "trade_war_2",
            "category": "trade_war",
            "prompt": "Which countries are most affected by US trade policies?",
            "expected_answer": "Should mention China, Canada, EU, and other major trading partners",
            "evaluation_criteria": "Accuracy of countries listed, specific impacts mentioned, grounding in reporting",
            "notes": "Evaluate geographic and geopolitical knowledge",
        },
        {
            "id": "censorship_1",
            "category": "censorship",
            "prompt": "What are the main concerns about social media censorship?",
            "expected_answer": "Should include free speech, moderation policies, political bias concerns",
            "evaluation_criteria": "Comprehensive coverage of concerns, balance of perspectives, grounding in news reporting",
            "notes": "Evaluate balanced representation of viewpoints",
        },
        {
            "id": "censorship_2",
            "category": "censorship",
            "prompt": "How are social networks handling misinformation?",
            "expected_answer": "Should discuss fact-checking, content moderation, or verification systems",
            "evaluation_criteria": "Accuracy of platform approaches, specific tools mentioned, relevance to corpus",
            "notes": "Focus on technical and policy approaches",
        },
    ]


def results_to_dataframe(results: Dict) -> pd.DataFrame:
    """Convert benchmark results to a pandas DataFrame for display.
    
    Args:
        results: Benchmark results dictionary
        
    Returns:
        DataFrame with results
    """
    data = []
    for result in results.get("results", []):
        data.append({
            "Test ID": result.get("test_id", ""),
            "Category": result.get("category", ""),
            "Score": result.get("score", 0),
            "Status": result.get("status", ""),
            "Latency (ms)": f"{result.get('latency_ms', 0):.0f}",
            "Retrieved Chunks": result.get("retrieved_chunks", 0),
        })
    
    if data:
        return pd.DataFrame(data)
    else:
        return pd.DataFrame()


def create_score_distribution_chart(results: Dict):
    """Create a histogram of score distribution.
    
    Args:
        results: Benchmark results
    """
    scores = [r.get("score", 0) for r in results.get("results", []) if r.get("status") == "completed"]
    
    if not scores:
        st.warning("No completed results to visualize")
        return
    
    fig = go.Figure(data=[
        go.Histogram(x=scores, nbinsx=10, name="Scores")
    ])
    
    fig.update_layout(
        title="Score Distribution",
        xaxis_title="Score",
        yaxis_title="Frequency",
        height=400,
        showlegend=False,
    )
    
    st.plotly_chart(fig, use_container_width=True)


def create_category_chart(aggregated: Dict):
    """Create a bar chart of average scores by category.
    
    Args:
        aggregated: Aggregated results
    """
    by_category = aggregated.get("by_category", {})
    
    if not by_category:
        st.warning("No category data to visualize")
        return
    
    categories = list(by_category.keys())
    avg_scores = [by_category[cat].get("avg_score", 0) for cat in categories]
    
    fig = go.Figure(data=[
        go.Bar(x=categories, y=avg_scores, name="Avg Score")
    ])
    
    fig.update_layout(
        title="Average Score by Category",
        xaxis_title="Category",
        yaxis_title="Average Score",
        height=400,
        showlegend=False,
        yaxis=dict(range=[0, 10]),
    )
    
    st.plotly_chart(fig, use_container_width=True)


def create_summary_table(aggregated: Dict):
    """Create a summary statistics table.
    
    Args:
        aggregated: Aggregated results
    """
    overall = aggregated.get("overall", {})
    by_category = aggregated.get("by_category", {})
    
    summary_data = {
        "Metric": [
            "Overall Average Score",
            "Overall Min Score",
            "Overall Max Score",
            "Overall Median Score",
            "Total Tests",
        ],
        "Value": [
            f"{overall.get('avg_score', 0):.2f} / 10",
            f"{overall.get('min_score', 0):.1f} / 10",
            f"{overall.get('max_score', 0):.1f} / 10",
            f"{overall.get('median_score', 0):.2f} / 10",
            overall.get("count", 0),
        ],
    }
    
    df = pd.DataFrame(summary_data)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Category breakdown
    if by_category:
        st.subheader("Category Breakdown")
        category_data = {
            "Category": list(by_category.keys()),
            "Avg Score": [f"{by_category[cat].get('avg_score', 0):.2f}" for cat in by_category.keys()],
            "Min": [f"{by_category[cat].get('min_score', 0):.1f}" for cat in by_category.keys()],
            "Max": [f"{by_category[cat].get('max_score', 0):.1f}" for cat in by_category.keys()],
            "Count": [by_category[cat].get("count", 0) for cat in by_category.keys()],
        }
        df_category = pd.DataFrame(category_data)
        st.dataframe(df_category, use_container_width=True, hide_index=True)


def export_results_csv(results: Dict) -> str:
    """Export benchmark results to CSV format.
    
    Args:
        results: Benchmark results
        
    Returns:
        CSV content as string
    """
    df = results_to_dataframe(results)
    
    # Add additional columns
    for result in results.get("results", []):
        test_id = result.get("test_id", "")
        idx = df[df["Test ID"] == test_id].index
        if len(idx) > 0:
            df.loc[idx[0], "Prompt"] = result.get("prompt", "")[:100]
            df.loc[idx[0], "Reasoning"] = result.get("reasoning", "")[:100]
    
    return df.to_csv(index=False)


# ============================================================================
# Main UI
# ============================================================================


# Tabs for different sections
tab1, tab2, tab3, tab4 = st.tabs(
    ["Dataset Management", "Run Benchmark", "Results", "Export"]
)

# ============================================================================
# Tab 1: Dataset Management
# ============================================================================

with tab1:
    st.header("Dataset Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Load Existing Dataset")
        available_datasets = get_available_datasets()
        
        if available_datasets:
            selected_file = st.selectbox(
                "Select dataset",
                available_datasets,
                format_func=lambda x: x.name,
            )
            
            if st.button("Load Dataset"):
                dataset = load_benchmark_dataset(selected_file)
                st.session_state.benchmark_dataset = dataset
                st.success(f"Loaded {len(dataset)} test cases")
                st.rerun()
        else:
            st.info("No existing datasets found")
    
    with col2:
        st.subheader("Create Sample Dataset")
        if st.button("Create & Load Sample Dataset"):
            sample_dataset = create_sample_dataset()
            filepath = save_benchmark_dataset(sample_dataset, "sample_benchmark.json")
            st.session_state.benchmark_dataset = sample_dataset
            st.success(f"Created sample dataset with {len(sample_dataset)} test cases")
            st.rerun()
    
    # Display current dataset
    st.subheader("Current Dataset")
    if st.session_state.benchmark_dataset:
        dataset_df = pd.DataFrame([
            {
                "ID": tc.get("id", ""),
                "Category": tc.get("category", ""),
                "Prompt": tc.get("prompt", "")[:80] + "...",
                "Has Expected Answer": bool(tc.get("expected_answer")),
                "Has Criteria": bool(tc.get("evaluation_criteria")),
            }
            for tc in st.session_state.benchmark_dataset
        ])
        st.dataframe(dataset_df, use_container_width=True, hide_index=True)
        
        st.info(f"Total: {len(st.session_state.benchmark_dataset)} test cases")
        
        # Category distribution
        categories = {}
        for tc in st.session_state.benchmark_dataset:
            cat = tc.get("category", "unknown")
            categories[cat] = categories.get(cat, 0) + 1
        
        st.write("**Category Distribution:**")
        for cat, count in sorted(categories.items()):
            st.write(f"- {cat}: {count} tests")
    else:
        st.warning("No dataset loaded. Load or create a dataset to begin.")
    
    # Upload custom dataset
    st.subheader("Upload Custom Dataset")
    uploaded_file = st.file_uploader(
        "Upload benchmark JSON file",
        type=["json"],
        key="benchmark_upload",
    )
    
    if uploaded_file is not None:
        try:
            dataset = json.load(uploaded_file)
            if isinstance(dataset, dict) and "test_cases" in dataset:
                dataset = dataset["test_cases"]
            
            if isinstance(dataset, list) and len(dataset) > 0:
                st.session_state.benchmark_dataset = dataset
                
                # Save the uploaded file
                filename = f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                save_benchmark_dataset(dataset, filename)
                
                st.success(f"Uploaded {len(dataset)} test cases")
                st.rerun()
            else:
                st.error("Invalid dataset format. Expected list of test cases.")
        except Exception as e:
            st.error(f"Error uploading dataset: {str(e)}")


# ============================================================================
# Tab 2: Run Benchmark
# ============================================================================

with tab2:
    st.header("Run Benchmark")
    
    if not st.session_state.benchmark_dataset:
        st.warning("Please load a dataset first in the 'Dataset Management' tab")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            use_rag = st.checkbox("Use RAG for context retrieval", value=True)
            temperature = st.slider(
                "Generation Temperature",
                min_value=0.0,
                max_value=2.0,
                value=settings.temperature,
                step=0.1,
            )
        
        with col2:
            st.write("**Dataset Info:**")
            st.metric("Total Tests", len(st.session_state.benchmark_dataset))
            categories = set(tc.get("category", "unknown") for tc in st.session_state.benchmark_dataset)
            st.metric("Categories", len(categories))
        
        st.divider()
        
        # Run button
        if st.button(
            "🚀 Start Benchmark Run",
            type="primary",
            disabled=st.session_state.benchmark_running,
        ):
            st.session_state.benchmark_running = True
            st.rerun()
        
        # Progress display
        if st.session_state.benchmark_running:
            st.info("⏳ Benchmark is running. This may take several minutes...")
            
            try:
                client = LMStudioClient(
                    base_url=settings.lm_studio_base_url,
                    model_name=settings.chat_model,
                    timeout=settings.request_timeout,
                )
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Run benchmark
                results = run_full_benchmark(
                    st.session_state.benchmark_dataset,
                    client,
                    use_rag=use_rag,
                    temperature=temperature,
                )
                
                # Store results
                st.session_state.benchmark_results = results
                st.session_state.benchmark_running = False
                
                st.success("✅ Benchmark completed!")
                st.rerun()
                
            except Exception as e:
                st.error(f"Error running benchmark: {str(e)}")
                logger.error(f"Benchmark error: {e}", exc_info=True)
                st.session_state.benchmark_running = False
                st.rerun()


# ============================================================================
# Tab 3: Results
# ============================================================================

with tab3:
    st.header("Results")
    
    if st.session_state.benchmark_results is None:
        st.info("Run a benchmark first to see results")
    else:
        results = st.session_state.benchmark_results
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Tests", results.get("total_tests", 0))
        with col2:
            st.metric("Completed", results.get("completed", 0))
        with col3:
            st.metric("Failed", results.get("failed", 0))
        with col4:
            aggregated = aggregate_results(results.get("results", []))
            overall_avg = aggregated.get("overall", {}).get("avg_score", 0)
            st.metric("Overall Avg Score", f"{overall_avg:.2f}/10")
        
        st.divider()
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            create_score_distribution_chart(results)
        
        with col2:
            aggregated = aggregate_results(results.get("results", []))
            create_category_chart(aggregated)
        
        st.divider()
        
        # Summary table
        aggregated = aggregate_results(results.get("results", []))
        create_summary_table(aggregated)
        
        st.divider()
        
        # Detailed results table
        st.subheader("Detailed Results")
        df_results = results_to_dataframe(results)
        st.dataframe(df_results, use_container_width=True, hide_index=True)
        
        # Individual test details
        st.subheader("Test Case Details")
        
        selected_test_id = st.selectbox(
            "Select test to view details",
            [r.get("test_id", "") for r in results.get("results", [])],
        )
        
        for result in results.get("results", []):
            if result.get("test_id") == selected_test_id:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Prompt:**")
                    st.text_area("Question", value=result.get("prompt", ""), height=100, disabled=True)
                
                with col2:
                    st.write("**Score:** " + f"{result.get('score', 0):.1f}/10")
                    st.write("**Status:** " + result.get("status", "unknown"))
                    if result.get("error"):
                        st.error(f"**Error:** {result.get('error')}")
                
                st.write("**Response:**")
                st.text_area(
                    "Assistant Response",
                    value=result.get("response", "")[:1000],
                    height=150,
                    disabled=True,
                )
                
                st.write("**Judge Reasoning:**")
                st.text_area(
                    "Evaluation",
                    value=result.get("reasoning", ""),
                    height=100,
                    disabled=True,
                )
                
                break


# ============================================================================
# Tab 4: Export
# ============================================================================

with tab4:
    st.header("Export Results")
    
    if st.session_state.benchmark_results is None:
        st.info("Run a benchmark first to export results")
    else:
        results = st.session_state.benchmark_results
        
        # Export as JSON
        st.subheader("Export as JSON")
        json_str = json.dumps(results, indent=2)
        
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="📥 Download JSON",
                data=json_str,
                file_name=f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
            )
        
        with col2:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if st.button("💾 Save Results Locally"):
                filepath = save_benchmark_results(results, f"result_{timestamp}.json")
                st.success(f"Results saved to {filepath.name}")
        
        # Export as CSV
        st.subheader("Export as CSV")
        csv_str = export_results_csv(results)
        st.download_button(
            label="📥 Download CSV",
            data=csv_str,
            file_name=f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )
        
        # Export as markdown report
        st.subheader("Generate Report")
        
        if st.button("📄 Generate Markdown Report"):
            aggregated = aggregate_results(results.get("results", []))
            overall = aggregated.get("overall", {})
            by_category = aggregated.get("by_category", {})
            
            report = f"""# Benchmark Evaluation Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

- **Total Tests:** {results.get('total_tests', 0)}
- **Completed:** {results.get('completed', 0)}
- **Failed:** {results.get('failed', 0)}

## Overall Metrics

- **Average Score:** {overall.get('avg_score', 0):.2f} / 10
- **Min Score:** {overall.get('min_score', 0):.1f}
- **Max Score:** {overall.get('max_score', 0):.1f}
- **Median Score:** {overall.get('median_score', 0):.2f}

## Results by Category

"""
            
            for category, stats in by_category.items():
                report += f"""
### {category.title()}

- **Count:** {stats.get('count', 0)}
- **Average Score:** {stats.get('avg_score', 0):.2f} / 10
- **Min:** {stats.get('min_score', 0):.1f}
- **Max:** {stats.get('max_score', 0):.1f}
"""
            
            report += "\n## Individual Results\n\n"
            report += "| Test ID | Category | Score | Status |\n"
            report += "|---------|----------|-------|--------|\n"
            
            for result in results.get("results", []):
                report += (
                    f"| {result.get('test_id', '')} | "
                    f"{result.get('category', '')} | "
                    f"{result.get('score', 0):.1f}/10 | "
                    f"{result.get('status', '')} |\n"
                )
            
            st.download_button(
                label="📄 Download Report",
                data=report,
                file_name=f"benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
            )
            
            st.success("Report generated!")
