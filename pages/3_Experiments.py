"""Experiments page for generation parameter analysis.

Supports controlled experiments comparing different temperatures,
repeated sampling, output variability analysis, and results visualization.
"""

import streamlit as st
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List
import pandas as pd
import numpy as np

from config import settings
from core.llm_client import LMStudioClient
from core.utils import setup_logging

# Configure logging
logger = setup_logging(level="INFO")

# Configure page
st.set_page_config(
    page_title="Experiments - AI News Intelligence Assistant",
    page_icon="🧪",
    layout="wide",
)

st.title("🧪 Experiments")
st.markdown("Analyze generation parameters through controlled experiments")


# Initialize session state
if "experiments" not in st.session_state:
    st.session_state.experiments = []

if "current_experiment" not in st.session_state:
    st.session_state.current_experiment = None

if "experiment_results" not in st.session_state:
    st.session_state.experiment_results = []


def save_experiment_to_disk(experiment_data: Dict) -> Path:
    """Save experiment data to disk as JSON.
    
    Args:
        experiment_data: Dictionary containing experiment metadata and results
        
    Returns:
        Path to the saved file
    """
    experiments_dir = settings.project_root / settings.data_dir / "experiments"
    experiments_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"experiment_{timestamp}.json"
    filepath = experiments_dir / filename
    
    with open(filepath, "w") as f:
        json.dump(experiment_data, f, indent=2)
    
    logger.info(f"Saved experiment to {filepath}")
    return filepath


def load_experiments_from_disk() -> List[Dict]:
    """Load all experiments from disk.
    
    Returns:
        List of experiment dictionaries
    """
    experiments_dir = settings.project_root / settings.data_dir / "experiments"
    experiments_dir.mkdir(parents=True, exist_ok=True)
    
    experiments = []
    for filepath in sorted(experiments_dir.glob("experiment_*.json"), reverse=True):
        try:
            with open(filepath, "r") as f:
                exp_data = json.load(f)
                experiments.append(exp_data)
        except Exception as e:
            logger.error(f"Failed to load experiment from {filepath}: {str(e)}")
    
    return experiments


def run_single_experiment(
    prompt: str,
    temperature: float,
    top_p: float,
    top_k: int,
    llm_client: LMStudioClient,
) -> Dict:
    """Run a single LLM generation with given parameters.
    
    Args:
        prompt: Input prompt for generation
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        llm_client: LM Studio client for generation
        
    Returns:
        Dictionary with response and metadata
    """
    try:
        response = llm_client.complete(
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=settings.max_tokens,
        )
        
        return {
            "text": response.text,
            "length": len(response.text),
            "tokens": int(response.tokens_used) if hasattr(response.tokens_used, 'item') else int(response.tokens_used),
            "latency_ms": float(response.latency_ms) if hasattr(response.latency_ms, 'item') else float(response.latency_ms),
            "finish_reason": response.finish_reason,
            "success": True,
            "error": None,
        }
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Generation failed: {error_msg}")
        return {
            "text": "",
            "length": 0,
            "tokens": 0,
            "latency_ms": 0,
            "finish_reason": "error",
            "success": False,
            "error": error_msg,
        }


def run_temperature_experiment(
    prompt: str,
    temperatures: List[float],
    num_runs: int,
    top_p: float,
    top_k: int,
) -> Dict:
    """Run an experiment comparing multiple temperatures.
    
    Args:
        prompt: Input prompt for generation
        temperatures: List of temperature values to test
        num_runs: Number of runs per temperature
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        
    Returns:
        Dictionary with experiment results
    """
    llm_client = LMStudioClient(
        base_url=settings.lm_studio_base_url,
        model_name=settings.chat_model,
        timeout=settings.request_timeout,
    )
    
    experiment_data = {
        "timestamp": datetime.now().isoformat(),
        "prompt": prompt,
        "temperatures": temperatures,
        "num_runs": num_runs,
        "top_p": top_p,
        "top_k": top_k,
        "results": [],
        "summary": {},
    }
    
    total_runs = len(temperatures) * num_runs
    progress_bar = st.progress(0)
    progress_text = st.empty()
    
    run_count = 0
    
    for temp_idx, temperature in enumerate(temperatures):
        temp_results = {
            "temperature": temperature,
            "runs": [],
        }
        
        for run_idx in range(num_runs):
            progress_text.text(
                f"Running: Temperature {temperature:.2f}, Run {run_idx + 1}/{num_runs}"
            )
            
            result = run_single_experiment(
                prompt=prompt,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                llm_client=llm_client,
            )
            
            result["run_index"] = run_idx + 1
            temp_results["runs"].append(result)
            
            run_count += 1
            progress_bar.progress(run_count / total_runs)
        
        # Compute temperature-level statistics
        successful_runs = [r for r in temp_results["runs"] if r["success"]]
        
        if successful_runs:
            lengths = [r["length"] for r in successful_runs]
            latencies = [r["latency_ms"] for r in successful_runs]
            
            temp_results["stats"] = {
                "avg_length": float(np.mean(lengths)),
                "min_length": int(np.min(lengths)),
                "max_length": int(np.max(lengths)),
                "std_length": float(np.std(lengths)),
                "avg_latency_ms": float(np.mean(latencies)),
                "success_rate": len(successful_runs) / len(temp_results["runs"]),
            }
        else:
            temp_results["stats"] = {
                "avg_length": 0,
                "min_length": 0,
                "max_length": 0,
                "std_length": 0,
                "avg_latency_ms": 0,
                "success_rate": 0,
            }
        
        experiment_data["results"].append(temp_results)
    
    progress_text.empty()
    progress_bar.empty()
    
    # Compute overall summary
    all_lengths = [
        r["length"]
        for temp_result in experiment_data["results"]
        for r in temp_result["runs"]
        if r["success"]
    ]
    
    if all_lengths:
        experiment_data["summary"] = {
            "total_runs": total_runs,
            "successful_runs": len(all_lengths),
            "overall_avg_length": float(np.mean(all_lengths)),
            "overall_std_length": float(np.std(all_lengths)),
            "min_length": int(np.min(all_lengths)),
            "max_length": int(np.max(all_lengths)),
        }
    
    logger.info(f"Experiment completed: {experiment_data['summary']}")
    
    return experiment_data


def format_experiment_results_df(experiment_data: Dict) -> pd.DataFrame:
    """Format experiment results into a DataFrame for display.
    
    Args:
        experiment_data: Experiment data dictionary
        
    Returns:
        Formatted pandas DataFrame
    """
    rows = []
    
    for temp_result in experiment_data["results"]:
        temperature = temp_result["temperature"]
        
        for run in temp_result["runs"]:
            rows.append({
                "Temperature": temperature,
                "Run": run.get("run_index", 0),
                "Length": run["length"],
                "Tokens": run.get("tokens", 0),
                "Latency (ms)": f"{run.get('latency_ms', 0):.0f}",
                "Status": "✓" if run["success"] else "✗",
            })
    
    return pd.DataFrame(rows)


def export_experiment_results(
    experiment_data: Dict,
    format_type: str = "json",
) -> Optional[bytes]:
    """Export experiment results in the specified format.
    
    Args:
        experiment_data: Experiment data dictionary
        format_type: Export format ('json', 'csv')
        
    Returns:
        Exported data as bytes
    """
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            return json.JSONEncoder.default(self, obj)

    try:
        if format_type == "json":
            return json.dumps(experiment_data, indent=2, cls=NumpyEncoder).encode()
        
        elif format_type == "csv":
            df = format_experiment_results_df(experiment_data)
            return df.to_csv(index=False).encode()
        
        else:
            logger.error(f"Unsupported export format: {format_type}")
            return None
    
    except Exception as e:
        logger.error(f"Export failed: {str(e)}")
        return None


# Main layout
tab1, tab2, tab3 = st.tabs(["🚀 Run Experiment", "📊 Results", "📈 Analysis"])


with tab1:
    st.subheader("Temperature Comparison Experiment")
    
    st.markdown(
        """
        Run controlled experiments to compare how different temperatures affect:
        - **Output length** and variability
        - **Generation latency**
        - **Response diversity and consistency**
        
        Each temperature setting will be run multiple times to capture variability.
        """
    )
    
    # Experiment configuration
    col1, col2 = st.columns([2, 1])
    
    with col1:
        prompt = st.text_area(
            "Prompt for experiment",
            value="Explain the impact of renewable energy on the U.S. trade war.",
            height=100,
            help="The same prompt will be used for all runs to ensure fair comparison",
        )
    
    with col2:
        st.markdown("**Parameters**")
        
        num_runs = st.number_input(
            "Runs per temperature",
            min_value=1,
            max_value=10,
            value=3,
            help="Number of times to run each temperature setting",
        )
    
    st.markdown("---")
    
    # Temperature configuration
    col1, col2, col3 = st.columns(3)
    
    with col1:
        temp_min = st.slider(
            "Min temperature",
            min_value=0.0,
            max_value=2.0,
            value=0.3,
            step=0.1,
        )
    
    with col2:
        temp_max = st.slider(
            "Max temperature",
            min_value=0.0,
            max_value=2.0,
            value=1.5,
            step=0.1,
        )
    
    with col3:
        temp_step = st.slider(
            "Temperature step",
            min_value=0.1,
            max_value=1.0,
            value=0.3,
            step=0.1,
        )
    
    # Generate temperature list
    temperatures = [
        round(t, 1)
        for t in np.arange(temp_min, temp_max + temp_step / 2, temp_step)
    ]
    
    st.info(f"**Temperatures to test:** {', '.join(str(t) for t in temperatures)}")
    
    st.markdown("---")
    
    # Other parameters
    col1, col2 = st.columns(2)
    
    with col1:
        top_p = st.slider(
            "Top-P",
            min_value=0.0,
            max_value=1.0,
            value=settings.top_p,
            step=0.05,
        )
    
    with col2:
        top_k = st.slider(
            "Top-K",
            min_value=1,
            max_value=100,
            value=settings.top_k,
            step=1,
        )
    
    st.markdown("---")
    
    # Run experiment button
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("🚀 Run Experiment", use_container_width=True, width='stretch', key="run_exp"):
            with st.spinner("Running experiment..."):
                try:
                    experiment_data = run_temperature_experiment(
                        prompt=prompt,
                        temperatures=temperatures,
                        num_runs=num_runs,
                        top_p=top_p,
                        top_k=top_k,
                    )
                    
                    st.session_state.experiment_results.append(experiment_data)
                    
                    # Save to disk
                    save_experiment_to_disk(experiment_data)
                    
                    st.success("✅ Experiment completed!")
                    
                except Exception as e:
                    error_msg = f"Experiment failed: {str(e)}"
                    st.error(error_msg)
                    logger.error(error_msg)
    
    with col2:
        if st.button("📂 Load Previous Experiments", use_container_width=True,  width='stretch', key="load_exp"):
            st.session_state.experiments = load_experiments_from_disk()
            st.info(f"Loaded {len(st.session_state.experiments)} experiment(s)")


with tab2:
    st.subheader("Experiment Results")
    
    if not st.session_state.experiment_results:
        if st.session_state.experiments:
            st.info("Select an experiment from the list below")
        else:
            st.info("No experiments yet. Run one from the 'Run Experiment' tab.")
    else:
        # Select experiment
        exp_options = [
            f"Exp {i+1}: {exp['timestamp'][:19]}"
            for i, exp in enumerate(st.session_state.experiment_results)
        ]
        
        selected_idx = st.selectbox(
            "Select experiment",
            options=range(len(st.session_state.experiment_results)),
            format_func=lambda i: exp_options[i],
        )
        
        if selected_idx is not None:
            selected_exp = st.session_state.experiment_results[selected_idx]
            
            # Display metadata
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Prompt Length", len(selected_exp["prompt"]))
            
            with col2:
                st.metric("Temperatures", len(selected_exp["temperatures"]))
            
            with col3:
                st.metric("Runs per Temp", selected_exp["num_runs"])
            
            with col4:
                st.metric("Total Runs", selected_exp["summary"].get("total_runs", 0))
            
            st.markdown("---")
            
            # Display summary stats
            if selected_exp["summary"]:
                summary = selected_exp["summary"]
                st.subheader("Summary Statistics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Avg Output Length",
                        f"{summary.get('overall_avg_length', 0):.0f}",
                        "chars",
                    )
                
                with col2:
                    st.metric(
                        "Std Dev",
                        f"{summary.get('overall_std_length', 0):.0f}",
                        "chars",
                    )
                
                with col3:
                    st.metric(
                        "Min Length",
                        f"{summary.get('min_length', 0):.0f}",
                        "chars",
                    )
                
                with col4:
                    st.metric(
                        "Max Length",
                        f"{summary.get('max_length', 0):.0f}",
                        "chars",
                    )
            
            st.markdown("---")
            
            # Display detailed results table
            st.subheader("Detailed Results")
            
            df = format_experiment_results_df(selected_exp)
            
            # Display with pagination and sorting
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.dataframe(df, use_container_width=True, width='stretch')
            
            with col2:
                if st.button("📋 Copy Results", key="copy_results"):
                    st.code(df.to_string())
            
            with col3:
                # Export options
                export_format = st.radio(
                    "Export as",
                    options=["JSON", "CSV"],
                    horizontal=True,
                )
            
            st.markdown("---")
            
            # Export button
            col1, col2, col3 = st.columns(3)
            
            with col1:
                export_data = export_experiment_results(
                    selected_exp,
                    format_type=export_format.lower(),
                )
                
                if export_data:
                    filename = (
                        f"experiment_{selected_exp['timestamp'][:10]}.{export_format.lower()}"
                    )
                    st.download_button(
                        label=f"📥 Download {export_format}",
                        data=export_data,
                        file_name=filename,
                        mime=f"application/{export_format.lower()}",
                    )
            
            with col2:
                if st.button("🗑️ Delete Experiment", key="delete_exp"):
                    st.session_state.experiment_results.pop(selected_idx)
                    st.success("Experiment deleted")
                    st.rerun()


with tab3:
    st.subheader("Analysis & Visualization")
    
    if not st.session_state.experiment_results:
        st.info("No experiments to analyze. Run one from the 'Run Experiment' tab.")
    else:
        # Select experiment
        exp_options = [
            f"Exp {i+1}: {exp['timestamp'][:19]}"
            for i, exp in enumerate(st.session_state.experiment_results)
        ]
        
        selected_idx = st.selectbox(
            "Select experiment for analysis",
            options=range(len(st.session_state.experiment_results)),
            format_func=lambda i: exp_options[i],
            key="analysis_select",
        )
        
        if selected_idx is not None:
            selected_exp = st.session_state.experiment_results[selected_idx]
            
            # Prepare data for charts
            analysis_data = []
            
            for temp_result in selected_exp["results"]:
                temperature = temp_result["temperature"]
                
                for run in temp_result["runs"]:
                    if run["success"]:
                        analysis_data.append({
                            "Temperature": temperature,
                            "Output Length": run["length"],
                            "Latency (ms)": run.get("latency_ms", 0),
                        })
            
            if analysis_data:
                df_analysis = pd.DataFrame(analysis_data)
                
                # 1. Output length distribution by temperature
                st.subheader("Output Length Distribution")
                
                chart_data = []
                for temp_result in selected_exp["results"]:
                    temp = temp_result["temperature"]
                    lengths = [
                        r["length"] for r in temp_result["runs"] if r["success"]
                    ]
                    for length in lengths:
                        chart_data.append({
                            "Temperature": f"{temp:.1f}",
                            "Length": length,
                        })
                
                if chart_data:
                    df_chart = pd.DataFrame(chart_data)
                    st.bar_chart(
                        data=df_chart.groupby("Temperature")["Length"].mean(),
                        use_container_width=True,
                        width='stretch'
                    )
                    
                    st.caption("Average output length by temperature")
                
                st.markdown("---")
                
                # 2. Latency analysis
                st.subheader("Generation Latency")
                
                latency_data = []
                for temp_result in selected_exp["results"]:
                    temp = temp_result["temperature"]
                    latencies = [
                        r.get("latency_ms", 0)
                        for r in temp_result["runs"]
                        if r["success"]
                    ]
                    for latency in latencies:
                        latency_data.append({
                            "Temperature": f"{temp:.1f}",
                            "Latency": latency,
                        })
                
                if latency_data:
                    df_latency = pd.DataFrame(latency_data)
                    st.line_chart(
                        data=df_latency.groupby("Temperature")["Latency"].mean(),
                        use_container_width=True,
                        width='stretch'
                    )
                    
                    st.caption("Average generation latency by temperature")
                
                st.markdown("---")
                
                # 3. Variability (standard deviation)
                st.subheader("Output Variability (Std Dev)")
                
                variability_data = {}
                for temp_result in selected_exp["results"]:
                    temp = temp_result["temperature"]
                    lengths = [
                        r["length"]
                        for r in temp_result["runs"]
                        if r["success"]
                    ]
                    if lengths:
                        variability_data[f"{temp:.1f}"] = np.std(lengths)
                
                if variability_data:
                    st.bar_chart(
                        pd.Series(variability_data),
                        use_container_width=True,
                        width='stretch'
                    )
                    
                    st.caption("Standard deviation of output lengths by temperature")
                
                st.markdown("---")
                
                # 4. Statistics table
                st.subheader("Temperature Statistics")
                
                stats_rows = []
                for temp_result in selected_exp["results"]:
                    stats = temp_result.get("stats", {})
                    stats_rows.append({
                        "Temperature": temp_result["temperature"],
                        "Avg Length": f"{stats.get('avg_length', 0):.0f}",
                        "Std Dev": f"{stats.get('std_length', 0):.0f}",
                        "Min": f"{stats.get('min_length', 0):.0f}",
                        "Max": f"{stats.get('max_length', 0):.0f}",
                        "Avg Latency": f"{stats.get('avg_latency_ms', 0):.0f}ms",
                        "Success Rate": f"{stats.get('success_rate', 0):.0%}",
                    })
                
                df_stats = pd.DataFrame(stats_rows)
                st.dataframe(df_stats, use_container_width=True,  width='stretch')
                
                st.markdown("---")
                
                # 5. Key insights
                st.subheader("Key Insights")
                
                # Find extremes
                all_lengths = [r["Output Length"] for r in analysis_data]
                avg_lengths_by_temp = {}
                
                for temp_result in selected_exp["results"]:
                    temp = temp_result["temperature"]
                    lengths = [
                        r["length"]
                        for r in temp_result["runs"]
                        if r["success"]
                    ]
                    if lengths:
                        avg_lengths_by_temp[temp] = np.mean(lengths)
                
                if avg_lengths_by_temp:
                    highest_temp = max(avg_lengths_by_temp, key=avg_lengths_by_temp.get)
                    lowest_temp = min(avg_lengths_by_temp, key=avg_lengths_by_temp.get)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric(
                            "Longest avg response",
                            f"Temp {highest_temp:.1f}",
                            f"{avg_lengths_by_temp[highest_temp]:.0f} chars",
                        )
                    
                    with col2:
                        st.metric(
                            "Shortest avg response",
                            f"Temp {lowest_temp:.1f}",
                            f"{avg_lengths_by_temp[lowest_temp]:.0f} chars",
                        )
            else:
                st.warning("No successful runs to analyze")


if __name__ == "__main__":
    pass
