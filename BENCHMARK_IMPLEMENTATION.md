# Benchmark Page Implementation

## Overview

The benchmark page (pages/4_Benchmark.py) is a comprehensive evaluation system for testing the AI News Intelligence Assistant on a curated test dataset using LLM-as-Judge evaluation.

## Key Features

### 1. Dataset Management (Tab 1)
- **Load Existing Datasets**: Browse and select from previously saved benchmark datasets
- **Create Sample Dataset**: Generate a pre-configured sample benchmark with 9 test cases across 3 categories
- **Upload Custom Datasets**: Upload custom JSON benchmark files
- **Dataset Preview**: View dataset statistics and category distribution
- **Dataset Validation**: Automatic validation of test case structure

Sample dataset includes:
- **green_energy** (3 tests): Climate policy, renewable adoption, EV impact
- **trade_war** (3 tests): Trade impacts, affected countries, consumer effects
- **censorship** (3 tests): Content moderation, misinformation, political discourse

### 2. Run Benchmark (Tab 2)
- **Configuration Options**:
  - Toggle RAG (Retrieval-Augmented Generation) on/off
  - Adjust generation temperature (0.0-2.0)
- **Test Execution**:
  - Run all test cases sequentially
  - For each test case:
    1. Execute assistant with optional RAG retrieval
    2. Collect response and latency metrics
    3. Generate judge evaluation using LLM-as-Judge
    4. Score response on 0-10 scale with reasoning
- **Progress Tracking**: Real-time feedback on execution status

### 3. Results Analysis (Tab 3)
- **Summary Metrics**:
  - Total tests, completed, failed
  - Overall average score (0-10)
  
- **Visualizations**:
  - Score distribution histogram
  - Category-wise average scores bar chart
  - Summary statistics table
  - Category breakdown with min/max/avg scores
  
- **Detailed Results**:
  - Results table with all key metrics
  - Individual test case details viewer
  - Response and judge reasoning display

### 4. Export & Reporting (Tab 4)
- **JSON Export**: Full results with all metadata
- **CSV Export**: Tabular format for spreadsheet analysis
- **Markdown Report**: Auto-generated report with summary and category breakdown
- **Local Persistence**: Save results to disk for later review

## Core Evaluator Module

### `core/evaluator.py`

#### `run_benchmark_case(test_case, client, use_rag, temperature)`
Executes a single test case:
1. Embeds the question
2. Optionally retrieves context via RAG
3. Generates response with configured temperature
4. Returns response with latency and retrieval metrics

#### `evaluate_response(response, test_case, client)`
Uses LLM-as-Judge to score the response:
1. Crafts evaluation prompt with test case, expected answer, and criteria
2. Calls LLM with low temperature (0.3) for consistency
3. Extracts JSON-formatted score (0-10) and reasoning
4. Returns structured evaluation result

#### `run_full_benchmark(test_cases, client, use_rag, temperature)`
Orchestrates full benchmark run:
1. Iterates through all test cases
2. Runs each case and evaluates response
3. Collects all results with timestamps
4. Returns comprehensive results dictionary

#### `aggregate_results(results)`
Aggregates statistics by category:
1. Calculates overall: average, min, max, median scores
2. Groups by category with category-specific statistics
3. Returns structured aggregation for reporting

## Dataset Format

```json
{
  "test_cases": [
    {
      "id": "unique_identifier",
      "category": "category_name",
      "prompt": "Question to ask the assistant",
      "expected_answer": "Reference/expected response",
      "evaluation_criteria": "Evaluation guidelines for judge",
      "notes": "Optional notes"
    }
  ]
}
```

## Results Format

```json
{
  "total_tests": 9,
  "completed": 9,
  "failed": 0,
  "timestamp": "2026-05-30T20:30:00.000Z",
  "results": [
    {
      "test_id": "test_id",
      "category": "category",
      "prompt": "question",
      "response": "assistant response",
      "score": 8.5,
      "reasoning": "judge explanation",
      "status": "completed",
      "latency_ms": 2340,
      "retrieved_chunks": 5
    }
  ]
}
```

## Usage Workflow

### Quick Start
1. Go to "Dataset Management" tab
2. Click "Create & Load Sample Dataset"
3. Go to "Run Benchmark" tab
4. Configure options (RAG, temperature)
5. Click "🚀 Start Benchmark Run"
6. Monitor progress
7. View results in "Results" tab
8. Export in "Export" tab

### Custom Evaluation
1. Prepare benchmark JSON file
2. Go to "Dataset Management" → Upload
3. Verify dataset in preview
4. Run benchmark with custom configuration
5. Export results

### Offline Analysis
1. Run benchmark and save results locally
2. Export as JSON or CSV
3. Generate Markdown report
4. Download for inclusion in course report

## Metrics & Scoring

### Judge Scoring Criteria
The LLM-as-Judge evaluates based on:
- Accuracy to corpus facts
- Specificity and detail
- Relevance to evaluation criteria
- Quality of reasoning

Scores: 0-10 scale where:
- 0-2: Completely incorrect or irrelevant
- 3-4: Partially correct with significant gaps
- 5-6: Mostly correct with some issues
- 7-8: Good response with minor improvements
- 9-10: Excellent, comprehensive, well-grounded

### Aggregation
- **Overall Average**: Mean score across all tests
- **Category Average**: Mean score per category
- **Min/Max/Median**: Spread and distribution analysis

## Integration Points

### With LLM Client
- Uses `LMStudioClient` for both generation and evaluation
- Configurable base URL, model, timeout
- Built-in retry logic

### With RAG Module
- Optional context retrieval per test case
- Top-k=5 by default
- Preserves retrieval metadata

### With Streamlit UI
- Session state management for datasets and results
- Tab-based interface for different workflows
- Real-time progress feedback

## Error Handling

- Missing knowledge base: Graceful degradation, continues without RAG
- Model failures: Caught and logged, test marked as failed
- Invalid datasets: Validation before loading
- Evaluation errors: Fallback scoring, logged for debugging

## Performance Considerations

- Each test case requires two LLM calls (generation + judge)
- Total time = (N tests) × (generation_time + judge_time)
- For 9 tests with 2-3s per call: ~45-60 seconds typical
- Results cached in session state for fast navigation

## Example Test Categories

### Green Energy
Questions about renewable energy, climate policy, EV technology, sustainable practices

### Trade War
Questions about tariffs, supply chains, geopolitical impacts, economic effects

### Censorship
Questions about content moderation, platform policies, free speech, misinformation

## Future Enhancements

- Batch processing with progress bar
- Multi-judge evaluation (ensemble scoring)
- A/B testing between configurations
- Historical trend analysis
- Custom evaluation rubrics
- Fine-grained per-category metrics
