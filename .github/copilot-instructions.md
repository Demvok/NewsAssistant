# Project Instructions for GitHub Copilot

## 1. Project goal

Build a single Python application that serves as a full-stack Generative AI course project and combines multiple laboratory works into one coherent system.

The project concept is an AI News Intelligence Assistant built on a corpus of CNN news articles. The system must support:
- multi-turn chat over the corpus,
- retrieval-augmented generation (RAG),
- a knowledge base with document ingestion and search,
- parameter experiments for generation quality analysis,
- benchmark evaluation with LLM-as-Judge,
- tool/function calling,
- optional multi-agent reasoning pipeline.

The application must be implemented as a Streamlit app with multiple pages and a clean modular backend.

## 2. Core idea and domain

The assistant works on a news corpus focused on:
- green energy,
- the U.S. trade war,
- censorship in social networks.

The system should behave like a structured analytical assistant for media intelligence, not like a generic chat bot. Its responses should be grounded in the document corpus whenever RAG is enabled.

## 3. Hard technical constraints

Use only the following stack unless explicitly necessary:
- Python 3.13.7
- virtual environment (`venv`)
- Streamlit for UI
- LM Studio as the local model runtime
- one chat/generation model: `gemma-4-e4b`
- one embedding model: `embeddinggemma-300M-GGUF`
- local vector storage: ChromaDB
- LangChain or LangGraph may be used, but only if they improve structure and maintainability

Do not introduce unnecessary cloud APIs, external paid services, or additional model providers.

The project should be designed so that model and embedding endpoints can be swapped later without changing core app logic.

## 4. High-level architecture

The application must be modular and layered.

Recommended layers:
- UI layer: Streamlit pages and widgets
- Application core: orchestration, retrieval, agents, tools, evaluation
- Infrastructure: local model access, embeddings, vector DB, storage, config
- Data layer: documents, indexes, test datasets, logs, evaluation results

The app must remain understandable and extensible. Each layer should depend on the layer below it, not the other way around.

## 5. Main application pages

Create four Streamlit pages:

### 5.1 Chat
This is the main user interface. It must:
- support multi-turn conversation,
- allow switching RAG on/off,
- allow switching multi-agent mode on/off,
- allow configuring generation parameters such as temperature, top_p, and top_k if supported,
- show chat history,
- display retrieval context when RAG is used,
- display agent traces or tool traces when those modes are enabled,
- clearly separate final answer from internal reasoning artifacts.

The chat page should make the assistant feel like a real analytical system, not just a text box.

### 5.2 Knowledge Base
This page manages documents and search.

It must support:
- uploading documents, preferably PDF and TXT at minimum,
- ingestion into chunks,
- embedding generation,
- persistence in ChromaDB,
- search over the corpus,
- display of retrieved chunks and similarity scores,
- basic corpus statistics,
- rebuild or refresh index actions.

If possible, include visual analytics such as:
- chunk length distribution,
- similarity distribution,
- document or topic statistics,
- a simple retrieval preview.

### 5.3 Experiments
This page is for generation parameter analysis.

It must support:
- running the same prompt under multiple temperatures or other decoding settings,
- repeating each setting multiple times,
- collecting output length, variability, and qualitative differences,
- comparing outputs in a table,
- showing charts for output length or diversity,
- preserving experiment metadata for later reporting.

This page corresponds to the laboratory work on generation parameters and should be presented as a controlled experiment environment.

### 5.4 Benchmark
This page evaluates assistant behavior on a test dataset.

It must support:
- a JSON or structured test set,
- question, expected answer, category, and evaluation criteria fields,
- running the assistant on all test cases,
- using an evaluator model as judge,
- collecting scores and reasoning,
- summarizing results by category,
- exporting results to a table or file.

The benchmark must be reproducible and suitable for report generation.

## 6. Core backend modules

The project should be organized into clear modules. Suggested structure:

```text
project/
├── app.py
├── config.py
├── requirements.txt
├── .env.example
├── README.md
├── data/
│   ├── raw/
│   ├── processed/
│   ├── chroma_db/
│   └── benchmark/
├── logs/
├── core/
│   ├── llm_client.py
│   ├── embeddings.py
│   ├── rag.py
│   ├── agents.py
│   ├── tools.py
│   ├── evaluator.py
│   ├── schemas.py
│   └── utils.py
├── ingestion/
│   ├── loader.py
│   ├── chunker.py
│   └── indexer.py
└── pages/
    ├── 1_Chat.py
    ├── 2_Knowledge_Base.py
    ├── 3_Experiments.py
    └── 4_Benchmark.py
````

If LangGraph is used, keep graph definitions in a dedicated module and do not mix them into UI code.

## 7. Local model integration

LM Studio should be treated as the only model runtime.

Assume:

* the chat model and reasoning model are the same local model,
* the embedding model is separate and local,
* the app talks to LM Studio through an OpenAI-compatible local API layer if available.

Implementation should:

* centralize model access in one client,
* avoid scattering API calls across the app,
* support configurable base URL, model name, timeout, and retry logic,
* gracefully handle model failures,
* log latency and response length where useful.

The code must not hardcode environment-specific secrets.

## 8. RAG requirements

The RAG pipeline must follow a clean sequence:

1. Load documents.
2. Clean and normalize text.
3. Split documents into chunks with overlap.
4. Generate embeddings.
5. Persist embeddings in ChromaDB.
6. Retrieve relevant chunks for a user query.
7. Inject retrieved context into the prompt.
8. Generate the final answer with citations or source references if possible.

RAG should be designed for news analytics:

* retrieve semantically relevant passages,
* preserve source metadata,
* keep chunk provenance,
* return top-k results,
* support index rebuilds.

The retrieval layer must be easy to inspect from the UI.

## 9. Multi-agent requirements

The project should support a multi-agent reasoning mode.

The multi-agent pipeline should be simple, explicit, and traceable. A recommended structure is:

* first agent: proposes an answer or analysis,
* second agent: critiques, fact-checks, or identifies weaknesses,
* third agent: synthesizes a final balanced answer.

Use this only when it helps the use case. Do not overcomplicate the implementation.

If LangGraph is used, it should manage agent transitions and state. If LangChain is used, keep the agent logic explicit and easy to debug.

The final system should clearly show agent handoffs or traces in the UI.

## 10. Tool calling requirements

Support function calling or tool use as a separate capability.

Tools should be implemented as a registry of Python functions with clear input/output contracts. Examples:

* filter documents by topic,
* filter by date range,
* count keyword mentions,
* search a document subset,
* return corpus statistics.

Tool usage should be optional and observable. The assistant should be able to decide when a tool is needed, but the implementation must remain deterministic enough for debugging.

Do not add fake tools. Every tool should have a concrete purpose related to the corpus.

## 11. Benchmarking and evaluation requirements

The benchmark module must evaluate the assistant on test cases with known expectations.

Each test case should contain:

* id,
* category,
* prompt,
* expected answer or expected facts,
* evaluation criteria,
* optional notes.

The evaluator should:

* run the assistant on each case,
* optionally ask a judge model to score the output,
* store scores and explanations,
* compute averages by category,
* present a summary of strengths and weaknesses.

Evaluation should be useful for the course report, not just technically correct.

## 12. Experiment module requirements

The experiments page should support controlled decoding experiments.

At minimum compare:

* different temperatures,
* repeated runs for the same prompt,
* response length variation,
* qualitative drift or consistency.

Store results in a structured format so they can be reused in reports and charts.

## 13. Engineering quality requirements

The code must be production-oriented enough for a student project:

* use type hints,
* keep functions small and readable,
* use dataclasses or schemas where helpful,
* separate configuration from logic,
* avoid duplication,
* make names descriptive,
* prefer explicitness over cleverness,
* include docstrings where useful,
* use logging instead of print statements for important runtime events.

The code should be easy to run, debug, and extend.

## 14. Error handling requirements

The app must handle common failures gracefully:

* missing model endpoint,
* empty document store,
* invalid file uploads,
* embedding failure,
* retrieval returning no results,
* malformed benchmark JSON,
* unsupported file types.

Errors should be shown in the UI clearly and safely without crashing the app.

## 15. Data handling requirements

The project must keep corpus data and generated artifacts organized.

Required folders or their equivalents:

* raw documents,
* processed documents,
* vector database,
* benchmark files,
* experiment outputs,
* logs.

Document metadata should be preserved where possible:

* source file name,
* article title,
* publication date if available,
* topic,
* chunk id,
* chunk order.

## 16. Streamlit requirements

The Streamlit app should:

* use sidebar controls for global settings,
* keep state across interactions,
* store current mode and toggles in session state,
* present outputs in a clean readable format,
* separate inputs, intermediate outputs, and final answers,
* avoid clutter.

The UI should feel like a real analytical dashboard and assistant combined.

## 17. Recommended implementation phases

Implement the project in stages.

### Phase 1: Project scaffold

Create:

* repository structure,
* virtual environment instructions,
* configuration module,
* Streamlit entry point,
* basic page navigation,
* local LM Studio client wrapper.

### Phase 2: Document ingestion and storage

Implement:

* file loading,
* text extraction,
* chunking,
* metadata tracking,
* ChromaDB persistence,
* index rebuilding.

### Phase 3: Retrieval and chat

Implement:

* query embedding,
* top-k retrieval,
* prompt assembly,
* chat history,
* basic answer generation.

### Phase 4: Experiments

Implement:

* prompt runner,
* temperature comparison,
* repeated sampling,
* results collection,
* plots and tables.

### Phase 5: Benchmark

Implement:

* benchmark dataset schema,
* batch runner,
* judge-based scoring,
* result aggregation,
* report-friendly output.

### Phase 6: Multi-agent and tool calling

Implement:

* agent roles,
* agent orchestration,
* tool registry,
* trace display,
* optional LangGraph workflow if needed.

### Phase 7: Polish

Implement:

* better UI,
* error handling,
* logging,
* result export,
* final cleanup for report and demonstration.

## 18. Design philosophy for the assistant

The assistant should behave like a specialized news analysis system.

It should:

* answer using the corpus when appropriate,
* distinguish between grounded facts and generated synthesis,
* preserve uncertainty when the corpus does not fully support a claim,
* remain focused on the three thematic areas,
* support comparative and analytical questions,
* avoid unnecessary general-purpose behavior.

This is not meant to be a generic assistant. It is a domain-focused analytical platform.

## 19. Preferred coding style for Copilot

When generating code, prefer:

* modular files over monolithic scripts,
* clear interfaces between layers,
* clean functions with one responsibility,
* explicit session-state management in Streamlit,
* reusable helper functions,
* minimal but meaningful abstraction.

Avoid:

* overengineering,
* deeply nested logic,
* hidden side effects,
* duplicated model calls,
* mixing UI and backend logic in one file,
* introducing frameworks that are not needed.

## 20. Practical constraints for the generated code

Generated code should assume:

* local development on Windows or Linux,
* environment variables loaded from `.env`,
* `venv` activation before running,
* Streamlit as the main run command,
* ChromaDB persistence on disk,
* LM Studio accessible locally.

The solution should be straightforward to run, understand, and explain in a course report.

## 21. What the final system should demonstrate

The finished project should demonstrate:

* local generative AI usage,
* retrieval-augmented generation,
* knowledge base construction,
* document search,
* parameter experimentation,
* multi-agent reasoning,
* tool calling,
* automatic evaluation,
* a realistic course-project architecture.

It should be strong enough to expand into a final course project with a clear story:
"An AI News Intelligence Assistant for thematic analysis of CNN articles using local LLMs, embeddings, RAG, multi-agent logic, and benchmark-driven evaluation."
