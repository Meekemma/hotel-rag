  -----------------------------------------------------------------------
  **PRODUCT REQUIREMENTS DOCUMENT**

  -----------------------------------------------------------------------

**PDF RAG System**

*Production-Ready Retrieval-Augmented Generation Platform*

  -------------------------- --------------------------------------------
  **Document Version**       1.0.0

  **Status**                 Draft

  **Author**                 Meeky

  **Date**                   24 March 2026

  **Project Type**           Backend Engineering / AI

  **Stack**                  Python, LangChain, FastAPI, Ollama
  -------------------------- --------------------------------------------

***CONFIDENTIAL***

**1. Executive Summary**

The PDF RAG System is a production-grade Retrieval-Augmented Generation
(RAG) platform designed to enable intelligent question-answering over
PDF documents. Building on the foundations established in the Hotel CSV
RAG prototype, this system introduces advanced retrieval strategies,
evaluation pipelines, observability tooling, persistent memory, and a
fully documented REST API layer.

The platform is designed to serve as a reusable backend infrastructure
for any document-intelligence use case --- including legal document Q&A,
research paper assistants, company policy chatbots, and multi-document
reasoning systems. All components follow production engineering
standards: containerised deployment, structured logging, rate limiting,
session management, and automated evaluation.

**2. Project Goals**

**2.1 Primary Objectives**

-   Build a fully functional RAG pipeline that ingests, chunks, embeds,
    and retrieves content from one or more PDF documents.

-   Implement advanced retrieval techniques (hybrid search, reranking,
    multi-query) that demonstrably outperform naive vector search.

-   Establish an automated evaluation pipeline using RAGAS to measure
    retrieval and generation quality.

-   Expose all functionality via a documented FastAPI REST interface
    with session-aware conversation history.

-   Integrate LangSmith observability for full trace-level debugging and
    performance monitoring.

**2.2 Non-Goals**

-   Fine-tuning or training custom language models.

-   Building a frontend user interface (API-first; UI is out of scope
    for this PRD).

-   Multi-tenant cloud SaaS deployment (single-user local/server
    deployment only).

**3. Feature Specifications**

**3.1 PDF Ingestion and Chunking**

The system shall accept PDF files as its primary document input format.
Given that PDFs vary widely in structure --- including multi-column
layouts, embedded tables, scanned pages, and mixed media --- robust
extraction and chunking are foundational to retrieval quality.

**Functional Requirements**

-   Accept single or multiple PDF files as input via file path or file
    upload endpoint.

-   Extract raw text using PyMuPDF (fitz) or pdfplumber, with fallback
    handling for scanned PDFs via OCR (pytesseract).

-   Apply Semantic Chunking (LangChain SemanticChunker) to split
    documents into contextually coherent units rather than fixed
    character windows.

-   Preserve document-level metadata per chunk: source filename, page
    number, chunk index, and document title where available.

-   Detect and skip duplicate documents by hashing file content before
    ingestion.

**Technical Notes**

-   Chunk size target: 400--800 tokens with 10% overlap for continuity
    across chunk boundaries.

-   Semantic chunking uses embedding distance thresholds to detect topic
    shifts rather than fixed token counts.

-   Tables within PDFs shall be extracted as structured text blocks to
    preserve row/column relationships.

**3.2 Advanced Retrieval**

Retrieval quality is the single most impactful variable in RAG system
performance. This project implements three retrieval enhancements over
standard vector similarity search, each addressing a distinct failure
mode.

**3.2.1 Hybrid Search**

Pure vector search fails on exact keyword queries (names, codes,
identifiers). Hybrid search combines dense vector retrieval with BM25
sparse keyword matching to cover both semantic and lexical similarity.

-   Integrate rank_bm25 for sparse retrieval alongside ChromaDB vector
    retrieval.

-   Implement Reciprocal Rank Fusion (RRF) to merge results from both
    retrieval heads into a single ranked list.

-   Configurable weight parameter (alpha) to shift balance between
    semantic and keyword retrieval depending on query type.

**3.2.2 Reranking**

Initial retrieval returns candidate documents but does not optimise for
answer-relevance. A cross-encoder reranker scores each candidate against
the query directly, significantly improving precision.

-   Retrieve an expanded candidate set (k=20) during initial retrieval.

-   Pass candidates through FlashrankRerank or Cohere Rerank API to
    produce a re-scored list.

-   Return only the top N (default: 5) documents after reranking.

**3.2.3 Multi-Query Retrieval**

Single queries often miss relevant documents due to vocabulary mismatch.
Multi-query retrieval instructs the LLM to generate three semantically
distinct reformulations of the user query, retrieves separately for
each, and deduplicates results.

-   Use LangChain MultiQueryRetriever with the base LLM as the query
    reformulator.

-   Deduplicate across all result sets using document ID hashing before
    passing to the reranker.

-   Log all generated sub-queries to LangSmith for traceability.

**3.3 Evaluation Pipeline (RAGAS)**

Without quantitative evaluation, improvements to retrieval or generation
cannot be measured reliably. The RAGAS evaluation framework provides
automated, reference-free metrics across four dimensions.

**Metrics to Track**

  ------------------ -------------------------- --------------------------
  **Metric**         **Measures**               **Target**

  Context Precision  Are retrieved docs         \> 0.80
                     relevant to the query?     

  Context Recall     Are all relevant docs      \> 0.75
                     retrieved?                 

  Faithfulness       Is the answer grounded in  \> 0.85
                     retrieved context?         

  Answer Relevancy   Does the answer address    \> 0.80
                     the question?              
  ------------------ -------------------------- --------------------------

-   Build a test dataset of at least 30 question-answer pairs grounded
    in the target PDF corpus.

-   Run RAGAS evaluation automatically on every significant change to
    retrieval strategy or prompt template.

-   Store evaluation results as JSON artefacts keyed by git commit hash
    for regression tracking.

**3.4 Observability and Tracing (LangSmith)**

Every production RAG system requires full pipeline visibility. LangSmith
provides trace-level logging for every chain invocation, surfacing
latency, token usage, retrieved documents, and prompt contents.

-   Instrument all LangChain chain and retriever calls with the
    LangSmith tracer via LANGCHAIN_TRACING_V2=true.

-   Tag traces by session_id, query_type, and retrieval_strategy to
    enable filtering in the LangSmith dashboard.

-   Configure alert thresholds for latency (\> 8s) and hallucination
    score (\< 0.75 faithfulness).

-   Export trace summaries to structured logs (JSON) for offline
    analysis.

**3.5 Persistent Conversation Memory**

The prototype used an in-memory Python list for conversation history
which is lost on process exit. The production system replaces this with
persistent, session-scoped memory backed by Redis or PostgreSQL.

**Memory Strategies**

-   ConversationBufferWindowMemory --- retains the last N turns
    verbatim. Suitable for short, focused sessions.

-   ConversationSummaryMemory --- uses the LLM to compress older history
    into a rolling summary. Suitable for long sessions where context
    window overflow is a risk.

**Storage Requirements**

-   Each session identified by a unique session_id (UUID v4).

-   History persisted to Redis (preferred for speed) or PostgreSQL
    (preferred for durability and querying).

-   TTL of 24 hours on session data by default; configurable via
    environment variable.

-   DELETE /history/{session_id} endpoint for explicit session
    termination.

**3.6 Guardrails**

Guardrails protect the system from off-topic abuse, hallucinated
outputs, and unsafe inputs, ensuring users receive only grounded,
relevant answers.

**Input Guardrails**

-   Topic classifier checks incoming queries against the document domain
    before retrieval. Off-topic queries return a structured refusal
    without consuming LLM tokens.

-   Input length cap at 512 tokens to prevent prompt injection via
    oversized queries.

-   PII detection: flag and optionally redact personal identifiable
    information in queries before logging.

**Output Guardrails**

-   Post-generation faithfulness check: compare answer against retrieved
    context using embedding cosine similarity. Flag answers below
    threshold (\< 0.70) for review.

-   Source attribution: every answer must cite the document and page
    number from which its content was retrieved.

-   Confidence scoring: return a confidence field in the API response
    indicating retrieval and generation quality.

**3.7 FastAPI REST Interface**

The system is exposed via a RESTful API built with FastAPI. All
endpoints are fully documented via auto-generated OpenAPI/Swagger UI at
/docs.

**Core Endpoints**

  ------------ ----------------------- ----------------------------------------
  **Method**   **Endpoint**            **Description**

  POST         /ingest                 Upload and ingest one or more PDF files
                                       into the vector store.

  POST         /chat                   Submit a query; returns answer, sources,
                                       confidence, and session context.

  GET          /history/{session_id}   Retrieve full conversation history for a
                                       session.

  DELETE       /history/{session_id}   Clear and terminate a session.

  GET          /eval                   Trigger a RAGAS evaluation run against
                                       the test dataset.

  GET          /health                 System health check including vector
                                       store and Redis connectivity.
  ------------ ----------------------- ----------------------------------------

-   All endpoints require Content-Type: application/json except /ingest
    which uses multipart/form-data.

-   Authentication via API key header (X-API-Key) using FastAPI
    dependency injection.

-   Rate limiting: 60 requests per minute per API key, enforced via
    slowapi middleware.

**4. Technical Architecture**

**4.1 Technology Stack**

  --------------------- ------------------------ ------------------------
  **Layer**             **Technology**           **Purpose**

  LLM                   Ollama (llama3.2)        Local inference;
                                                 swappable with GPT-4 or
                                                 Claude via LangChain

  Embeddings            mxbai-embed-large        Dense vector
                                                 representations via
                                                 Ollama

  Vector Store          ChromaDB                 Persistent vector
                                                 database with metadata
                                                 filtering

  Sparse Search         rank_bm25                BM25 keyword retrieval
                                                 for hybrid search

  Reranker              FlashrankRerank          Cross-encoder reranking
                                                 of candidate documents

  Orchestration         LangChain                Chain composition,
                                                 retrievers, memory, and
                                                 tools

  API Framework         FastAPI                  REST API with async
                                                 support and OpenAPI docs

  Memory Store          Redis                    Persistent
                                                 session-scoped
                                                 conversation history

  Evaluation            RAGAS                    Automated RAG quality
                                                 metrics

  Observability         LangSmith                Full pipeline tracing
                                                 and performance
                                                 monitoring

  PDF Parsing           PyMuPDF / pdfplumber     Text and table
                                                 extraction from PDF
                                                 files
  --------------------- ------------------------ ------------------------

**4.2 Project Directory Structure**

The recommended directory structure follows domain-driven separation of
concerns:

> **pdf_rag/**
>
> ├── main.py \# Entry point and CLI
>
> ├── api/
>
> │ ├── routes.py \# FastAPI route definitions
>
> │ └── schemas.py \# Pydantic request/response models
>
> ├── rag/
>
> │ ├── ingestor.py \# PDF parsing and chunking
>
> │ ├── retriever.py \# Hybrid search + reranking
>
> │ ├── chain.py \# LangChain chain composition
>
> │ └── memory.py \# Redis-backed session memory
>
> ├── eval/
>
> │ ├── ragas_runner.py \# Evaluation pipeline
>
> │ └── test_dataset.json \# Ground-truth QA pairs
>
> ├── guardrails/
>
> │ ├── input_guard.py \# Topic classification + PII
>
> │ └── output_guard.py \# Faithfulness check
>
> ├── vector/ \# ChromaDB persistence
>
> ├── config.py \# Environment and settings
>
> ├── requirements.txt
>
> └── .env

**5. Build Roadmap**

The project is divided into four iterative phases. Each phase produces a
testable, runnable system before the next is begun.

**Phase 1 --- Foundation (Week 1--2)**

-   PDF ingestion pipeline with PyMuPDF and SemanticChunker

-   ChromaDB vector store with document-level metadata

-   Basic retrieval and LangChain chain (equivalent to the CSV prototype
    but over PDFs)

-   CLI interface for local testing

**Phase 2 --- Advanced Retrieval (Week 3--4)**

-   Hybrid search (ChromaDB + BM25 with RRF merging)

-   FlashrankRerank cross-encoder integration

-   MultiQueryRetriever for query reformulation

-   LangSmith tracing instrumented across all components

**Phase 3 --- API and Memory (Week 5--6)**

-   FastAPI application with all six core endpoints

-   Redis-backed persistent session memory

-   API key authentication and rate limiting

-   Input and output guardrails

**Phase 4 --- Evaluation and Hardening (Week 7--8)**

-   RAGAS evaluation pipeline with 30-question test dataset

-   Automated eval run on CI (GitHub Actions or pre-commit hook)

-   Performance benchmarking and latency profiling

-   README and API documentation

**6. Acceptance Criteria**

The project is considered complete when all of the following criteria
are met:

1.  The /ingest endpoint successfully processes a 50-page PDF and stores
    all chunks with metadata in ChromaDB.

2.  The /chat endpoint returns a cited answer with source document and
    page number for any question answerable from the ingested corpus.

3.  RAGAS evaluation scores meet all four target thresholds defined in
    Section 3.3.

4.  Conversation history persists across process restarts and is
    correctly scoped per session_id.

5.  LangSmith traces are visible and correctly tagged for a minimum of
    20 test queries.

6.  Off-topic queries are rejected by the input guardrail without
    reaching the LLM.

7.  Rate limiting correctly enforces 60 requests per minute per API key.

8.  All endpoints are documented in OpenAPI/Swagger and respond with
    correct status codes.

**7. Risks and Mitigations**

  ----------------------- ----------------------- --------------------------------
  **Risk**                **Likelihood**          **Mitigation**

  Poor PDF text           Medium                  Integrate pytesseract OCR
  extraction on scanned                           fallback for image-based PDFs
  docs                                            

  Reranker latency        Medium                  Cache reranked results per query
  increases response time                         hash; async pipeline

  RAGAS scores below      Low                     Freeze eval dataset; run evals
  target after retrieval                          before and after each change
  changes                                         

  Redis unavailable in    Low                     Fallback to in-memory
  local dev environment                           ConversationBufferWindowMemory

  LLM hallucinations on   Medium                  Output faithfulness guardrail;
  low-context queries                             require minimum context score
  ----------------------- ----------------------- --------------------------------

**8. Glossary**

**RAG:** Retrieval-Augmented Generation. An LLM architecture that
retrieves relevant documents before generating an answer, grounding the
response in external knowledge.

**Chunking:** The process of splitting a large document into smaller
segments (chunks) that can be individually embedded and retrieved.

**Hybrid Search:** A retrieval strategy combining dense vector
similarity search with sparse BM25 keyword search, merged via Reciprocal
Rank Fusion.

**Reranking:** A second-pass scoring step that uses a cross-encoder
model to re-score retrieved candidates by direct query relevance.

**RAGAS:** Retrieval-Augmented Generation Assessment. A framework
providing reference-free evaluation metrics for RAG systems.

**LangSmith:** An observability and debugging platform for LangChain
applications providing trace-level visibility into pipeline execution.

**Faithfulness:** A RAGAS metric measuring whether every claim in the
generated answer is supported by the retrieved context.

**Session:** A unique conversation thread identified by a session_id,
with its own isolated conversation history stored in Redis.

**Document Sign-Off**

This document has been reviewed and approved for use as the guiding
specification for the PDF RAG System project.

  ----------------------------------- -----------------------------------
  **Author**                          **Reviewer**

  Name:                               Name:

  Signature:                          Signature:

  Date:                               Date:
  ----------------------------------- -----------------------------------
