"""
Smoke test for the Phase 2 retrieval pipeline.

Run from the project root using the project venv:
    env/Scripts/python scripts/smoke_test.py          (Windows)
    source env/bin/activate && python scripts/smoke_test.py  (Mac/Linux)

What this tests:
  1. ChromaDB is reachable and contains documents.
  2. The hybrid retriever (BM25 + vector) returns results for a sample query.
  3. The full advanced retriever (multi-query + rerank) returns results.
  4. The end-to-end RAG chain produces a grounded answer.

If LangSmith tracing is enabled (LANGSMITH_TRACING=true and LANGSMITH_API_KEY set),
every retrieval and LLM call in this script will appear as a trace run in your
LangSmith dashboard at https://smith.langchain.com.
"""

# Add the project root to sys.path so `app.*` imports resolve correctly
# regardless of where Python is invoked from.
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# On Windows, Python's SSL stack often can't find the system CA bundle,
# causing certificate errors when downloading models from HuggingFace or
# sending traces to LangSmith. Pointing to certifi's bundle fixes this.
import certifi
os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())

# load_dotenv before any LangChain imports — same reason as main.py.
from dotenv import load_dotenv
load_dotenv()

import sys
import textwrap

# ---- helpers ----------------------------------------------------------------

def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)

def ok(msg: str) -> None:
    print(f"  [PASS] {msg}")

def fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")
    sys.exit(1)

# ---- test 1: ChromaDB contains documents ------------------------------------

section("1 / 4 — ChromaDB health check")

from app.rag.vector_store import get_vector_store

try:
    store = get_vector_store()
    raw = store.get(include=["documents"])
    doc_count = len(raw.get("documents") or [])
except Exception as e:
    fail(f"Could not connect to ChromaDB: {e}")

if doc_count == 0:
    fail(
        "ChromaDB is empty. Run the ingestion pipeline first:\n"
        "  python -c \"from app.rag.ingestor import ingest; ingest()\""
    )

ok(f"ChromaDB contains {doc_count} chunks.")

# ---- test 2: hybrid retriever -----------------------------------------------

section("2 / 4 — Hybrid retriever (BM25 + vector)")

from app.rag.retriever import get_hybrid_retriever

TEST_QUERY = "What time is check-in?"

try:
    hybrid = get_hybrid_retriever()
    hybrid_results = hybrid.invoke(TEST_QUERY)
except Exception as e:
    fail(f"Hybrid retriever raised an exception: {e}")

if not hybrid_results:
    fail("Hybrid retriever returned no documents.")

ok(f"Returned {len(hybrid_results)} documents for: '{TEST_QUERY}'")
print(f"\n  Top result preview:")
preview = textwrap.shorten(hybrid_results[0].page_content, width=120, placeholder="...")
print(f"  {preview}")

# ---- test 3: advanced retriever (multi-query + rerank) ----------------------

section("3 / 4 — Advanced retriever (multi-query + rerank)")

from app.rag.retriever import get_advanced_retriever

# Use an exact keyword query to stress-test BM25 and the reranker.
KEYWORD_QUERY = "breakfast menu"

try:
    advanced = get_advanced_retriever()
    advanced_results = advanced.invoke(KEYWORD_QUERY)
except Exception as e:
    fail(f"Advanced retriever raised an exception: {e}")

if not advanced_results:
    fail("Advanced retriever returned no documents.")

ok(f"Returned {len(advanced_results)} documents for: '{KEYWORD_QUERY}'")
print(f"\n  Top result preview:")
preview = textwrap.shorten(advanced_results[0].page_content, width=120, placeholder="...")
print(f"  {preview}")

# ---- test 4: end-to-end chain -----------------------------------------------

section("4 / 4 — End-to-end RAG chain")

from app.rag.chain import get_rag_chain

E2E_QUESTION = "What amenities does the hotel offer?"

try:
    chain = get_rag_chain()
    answer = chain.invoke({"question": E2E_QUESTION})
except Exception as e:
    fail(f"RAG chain raised an exception: {e}")

if not answer or not answer.strip():
    fail("Chain returned an empty answer.")

ok(f"Chain produced an answer for: '{E2E_QUESTION}'")
print(f"\n  Answer:\n")
for line in textwrap.wrap(answer, width=70):
    print(f"    {line}")

# ---- summary ----------------------------------------------------------------

section("All tests passed")
print()
print("  If LANGSMITH_API_KEY is set, check your traces at:")
print("  https://smith.langchain.com")
print()
