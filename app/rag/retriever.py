"""
Advanced retrieval pipeline — Phase 2.

Three progressive layers, each fixing a specific weakness of basic vector search:

  Layer 1 | get_hybrid_retriever()   — BM25 + ChromaDB via EnsembleRetriever
           |   Problem solved: pure vector search fails on exact keyword queries
           |   (room numbers, names, policy codes) because embeddings encode
           |   meaning, not spelling.

  Layer 2 | MultiQueryRetriever      — LLM generates 3 query rephrasings
           |   Problem solved: vocabulary mismatch. A guest says "when can I
           |   arrive?" but the document says "check-in begins at 3 PM."
           |   A single embedding may miss it; three rephrasings rarely all miss.

  Layer 3 | ContextualCompressionRetriever + FlashrankRerank
           |   Problem solved: coarse precision. BM25 and vector search score
           |   each document independently of the query context. A cross-encoder
           |   (what FlashrankRerank uses) reads the query and document TOGETHER
           |   and produces a much more accurate relevance score.

chain.py should only call get_advanced_retriever() — it replaces the old
get_retriever(get_vector_store()) call.
"""

import logging

# In LangChain 1.x these moved from langchain.retrievers to langchain_classic.
from langchain_classic.retrievers import (
    ContextualCompressionRetriever,
    EnsembleRetriever,
    MultiQueryRetriever,
)
from langchain_community.document_compressors import FlashrankRerank
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_ollama import ChatOllama

from app.core.config import settings
from app.rag.vector_store import get_vector_store

logger = logging.getLogger(__name__)


class EmptyKnowledgeBaseError(RuntimeError):
    """Raised when ChromaDB has no indexed documents to retrieve from.

    Distinct from a generic RuntimeError so callers (e.g. the API layer) can
    catch this specifically and return a 503 with an actionable message,
    instead of lumping "nothing ingested yet" in with unrelated 500s.
    """


# ---------------------------------------------------------------------------
# PRIVATE HELPER
# ---------------------------------------------------------------------------

def _load_all_documents() -> list[Document]:
    """Return every document stored in ChromaDB as a list of LangChain Documents.

    BM25 is not a vector database — it builds an inverted index from raw text
    and needs the full corpus upfront. For this hotel system (hundreds to low
    thousands of chunks) that is fine. A very large collection would need
    batching or a pre-built BM25 index persisted to disk.
    """
    store = get_vector_store()

    # .get() without a query returns ALL records in the collection.
    # include= controls which fields are returned — we skip embeddings
    # because they are large and BM25 has no use for them.
    raw = store.get(include=["documents", "metadatas"])

    if not raw.get("documents"):
        logger.warning(
            "ChromaDB collection is empty — BM25 will have nothing to index. "
            "Run the ingestion pipeline first."
        )
        return []

    return [
        Document(page_content=text, metadata=meta or {})
        for text, meta in zip(raw["documents"], raw["metadatas"])
    ]


# ---------------------------------------------------------------------------
# LAYER 1 — HYBRID SEARCH
# ---------------------------------------------------------------------------

def get_hybrid_retriever() -> EnsembleRetriever:
    """BM25 keyword search + ChromaDB vector search merged by Reciprocal Rank Fusion.

    Why combine them?
    - Vector search captures meaning: "gym" and "fitness centre" are close.
    - BM25 captures exact terms: "room 204" or "promo code HOTEL20".
    - Neither alone is sufficient; together they cover both retrieval modes.
    """
    store = get_vector_store()

    # --- BM25 side ---
    # BM25 scores a document by asking two questions per query term:
    #   1. How often does this term appear in the document? (term frequency)
    #   2. How rare is this term across the whole collection? (inverse doc freq)
    # A rare term that appears several times in one chunk = very high score.
    # Common words like "the" or "is" have near-zero IDF and are ignored.
    all_docs = _load_all_documents()
    if not all_docs:
        raise EmptyKnowledgeBaseError(
            "ChromaDB contains no indexed documents. Run the ingestion pipeline before querying."
        )
    bm25_retriever = BM25Retriever.from_documents(all_docs)

    # Cast a wide net here — retrieval_candidates (20) gives the reranker
    # (Layer 3) a rich pool to work with. The reranker trims to retriever_k (5).
    bm25_retriever.k = settings.retrieval_candidates

    # --- Vector side ---
    # Same ChromaDB cosine-similarity search as Phase 1, but fetching more
    # candidates because the reranker will narrow them down later.
    vector_retriever = store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": settings.retrieval_candidates},
    )

    # --- Merge via Reciprocal Rank Fusion ---
    # Each retriever returns its own ranked list. RRF converts each rank to a
    # score using: 1 / (rank + 60). The constant 60 prevents the rank-1 doc
    # from dominating when the other list disagrees. Scores are summed across
    # both lists — documents that rank highly in BOTH float to the top.
    #
    # weights=[0.5, 0.5] = equal trust in BM25 and vector search.
    # Tune toward [0.6, 0.4] (more BM25) if guests use exact policy language,
    # or [0.4, 0.6] (more vector) if queries tend to be conversational.
    return EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5],
    )


# ---------------------------------------------------------------------------
# LAYER 2 + 3 — FULL ADVANCED PIPELINE
# ---------------------------------------------------------------------------

def get_advanced_retriever() -> BaseRetriever:
    """Full Phase 2 retrieval pipeline. This is the only public function chain.py needs.

    Execution order when a query arrives:
      1. MultiQueryRetriever asks the LLM to rephrase the query 3 ways.
      2. For each rephrasing, get_hybrid_retriever() fetches up to 20 candidates.
      3. Results from all 3 rephrasings are deduplicated (by document content).
      4. FlashrankRerank re-scores the deduplicated pool using a cross-encoder
         and returns the top reranker_top_n (5) documents.
    """

    # --- MultiQueryRetriever (Layer 2) ---
    # Uses the LLM to rephrase the user's question before retrieval, not after.
    # This is different from the answer-generation LLM — here the model's job
    # is purely to produce alternative phrasings of the same intent.
    #
    # temperature=0 is important: we want predictable, literal rephrasings,
    # not creative variations that drift from the user's actual question.
    llm = ChatOllama(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        temperature=0,
    )

    multi_query = MultiQueryRetriever.from_llm(
        # Pass hybrid retrieval as the base — each rephrasing triggers a
        # full BM25 + vector search independently.
        retriever=get_hybrid_retriever(),
        llm=llm,
    )

    # --- FlashrankRerank via ContextualCompressionRetriever (Layer 3) ---
    # Cross-encoders (what FlashrankRerank uses) are too slow to search the
    # entire corpus, but fast enough to re-score a short candidate list.
    # The two-stage pattern — fast broad retrieval, slow precise reranking —
    # is the production standard for high-quality RAG systems.
    #
    # top_n=reranker_top_n (5): keep the 5 most relevant chunks after reranking.
    compressor = FlashrankRerank(top_n=settings.reranker_top_n)

    # ContextualCompressionRetriever wires the retriever and compressor together:
    #   Step 1 — base_retriever (multi_query) fetches the candidate pool.
    #   Step 2 — base_compressor (flashrank) reranks and drops all but top_n.
    # The caller receives only the final top_n documents — the intermediate
    # candidates are never exposed outside this function.
    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=multi_query,
    )
