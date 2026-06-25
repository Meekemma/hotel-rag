from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from app.core.config import settings

# One fixed name for our ChromaDB collection.
# Think of it like a table name in a database — it namespaces our hotel
# documents so they don't collide with any future collections in the same DB.
_COLLECTION_NAME = "hotel_docs"


def _build_embeddings() -> OllamaEmbeddings:
    # Isolated in its own function so there is exactly ONE place in the codebase
    # that decides which embedding model is used. If you ever switch models,
    # you change it here and nowhere else.
    # WARNING: this must always match what was used during ingestion.
    # Storing with model A and searching with model B produces garbage results
    # because the vectors live in completely different mathematical spaces.
    return OllamaEmbeddings(
        base_url=settings.ollama_base_url,
        model=settings.ollama_embedding_model,
    )


def get_vector_store() -> Chroma:
    # Chroma will create the persist_directory folder on first run and load
    # existing data on every subsequent run — so this is safe to call at
    # startup without any "does it exist yet?" checks.
    # We wrap this in a function (not a module-level variable) so that Ollama
    # is only contacted when this is explicitly called, not on every import.
    return Chroma(
        collection_name=_COLLECTION_NAME,
        embedding_function=_build_embeddings(),
        persist_directory=settings.chroma_persist_dir,
    )


def add_documents(store: Chroma, docs: list[Document]) -> None:
    # Guard against an empty ingest (e.g. a duplicate file that returned []).
    # Calling store.add_documents([]) is harmless but wastes a round-trip.
    if not docs:
        return

    # LangChain's Chroma wrapper handles batching, embedding each chunk,
    # and writing both the vector and the metadata to disk in one call.
    store.add_documents(docs)


def get_retriever(store: Chroma) -> VectorStoreRetriever:
    # as_retriever() wraps the store in a LangChain Retriever interface so
    # the chain can call it without knowing it is backed by ChromaDB.
    # search_type="similarity" means cosine distance between query vector
    # and stored vectors — the default and correct choice for Phase 1.
    # In Phase 2 we will replace this with a hybrid retriever (vector + BM25).
    return store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": settings.retriever_k},
    )
