import hashlib
import json
from pathlib import Path

import fitz  # PyMuPDF
from langchain_ollama import OllamaEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.documents import Document

from app.core.config import settings

# Tracks which PDFs have already been ingested so we never process the same
# file twice, even across restarts. Lives next to the ChromaDB folder.
_SEEN_HASHES_FILE = Path(settings.chroma_persist_dir).parent / "ingested_hashes.json"


def _load_seen_hashes() -> set[str]:
    if _SEEN_HASHES_FILE.exists():
        return set(json.loads(_SEEN_HASHES_FILE.read_text()))
    return set()


def _save_seen_hashes(hashes: set[str]) -> None:
    _SEEN_HASHES_FILE.parent.mkdir(parents=True, exist_ok=True)
    _SEEN_HASHES_FILE.write_text(json.dumps(list(hashes)))


def ingest_pdf(file_path: str) -> list[Document]:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    # Hash the raw file bytes — not the text — so two files with identical
    # content but different names are still treated as duplicates.
    file_hash = hashlib.md5(path.read_bytes()).hexdigest()
    seen = _load_seen_hashes()
    if file_hash in seen:
        # Return empty list so the caller knows nothing new was stored.
        return []

    # --- PDF Extraction ---
    # We open page by page (not all at once) so we can record the page number
    # alongside each block of text. Without this, citations become impossible
    # later — the LLM has no way to recover "this came from page 7".
    pdf = fitz.open(str(path))
    pages: list[tuple[int, str]] = []

    for page_num, page in enumerate(pdf, start=1):
        text = page.get_text()
        if text.strip():  # blank/scanned pages produce empty strings — skip them
            pages.append((page_num, text))

    pdf.close()

    if not pages:
        raise ValueError(f"No extractable text found in: {file_path}")

    # --- Semantic Chunking ---
    # SemanticChunker uses the embedding model to measure the meaning distance
    # between consecutive sentences. It only cuts when the topic actually shifts,
    # so a paragraph about room pricing stays in one chunk rather than being
    # split mid-sentence by a fixed character count.
    # NOTE: this requires a live Ollama server — it calls the embedding model
    # during chunking, not just at retrieval time.
    embeddings = OllamaEmbeddings(
        base_url=settings.ollama_base_url,
        model=settings.ollama_embedding_model,  # mxbai-embed-large, NOT the chat LLM
    )
    splitter = SemanticChunker(embeddings)

    # We chunk each page separately and pass its metadata in parallel so that
    # every resulting chunk automatically inherits the correct page number.
    # If we merged all pages into one string first, we would lose that mapping.
    texts = [text for _, text in pages]
    metadatas = [
        {"source": path.name, "page": page_num}
        for page_num, _ in pages
    ]

    raw_docs: list[Document] = splitter.create_documents(
        texts=texts,
        metadatas=metadatas,
    )

    # Stamp a global chunk_index so every chunk has a unique position identifier
    # across the entire document, useful for debugging and deduplication later.
    documents: list[Document] = []
    for i, doc in enumerate(raw_docs):
        doc.metadata["chunk_index"] = i
        documents.append(doc)

    # Only mark as seen after successful processing so a crash mid-way
    # doesn't silently block re-ingestion on the next run.
    seen.add(file_hash)
    _save_seen_hashes(seen)

    return documents
