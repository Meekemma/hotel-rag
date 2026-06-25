# Grand Lekki Hotel — RAG Chatbot

A production-grade Retrieval-Augmented Generation (RAG) chatbot for Grand Lekki Hotel. Guests ask questions in natural language; the system retrieves relevant passages from the hotel's documents and generates grounded answers using a locally-running LLM — no OpenAI, no cloud API costs.

---

## How it works

```
Guest question
      |
      ▼
[FastAPI /chat endpoint]
      |
      ▼
[ChromaDB retriever] — finds the 5 most relevant chunks from hotel documents
      |
      ▼
[Prompt template] — inserts chunks as context alongside the question
      |
      ▼
[Ollama / llama3.2] — generates a grounded answer
      |
      ▼
Answer returned to guest
```

---

## Stack

| Layer | Technology |
|---|---|
| API | FastAPI + Uvicorn |
| LLM | Ollama — llama3.2 (chat) |
| Embeddings | Ollama — mxbai-embed-large |
| Vector store | ChromaDB (persistent, local) |
| Chunking | LangChain SemanticChunker |
| Orchestration | LangChain LCEL |
| Observability | LangSmith (optional) |

---

## Project structure

```
Hotel_system/
├── main.py                  # FastAPI app entry point
├── requirements.txt
├── .env                     # Local config (copy from .env.example)
├── .env.example
├── data/
│   ├── grand_lekki_hotel_corpus.pdf   # Source document
│   ├── chroma_db/                     # Vector store (auto-created on first ingest)
│   └── ingested_hashes.json           # Dedup tracker (auto-created on first ingest)
└── app/
    ├── core/
    │   └── config.py        # All settings via pydantic-settings + .env
    ├── rag/
    │   ├── ingestor.py      # PDF load → semantic chunking → Document list
    │   ├── vector_store.py  # ChromaDB wrapper — add docs, get retriever
    │   └── chain.py         # Retriever | Prompt | LLM | Parser
    └── api/
        └── routes.py        # POST /chat and GET /health
```

---

## Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) installed and running locally
- The following models pulled in Ollama:

```bash
ollama pull llama3.2
ollama pull mxbai-embed-large
```

---

## Setup

```bash
# 1. Clone and enter the project
git clone <repo-url>
cd Hotel_system

# 2. Create and activate a virtual environment
python -m venv env
# Windows:
.\env\Scripts\activate
# macOS/Linux:
source env/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env if your Ollama URL or model names differ from the defaults

# 5. Start the server
uvicorn main:app --reload
```

The API will be live at `http://127.0.0.1:8000`.

---

## Ingesting hotel documents

Before the chatbot can answer questions, the hotel PDF must be loaded into ChromaDB. Run the ingestor once:

```python
from app.rag.ingestor import ingest_pdf
from app.rag.vector_store import get_vector_store, add_documents

docs = ingest_pdf("data/grand_lekki_hotel_corpus.pdf")
store = get_vector_store()
add_documents(store, docs)
```

The ingestor tracks file hashes — re-running it on the same file is safe and does nothing.

---

## API reference

### `GET /health`
Returns `200 OK` when the server is running.

```json
{"status": "ok"}
```

### `POST /chat`
Send a guest question, receive a grounded answer.

**Request body:**
```json
{"question": "What time is check-in?"}
```

**Response:**
```json
{"answer": "Check-in is available from 2:00 PM..."}
```

The interactive docs (with a built-in test UI) are available at:
```
http://127.0.0.1:8000/docs
```

---

## Configuration

All settings live in `.env` and are typed in `app/core/config.py`.

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3.2` | Chat model |
| `OLLAMA_EMBEDDING_MODEL` | `mxbai-embed-large` | Embedding model |
| `CHROMA_PERSIST_DIR` | `data/chroma_db` | Where ChromaDB stores vectors |
| `RETRIEVER_K` | `5` | Number of chunks retrieved per query |
| `LANGCHAIN_TRACING_V2` | `false` | Enable LangSmith tracing |
| `LANGCHAIN_API_KEY` | `` | LangSmith API key (if tracing enabled) |
