import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from pydantic import BaseModel

from app.core.config import settings
from app.core.limiter import limiter
from app.core.security import verify_api_key
from app.rag.chain import to_messages, get_rag_chain
from app.rag.ingestor import ingest_pdf
from app.rag.retriever import EmptyKnowledgeBaseError
from app.rag.vector_store import add_documents, get_vector_store
from app.guardrails.exceptions import GuardrailViolation
from app.guardrails.input_guard import check_input
from app.guardrails.output_guard import check_output
from app.memory.session_store import append_turn, create_session, delete_session, get_history

router = APIRouter()

# Where uploaded PDFs get saved before ingest_pdf() reads them back off disk.
# Lives next to ChromaDB's own data (same parent as chroma_persist_dir), same
# convention ingestor.py already uses for its ingested_hashes.json file.
_UPLOAD_DIR = Path(settings.chroma_persist_dir).parent / "uploads"


# ---------------------------------------------------------------------------
# REQUEST & RESPONSE MODELS
# ---------------------------------------------------------------------------
# Pydantic models define the exact shape of JSON coming in and going out.
# FastAPI uses them to automatically validate requests (returning a clear 422
# error if the client sends the wrong shape) and to generate the /docs schema.
#
# Without these, we'd have to manually parse and validate raw JSON — Pydantic
# handles that for free in exchange for defining the model once.
class ChatRequest(BaseModel):
    question: str
    # None on a guest's first message — the server creates a new session and
    # returns its id. The client must echo that id back on every following
    # call in this same conversation so history can be looked up in Redis.
    session_id: str | None = None


class ChatResponse(BaseModel):
    answer: str
    # Always returned, even on the first call, so the client has something
    # to echo back next time.
    session_id: str


# ---------------------------------------------------------------------------
# HEALTH CHECK
# ---------------------------------------------------------------------------
# A dead-simple endpoint that returns 200 OK when the server is alive.
# Load balancers, Docker health checks, and monitoring tools all hit this
# to decide whether to send real traffic to the instance.
@router.get("/health")
async def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# CHAT ENDPOINT
# ---------------------------------------------------------------------------
# Depends(verify_api_key): FastAPI resolves this before the function body
# runs. A missing/wrong X-API-Key header gets rejected with a 401 here,
# before the guardrails, Redis, or the RAG chain ever see the request.
#
# @limiter.limit(...): slowapi wraps this function; on each call it checks
# the caller's request count (keyed by client IP, see app/core/limiter.py)
# against settings.rate_limit and raises RateLimitExceeded (turned into a
# 429 by the handler registered in main.py) if they're over budget. slowapi
# needs a genuine Starlette `Request` object to read the caller's IP from —
# that's the `request: Request` parameter below. It's unrelated to our
# `ChatRequest` Pydantic model, which is why the JSON body parameter is
# named `payload` instead of `request` — the two would otherwise collide.
@router.post("/chat", response_model=ChatResponse)
@limiter.limit(settings.rate_limit)
async def chat(request: Request, payload: ChatRequest, _=Depends(verify_api_key)):
    """Receive a guest question and return an answer grounded in hotel documents."""

    # Input guardrail: rejects empty/too-long questions and obvious prompt-
    # injection attempts before they ever reach the chain. See
    # app/guardrails/input_guard.py for what's checked and why.
    try:
        check_input(payload.question)
    except GuardrailViolation as exc:
        raise HTTPException(status_code=400, detail=exc.reason) from exc

    # Session lifecycle: a fresh guest (no session_id sent) gets a brand new
    # session created in Redis. A returning guest's session_id is used as-is
    # to look up whatever history already exists for their conversation.
    session_id = payload.session_id or create_session()

    # Pull this session's prior turns (oldest first, already capped to the
    # last WINDOW_SIZE by session_store.append_turn) and convert them from
    # Redis's plain-dict storage format into the BaseMessage objects the
    # chain's MessagesPlaceholder("history") expects.
    history = to_messages(get_history(session_id))

    # get_rag_chain() builds a fresh chain each call (retriever + LLM wired
    # together). invoke() runs the full pipeline — retrieve → prompt → LLM →
    # parse — and returns a plain string answer.
    #
    # We wrap this in try/except so that if Ollama is down or ChromaDB has an
    # issue, the API returns a clean JSON error instead of a raw Python
    # traceback, which would expose internals to the caller.
    #
    # EmptyKnowledgeBaseError is caught separately: it means the system is
    # working correctly but nothing has been ingested yet, which is a 503
    # (temporarily unavailable, retry after ingestion) — not a 500 (something
    # actually broke).
    try:
        answer = get_rag_chain().invoke(
            {"question": payload.question, "history": history}
        )
    except EmptyKnowledgeBaseError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Chain error: {exc}") from exc

    # Output guardrail: catches leaked system-prompt text or an empty reply
    # and swaps in a safe fallback. Never raises — a bad output isn't the
    # guest's fault, so the request still succeeds with a 200.
    answer = check_output(answer)

    # Record this exchange so it's part of "history" on the guest's next
    # call. Done after the guardrail substitution so a fallback message
    # never gets remembered as if the LLM actually said it.
    append_turn(session_id, payload.question, answer)

    return ChatResponse(answer=answer, session_id=session_id)


# ---------------------------------------------------------------------------
# HISTORY ENDPOINTS
# ---------------------------------------------------------------------------
# Both routes sit behind the same Depends(verify_api_key) as /chat. A
# session's history is a guest's conversation content — the same trust
# boundary as the chat itself, so it gets the same protection.
#
# {session_id} in the decorator path is a "path parameter": FastAPI matches
# whatever the client puts in that URL segment (e.g. GET /history/abc-123)
# and passes it into the function as the session_id argument, converting it
# to whatever type you annotate (str here — a session_id is just a UUID
# string, no conversion needed).
@router.get("/history/{session_id}")
async def get_session_history(session_id: str, _=Depends(verify_api_key)):
    """Return the stored turns for a session (oldest first).

    We don't distinguish "session never existed" from "session expired"
    from "session exists but is empty" — get_history() already returns []
    for all three cases (see session_store.py), and that's the right
    behavior here too: there's nothing wrong with asking for history that
    isn't there, so this is always a 200, never a 404.
    """
    return {"session_id": session_id, "history": get_history(session_id)}


@router.delete("/history/{session_id}")
async def delete_session_history(session_id: str, _=Depends(verify_api_key)):
    """Delete a session's history.

    Design choice: this always returns 200, even if there was nothing to
    delete. DELETE is supposed to be idempotent — calling it once or five
    times on the same session_id should leave the world in the same state
    (no history) and should not behave differently based on some race
    between your request and Redis's own TTL expiry. A 404 here would force
    every caller to treat "I deleted it" and "someone/something already beat
    me to it" as different cases, when neither is actually an error.

    We still surface whether a session was actually found via the "deleted"
    field, in case a caller *does* care (e.g. a debugging tool that wants
    to know "was that session_id real?").
    """
    deleted = delete_session(session_id)
    return {"session_id": session_id, "deleted": deleted}


# ---------------------------------------------------------------------------
# INGEST ENDPOINT
# ---------------------------------------------------------------------------
# Lets an admin add a new PDF to the knowledge base over HTTP, instead of the
# old way (manually dropping a file into data/ and running ingest_pdf() by
# hand). Behind the same verify_api_key as everything else — this endpoint
# changes what every guest gets told, so it deserves at least that much
# protection.
@router.post("/ingest")
async def ingest_document(file: UploadFile = File(...), _=Depends(verify_api_key)):
    """Upload a PDF and add its contents to the knowledge base."""

    # Reject non-PDFs early, before we bother saving anything to disk.
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    # ingest_pdf() expects a real file_path on disk, not the in-memory/
    # temp-spooled stream FastAPI gives us as `file`. So step one is writing
    # the upload out to a path we control.
    #
    # We prefix the saved filename with a uuid so two different uploads
    # named e.g. "menu.pdf" never overwrite each other on disk. This is
    # purely about not clobbering a file — it's unrelated to ingest_pdf()'s
    # own duplicate check below, which compares file CONTENT (a hash of the
    # bytes), not the filename.
    _UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest_path = _UPLOAD_DIR / f"{uuid.uuid4()}_{file.filename}"

    contents = await file.read()
    dest_path.write_bytes(contents)

    # ingest_pdf() does the real work: extract text page-by-page, semantically
    # chunk it, return LangChain Documents ready to embed. It can fail in a
    # way that's the CALLER's fault — e.g. a scanned/image-only PDF with no
    # extractable text raises ValueError — so we translate that into a 400
    # instead of letting it look like a server bug. Anything else unexpected
    # (Ollama down, disk issue, ...) falls through to a generic 500, same
    # pattern as the /chat endpoint above.
    try:
        documents = ingest_pdf(str(dest_path))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ingestion error: {exc}") from exc

    # An empty list means ingest_pdf() recognized this exact file content
    # (by hash) as one it already processed before — not an error, just
    # nothing new to add. See the _load_seen_hashes()/_save_seen_hashes()
    # dedup logic in ingestor.py.
    if not documents:
        return {"filename": file.filename, "status": "duplicate", "chunks_added": 0}

    # add_documents() embeds each chunk (calls the Ollama embedding model,
    # one round trip per chunk) and writes both the vector and its metadata
    # into ChromaDB. After this call, the new content is immediately
    # retrievable by /chat.
    store = get_vector_store()
    add_documents(store, documents)

    return {
        "filename": file.filename,
        "status": "ingested",
        "chunks_added": len(documents),
    }
