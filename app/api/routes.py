from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.rag.chain import get_rag_chain

router = APIRouter()


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


class ChatResponse(BaseModel):
    answer: str


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
@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Receive a guest question and return an answer grounded in hotel documents."""

    # Reject blank questions immediately — the chain would still run but the
    # retriever would return low-quality results for an empty query string.
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # get_rag_chain() builds a fresh chain each call (retriever + LLM wired
    # together). invoke() runs the full pipeline — retrieve → prompt → LLM →
    # parse — and returns a plain string answer.
    #
    # We wrap this in try/except so that if Ollama is down or ChromaDB has an
    # issue, the API returns a clean 500 JSON error instead of a raw Python
    # traceback, which would expose internals to the caller.
    try:
        answer = get_rag_chain().invoke({"question": request.question})
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Chain error: {exc}") from exc

    return ChatResponse(answer=answer)
