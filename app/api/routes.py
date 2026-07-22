from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.rag.chain import to_messages, get_rag_chain
from app.rag.retriever import EmptyKnowledgeBaseError
from app.guardrails.exceptions import GuardrailViolation
from app.guardrails.input_guard import check_input
from app.guardrails.output_guard import check_output
from app.memory.session_store import append_turn, create_session, get_history

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
@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Receive a guest question and return an answer grounded in hotel documents."""

    # Input guardrail: rejects empty/too-long questions and obvious prompt-
    # injection attempts before they ever reach the chain. See
    # app/guardrails/input_guard.py for what's checked and why.
    try:
        check_input(request.question)
    except GuardrailViolation as exc:
        raise HTTPException(status_code=400, detail=exc.reason) from exc

    # Session lifecycle: a fresh guest (no session_id sent) gets a brand new
    # session created in Redis. A returning guest's session_id is used as-is
    # to look up whatever history already exists for their conversation.
    session_id = request.session_id or create_session()

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
            {"question": request.question, "history": history}
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
    append_turn(session_id, request.question, answer)

    return ChatResponse(answer=answer, session_id=session_id)
