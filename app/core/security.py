import secrets

from fastapi import Header, HTTPException

from app.core.config import settings


# A FastAPI dependency: when used via Depends(verify_api_key), FastAPI calls
# this function before the route handler runs. If it raises, the handler
# never executes and the client gets the exception's response instead —
# exactly what we want for "reject bad requests before they touch the RAG
# chain."
#
# `x_api_key: str = Header(...)` tells FastAPI to read the `X-API-Key` HTTP
# header (FastAPI converts the snake_case parameter name to kebab-case for
# header lookups) and treat it as required — a request with no header at all
# gets an automatic 422 before this function body even runs.
def verify_api_key(x_api_key: str = Header(...)) -> None:
    # secrets.compare_digest instead of `==`: a plain string comparison
    # returns False the instant it hits the first mismatched character,
    # which makes the comparison take slightly less time for a "more wrong"
    # guess than a "less wrong" one. An attacker who can measure that timing
    # difference could recover the key one character at a time. compare_digest
    # always takes the same amount of time regardless of where (or whether)
    # the strings differ, so no timing signal leaks.
    if not secrets.compare_digest(x_api_key, settings.api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")
