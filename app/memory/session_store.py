import json
from uuid import uuid4

import redis

from app.core.config import settings

# ---------------------------------------------------------------------------
# WHAT THIS FILE IS RESPONSIBLE FOR
# ---------------------------------------------------------------------------
# This module's only job is talking to Redis. It knows nothing about
# LangChain, prompts, or FastAPI — it just stores and returns conversation
# turns for a given session_id. chain.py and routes.py are the callers that
# give this raw history meaning (formatting it into a prompt, accepting it
# over HTTP). Keeping this boundary clean means you could swap Redis for
# another store later without touching the RAG or API layers at all.

# ---------------------------------------------------------------------------
# WINDOW SIZE
# ---------------------------------------------------------------------------
# We're using a "buffer window" strategy (decided over summary-memory):
# keep the last WINDOW_SIZE raw question/answer pairs, drop anything older.
# For a concierge bot, recent turns are what disambiguate a follow-up
# question ("and what about breakfast?") — there's no need to pay for an
# extra LLM call per turn to summarize history we don't need to keep.
WINDOW_SIZE = 5

# ---------------------------------------------------------------------------
# REDIS CLIENT — why a module-level client here is SAFE
# ---------------------------------------------------------------------------
# You already ran into a bug in ingestor.py where a module-level DB
# connection broke at import time, because that connection tried to
# eagerly open a socket the moment the module was loaded — before the app
# even had a chance to configure things.
#
# redis.Redis.from_url() does NOT do that. It's lazy: calling it only
# builds a connection pool object in memory. No actual network connection
# is opened until the first command (get/set/expire/...) is sent. So
# creating the client at import time is safe — if Redis happens to be down,
# nothing breaks until a function below actually tries to use it, at which
# point it raises a normal redis.RedisError that bubbles up like any other
# runtime error (routes.py's existing try/except already turns unexpected
# exceptions into a 500).
#
# decode_responses=True makes Redis return Python str instead of bytes,
# so we don't have to manually .decode() everything we read back.
#
# protocol=2 pins the client to RESP2. redis-py defaults to negotiating
# RESP3 by sending a HELLO command on connect, but the Redis server on this
# machine is the old Microsoft Windows port (5.0.14) — RESP3/HELLO didn't
# exist until Redis 6. Without pinning to RESP2, every connection attempt
# fails with "unknown command HELLO" before a single real command runs.
_client = redis.Redis.from_url(
    settings.redis_url, decode_responses=True, protocol=2
)


def _key(session_id: str) -> str:
    """Build the Redis key for a session's history.

    Namespacing with a "session:" prefix keeps our keys visually distinct
    from any other data that might end up in the same Redis instance later
    (e.g. rate-limit counters in Phase 3 piece 3).
    """
    return f"session:{session_id}"


# ---------------------------------------------------------------------------
# CREATE SESSION
# ---------------------------------------------------------------------------
def create_session() -> str:
    """Create a new, empty conversation session and return its session_id.

    uuid4() generates a random 128-bit ID — for our purposes (a public,
    unguessable-enough identifier a client just echoes back) this is fine;
    we don't need it to be cryptographically unpredictable like an API key.
    """
    session_id = str(uuid4())

    # Store an empty history list as a JSON string. Redis strings can't
    # hold a Python list directly — json.dumps() serializes it to text,
    # json.loads() (in get_history) turns it back into a Python list.
    _client.setex(
        _key(session_id),
        settings.session_ttl_seconds,
        json.dumps([]),
    )

    return session_id


# ---------------------------------------------------------------------------
# GET HISTORY
# ---------------------------------------------------------------------------
def get_history(session_id: str) -> list[dict]:
    """Return the stored turns for a session, oldest first.

    Each turn is a dict like {"question": "...", "answer": "..."}.

    If the session_id doesn't exist — either it was never created, or its
    TTL already expired — we return [] instead of raising. From the RAG
    chain's point of view, "no history" and "unknown session" should behave
    identically: just answer the current question with no prior context.
    """
    raw = _client.get(_key(session_id))

    if raw is None:
        return []

    return json.loads(raw)


# ---------------------------------------------------------------------------
# APPEND TURN
# ---------------------------------------------------------------------------
def append_turn(session_id: str, question: str, answer: str) -> None:
    """Record a new question/answer pair and reset the session's TTL.

    This is the "sliding TTL" behavior: every time a session is actively
    used, its expiry clock resets back to the full session_ttl_seconds.
    A guest mid-conversation never gets logged out; only a session that's
    truly gone quiet for a full TTL window gets cleaned up by Redis.
    """
    history = get_history(session_id)

    history.append({"question": question, "answer": answer})

    # Trim to the last WINDOW_SIZE turns. Negative slicing keeps the most
    # RECENT entries — history[-5:] means "the last 5 items", which is
    # exactly the sliding window we want (oldest turns fall off the front).
    history = history[-WINDOW_SIZE:]

    # setex = SET + EXPIRE in one call. Overwrites the old value and resets
    # the TTL in a single round trip, which is both simpler and avoids a
    # race where another process could read the key in the gap between two
    # separate SET and EXPIRE calls.
    _client.setex(
        _key(session_id),
        settings.session_ttl_seconds,
        json.dumps(history),
    )
