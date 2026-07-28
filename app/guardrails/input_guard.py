import re

from app.guardrails.exceptions import GuardrailViolation

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
# A hard ceiling on question length. Two reasons this matters, not just one:
#   1. Cost/latency — a 50,000-character question means a 50,000-character
#      prompt, which is slower and more expensive for no real benefit (guest
#      questions are naturally short).
#   2. Abuse surface — extremely long inputs are a common vector for trying
#      to bury an injection attempt where a naive filter won't catch it, or
#      for simple resource-exhaustion abuse against your Ollama server.
MAX_QUESTION_LENGTH = 1000

# ---------------------------------------------------------------------------
# PROMPT INJECTION DENYLIST
# ---------------------------------------------------------------------------
# This is NOT a robust defense against a determined attacker — a small
# keyword/regex list can always be phrased around. What it DOES do cheaply:
#   - Blocks the lazy, copy-pasted jailbreak attempts that are extremely
#     common in the wild ("ignore previous instructions", "you are now DAN").
#   - Costs ~0ms and no extra dependency, unlike an LLM-based classifier
#     which would double your latency and Ollama load on every single
#     request just to screen it.
# Treat this as a cheap first line of defense, not the only one — the
# system prompt in chain.py ("answer using ONLY the context") is the second
# line, and the output guardrail (output_guard.py) is the third.
#
# Patterns are matched case-insensitively against the raw question text.
_INJECTION_PATTERNS = [
    r"ignore (all|any|the)?\s*(previous|prior|above)\s*(instructions?|prompts?)",
    r"disregard (all|any|the)?\s*(previous|prior|above)\s*(instructions?|prompts?)",
    r"reveal (your|the)\s*(system\s*)?prompt",
    r"show (me\s*)?(your|the)\s*(system\s*)?prompt",
    r"what (are|is)\s*your\s*(system\s*)?(instructions?|prompt)",
    r"you are now\s+\w+",  # e.g. "you are now DAN"
    r"act as (if you|though you)? ?(are|were)\s+(an?\s+)?(unrestricted|unfiltered|jailbroken)",
    r"pretend (you have|to have)\s+no\s+(rules|restrictions|guidelines)",
]

# Pre-compile once at import time rather than on every call — re.compile()
# does real work (parsing the pattern into a matching engine), and this
# module only needs to pay that cost once, not on every guest question.
_COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]


def check_input(question: str) -> None:
    """Validate a guest question before it reaches the RAG chain.

    Raises GuardrailViolation if the question should be rejected.
    Returns None (silently) if the question is clean — the caller doesn't
    need a value back, just the guarantee that no exception means "safe to
    proceed."
    """

    # --- Check 1: empty / whitespace-only ---
    # routes.py already checks this today, but once this guardrail is wired
    # in, it becomes the single source of truth for "is this question valid"
    # so the check belongs here too.
    stripped = question.strip()
    if not stripped:
        raise GuardrailViolation("Question cannot be empty.")

    # --- Check 2: length ceiling ---
    if len(stripped) > MAX_QUESTION_LENGTH:
        raise GuardrailViolation(
            f"Question is too long ({len(stripped)} characters). "
            f"Please limit it to {MAX_QUESTION_LENGTH} characters."
        )

    # --- Check 3: prompt injection denylist ---
    # We search (not match) each pattern against the stripped question, so a
    # pattern can hit anywhere in the text, not just at the start.
    for pattern in _COMPILED_PATTERNS:
        if pattern.search(stripped):
            raise GuardrailViolation(
                "Your question couldn't be processed. Please rephrase and "
                "ask about the hotel directly."
            )
