import re

# ---------------------------------------------------------------------------
# WHY THIS FILE EXISTS SEPARATELY FROM input_guard.py
# ---------------------------------------------------------------------------
# input_guard.py decides whether a question is even allowed to reach the
# chain. This file assumes the question WAS allowed and the chain DID run —
# its job is to catch the (rarer, but still real) case where a clean-looking
# question still tricked the LLM into producing something it shouldn't.
#
# Concretely: an attacker can phrase a jailbreak in a way that dodges the
# input denylist entirely (e.g. spelled out across multiple sentences,
# or using synonyms the regex doesn't cover) but the LLM still complies and
# leaks its instructions in the *answer*. That's a failure the input
# guardrail structurally cannot catch, because it never sees the output.
#
# check_output() never raises — unlike check_input(), a bad OUTPUT isn't the
# guest's fault, so there's nothing to reject with a 400. Instead we replace
# the leaked/empty answer with a safe fallback and let the request succeed
# normally from the guest's point of view.

# The literal opening of the system prompt in chain.py. If the model ever
# echoes this back verbatim, it's leaking its own instructions — a classic
# tell that an injection attempt partially succeeded even though the guest's
# original question passed the input guardrail.
_SYSTEM_PROMPT_LEAK_PATTERNS = [
    r"you are a helpful concierge assistant for grand lekki hotel",
    r"answer the guest'?s question using only the context",
]

_COMPILED_LEAK_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in _SYSTEM_PROMPT_LEAK_PATTERNS
]

_FALLBACK_ANSWER = (
    "I'm sorry, I wasn't able to put together an answer for that. "
    "Could you try rephrasing your question, or reach out to the front desk?"
)


def check_output(answer: str) -> str:
    """Inspect an LLM-generated answer before it is returned to the caller.

    Always returns a string — either the original answer (if it's clean) or
    a safe fallback (if it's empty or leaks internal instructions). Never
    raises: a bad output is a system-quality problem, not a guest error, so
    the guest still gets a 200 response, just with a safe message instead of
    the compromised one.
    """

    stripped = answer.strip()

    # --- Check 1: empty answer ---
    # Can happen if the LLM returns nothing meaningful (e.g. just
    # whitespace, which the guest should never see as a "successful" reply).
    if not stripped:
        return _FALLBACK_ANSWER

    # --- Check 2: system prompt leakage ---
    for pattern in _COMPILED_LEAK_PATTERNS:
        if pattern.search(stripped):
            return _FALLBACK_ANSWER

    return stripped
