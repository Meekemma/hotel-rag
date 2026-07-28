# A dedicated exception type for guardrail failures.
#
# Why not just raise a plain Exception or ValueError? Because routes.py needs
# to tell "the guest broke a rule" (400 Bad Request) apart from "something
# actually broke" (500 Internal Server Error). A custom exception class is
# the cleanest way to do that — the except clause in routes.py can catch
# GuardrailViolation specifically, the same way it already catches
# EmptyKnowledgeBaseError specifically (see app/rag/retriever.py).
class GuardrailViolation(Exception):
    """Raised when input or output fails a guardrail check.

    `reason` is safe to show the caller — it should never contain internal
    details (stack traces, file paths, prompt text), only a short, guest-
    facing explanation of what was rejected and why.
    """

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)
