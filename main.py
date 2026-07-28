import os
import certifi

# On Windows, Python's SSL stack often can't find the system CA bundle.
# Set certifi's bundle before any network-touching imports (LangSmith, HuggingFace).
os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())

# load_dotenv() must be called before any LangChain imports.
# pydantic_settings reads .env into the Settings object, but it does NOT
# write those values into os.environ. LangSmith's tracing is activated by
# os.environ["LANGSMITH_TRACING"], not by our Settings object, so we
# need load_dotenv() here to bridge that gap.
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler

from app.api.routes import router
from app.core.limiter import limiter

app = FastAPI(title="Grand Lekki Hotel Assistant")

# Wire the shared limiter into this app instance. app.state.limiter is where
# slowapi's internals (and the @limiter.limit decorator used in routes.py)
# expect to find it. The exception handler catches RateLimitExceeded and
# turns it into slowapi's standard 429 "Too Many Requests" JSON response
# instead of an unhandled-exception 500.
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.include_router(router)
