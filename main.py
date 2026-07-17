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
from app.api.routes import router

app = FastAPI(title="Grand Lekki Hotel Assistant")
app.include_router(router)
