from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    ollama_embedding_model: str = "mxbai-embed-large"

    # ChromaDB
    chroma_persist_dir: str = str(BASE_DIR / "data" / "chroma_db")
    corpus_path: str = str(BASE_DIR / "data" / "grand_lekki_hotel_corpus.pdf")

    # Chunking
    chunk_size: int = 500
    chunk_overlap: int = 50

    # Retrieval
    retriever_k: int = 5
    reranker_top_n: int = 5
    retrieval_candidates: int = 20

    # Session memory
    session_ttl_seconds: int = 86400  # 24 hours
    redis_url: str = "redis://localhost:6379"

    # LangSmith observability
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "hotel-rag"

    # API security
    api_key: str = "dev-key"
    rate_limit: str = "60/minute"


settings = Settings()
