from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM (Groq)
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    max_tokens: int = 512
    temperature: float = 0.2

    # Embeddings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Qdrant vector store
    # qdrant_mode: "memory" (HF Spaces / CI), "local" (dev), "remote" (Docker/prod)
    qdrant_mode: str = "local"
    qdrant_host: str = ""
    qdrant_port: int = 6333
    qdrant_path: str = "qdrant_storage"
    qdrant_collection: str = "pubmed"

    # Retrieval
    top_k_default: int = 5
    top_k_max: int = 10

    # Chunking
    chunk_size: int = 400
    chunk_overlap: int = 80

    # Confidence thresholds (cosine similarity, 0–1)
    confidence_high: float = 0.70
    confidence_medium: float = 0.50

    # API security — empty string disables authentication
    api_key: str = ""

    # Rate limiting (slowapi format e.g. "30/minute")
    rate_limit: str = "30/minute"

    # Observability
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
