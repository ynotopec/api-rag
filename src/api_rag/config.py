from __future__ import annotations

import os
from dataclasses import dataclass


def _as_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    openai_api_base: str = os.getenv("OPENAI_API_BASE", "http://localhost:8000/v1")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "changeme")
    api_auth_token: str = os.getenv("API_AUTH_TOKEN", "")

    model_rag: str = os.getenv("MODEL_RAG", "ai-rag")
    upstream_model_rag: str = os.getenv("UPSTREAM_MODEL_RAG", "gpt-4o-mini")

    vectorstore_dir: str = os.getenv("VECTORSTORE_DIR", "vectorstore_db")
    ingestion_paths: str = os.getenv("INGESTION_TEXT_PATHS", "wiki.txt")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "900"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "120"))
    rag_top_k: int = int(os.getenv("RAG_TOP_K", "6"))
    force_rebuild: bool = _as_bool("RAG_FORCE_REBUILD", False)

    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    port: int = int(os.getenv("PORT", "8080"))

    @property
    def chat_completions_url(self) -> str:
        return f"{self.openai_api_base.rstrip('/')}/chat/completions"


SETTINGS = Settings()
