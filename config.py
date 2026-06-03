"""
config.py
---------
Centralised settings for the RAG-Powered Financial Intelligence System.
All values are loaded from environment variables (or a .env file).

Usage:
    from config import settings
    print(settings.faiss_index_path)
"""

import os
from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings — loaded from .env / environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM Provider ──────────────────────────────────────────────────────────
    openai_api_key: Optional[str] = None

    # ── OpenAI settings ───────────────────────────────────────────────────────
    openai_llm_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_max_tokens: int = 512
    openai_temperature: float = 0.1

    # ── Ollama / local settings ────────────────────────────────────────────────
    ollama_model: str = "llama3"
    ollama_base_url: str = "http://localhost:11434"

    # ── HuggingFace embedding model ───────────────────────────────────────────
    hf_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ── FAISS ─────────────────────────────────────────────────────────────────
    faiss_index_path: str = "faiss_index"

    # ── Retrieval ─────────────────────────────────────────────────────────────
    top_k_chunks: int = 5
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # ── FastAPI server ────────────────────────────────────────────────────────
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_reload: bool = True
    app_log_level: str = "info"

    # ── Upload directory ──────────────────────────────────────────────────────
    upload_dir: str = "uploaded_docs"

    # ── Derived helpers ───────────────────────────────────────────────────────
    @property
    def use_openai(self) -> bool:
        """True when an OpenAI API key is present."""
        return bool(self.openai_api_key)

    @property
    def llm_provider(self) -> str:
        return "openai" if self.use_openai else "ollama"

    def ensure_dirs(self) -> None:
        """Create runtime directories if they don't exist."""
        os.makedirs(self.faiss_index_path, exist_ok=True)
        os.makedirs(self.upload_dir, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()


# Module-level convenience alias
settings: Settings = get_settings()
