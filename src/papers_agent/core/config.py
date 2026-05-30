"""Runtime configuration loaded from environment variables.

Backed by pydantic-settings: the resulting Settings object is frozen
(immutable after construction), tolerates extra env keys, and wraps
credentials in SecretStr so they never leak through repr or logs.
"""

import pathlib
from functools import lru_cache
from typing import Self

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Strongly-typed view over the runtime configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        frozen=True,
        extra="ignore",
        case_sensitive=False,
    )

    # === Gemini ===
    gemini_api_key: SecretStr
    gemini_model: str = "gemini-2.5-flash-lite"
    gemini_embedding_model: str = "gemini-embedding-001"

    # === ChromaDB ===
    chroma_host: str = "localhost"
    chroma_port: int = Field(default=8000, ge=1, le=65535)
    chroma_collection: str = "papers"

    # === Data ===
    pdf_dir: pathlib.Path = pathlib.Path("/app/pdfs")

    # === SQLite ===
    database_url: str = "sqlite+aiosqlite:////data/papers_agent.sqlite"

    # === Ingestion / RAG ===
    chunk_size: int = Field(default=800, ge=1)
    chunk_overlap: int = Field(default=100, ge=0)
    top_k: int = Field(default=5, ge=1, le=20)

    # === API ===
    api_host: str = "0.0.0.0"
    api_port: int = Field(default=8080, ge=1, le=65535)

    # === Logging ===
    log_level: str = "INFO"

    @model_validator(mode="after")
    def _chunk_overlap_must_be_smaller_than_chunk_size(self) -> Self:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be strictly smaller "
                f"than chunk_size ({self.chunk_size})"
            )
        return self


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide Settings singleton, cached after first load."""
    # pydantic-settings loads required fields from the environment, but mypy
    # cannot see through BaseSettings.__init__ and flags gemini_api_key as a
    # missing positional argument. This is the documented escape hatch.
    return Settings()  # type: ignore[call-arg]
