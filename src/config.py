"""Central configuration loaded from environment via pydantic-settings."""

from __future__ import annotations

from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    ANTHROPIC_API_KEY: str = ""
    CHROMA_PERSIST_DIR: Path = Path("./chroma_store")
    UPLOAD_DIR: Path = Path("./uploads")
    CHAT_MODEL: str = "claude-3-5-sonnet-20241022"
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64
    TOP_K: int = 5

    @field_validator("ANTHROPIC_API_KEY")
    @classmethod
    def api_key_must_be_set(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError(
                "ANTHROPIC_API_KEY is not set. "
                "Export it as an environment variable or add it to a .env file."
            )
        return v


_settings = Settings()

# Module-level names kept for backward compatibility with existing imports.
ANTHROPIC_API_KEY: str = _settings.ANTHROPIC_API_KEY
CHROMA_PERSIST_DIR: Path = _settings.CHROMA_PERSIST_DIR
UPLOAD_DIR: Path = _settings.UPLOAD_DIR
CHAT_MODEL: str = _settings.CHAT_MODEL
CHUNK_SIZE: int = _settings.CHUNK_SIZE
CHUNK_OVERLAP: int = _settings.CHUNK_OVERLAP
TOP_K: int = _settings.TOP_K

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
