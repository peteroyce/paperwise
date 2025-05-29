"""Central configuration loaded from environment."""

from __future__ import annotations
import os
from pathlib import Path

ANTHROPIC_API_KEY: str = os.environ.get("ANTHROPIC_API_KEY", "")
CHROMA_PERSIST_DIR: Path = Path(os.environ.get("CHROMA_PERSIST_DIR", "./chroma_store"))
UPLOAD_DIR: Path = Path(os.environ.get("UPLOAD_DIR", "./uploads"))
CHAT_MODEL: str = os.environ.get("CHAT_MODEL", "claude-3-5-sonnet-20241022")
CHUNK_SIZE: int = int(os.environ.get("CHUNK_SIZE", "512"))
CHUNK_OVERLAP: int = int(os.environ.get("CHUNK_OVERLAP", "64"))
TOP_K: int = int(os.environ.get("TOP_K", "5"))

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
