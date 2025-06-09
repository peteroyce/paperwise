"""Root conftest — set dummy env vars before any src module is imported."""
from __future__ import annotations

import os

# Must be set before src.config is imported anywhere in the test session.
os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-test-key-for-pytest")
