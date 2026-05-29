"""Shared fixtures for the unit-test suite."""

from unittest.mock import AsyncMock

import pytest

from papers_agent.infra.chroma_client import VectorStoreClient
from papers_agent.infra.gemini_client import EmbeddingClient, LLMClient


@pytest.fixture
def mock_embedder() -> AsyncMock:
    """Mock EmbeddingClient that returns a single 768-dim vector by default."""
    mock = AsyncMock(spec=EmbeddingClient)
    mock.embed.return_value = [[0.1] * 768]
    return mock


@pytest.fixture
def mock_vs() -> AsyncMock:
    """Mock VectorStoreClient. Tests override return values per case."""
    return AsyncMock(spec=VectorStoreClient)


@pytest.fixture
def mock_llm() -> AsyncMock:
    """Mock LLMClient (used by F4-B analyst tools)."""
    return AsyncMock(spec=LLMClient)
