"""Unit tests for SearchDocumentsTool."""

from unittest.mock import AsyncMock

from papers_agent.infra.chroma_client import QueryResult
from papers_agent.tools.search_documents import (
    SearchDocumentsInput,
    SearchDocumentsTool,
)


async def test_search_returns_chunks(
    mock_embedder: AsyncMock, mock_vs: AsyncMock
) -> None:
    mock_vs.query.return_value = QueryResult(
        ids=["attention:0001"],
        distances=[0.12],
        metadatas=[{"paper_id": "attention", "section": "method", "page": 3}],
        documents=["self-attention is the core mechanism..."],
    )
    tool = SearchDocumentsTool(mock_embedder, mock_vs)
    result = await tool.run(SearchDocumentsInput(query="self-attention", top_k=3))

    assert result.success is True
    assert result.tool_name == "search_documents"
    assert len(result.data.chunks) == 1
    assert result.data.chunks[0].paper_id == "attention"
    assert result.data.scores == [0.12]
    assert result.metadata["n_chunks"] == 1
    mock_embedder.embed.assert_awaited_once_with(["self-attention"])


async def test_search_filters_by_paper_ids(
    mock_embedder: AsyncMock, mock_vs: AsyncMock
) -> None:
    mock_vs.query.return_value = QueryResult(
        ids=[], distances=[], metadatas=[], documents=[],
    )
    tool = SearchDocumentsTool(mock_embedder, mock_vs)
    await tool.run(
        SearchDocumentsInput(
            query="anything", paper_ids=["react", "toolformer"], top_k=5
        )
    )

    call = mock_vs.query.await_args
    assert call.kwargs["where"] == {"paper_id": {"$in": ["react", "toolformer"]}}


async def test_search_returns_failure_on_embedder_error(
    mock_embedder: AsyncMock, mock_vs: AsyncMock
) -> None:
    mock_embedder.embed.side_effect = RuntimeError("embedder down")
    tool = SearchDocumentsTool(mock_embedder, mock_vs)
    result = await tool.run(SearchDocumentsInput(query="x"))

    assert result.success is False
    assert "embedder down" in (result.error or "")
    assert result.data.chunks == []
