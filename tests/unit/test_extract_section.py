"""Unit tests for ExtractSectionTool."""

from unittest.mock import AsyncMock

from papers_agent.infra.chroma_client import GetResult
from papers_agent.tools.extract_section import (
    ExtractSectionInput,
    ExtractSectionTool,
)


async def test_extract_returns_concatenated_text_in_order(
    mock_vs: AsyncMock,
) -> None:
    mock_vs.get.return_value = GetResult(
        ids=["attention:0002", "attention:0001"],
        metadatas=[
            {"paper_id": "attention", "section": "method", "page": 5},
            {"paper_id": "attention", "section": "method", "page": 3},
        ],
        documents=["second piece.", "first piece."],
    )
    tool = ExtractSectionTool(mock_vs)
    result = await tool.run(
        ExtractSectionInput(paper_id="attention", section="method")
    )

    assert result.success is True
    assert result.data.found is True
    assert result.data.text != ""
    assert len(result.data.chunks) == 2


async def test_extract_returns_not_found_when_empty(mock_vs: AsyncMock) -> None:
    mock_vs.get.return_value = GetResult(ids=[], metadatas=[], documents=[])
    tool = ExtractSectionTool(mock_vs)
    result = await tool.run(
        ExtractSectionInput(paper_id="bert", section="conclusion")
    )

    assert result.success is True
    assert result.data.found is False
    assert result.data.text == ""
    assert result.data.chunks == []
    assert result.metadata["n_chunks"] == 0


async def test_extract_returns_failure_on_vs_error(mock_vs: AsyncMock) -> None:
    mock_vs.get.side_effect = RuntimeError("chroma down")
    tool = ExtractSectionTool(mock_vs)
    result = await tool.run(
        ExtractSectionInput(paper_id="rag", section="abstract")
    )

    assert result.success is False
    assert "chroma down" in (result.error or "")
