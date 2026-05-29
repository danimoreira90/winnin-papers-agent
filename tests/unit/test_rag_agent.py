"""Unit tests for RAGAgent."""

from unittest.mock import AsyncMock

from papers_agent.agents.rag_agent import RAGAgent
from papers_agent.core.models import Chunk, ToolResult
from papers_agent.tools.extract_section import ExtractSectionOutput
from papers_agent.tools.search_documents import SearchDocumentsOutput


def _chunk(pid: str, text: str, section: str | None = "method") -> Chunk:
    return Chunk(
        chunk_id=f"{pid}:0001",
        paper_id=pid,
        text=text,
        section=section,
        page=1,
        char_start=0,
        char_end=len(text),
    )


async def test_retrieve_context_formats_chunks() -> None:
    search = AsyncMock()
    search.run.return_value = ToolResult(
        tool_name="search_documents",
        success=True,
        data=SearchDocumentsOutput(
            chunks=[_chunk("attention", "self-attention core")], scores=[0.1]
        ),
    )
    extract = AsyncMock()
    agent = RAGAgent(search, extract)
    out = await agent.rag_retrieve_context("attention mechanism")
    assert "[attention]" in out
    assert "self-attention core" in out


async def test_retrieve_context_empty_returns_message() -> None:
    search = AsyncMock()
    search.run.return_value = ToolResult(
        tool_name="search_documents",
        success=True,
        data=SearchDocumentsOutput(chunks=[], scores=[]),
    )
    agent = RAGAgent(search, AsyncMock())
    out = await agent.rag_retrieve_context("nonexistent")
    assert "Nenhum trecho relevante" in out


async def test_extract_section_invalid_params_returns_message() -> None:
    agent = RAGAgent(AsyncMock(), AsyncMock())
    out = await agent.rag_extract_section("attention", "nonexistent_section")
    assert "invalidos" in out.lower() or "invalid" in out.lower()


async def test_extract_section_happy_path() -> None:
    extract = AsyncMock()
    extract.run.return_value = ToolResult(
        tool_name="extract_section",
        success=True,
        data=ExtractSectionOutput(
            paper_id="attention",
            section="abstract",
            text="abstract body here",
            found=True,
            chunks=[],
        ),
    )
    agent = RAGAgent(AsyncMock(), extract)
    out = await agent.rag_extract_section("attention", "abstract")
    assert "## abstract (attention)" in out
    assert "abstract body here" in out
