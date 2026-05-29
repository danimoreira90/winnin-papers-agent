"""Unit tests for AnalystAgent.

Catalog uses MagicMock (sync methods); RAGAgent and tools use AsyncMock.
"""

from unittest.mock import AsyncMock, MagicMock

from papers_agent.agents.analyst_agent import AnalystAgent
from papers_agent.core.models import Chunk, ToolResult
from papers_agent.tools.summarize import SummarizeOutput


def _chunk(pid: str) -> Chunk:
    return Chunk(
        chunk_id=f"{pid}:0001",
        paper_id=pid,
        text=f"text {pid}",
        section="abstract",
        page=1,
        char_start=0,
        char_end=6,
    )


async def test_summarize_paper_happy_path() -> None:
    rag = AsyncMock()
    rag.fetch_section_chunks.return_value = [_chunk("bert")]
    rag.fetch_chunks_for_query.return_value = [_chunk("bert")]
    summarize = AsyncMock()
    summarize.run.return_value = ToolResult(
        tool_name="summarize",
        success=True,
        data=SummarizeOutput(paper_id="bert", title="BERT", bullets=["b1", "b2"]),
    )
    catalog = MagicMock()
    catalog.get_title.return_value = "BERT"
    agent = AnalystAgent(AsyncMock(), summarize, AsyncMock(), rag, catalog)
    out = await agent.analyst_summarize_paper("bert")
    assert "BERT" in out
    assert "b1" in out


async def test_summarize_paper_unknown_id() -> None:
    catalog = MagicMock()
    catalog.get_title.side_effect = KeyError("nope")
    agent = AnalystAgent(
        AsyncMock(), AsyncMock(), AsyncMock(), AsyncMock(), catalog
    )
    out = await agent.analyst_summarize_paper("xyz")
    assert "desconhecido" in out.lower()
