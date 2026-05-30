"""Unit tests for SummarizeTool."""

import json
from unittest.mock import AsyncMock

from papers_agent.core.models import Chunk
from papers_agent.tools.summarize import SummarizeInput, SummarizeTool


async def test_summarize_truncates_to_max_bullets(mock_llm: AsyncMock) -> None:
    mock_llm.generate.return_value = json.dumps(
        {
            "paper_id": "bert",
            "title": "BERT",
            "bullets": ["b1", "b2", "b3", "b4", "b5", "b6", "b7"],
        }
    )
    tool = SummarizeTool(mock_llm)
    ch = Chunk(
        chunk_id="bert:0001",
        paper_id="bert",
        text="x",
        section=None,
        page=1,
        char_start=0,
        char_end=1,
    )
    result = await tool.run(
        SummarizeInput(paper_id="bert", title="BERT", context_chunks=[ch], max_bullets=5)
    )

    assert result.success
    assert len(result.data.bullets) == 5


async def test_summarize_failure_returns_placeholder(mock_llm: AsyncMock) -> None:
    mock_llm.generate.side_effect = RuntimeError("boom")
    tool = SummarizeTool(mock_llm)
    ch = Chunk(
        chunk_id="x:0001",
        paper_id="rag",
        text="x",
        section=None,
        page=1,
        char_start=0,
        char_end=1,
    )
    result = await tool.run(SummarizeInput(paper_id="rag", title="RAG", context_chunks=[ch]))

    assert not result.success
    assert result.data.bullets == ["[summarization failed]"]
