"""Unit tests for ComparePapersTool."""

import json
from unittest.mock import AsyncMock

from papers_agent.core.models import Chunk, PaperContextBundle, PaperContextItem
from papers_agent.tools.compare_papers import ComparePapersInput, ComparePapersTool


def _make_bundle(paper_ids: list[str]) -> PaperContextBundle:
    items = [
        PaperContextItem(
            paper_id=pid,
            chunks=[
                Chunk(
                    chunk_id=f"{pid}:0001",
                    paper_id=pid,
                    text=f"text from {pid}",
                    section="method",
                    page=1,
                    char_start=0,
                    char_end=15,
                )
            ],
        )
        for pid in paper_ids
    ]
    return PaperContextBundle(items=items)


async def test_compare_returns_structured(mock_llm: AsyncMock) -> None:
    mock_llm.generate.return_value = json.dumps(
        {
            "aspect": "use of tools",
            "perspectives": [
                {
                    "paper_id": "react",
                    "summary": "reasoning + action loop",
                    "key_quotes": ["short quote"],
                },
                {
                    "paper_id": "toolformer",
                    "summary": "fine-tuned tool use",
                    "key_quotes": [],
                },
            ],
            "synthesis": "ReAct interleaves; Toolformer learns offline.",
        }
    )
    tool = ComparePapersTool(mock_llm)
    result = await tool.run(
        ComparePapersInput(
            paper_ids=["react", "toolformer"],
            aspect="use of tools",
            contexts=_make_bundle(["react", "toolformer"]),
        )
    )

    assert result.success
    assert len(result.data.perspectives) == 2
    assert result.metadata["n_papers"] == 2


async def test_compare_failure_on_llm_error(mock_llm: AsyncMock) -> None:
    mock_llm.generate.side_effect = RuntimeError("llm down")
    tool = ComparePapersTool(mock_llm)
    result = await tool.run(
        ComparePapersInput(
            paper_ids=["a", "b"],
            aspect="x" * 5,
            contexts=_make_bundle(["a", "b"]),
        )
    )

    assert not result.success
    assert "llm down" in (result.error or "")
