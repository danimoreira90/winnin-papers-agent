"""Unit tests for RankPapersTool."""

import json
from unittest.mock import AsyncMock

from papers_agent.core.models import Chunk, PaperContextBundle, PaperContextItem
from papers_agent.tools.rank_papers import RankPapersInput, RankPapersTool


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


async def test_rank_resorts_when_out_of_order(mock_llm: AsyncMock) -> None:
    mock_llm.generate.return_value = json.dumps(
        {
            "criterion": "tool use relevance",
            "ranking": [
                {
                    "rank": 2,
                    "paper_id": "toolformer",
                    "justification": "fine-tune",
                },
                {
                    "rank": 1,
                    "paper_id": "react",
                    "justification": "loop pattern",
                },
            ],
        }
    )
    tool = RankPapersTool(mock_llm)
    result = await tool.run(
        RankPapersInput(
            criterion="tool use relevance",
            paper_ids=["react", "toolformer"],
            contexts=_make_bundle(["react", "toolformer"]),
        )
    )

    assert result.success
    assert result.data.ranking[0].rank == 1
    assert result.data.ranking[0].paper_id == "react"


async def test_rank_failure_on_llm_error(mock_llm: AsyncMock) -> None:
    mock_llm.generate.side_effect = RuntimeError("api 503")
    tool = RankPapersTool(mock_llm)
    result = await tool.run(
        RankPapersInput(
            criterion="x" * 5,
            paper_ids=["a", "b"],
            contexts=_make_bundle(["a", "b"]),
        )
    )

    assert not result.success
