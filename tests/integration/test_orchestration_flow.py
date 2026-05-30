"""Integration tests for the orchestration chain (SR-27).

These cover Agent -> Tool -> infra(mock) with the REAL graph: tools,
agents, and PaperCatalog wired exactly the way build_orchestrator
registers them. Externals (LLM, embedder, vector store) are
AsyncMock -- zero real calls.

OrchestratorAgent.handle is NOT exercised here: it lazy-imports
google.genai, whose module-load crashes pytest on Windows hosts via
the OpenSSL Applink DLL issue (see T2.6 retroactive lazy-import fix).
The substance handle() dispatches -- the bound methods rag_* and
analyst_* that the SDK registers as tools -- is exactly what these
tests run end to end through the real chain.
"""

import json
import pathlib
from typing import Any
from unittest.mock import AsyncMock

from papers_agent.agents.analyst_agent import AnalystAgent
from papers_agent.agents.rag_agent import RAGAgent
from papers_agent.core.catalog import build_catalog
from papers_agent.core.ports import QueryResult
from papers_agent.tools.compare_papers import ComparePapersTool
from papers_agent.tools.extract_section import ExtractSectionTool
from papers_agent.tools.rank_papers import RankPapersTool
from papers_agent.tools.search_documents import SearchDocumentsTool
from papers_agent.tools.summarize import SummarizeTool


def _query_side_effect(**kwargs: Any) -> QueryResult:
    """Return a per-paper canned QueryResult derived from the where filter."""
    where = kwargs.get("where")
    paper_id = where["paper_id"]["$in"][0] if where and "paper_id" in where else "attention"
    return QueryResult(
        ids=[f"{paper_id}:0001"],
        distances=[0.12],
        metadatas=[{"paper_id": paper_id, "section": "method", "page": 1}],
        documents=[f"core mechanism from {paper_id}: self-attention layers"],
    )


def _build_graph() -> tuple[RAGAgent, AnalystAgent, AsyncMock, AsyncMock, AsyncMock]:
    """Wire the real graph against mocked infra clients."""
    embedder = AsyncMock()
    embedder.embed.return_value = [[0.1] * 768]
    vs = AsyncMock()
    vs.query.side_effect = _query_side_effect
    llm = AsyncMock()
    search = SearchDocumentsTool(embedder, vs)
    extract = ExtractSectionTool(vs)
    compare = ComparePapersTool(llm)
    summarize = SummarizeTool(llm)
    rank = RankPapersTool(llm)
    rag = RAGAgent(search, extract)
    catalog = build_catalog(pathlib.Path("/tmp"))
    analyst = AnalystAgent(compare, summarize, rank, rag, catalog)
    return rag, analyst, embedder, vs, llm


async def test_compare_flow_agent_to_tool_to_infra() -> None:
    """analyst_compare_papers exercises AnalystAgent -> RAG fetch ->
    SearchDocumentsTool -> vs mock; then ComparePapersTool -> llm mock."""
    _rag, analyst, _embedder, vs, llm = _build_graph()
    llm.generate.return_value = json.dumps(
        {
            "aspect": "mecanismo de atencao",
            "perspectives": [
                {
                    "paper_id": "attention",
                    "summary": "self-attention nucleo do transformer",
                    "key_quotes": [],
                },
                {
                    "paper_id": "bert",
                    "summary": "atencao bidirecional sobre tokens",
                    "key_quotes": [],
                },
            ],
            "synthesis": (
                "Attention propoe self-attention; BERT a aplica em encoders bidirecionais."
            ),
        }
    )

    out = await analyst.analyst_compare_papers(["attention", "bert"], "mecanismo de atencao")

    assert "[attention]" in out
    assert "[bert]" in out
    assert "Sintese" in out
    # The vector store was hit once per paper_id.
    assert vs.query.await_count == 2
    # The LLM was called exactly once by the compare tool.
    assert llm.generate.await_count == 1


async def test_retrieve_context_flow_agent_to_tool_to_infra() -> None:
    """rag_retrieve_context exercises RAGAgent -> SearchDocumentsTool ->
    embedder/vs mocks, then formats chunks as markdown."""
    rag, _analyst, embedder, vs, _llm = _build_graph()

    out = await rag.rag_retrieve_context("attention mechanism", top_k=3)

    assert "[attention]" in out
    assert "self-attention" in out
    assert embedder.embed.await_count == 1
    assert vs.query.await_count == 1
