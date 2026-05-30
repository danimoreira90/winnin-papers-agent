"""SearchDocumentsTool: one atomic vector search (SR-5).

Embed the query (1 call), query Chroma (1 call), wrap results. No
decisions, no composition - that lives in the RAGAgent (F5).
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from papers_agent.core.logging import get_logger
from papers_agent.core.models import Chunk, ToolResult
from papers_agent.core.ports import EmbeddingClient, VectorStoreClient
from papers_agent.tools.base import Tool

log = get_logger(__name__)


class SearchDocumentsInput(BaseModel):
    """Input for SearchDocumentsTool."""

    model_config = ConfigDict(frozen=True)

    query: str = Field(min_length=1, max_length=1000)
    paper_ids: list[str] | None = None
    top_k: int = Field(default=5, ge=1, le=20)


class SearchDocumentsOutput(BaseModel):
    """Output for SearchDocumentsTool."""

    model_config = ConfigDict(frozen=True)

    chunks: list[Chunk]
    scores: list[float]


class SearchDocumentsTool(Tool[SearchDocumentsInput, SearchDocumentsOutput]):
    """Semantic search over the paper corpus. Returns top_k chunks."""

    name = "search_documents"
    description = "Semantic search over the paper corpus. Returns top_k chunks."
    input_schema = SearchDocumentsInput
    output_schema = SearchDocumentsOutput

    def __init__(self, embedder: EmbeddingClient, vs: VectorStoreClient) -> None:
        self._embedder = embedder
        self._vs = vs

    async def run(self, inp: SearchDocumentsInput) -> ToolResult:
        try:
            vec, embed_ms = await self._measured(self._embedder.embed([inp.query]))
            where: dict[str, Any] | None = None
            if inp.paper_ids:
                where = {"paper_id": {"$in": inp.paper_ids}}
            result, query_ms = await self._measured(
                self._vs.query(embedding=vec[0], top_k=inp.top_k, where=where)
            )
            chunks = result.to_chunks()
            return ToolResult(
                tool_name=self.name,
                success=True,
                data=SearchDocumentsOutput(chunks=chunks, scores=result.distances),
                metadata={
                    "duration_ms": embed_ms + query_ms,
                    "n_chunks": len(chunks),
                },
            )
        except Exception as exc:
            # Tool boundary per PLAN sec.1.7: capture, log, return success=False.
            log.exception("tool.search_documents.error", query=inp.query)
            return ToolResult(
                tool_name=self.name,
                success=False,
                data=SearchDocumentsOutput(chunks=[], scores=[]),
                error=str(exc),
            )
