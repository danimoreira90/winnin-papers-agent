"""Ports (abstract interfaces) for outbound dependencies.

Adapters live in infra/; consumers depend on these abstractions, not on
concrete infra -- Dependency Inversion. Tools and agents import the
Protocols and the result types from here; the api/composition root is
the only layer that knows about the concrete adapters.
"""

from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict

from papers_agent.core.models import Chunk


def _chunk_from_raw(chunk_id: str, document: str, metadata: dict[str, Any]) -> Chunk:
    """Rebuild a domain Chunk from one raw vector-store row."""
    return Chunk(
        chunk_id=chunk_id,
        paper_id=metadata["paper_id"],
        text=document,
        section=metadata.get("section"),
        page=metadata.get("page"),
        char_start=0,
        char_end=len(document),
    )


class QueryResult(BaseModel):
    """Wrapped result of an embedding-based query (one query in, top_k out)."""

    model_config = ConfigDict(frozen=True)

    ids: list[str]
    distances: list[float]
    metadatas: list[dict[str, Any]]
    documents: list[str]

    def to_chunks(self) -> list[Chunk]:
        """Rebuild Chunk entities from the query result, in rank order."""
        return [
            _chunk_from_raw(cid, doc, meta)
            for cid, doc, meta in zip(self.ids, self.documents, self.metadatas, strict=True)
        ]


class GetResult(BaseModel):
    """Wrapped result of a metadata-filter fetch (no scoring)."""

    model_config = ConfigDict(frozen=True)

    ids: list[str]
    metadatas: list[dict[str, Any]]
    documents: list[str]

    def to_chunks(self) -> list[Chunk]:
        """Rebuild Chunk entities from the get result."""
        return [
            _chunk_from_raw(cid, doc, meta)
            for cid, doc, meta in zip(self.ids, self.documents, self.metadatas, strict=True)
        ]


class EmbeddingClient(Protocol):
    """Async embedding contract."""

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class LLMClient(Protocol):
    """Async LLM generation contract."""

    async def generate(
        self,
        prompt: str,
        response_schema: type[BaseModel] | None = None,
    ) -> str: ...


class VectorStoreClient(Protocol):
    """Async vector-store contract used by tools and agents."""

    async def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None: ...

    async def query(
        self,
        embedding: list[float],
        top_k: int,
        where: dict[str, Any] | None = None,
    ) -> QueryResult: ...

    async def get(self, where: dict[str, Any]) -> GetResult: ...

    async def count(self) -> int: ...
