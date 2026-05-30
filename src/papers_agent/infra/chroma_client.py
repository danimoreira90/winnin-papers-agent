"""ChromaDB async adapter.

Concrete VectorStoreClient implementation backed by chromadb's
AsyncHttpClient (server mode, 1.5.x). The Protocol and the GetResult /
QueryResult value types live in papers_agent.core.ports so consumers
depend on the abstraction, not on this adapter. setup() must be invoked
once before any I/O method; production wiring calls it from the FastAPI
lifespan (T6.5) or from the ingestion script (T3.2).
"""

from typing import Any, cast

import chromadb
from chromadb.api import AsyncClientAPI
from chromadb.api.models.AsyncCollection import AsyncCollection

from papers_agent.core.logging import get_logger
from papers_agent.core.ports import GetResult, QueryResult

log = get_logger(__name__)


class ChromaVectorStore:
    """VectorStoreClient implementation backed by ChromaDB AsyncHttpClient."""

    def __init__(self, host: str, port: int, collection_name: str) -> None:
        self._host = host
        self._port = port
        self._collection_name = collection_name
        self._client: AsyncClientAPI | None = None
        self._collection: AsyncCollection | None = None

    async def setup(self) -> None:
        """Connect to the Chroma server and resolve the collection.

        Idempotent: a second call is a no-op once the collection handle exists.
        """
        if self._collection is not None:
            return
        self._client = await chromadb.AsyncHttpClient(host=self._host, port=self._port)
        self._collection = await self._client.get_or_create_collection(name=self._collection_name)
        log.info(
            "chroma.setup.done",
            host=self._host,
            port=self._port,
            collection=self._collection_name,
        )

    def _require_collection(self, op: str) -> AsyncCollection:
        if self._collection is None:
            raise RuntimeError(f"ChromaVectorStore.setup() must be called before {op}()")
        return self._collection

    async def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        collection = self._require_collection("upsert")
        # Chroma's SDK declares embeddings as Sequence[...] and metadatas as
        # Mapping[str, <strict-union>]; list/dict are structurally compatible
        # but mypy invariance rejects the narrower outer container.
        await collection.upsert(
            ids=ids,
            embeddings=embeddings,  # type: ignore[arg-type]
            documents=documents,
            metadatas=metadatas,  # type: ignore[arg-type]
        )
        log.info("chroma.upsert.done", n=len(ids))

    async def query(
        self,
        embedding: list[float],
        top_k: int,
        where: dict[str, Any] | None = None,
    ) -> QueryResult:
        collection = self._require_collection("query")
        raw = await collection.query(
            query_embeddings=[embedding],  # type: ignore[arg-type]
            n_results=top_k,
            where=where,
        )
        # Chroma batches by query: each field is List[List[...]] with one
        # inner list per query embedding. We always send one embedding, so
        # we narrow with [0]. Default include= covers metadatas/docs/distances,
        # and casts drop the Optional layer the TypedDict carries.
        result = QueryResult(
            ids=raw["ids"][0],
            distances=cast(list[list[float]], raw["distances"])[0],
            metadatas=cast(list[list[dict[str, Any]]], raw["metadatas"])[0],
            documents=cast(list[list[str]], raw["documents"])[0],
        )
        log.info("chroma.query.done", n_results=len(result.ids), top_k=top_k)
        return result

    async def get(self, where: dict[str, Any]) -> GetResult:
        collection = self._require_collection("get")
        # Unlike .query, .get returns flat List[...] (no per-query nesting).
        # Casts strip the Optional layer the TypedDict carries for fields the
        # default include= asks for (metadatas, documents).
        raw = await collection.get(where=where)
        result = GetResult(
            ids=raw["ids"],
            metadatas=cast(list[dict[str, Any]], raw["metadatas"]),
            documents=cast(list[str], raw["documents"]),
        )
        log.info("chroma.get.done", n_results=len(result.ids))
        return result

    async def count(self) -> int:
        collection = self._require_collection("count")
        n: int = await collection.count()
        return n
