"""Ingestion pipeline: download 5 PDFs, parse, chunk, embed, upsert into Chroma.

Entry point invoked by `make ingest` via `python -m scripts.ingest`.
Idempotent at the PDF level (skips already-downloaded files) and at the
vector level (Chroma upsert by chunk_id). A post-ingest smoke check
issues one query per paper to confirm retrieval works end-to-end.
"""

import asyncio

import httpx

from papers_agent.core.catalog import get_paper_catalog
from papers_agent.core.config import get_settings
from papers_agent.core.logging import configure_logging, get_logger
from papers_agent.core.models import PaperMetadata
from papers_agent.infra.chroma_client import ChromaVectorStore
from papers_agent.infra.gemini_client import GeminiEmbeddingClient
from papers_agent.infra.pdf_parser import build_chunks, parse_pdf

log = get_logger(__name__)

EMBED_BATCH = 10
_EMBED_BATCH_DELAY_SECONDS = 6.0
ARXIV_PDF_URL = "https://arxiv.org/pdf/{arxiv_id}.pdf"


async def download_pdf(client: httpx.AsyncClient, paper: PaperMetadata) -> None:
    """Fetch the PDF for one paper. Skips if the file already has content."""
    if paper.pdf_path.exists() and paper.pdf_path.stat().st_size > 0:
        log.info(
            "ingest.pdf.skip",
            paper_id=paper.paper_id,
            bytes=paper.pdf_path.stat().st_size,
        )
        return
    paper.pdf_path.parent.mkdir(parents=True, exist_ok=True)
    url = ARXIV_PDF_URL.format(arxiv_id=paper.arxiv_id)
    response = await client.get(url, follow_redirects=True)
    response.raise_for_status()
    paper.pdf_path.write_bytes(response.content)
    log.info(
        "ingest.pdf.downloaded",
        paper_id=paper.paper_id,
        bytes=len(response.content),
    )


async def _ingest_paper(
    paper: PaperMetadata,
    embedder: GeminiEmbeddingClient,
    vs: ChromaVectorStore,
) -> int:
    """Parse one paper, chunk it, embed in batches, upsert. Returns chunk count."""
    pages = await parse_pdf(paper.pdf_path)
    chunks = build_chunks(paper.paper_id, pages)
    if not chunks:
        log.warning("ingest.paper.empty", paper_id=paper.paper_id)
        return 0
    for start in range(0, len(chunks), EMBED_BATCH):
        batch = chunks[start : start + EMBED_BATCH]
        vectors = await embedder.embed([c.text for c in batch])
        await vs.upsert(
            ids=[c.chunk_id for c in batch],
            embeddings=vectors,
            documents=[c.text for c in batch],
            metadatas=[
                {
                    "paper_id": c.paper_id,
                    "section": c.section if c.section is not None else "",
                    "page": c.page if c.page is not None else 0,
                }
                for c in batch
            ],
        )
        # Throttle between batches (not after the last) to stay under the
        # Gemini free-tier ~10 RPM ceiling; the rate-limit retry in
        # gemini_client.py covers any residual 429s.
        if start + EMBED_BATCH < len(chunks):
            log.info("ingest.throttle", seconds=_EMBED_BATCH_DELAY_SECONDS)
            await asyncio.sleep(_EMBED_BATCH_DELAY_SECONDS)
    log.info("ingest.paper.done", paper_id=paper.paper_id, n_chunks=len(chunks))
    return len(chunks)


async def _smoke_check(
    catalog_papers: list[PaperMetadata],
    embedder: GeminiEmbeddingClient,
    vs: ChromaVectorStore,
) -> None:
    """One query per paper; assert at least one hit filtered by paper_id."""
    for paper in catalog_papers:
        probe_text = f"main contribution of {paper.title}"
        vector = (await embedder.embed([probe_text]))[0]
        result = await vs.query(
            embedding=vector,
            top_k=3,
            where={"paper_id": paper.paper_id},
        )
        if not result.ids:
            log.error("ingest.smoke.fail", paper_id=paper.paper_id)
            raise RuntimeError(f"smoke check failed for {paper.paper_id}")
        log.info("ingest.smoke.ok", paper_id=paper.paper_id, top_id=result.ids[0])


async def main() -> None:
    """Run the full ingestion pipeline."""
    settings = get_settings()
    configure_logging(settings.log_level)
    log.info("ingest.start")

    catalog = get_paper_catalog()
    papers = catalog.all()

    async with httpx.AsyncClient(timeout=60.0) as client:
        await asyncio.gather(*(download_pdf(client, p) for p in papers))
    log.info("ingest.downloads.done", n_papers=len(papers))

    vs = ChromaVectorStore(
        host=settings.chroma_host,
        port=settings.chroma_port,
        collection_name=settings.chroma_collection,
    )
    await vs.setup()

    embedder = GeminiEmbeddingClient(
        api_key=settings.gemini_api_key,
        model=settings.gemini_embedding_model,
    )

    for paper in papers:
        await _ingest_paper(paper, embedder, vs)

    await _smoke_check(papers, embedder, vs)

    total = await vs.count()
    log.info("ingest.done", total_chunks=total)


if __name__ == "__main__":
    asyncio.run(main())
