"""Composition root + per-request FastAPI dependencies.

build_orchestrator wires the full object graph (tools, agents, infra
clients) and runs vs.setup(); it is called from the lifespan (T6.5)
exactly once and the result is stashed on app.state. get_orchestrator
hands it back per-request; get_repository scopes a ThreadRepository to
the per-request AsyncSession from infra.db.get_session.
"""

from typing import Annotated, cast

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from papers_agent.agents.analyst_agent import AnalystAgent
from papers_agent.agents.orchestrator import OrchestratorAgent
from papers_agent.agents.rag_agent import RAGAgent
from papers_agent.core.catalog import get_paper_catalog
from papers_agent.core.config import Settings
from papers_agent.infra.chroma_client import ChromaVectorStore
from papers_agent.infra.db import get_session
from papers_agent.infra.gemini_client import GeminiEmbeddingClient, GeminiLLMClient
from papers_agent.infra.repository import ThreadRepository
from papers_agent.tools.compare_papers import ComparePapersTool
from papers_agent.tools.extract_section import ExtractSectionTool
from papers_agent.tools.rank_papers import RankPapersTool
from papers_agent.tools.search_documents import SearchDocumentsTool
from papers_agent.tools.summarize import SummarizeTool


async def build_orchestrator(settings: Settings) -> OrchestratorAgent:
    """Build the full agent graph and connect to Chroma. Lifespan-only."""
    embedder = GeminiEmbeddingClient(
        api_key=settings.gemini_api_key,
        model=settings.gemini_embedding_model,
    )
    llm = GeminiLLMClient(
        api_key=settings.gemini_api_key,
        model=settings.gemini_model,
    )
    vs = ChromaVectorStore(
        host=settings.chroma_host,
        port=settings.chroma_port,
        collection_name=settings.chroma_collection,
    )
    await vs.setup()
    rag = RAGAgent(SearchDocumentsTool(embedder, vs), ExtractSectionTool(vs))
    analyst = AnalystAgent(
        ComparePapersTool(llm),
        SummarizeTool(llm),
        RankPapersTool(llm),
        rag,
        get_paper_catalog(),
    )
    return OrchestratorAgent(
        rag_agent=rag,
        analyst_agent=analyst,
        api_key=settings.gemini_api_key,
        model=settings.gemini_model,
    )


def get_orchestrator(request: Request) -> OrchestratorAgent:
    """Return the singleton orchestrator stashed on app.state by lifespan."""
    return cast(OrchestratorAgent, request.app.state.orchestrator)


async def get_repository(
    session: Annotated[AsyncSession, Depends(get_session)],
) -> ThreadRepository:
    """Per-request ThreadRepository bound to the request's AsyncSession."""
    return ThreadRepository(session)
