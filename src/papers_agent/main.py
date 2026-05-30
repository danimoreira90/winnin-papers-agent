"""FastAPI application entry point.

Lifespan brings up structured logging, DDL bootstrap, and the agent
graph (composition root). All four /threads* routes plus /health are
mounted here.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from papers_agent.api.dependencies import build_orchestrator
from papers_agent.api.routes import router
from papers_agent.core.config import get_settings
from papers_agent.core.logging import configure_logging, get_logger
from papers_agent.infra.db import init_db

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Init logging, DB schema, and the agent graph; teardown is best-effort."""
    settings = get_settings()
    configure_logging(settings.log_level)
    await init_db()
    app.state.orchestrator = await build_orchestrator(settings)
    log.info("app.startup.done")
    yield


app = FastAPI(title="Papers Agent", lifespan=lifespan)
app.include_router(router)


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe used by Docker HEALTHCHECK and external monitoring."""
    return {"status": "ok"}
