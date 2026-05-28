"""FastAPI application shell for the papers-agent service.

The lifespan hook stays a no-op at this stage; database init, repository
wiring, and router registration land in T6.5 once those modules exist.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Lifespan hook - init/dispose will arrive in T6.5 (db init, etc)."""
    yield


app = FastAPI(title="Papers Agent", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe used by Docker HEALTHCHECK and external monitoring."""
    return {"status": "ok"}
