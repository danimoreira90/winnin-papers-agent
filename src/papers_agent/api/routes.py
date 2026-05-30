"""HTTP routes: 4 endpoints (threads + messages).

Boundary layer: catches 404 for unknown thread_ids and lets all other
errors propagate to FastAPI's default exception handlers (which logs +
returns 500). Per-request DB session and orchestrator come via Depends.
"""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from papers_agent.agents.orchestrator import OrchestratorAgent
from papers_agent.api.dependencies import get_orchestrator, get_repository
from papers_agent.api.schemas import (
    CreateThreadResponse,
    PostMessageRequest,
    PostMessageResponse,
)
from papers_agent.core.logging import get_logger
from papers_agent.core.models import Message, Role, Thread
from papers_agent.infra.repository import ThreadRepository

log = get_logger(__name__)
router = APIRouter()

RepoDep = Annotated[ThreadRepository, Depends(get_repository)]
OrchestratorDep = Annotated[OrchestratorAgent, Depends(get_orchestrator)]


@router.post("/threads", response_model=CreateThreadResponse, status_code=201)
async def create_thread(repo: RepoDep) -> CreateThreadResponse:
    thread = await repo.create_thread()
    log.info("api.thread.created", thread_id=str(thread.thread_id))
    return CreateThreadResponse(thread_id=thread.thread_id)


@router.get("/threads", response_model=list[Thread])
async def list_threads(repo: RepoDep) -> list[Thread]:
    return await repo.list_threads()


@router.post("/threads/{thread_id}/messages", response_model=PostMessageResponse)
async def post_message(
    thread_id: uuid.UUID,
    body: PostMessageRequest,
    repo: RepoDep,
    orchestrator: OrchestratorDep,
) -> PostMessageResponse:
    if not await repo.thread_exists(thread_id):
        raise HTTPException(status_code=404, detail="Thread not found")
    # history must reflect prior turns only; persist the new user message
    # after capturing it so it does not show up as a previous turn.
    history = await repo.list_messages(thread_id)
    await repo.add_message(thread_id, Role.USER, body.content)
    answer = await orchestrator.handle(body.content, history)
    await repo.add_message(thread_id, Role.ASSISTANT, answer)
    log.info(
        "api.message.handled",
        thread_id=str(thread_id),
        prior_turns=len(history),
        answer_chars=len(answer),
    )
    return PostMessageResponse(thread_id=thread_id, response=answer)


@router.get("/threads/{thread_id}/messages", response_model=list[Message])
async def list_messages(thread_id: uuid.UUID, repo: RepoDep) -> list[Message]:
    if not await repo.thread_exists(thread_id):
        raise HTTPException(status_code=404, detail="Thread not found")
    return await repo.list_messages(thread_id)
