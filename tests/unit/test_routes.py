"""Unit tests for the API route handlers.

Calls each handler coroutine directly with AsyncMock repo + orchestrator.
Bypasses FastAPI's TestClient on purpose: TestClient would spin up the
lifespan, which calls build_orchestrator -> Gemini client + Chroma
connect; the Gemini path lazy-imports google.genai whose module-load
crashes pytest on Windows hosts (T2.6 lazy fix). Direct handler calls
keep the suite deterministic and host-portable.
"""

import datetime
import uuid
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from papers_agent.api.routes import (
    create_thread,
    list_messages,
    list_threads,
    post_message,
)
from papers_agent.api.schemas import PostMessageRequest
from papers_agent.core.models import Message, Role, Thread


def _make_thread() -> Thread:
    return Thread(
        thread_id=uuid.uuid4(),
        created_at=datetime.datetime.now(datetime.UTC),
        message_count=0,
    )


def _make_message(thread_id: uuid.UUID, role: Role, content: str) -> Message:
    return Message(
        message_id=uuid.uuid4(),
        thread_id=thread_id,
        role=role,
        content=content,
        created_at=datetime.datetime.now(datetime.UTC),
    )


async def test_create_thread_returns_id() -> None:
    repo = AsyncMock()
    thread = _make_thread()
    repo.create_thread.return_value = thread

    response = await create_thread(repo)

    assert response.thread_id == thread.thread_id
    repo.create_thread.assert_awaited_once()


async def test_list_threads_returns_all() -> None:
    repo = AsyncMock()
    t1 = _make_thread()
    t2 = _make_thread()
    repo.list_threads.return_value = [t1, t2]

    out = await list_threads(repo)

    assert out == [t1, t2]
    repo.list_threads.assert_awaited_once()


async def test_post_message_persists_then_returns() -> None:
    thread_id = uuid.uuid4()
    body = PostMessageRequest(content="qual o mecanismo central do attention?")
    prior = [
        _make_message(thread_id, Role.USER, "primeiro turno"),
        _make_message(thread_id, Role.ASSISTANT, "primeira resposta"),
    ]
    repo = AsyncMock()
    repo.thread_exists.return_value = True
    repo.list_messages.return_value = prior
    orchestrator = AsyncMock()
    orchestrator.handle.return_value = "self-attention [attention]"

    response = await post_message(thread_id, body, repo, orchestrator)

    # 1. Response shape correct.
    assert response.thread_id == thread_id
    assert response.response == "self-attention [attention]"
    # 2. Orchestrator received PRIOR history (not the brand-new user turn).
    orchestrator.handle.assert_awaited_once_with(body.content, prior)
    # 3. Two persistence calls, in order: USER first, ASSISTANT second.
    assert repo.add_message.await_count == 2
    user_call, assistant_call = repo.add_message.await_args_list
    assert user_call.args == (thread_id, Role.USER, body.content)
    assert assistant_call.args == (
        thread_id,
        Role.ASSISTANT,
        "self-attention [attention]",
    )


async def test_post_message_unknown_thread_404() -> None:
    thread_id = uuid.uuid4()
    body = PostMessageRequest(content="anything")
    repo = AsyncMock()
    repo.thread_exists.return_value = False
    orchestrator = AsyncMock()

    with pytest.raises(HTTPException) as exc_info:
        await post_message(thread_id, body, repo, orchestrator)

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Thread not found"
    orchestrator.handle.assert_not_awaited()
    repo.add_message.assert_not_awaited()
    repo.list_messages.assert_not_awaited()


async def test_list_messages_returns_history() -> None:
    thread_id = uuid.uuid4()
    repo = AsyncMock()
    repo.thread_exists.return_value = True
    msgs = [
        _make_message(thread_id, Role.USER, "a"),
        _make_message(thread_id, Role.ASSISTANT, "b"),
    ]
    repo.list_messages.return_value = msgs

    out = await list_messages(thread_id, repo)

    assert out == msgs


async def test_list_messages_unknown_thread_404() -> None:
    thread_id = uuid.uuid4()
    repo = AsyncMock()
    repo.thread_exists.return_value = False

    with pytest.raises(HTTPException) as exc_info:
        await list_messages(thread_id, repo)

    assert exc_info.value.status_code == 404
    repo.list_messages.assert_not_awaited()
