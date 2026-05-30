"""Unit tests for ThreadRepository.

Uses a fresh aiosqlite file in tmp_path per test for real DB coverage
(avoids the connection-sharing pitfalls of :memory: + async).
"""

import uuid
from collections.abc import AsyncIterator

import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from papers_agent.core.models import Role
from papers_agent.infra.db import Base
from papers_agent.infra.repository import ThreadRepository


@pytest_asyncio.fixture
async def repo(tmp_path: object) -> AsyncIterator[ThreadRepository]:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/t.sqlite")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as session:
        yield ThreadRepository(session)
    await engine.dispose()


async def test_create_and_list_threads(repo: ThreadRepository) -> None:
    t = await repo.create_thread()
    assert t.message_count == 0
    threads = await repo.list_threads()
    assert len(threads) == 1
    assert threads[0].thread_id == t.thread_id


async def test_add_and_list_messages_in_order(repo: ThreadRepository) -> None:
    t = await repo.create_thread()
    await repo.add_message(t.thread_id, Role.USER, "pergunta 1")
    await repo.add_message(t.thread_id, Role.ASSISTANT, "resposta 1")
    msgs = await repo.list_messages(t.thread_id)
    assert [m.role for m in msgs] == [Role.USER, Role.ASSISTANT]
    assert msgs[0].content == "pergunta 1"
    threads = await repo.list_threads()
    assert threads[0].message_count == 2


async def test_thread_exists(repo: ThreadRepository) -> None:
    t = await repo.create_thread()
    assert await repo.thread_exists(t.thread_id) is True
    assert await repo.thread_exists(uuid.uuid4()) is False
